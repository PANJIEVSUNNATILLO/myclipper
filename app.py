import math
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
import streamlit as st
import yt_dlp

# --- Page Configuration ---
st.set_page_config(page_title="ClipForge AI", page_icon="🎬", layout="wide")

st.title("🎬 ClipForge AI")
st.subheader("Turn long videos into short vertical clips")

# --- Initialize Session State ---
if "processed_clips" not in st.session_state:
    st.session_state.processed_clips = []
if "zip_data" not in st.session_state:
    st.session_state.zip_data = None


# --- Helper Functions ---
def run_command(command, timeout=300):
    return subprocess.run(
        command, capture_output=True, text=True, timeout=timeout
    )


def download_youtube_video(url, output_dir):
    """Downloads video using android/mweb clients and 10MB chunking to bypass CDN 403 blocks."""
    output_template = str(output_dir / "source.%(ext)s")
    cookie_file = None
    ffmpeg_bin = shutil.which("ffmpeg")

    ydl_opts = {
        # Prioritize 720p/MP4 without strictly rejecting other formats
        "format_sort": ["res:720", "ext:mp4:m4a"],
        "outtmpl": output_template,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "hls_prefer_native": True,
        # 10MB chunking prevents googlevideo CDN connection drops on cloud servers
        "http_chunk_size": 10485760,
        # Use mobile clients that do not enforce strict desktop cookie IP-binding
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "mweb"],
            }
        },
    }

    if ffmpeg_bin:
        ydl_opts["ffmpeg_location"] = ffmpeg_bin

    # Load cookies from Streamlit Secrets if configured
    if "YOUTUBE_COOKIES" in st.secrets:
        cookie_file = output_dir / "cookies.txt"
        cookie_file.write_text(st.secrets["YOUTUBE_COOKIES"])
        ydl_opts["cookiefile"] = str(cookie_file)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Check length first to protect cloud container RAM and disk
            info = ydl.extract_info(url, download=False)
            duration = info.get("duration", 0)

            # If video/stream is longer than 15 mins, download only first 10 mins
            if duration and duration > 900:
                st.info(
                    f"⚠️ Long video/stream detected ({int(duration // 60)} minutes). "
                    "Downloading the first 10 minutes to stay within cloud server memory limits."
                )

                def download_slice(info_dict, ydl_instance):
                    return [{"start_time": 0, "end_time": 600}]

                ydl_opts["download_ranges"] = download_slice

        # Download with finalized options
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

    except Exception as e:
        raise RuntimeError(
            f"YouTube download failed: {e}\n\n"
            "If platform anti-bot blocks persist, please switch to the '📁 Upload Video File' tab."
        )
    finally:
        if cookie_file and cookie_file.exists():
            cookie_file.unlink()

    files = list(output_dir.glob("source.*"))
    if not files:
        raise RuntimeError("Download completed but no video file was saved.")

    return max(files, key=lambda f: f.stat().st_size)


def get_duration(video_path):
    result = run_command([
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ])
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError("Could not detect video duration.")
    return float(result.stdout.strip())


def extract_audio_track(video_path, output_audio_path):
    """Extracts a lightweight mono audio track for fast volume scanning."""
    run_command([
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-b:a",
        "64k",
        str(output_audio_path),
    ])


def analyze_audio_section(audio_path, start, length):
    result = run_command([
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-ss",
        str(start),
        "-i",
        str(audio_path),
        "-t",
        str(length),
        "-af",
        "volumedetect",
        "-f",
        "null",
        "-",
    ])
    match = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", result.stderr)
    return float(match.group(1)) if match else -100.0


def find_highlights(audio_path, duration, length, count):
    if duration < length:
        raise ValueError("Video is shorter than your selected clip duration.")

    step = max(length // 2, math.ceil((duration - length) / 40))
    starts = list(range(0, int(duration - length) + 1, step))

    last_start = max(0, int(duration - length))
    if last_start not in starts:
        starts.append(last_start)

    scored = []
    progress_bar = st.progress(0, text="Scanning for high-energy moments...")

    for i, start in enumerate(starts):
        score = analyze_audio_section(audio_path, start, length)
        scored.append((score, start))
        progress_bar.progress(
            (i + 1) / len(starts), text=f"Analyzing section {i + 1}/{len(starts)}"
        )

    progress_bar.empty()
    scored.sort(reverse=True, key=lambda x: x[0])

    selected = []
    for score, start in scored:
        if all(
            abs(start - other_start) >= length * 0.75
            for _, other_start in selected
        ):
            selected.append((score, start))
        if len(selected) >= count:
            break

    return selected


def create_vertical_clip(video_path, output_path, start, length):
    """Crops 16:9 into 9:16 vertical (720x1280) with ultrafast encoding."""
    result = run_command(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(start),
            "-i",
            str(video_path),
            "-t",
            str(length),
            "-vf",
            "scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "26",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        timeout=300,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg encoding error: {result.stderr[-1000:] if result.stderr else 'Unknown'}"
        )


# --- User Interface Tabs ---
tab_url, tab_upload = st.tabs(["🔗 Video URL", "📁 Upload Video File"])

with tab_url:
    video_url = st.text_input(
        "Paste YouTube or video URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a video directly (100% reliable fallback)",
        type=["mp4", "mov", "mkv"],
    )

col1, col2 = st.columns(2)
with col1:
    clip_length = st.selectbox(
        "Clip duration (seconds)", [15, 30, 45, 60], index=1
    )
with col2:
    number_of_clips = st.slider("Number of clips to generate", 1, 4, 2)

process_btn = st.button(
    "🚀 Generate Clips", type="primary", use_container_width=True
)

# --- Execution Workflow ---
if process_btn:
    missing_tools = [
        tool for tool in ["ffmpeg", "ffprobe"] if shutil.which(tool) is None
    ]

    if missing_tools:
        st.error(
            f"Missing system tools: {', '.join(missing_tools)}. "
            "Please ensure you created a 'packages.txt' file with 'ffmpeg' in your repository."
        )
    elif not video_url.strip() and uploaded_file is None:
        st.warning("Please provide a video link or upload a file.")
    else:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                work_dir = Path(temp_dir)
                source_video = None

                # 1. Source acquisition
                if uploaded_file is not None:
                    source_video = work_dir / f"source_{uploaded_file.name}"
                    with open(source_video, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                else:
                    with st.spinner("Downloading video..."):
                        source_video = download_youtube_video(
                            video_url.strip(), work_dir
                        )

                # 2. Extract audio and get length
                with st.spinner("Analyzing audio levels..."):
                    audio_track = work_dir / "temp_audio.aac"
                    extract_audio_track(source_video, audio_track)
                    duration = get_duration(source_video)

                # 3. Detect Highlights
                with st.spinner("Locating peak moments..."):
                    highlights = find_highlights(
                        audio_track, duration, clip_length, number_of_clips
                    )

                if not highlights:
                    raise RuntimeError("No highlights detected in this video.")

                # 4. Generate vertical clips
                generated_clips = []
                clip_progress = st.progress(0, text="Rendering vertical clips...")

                for idx, (score, start_sec) in enumerate(highlights, 1):
                    clip_progress.progress(
                        (idx - 1) / len(highlights),
                        text=f"Rendering clip {idx}/{len(highlights)}...",
                    )
                    filename = f"clip_{idx}.mp4"
                    output_file = work_dir / filename

                    create_vertical_clip(
                        source_video, output_file, start_sec, clip_length
                    )

                    with open(output_file, "rb") as f:
                        generated_clips.append({
                            "name": filename,
                            "start": start_sec,
                            "bytes": f.read(),
                        })

                clip_progress.empty()

                # 5. Build ZIP archive
                zip_path = work_dir / "clipforge_clips.zip"
                with zipfile.ZipFile(
                    zip_path, "w", zipfile.ZIP_DEFLATED
                ) as archive:
                    for clip in generated_clips:
                        archive.writestr(clip["name"], clip["bytes"])

                # Save to session_state so interaction doesn't wipe results
                st.session_state.processed_clips = generated_clips
                st.session_state.zip_data = zip_path.read_bytes()

        except subprocess.TimeoutExpired:
            st.error("Operation timed out. Please try a shorter video.")
        except Exception as err:
            st.error(f"Error: {err}")

# --- Render Results (Persisted Across Button Clicks) ---
if st.session_state.processed_clips:
    st.divider()
    st.success(
        f"Generated {len(st.session_state.processed_clips)} vertical clips!"
    )

    if st.session_state.zip_data:
        st.download_button(
            "⬇️ Download All Clips (ZIP)",
            data=st.session_state.zip_data,
            file_name="clipforge_clips.zip",
            mime="application/zip",
            use_container_width=True,
        )

    cols = st.columns(len(st.session_state.processed_clips))
    for i, clip in enumerate(st.session_state.processed_clips):
        with cols[i]:
            timestamp = f"{int(clip['start'] // 60):02d}:{int(clip['start'] % 60):02d}"
            st.markdown(f"**Clip {i+1}** (starts at {timestamp})")
            st.video(clip["bytes"])
            st.download_button(
                f"Download Clip {i+1}",
                data=clip["bytes"],
                file_name=clip["name"],
                mime="video/mp4",
                key=f"dl_single_clip_{i}",
            )

st.divider()
st.caption(
    "ClipForge AI | Only process videos you own or have permission to use."
)
