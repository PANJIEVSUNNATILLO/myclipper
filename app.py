import math
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from faster_whisper import WhisperModel
import streamlit as st
import yt_dlp

# --- Page Configuration ---
st.set_page_config(page_title="ClipForge AI", page_icon="🎬", layout="wide")

st.title("🎬 ClipForge AI")
st.subheader("Turn long videos into short vertical clips with subtitles")

# --- Initialize Session State ---
if "processed_clips" not in st.session_state:
    st.session_state.processed_clips = []
if "zip_data" not in st.session_state:
    st.session_state.zip_data = None


# --- Cache Whisper Model in Memory ---
@st.cache_resource
def load_whisper_model():
    """Loads the quantized tiny model (uses only ~150MB RAM and runs fast on CPU)."""
    return WhisperModel("tiny", device="cpu", compute_type="int8")


# --- Helper Functions ---
def run_command(command, timeout=300, cwd=None):
    return subprocess.run(
        command, capture_output=True, text=True, timeout=timeout, cwd=cwd
    )


def download_youtube_video(url, output_dir):
    """Downloads video anonymously via the Android client without IP-pinned cookies."""
    output_template = str(output_dir / "source.%(ext)s")
    ffmpeg_bin = shutil.which("ffmpeg")

    ydl_opts = {
        # Select best 720p or fallback
        "format": "best[height<=720]/bestvideo[height<=720]+bestaudio/best",
        "outtmpl": output_template,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        # Force Android client which does NOT require cookies or IP matching
        "extractor_args": {
            "youtube": {
                "player_client": ["android"],
            }
        },
        "http_headers": {
            "User-Agent": (
                "com.google.android.youtube/19.29.37 (Linux; U; Android 14; US)"
                " gzip"
            ),
        },
    }

    if ffmpeg_bin:
        ydl_opts["ffmpeg_location"] = ffmpeg_bin

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Check length first to protect cloud RAM
            info = ydl.extract_info(url, download=False)
            duration = info.get("duration", 0)

            # Limit long streams (> 15 mins) to first 10 mins
            if duration and duration > 900:
                st.info(
                    f"⚠️ Long video/stream detected ({int(duration // 60)} mins). "
                    "Downloading the first 10 minutes to stay within server limits."
                )

                def download_slice(info_dict, ydl_instance):
                    return [{"start_time": 0, "end_time": 600}]

                ydl_opts["download_ranges"] = download_slice

        # Execute download
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

    except Exception as e:
        raise RuntimeError(
            f"YouTube download failed: {e}\n\n"
            "Because YouTube frequently blocks cloud hosting providers (AWS), "
            "please use the '📁 Upload Video File' tab to process your video."
        )

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
    """Extracts a lightweight mono audio track for volume scanning."""
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


def format_srt_time(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def generate_subtitles(clip_audio_path, srt_output_path, whisper_model):
    """Transcribes audio using faster-whisper and writes standard SRT."""
    segments, _ = whisper_model.transcribe(
        str(clip_audio_path), beam_size=1, word_timestamps=False
    )

    srt_content = []
    for idx, segment in enumerate(segments, start=1):
        start_time = format_srt_time(segment.start)
        end_time = format_srt_time(segment.end)
        text = segment.text.strip()
        if text:
            srt_content.append(f"{idx}\n{start_time} --> {end_time}\n{text}\n")

    srt_output_path.write_text("\n".join(srt_content), encoding="utf-8")
    return len(srt_content) > 0


def create_vertical_clip(
    video_path,
    output_path,
    start,
    length,
    work_dir,
    whisper_model=None,
    add_subs=True,
):
    """Crops to 9:16 vertical and burns in TikTok-style subtitles."""
    srt_filename = f"{output_path.stem}.srt"
    srt_path = work_dir / srt_filename
    temp_clip_audio = work_dir / f"{output_path.stem}_audio.wav"

    has_subtitles = False
    if add_subs and whisper_model:
        # Extract audio slice for transcription
        run_command([
            "ffmpeg",
            "-y",
            "-ss",
            str(start),
            "-i",
            str(video_path),
            "-t",
            str(length),
            "-vn",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(temp_clip_audio),
        ])

        if temp_clip_audio.exists():
            has_subtitles = generate_subtitles(
                temp_clip_audio, srt_path, whisper_model
            )

    # Build video filter chain: 9:16 Crop + Optional Subtitle Burn
    vf_chain = (
        "scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280"
    )

    if has_subtitles:
        # Style: Yellow font, black outline, elevated 150px safe zone
        sub_style = (
            f"subtitles={srt_filename}:force_style="
            "'FontName=Arial,FontSize=20,Bold=1,"
            "PrimaryColour=&H0000FFFF,OutlineColour=&H00000000,BorderStyle=1,"
            "Outline=2,Alignment=2,MarginV=150'"
        )
        vf_chain += f",{sub_style}"

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
            vf_chain,
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
        cwd=str(work_dir),  # Guarantees FFmpeg subtitle filter finds srt_filename cleanly
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg encoding error: {result.stderr[-1000:] if result.stderr else 'Unknown'}"
        )


# --- User Interface ---
tab_upload, tab_url = st.tabs(["📁 Upload Video File", "🔗 Video URL"])

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a video file (MP4, MOV, MKV)",
        type=["mp4", "mov", "mkv"],
    )

with tab_url:
    video_url = st.text_input(
        "Paste YouTube or video link",
        placeholder="https://www.youtube.com/watch?v=...",
    )

col1, col2, col3 = st.columns(3)
with col1:
    clip_length = st.selectbox(
        "Clip duration (seconds)", [15, 30, 45, 60], index=1
    )
with col2:
    number_of_clips = st.slider("Number of clips", 1, 4, 2)
with col3:
    enable_subtitles = st.checkbox(
        "✨ Auto-generate Subtitles",
        value=True,
        help="Transcribes speech and burns captions into the vertical clip.",
    )

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
            "Please ensure you created a 'packages.txt' file with 'ffmpeg'."
        )
    elif uploaded_file is None and not video_url.strip():
        st.warning("Please upload a file or paste a video link.")
    else:
        try:
            whisper_model = (
                load_whisper_model() if enable_subtitles else None
            )

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

                # 4. Generate vertical clips + subtitles
                generated_clips = []
                clip_progress = st.progress(
                    0, text="Rendering vertical clips & captions..."
                )

                for idx, (score, start_sec) in enumerate(highlights, 1):
                    clip_progress.progress(
                        (idx - 1) / len(highlights),
                        text=f"Rendering clip {idx}/{len(highlights)} (with captions)...",
                    )
                    filename = f"clip_{idx}.mp4"
                    output_file = work_dir / filename

                    create_vertical_clip(
                        source_video,
                        output_file,
                        start_sec,
                        clip_length,
                        work_dir,
                        whisper_model=whisper_model,
                        add_subs=enable_subtitles,
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

                st.session_state.processed_clips = generated_clips
                st.session_state.zip_data = zip_path.read_bytes()

        except subprocess.TimeoutExpired:
            st.error("Operation timed out. Please try a shorter video.")
        except Exception as err:
            st.error(f"Error: {err}")

# --- Render Results ---
if st.session_state.processed_clips:
    st.divider()
    st.success(
        f"Generated {len(st.session_state.processed_clips)} vertical clips with subtitles!"
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
