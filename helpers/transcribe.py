"""Transcribe a video with ElevenLabs Scribe, or free local Whisper when no key is set.

Extracts mono 16kHz audio via ffmpeg, uploads to Scribe with verbatim +
diarize + audio events + word-level timestamps, writes the full response
to <edit_dir>/transcripts/<video_stem>.json.

Cached: if the output file already exists, the upload is skipped.

Usage:
    python helpers/transcribe.py <video_path>
    python helpers/transcribe.py <video_path> --edit-dir /custom/edit
    python helpers/transcribe.py <video_path> --language en
    python helpers/transcribe.py <video_path> --num-speakers 2
"""

from __future__ import annotations

import argparse
import array
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import requests


SCRIBE_URL = "https://api.elevenlabs.io/v1/speech-to-text"


def load_api_key() -> str:
    for candidate in [Path(__file__).resolve().parent.parent / ".env", Path(".env")]:
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "ELEVENLABS_API_KEY":
                    return v.strip().strip('"').strip("'")
    # No key → "" and transcribe_one falls back to free local Whisper (mlx-whisper).
    return os.environ.get("ELEVENLABS_API_KEY", "")


def count_audio_tracks(video_path: Path) -> int:
    """How many audio streams the container holds."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True,
    )
    return len([ln for ln in out.stdout.splitlines() if ln.strip()])


def peak_dbfs(wav_path: Path) -> float:
    """Peak level of a 16-bit PCM wav, in dBFS. -inf for digital silence."""
    peak = 0
    with wave.open(str(wav_path), "rb") as w:
        # A chunk at a time: batch mode runs several of these at once, and a two-hour
        # take is 230 MB of 16 kHz mono before the array copy doubles it.
        while frames := w.readframes(1 << 16):
            samples = array.array("h", frames)
            peak = max(peak, max(samples), -min(samples))
    return 20 * math.log10(peak / 32768) if peak > 0 else float("-inf")


def extract_audio(video_path: Path, dest: Path, audio_track: int = 0) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-map", f"0:a:{audio_track}",
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(dest),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def call_scribe(
    audio_path: Path,
    api_key: str,
    language: str | None = None,
    num_speakers: int | None = None,
) -> dict:
    data: dict[str, str] = {
        "model_id": "scribe_v1",
        "diarize": "true",
        "tag_audio_events": "true",
        "timestamps_granularity": "word",
    }
    if language:
        data["language_code"] = language
    if num_speakers:
        data["num_speakers"] = str(num_speakers)

    with open(audio_path, "rb") as f:
        resp = requests.post(
            SCRIBE_URL,
            headers={"xi-api-key": api_key},
            files={"file": (audio_path.name, f, "audio/wav")},
            data=data,
            timeout=1800,
        )

    if resp.status_code != 200:
        raise RuntimeError(f"Scribe returned {resp.status_code}: {resp.text[:500]}")

    return resp.json()


LOCAL_MODEL = os.environ.get("VIDEO_USE_WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")
# Whisper drops disfluencies unless the prompt shows them; this nudges it verbatim so
# filler-word cuts still work without Scribe.
VERBATIM_PROMPT = "Umm, uh, like, you know."


# Whisper's stock outputs on silence — YouTube outro lines it memorised in many languages.
STOCK_HALLUCINATIONS = (
    "thanks for watching", "thank you for watching", "see you next time", "see you in the next",
    "please subscribe", "subtitles by", "subtitled by", "transcribed by", "amara.org",
    "다음 영상에서 만나요", "시청해 주셔서", "ご視聴ありがとうございました", "チャンネル登録",
    "продолжение следует", "субтитры", "спасибо за просмотр", "редактор субтитров",
    "谢谢观看", "字幕", "sous-titres", "untertitel",
)


def _is_hallucination(seg: dict) -> bool:
    """Whisper invents text on silence / music / wind. Drop the tell-tale segments."""
    text = (seg.get("text") or "").strip()
    if not text:
        return True
    low = text.lower()
    if any(p in low for p in STOCK_HALLUCINATIONS):
        return True
    # mlx-whisper reports no_speech_prob as 0, so judge by decode confidence instead.
    # Calibrated on real footage: invented phrases on wind/music/silence ("Thank you.",
    # "We'll see you next time.", "you") averaged <=0.52 word probability and were short;
    # real speech averaged 0.58-0.99.
    if seg.get("avg_logprob", 0) < -1.0:
        return True
    probs = [w.get("probability", 1.0) for w in seg.get("words", [])]
    mean_p = sum(probs) / len(probs) if probs else 0.0
    n_words = len(text.split())
    if mean_p < 0.35 or (mean_p < 0.55 and n_words <= 4):
        return True
    # looping output ("I have a look, I have a look, ...") compresses far too well
    if seg.get("compression_ratio", 0) > 2.4:
        return True
    # the initial prompt echoed back on a silent clip
    words = [w for w in "".join(c if c.isalnum() else " " for c in text.lower()).split()]
    prompt = set("".join(c if c.isalnum() else " " for c in VERBATIM_PROMPT.lower()).split())
    if words and all(w in prompt for w in words) and len(words) >= 3:
        return True
    # the same token over and over (e.g. a wall of one kana)
    if len(words) >= 6 and len(set(words)) <= 2:
        return True
    return False


def call_local_whisper(audio_path: Path, language: str | None = None) -> dict:
    """Free offline fallback. Returns a Scribe-shaped dict (words with type/start/end/speaker_id).

    No diarization or audio events: every word is speaker_0.
    """
    try:
        import mlx_whisper
    except ImportError:
        raise RuntimeError("no ELEVENLABS_API_KEY and mlx-whisper not installed — run `uv sync` in the video-use repo")

    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=LOCAL_MODEL,
        word_timestamps=True,
        language=language,
        initial_prompt=VERBATIM_PROMPT,
        condition_on_previous_text=False,
        hallucination_silence_threshold=2.0,
    )
    words: list[dict] = []
    prev_end: float | None = None
    for seg in result.get("segments", []):
        if _is_hallucination(seg):
            continue
        for w in seg.get("words", []):
            text = (w.get("word") or "").strip()
            if not text:
                continue
            start, end = float(w["start"]), float(w["end"])
            if prev_end is not None:
                words.append({"text": " ", "start": prev_end, "end": max(prev_end, start),
                              "type": "spacing", "speaker_id": "speaker_0"})
            words.append({"text": text, "start": start, "end": end, "type": "word",
                          "speaker_id": "speaker_0", "logprob": w.get("probability")})
            prev_end = end
    return {
        "language_code": result.get("language"),
        "text": " ".join(w["text"] for w in words if w["type"] == "word"),
        "words": words,
        "backend": f"local:{LOCAL_MODEL}",
    }


def transcript_path(edit_dir: Path, video: Path, audio_track: int = 0) -> Path:
    """Where a video's transcript lands.

    The track belongs in the name, or a rerun with --audio-track hands back the transcript of
    the track it is meant to replace. Track 0 keeps the plain name, so transcripts made before
    the flag existed stay valid. Batch mode tests its cache with this too — one function, so
    the two cannot drift apart.
    """
    suffix = "" if audio_track == 0 else f".track{audio_track}"
    return edit_dir / "transcripts" / f"{video.stem}{suffix}.json"


def transcribe_one(
    video: Path,
    edit_dir: Path,
    api_key: str,
    language: str | None = None,
    num_speakers: int | None = None,
    verbose: bool = True,
    audio_track: int = 0,
) -> Path:
    """Transcribe a single video. Returns path to transcript JSON.

    Cached: returns existing path immediately if the transcript already exists.
    """
    transcripts_dir = edit_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    out_path = transcript_path(edit_dir, video, audio_track)

    if out_path.exists():
        if verbose:
            print(f"cached: {out_path.name}")
        return out_path

    if verbose:
        print(f"  extracting audio from {video.name}", flush=True)

    n_tracks = count_audio_tracks(video)
    if n_tracks > 1 and verbose:
        print(f"  note: {video.name} has {n_tracks} audio tracks, using track "
              f"{audio_track + 1} (--audio-track to change)", flush=True)

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / f"{video.stem}.wav"
        extract_audio(video, audio, audio_track)

        # Uploading silence costs the same as uploading speech and returns
        # nothing, so catch the wrong-track case before paying for it.
        peak = peak_dbfs(audio)
        if peak < -60.0:
            raise RuntimeError(
                f"track {audio_track + 1} of {video.name} is silent "
                f"(peak {peak:.1f} dBFS) - not uploading. "
                + (f"The file has {n_tracks} audio tracks; try --audio-track "
                   + " or ".join(str(i) for i in range(n_tracks) if i != audio_track) + "."
                   if n_tracks > 1 else "Check the source audio.")
            )

        size_mb = audio.stat().st_size / (1024 * 1024)
        if api_key:
            if verbose:
                print(f"  uploading {video.stem}.wav ({size_mb:.1f} MB)", flush=True)
            payload = call_scribe(audio, api_key, language, num_speakers)
        else:
            if verbose:
                print(f"  transcribing {video.stem}.wav locally ({LOCAL_MODEL})", flush=True)
            payload = call_local_whisper(audio, language)

    out_path.write_text(json.dumps(payload, indent=2))
    dt = time.time() - t0

    if verbose:
        kb = out_path.stat().st_size / 1024
        print(f"  saved: {out_path.name} ({kb:.1f} KB) in {dt:.1f}s")
        if isinstance(payload, dict) and "words" in payload:
            print(f"    words: {len(payload['words'])}")

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe a video with ElevenLabs Scribe")
    ap.add_argument("video", type=Path, help="Path to video file")
    ap.add_argument(
        "--edit-dir",
        type=Path,
        default=None,
        help="Edit output directory (default: <video_parent>/edit)",
    )
    ap.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional ISO language code (e.g., 'en'). Omit to auto-detect.",
    )
    ap.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Optional number of speakers when known. Improves diarization accuracy.",
    )
    ap.add_argument(
        "--audio-track",
        type=int,
        default=0,
        help="Zero-based audio track to transcribe. OBS writes the game on track 0 "
             "and the mic on track 1; without this ffmpeg applies its default audio "
             "stream selection, which picks the track with the most channels.",
    )
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")

    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()
    api_key = load_api_key()

    transcribe_one(
        video=video,
        edit_dir=edit_dir,
        api_key=api_key,
        language=args.language,
        num_speakers=args.num_speakers,
        audio_track=args.audio_track,
    )


if __name__ == "__main__":
    main()
