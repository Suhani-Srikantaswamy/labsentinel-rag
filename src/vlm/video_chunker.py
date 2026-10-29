"""
Video Chunker — handles arbitrarily long videos with high-accuracy frame extraction.

Accuracy improvements over naive uniform sampling:
  1. Uniform temporal sampling   — guarantees coverage of every N seconds
  2. Motion-based sampling       — captures frames when activity spikes (incidents often happen fast)
  3. Scene-change detection      — catches abrupt scene cuts / camera switches
  4. First-frame guarantee       — always samples the very first frame of every scene/chunk
  5. Overlap between chunks      — 2-second overlap prevents missing events at chunk boundaries
  6. No frame deduplication      — slightly redundant frames are fine; missing frames are not
"""

from __future__ import annotations
import base64
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from config.settings import settings, OUTPUT_DIR

# Overlap between consecutive chunks (seconds) — catches events at boundaries
_CHUNK_OVERLAP_SEC = 2.0


def extract_frames_chunked(
    video_path: str,
    output_dir: str | None = None,
    sample_rate: int | None = None,
    max_frames: int | None = None,
    chunk_duration_sec: int | None = None,
    motion_threshold: float | None = None,
    on_progress=None,
) -> list[dict]:
    """
    Extract representative frames from any-length video.

    Accuracy strategy:
    - Uniform sampling every `sample_rate` seconds (baseline coverage)
    - Extra frames at high-motion moments (catch sudden incidents)
    - Extra frames at scene changes (detect camera cuts, new people entering)
    - 2-second overlap between processing chunks (avoid missing boundary events)
    - Always include the first frame of every chunk

    Returns list of dicts: {frame_index, timestamp_sec, frame_path, base64}
    """
    sample_rate        = sample_rate        or settings.video.frame_sample_rate
    max_frames         = max_frames         or settings.video.max_frames
    motion_threshold   = motion_threshold   or settings.video.motion_threshold
    chunk_sec          = chunk_duration_sec or settings.video.chunk_duration_sec
    video_path         = str(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    cap.release()

    logger.info(
        f"Video: {Path(video_path).name} | {duration_sec:.0f}s | {fps:.1f} FPS | "
        f"sample_rate={sample_rate}s | max_frames={max_frames} | "
        f"chunk={chunk_sec}s"
    )

    if output_dir is None:
        stem       = Path(video_path).stem
        output_dir = str(OUTPUT_DIR / "frames" / stem)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Build chunks with overlap ─────────────────────────────────────────────
    chunks: list[tuple[float, float]] = []
    t = 0.0
    while t < duration_sec:
        end = min(t + chunk_sec, duration_sec)
        chunks.append((t, end))
        t = end - _CHUNK_OVERLAP_SEC  # overlap so we don't miss boundary events
        if t <= chunks[-1][0]:        # avoid infinite loop on very short videos
            break

    logger.info(f"Processing {len(chunks)} chunk(s) (with {_CHUNK_OVERLAP_SEC}s overlap)")

    # ── Proportional frame budget per chunk ───────────────────────────────────
    total_span = sum(end - start for start, end in chunks)
    budgets = [max(2, round(max_frames * (end - start) / total_span))
               for start, end in chunks]
    # Trim to max_frames
    while sum(budgets) > max_frames:
        budgets[budgets.index(max(budgets))] -= 1

    # ── Extract ───────────────────────────────────────────────────────────────
    all_frames: list[dict] = []
    seen_timestamps: set[float] = set()   # deduplicate frames from overlapping chunks

    for ci, ((start_sec, end_sec), budget) in enumerate(zip(chunks, budgets)):
        if on_progress:
            on_progress(ci / len(chunks), f"Extracting frames {start_sec:.0f}s – {end_sec:.0f}s")

        chunk_frames = _extract_chunk(
            video_path   = video_path,
            start_sec    = start_sec,
            end_sec      = end_sec,
            output_dir   = output_dir,
            sample_rate  = sample_rate,
            max_frames   = budget,
            motion_thr   = motion_threshold,
            fps          = fps,
        )
        # Only add frames not already captured by a previous chunk's overlap
        for f in chunk_frames:
            if f["timestamp_sec"] not in seen_timestamps:
                seen_timestamps.add(f["timestamp_sec"])
                all_frames.append(f)

    # Sort by timestamp so the pipeline processes chronologically
    all_frames.sort(key=lambda x: x["timestamp_sec"])

    if on_progress:
        on_progress(1.0, f"Frame extraction complete — {len(all_frames)} frames")

    logger.success(f"Extracted {len(all_frames)} unique frames from {Path(video_path).name}")
    return all_frames


def _extract_chunk(
    video_path: str,
    start_sec: float,
    end_sec: float,
    output_dir: str,
    sample_rate: int,
    max_frames: int,
    motion_thr: float,
    fps: float,
) -> list[dict]:
    """
    Extract frames from [start_sec, end_sec] using three triggers:
      1. Uniform interval (every sample_rate seconds)
      2. High motion  (mean pixel diff > motion_thr)
      3. Scene change (histogram correlation drop < 0.6)
    Always includes the very first frame of the chunk.
    """
    cap         = cv2.VideoCapture(video_path)
    start_frame = int(start_sec * fps)
    end_frame   = int(end_sec   * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    interval   = max(1, int(fps * sample_rate))
    extracted: list[dict] = []
    prev_gray  = None
    prev_hist  = None
    fi         = start_frame
    is_first   = True

    while cap.isOpened() and fi < end_frame and len(extracted) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        local_fi = fi - start_frame
        gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Trigger 1: uniform interval
        sample = (local_fi % interval == 0)

        # Trigger 2: first frame of chunk — always sample
        if is_first:
            sample   = True
            is_first = False

        # Trigger 3: motion detection
        if prev_gray is not None and not sample:
            diff = cv2.absdiff(prev_gray, gray)
            if float(np.mean(diff)) > motion_thr:
                sample = True

        # Trigger 4: scene change via histogram correlation
        if prev_hist is not None and not sample:
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            cv2.normalize(hist, hist)
            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if corr < 0.60:   # low correlation = scene change
                sample = True
                logger.debug(f"Scene change detected @ {fi/fps:.1f}s (corr={corr:.2f})")

        if sample:
            ts      = round(fi / fps, 2)
            resized = cv2.resize(
                frame,
                (settings.video.resize_width, settings.video.resize_height),
                interpolation=cv2.INTER_AREA,
            )
            fname      = f"frame_{fi:07d}_{ts:.2f}s.jpg"
            fpath      = str(Path(output_dir) / fname)
            cv2.imwrite(fpath, resized, [cv2.IMWRITE_JPEG_QUALITY, 90])   # 90 quality for accuracy

            with open(fpath, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode("utf-8")

            extracted.append({
                "frame_index":   fi,
                "timestamp_sec": ts,
                "frame_path":    fpath,
                "base64":        b64,
            })

        # Update reference frames
        prev_gray = gray
        hist_now  = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist_now, hist_now)
        prev_hist = hist_now
        fi += 1

    cap.release()
    return extracted


def get_video_info(video_path: str) -> dict:
    cap     = cv2.VideoCapture(str(video_path))
    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    size_mb = (Path(video_path).stat().st_size / 1048576
               if Path(video_path).exists() else 0)
    dur     = total / fps
    return {
        "path":         str(video_path),
        "filename":     Path(video_path).name,
        "fps":          round(fps, 2),
        "total_frames": total,
        "duration_sec": round(dur, 2),
        "duration_hms": _hms(dur),
        "width":        width,
        "height":       height,
        "size_mb":      round(size_mb, 1),
    }


def _hms(secs: float) -> str:
    h, rem = divmod(int(secs), 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
