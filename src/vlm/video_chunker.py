"""
Video Chunker — handles arbitrarily long videos with high-accuracy frame extraction.

Accuracy improvements over naive uniform sampling:
  1. Uniform temporal sampling   — guarantees coverage of every N seconds
  2. Motion-based sampling       — captures frames when activity spikes (incidents often happen fast)
  3. Scene-change detection      — catches abrupt scene cuts / camera switches
  4. First-frame guarantee       — always samples the very first frame of every scene/chunk
  5. Overlap between chunks      — 2-second overlap prevents missing events at chunk boundaries
  6. Coverage guarantee          — ~75% of each chunk's budget is spread evenly across the
                                   whole chunk; motion frames only use the remainder and are
                                   rate-limited, so bursts can't starve the rest of the video
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
    motion_threshold   = motion_threshold   or settings.video.motion_threshold
    chunk_sec          = chunk_duration_sec or settings.video.chunk_duration_sec
    video_path         = str(video_path)

    fps, total_frames = probe_video(video_path)
    duration_sec = total_frames / fps
    if total_frames <= 0:
        raise ValueError(f"Could not read any frames from video: {video_path}")
    max_frames = max_frames or auto_frame_budget(duration_sec)

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
    budgets = [max(1, round(max_frames * (end - start) / total_span))
               for start, end in chunks]
    # Trim to max_frames
    while sum(budgets) > max_frames and max(budgets) > 0:
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


# Share of each chunk's frame budget reserved for evenly spaced frames (coverage);
# the rest may be spent on motion / scene-change frames.
_UNIFORM_SHARE = 0.75
# Minimum spacing between an event-triggered frame and any other sampled frame,
# so a burst of motion can't eat the whole budget in one second.
_MIN_EVENT_GAP_SEC = 2.0
_ANALYSIS_SIZE = (160, 90)   # motion/scene detection runs on a thumbnail (much faster)


def auto_frame_budget(duration_sec: float) -> int:
    """Frames to analyse when no fixed budget is set: scales with the video's length."""
    v = settings.video
    n = round(duration_sec / max(v.auto_frame_spacing_sec, 0.5))
    return int(max(v.auto_min_frames, min(v.auto_max_frames, n)))


def probe_video(video_path: str) -> tuple[float, int]:
    """
    (fps, total_frames) that work for any container. Some files (webm/mkv streams,
    screen recordings) report 0 / bogus frame counts or FPS — fall back to counting.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not (0.5 < fps < 1000):
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0 or total > 10**9:
        total = 0
        while cap.grab():
            total += 1
        logger.info(f"Frame count not in container metadata — counted {total} frames")
    cap.release()
    return fps, total


def _fit_size(w: int, h: int) -> tuple[int, int]:
    """Scale (w, h) down to fit inside the configured box, keeping the aspect ratio."""
    box_w, box_h = settings.video.resize_width, settings.video.resize_height
    scale = min(box_w / max(w, 1), box_h / max(h, 1), 1.0)
    return max(2, int(round(w * scale))), max(2, int(round(h * scale)))


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
    Extract frames from [start_sec, end_sec] with guaranteed coverage:
      1. Evenly spaced frames across the WHOLE chunk (≈75% of the budget).
         Spacing is max(sample_rate, chunk_length / uniform_budget), so the
         budget always stretches to the end of the chunk.
      2. Extra frames on high motion or scene change (remaining budget),
         at least 2s away from any other sampled frame.
    The first frame of the chunk is always included.
    """
    if max_frames <= 0:
        return []

    span = max(end_sec - start_sec, 1e-6)
    uniform_budget = max(1, min(max_frames, round(max_frames * _UNIFORM_SHARE)))
    event_budget = max_frames - uniform_budget
    # If uniform sampling at sample_rate needs fewer frames than budgeted, give the rest to events
    natural_uniform = int(span // max(sample_rate, 1e-6)) + 1
    if natural_uniform < uniform_budget:
        event_budget += uniform_budget - natural_uniform
        uniform_budget = natural_uniform
    uniform_interval_sec = max(float(sample_rate), span / uniform_budget)
    interval = max(1, int(round(fps * uniform_interval_sec)))
    min_gap_frames = int(fps * _MIN_EVENT_GAP_SEC)

    cap         = cv2.VideoCapture(video_path)
    start_frame = int(start_sec * fps)
    end_frame   = int(end_sec   * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    extracted: list[dict] = []
    # Adaptive motion baseline (running mean / variance of frame-to-frame change)
    m_mean, m_var, m_n = 0.0, 0.0, 0
    n_uniform  = 0
    n_event    = 0
    last_sampled_fi = -10**9
    prev_small = None
    prev_hist  = None
    fi         = start_frame

    while cap.isOpened() and fi < end_frame and len(extracted) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        local_fi = fi - start_frame
        small = cv2.cvtColor(cv2.resize(frame, _ANALYSIS_SIZE, interpolation=cv2.INTER_AREA),
                             cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([small], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)

        motion = float(np.mean(cv2.absdiff(prev_small, small))) if prev_small is not None else 0.0
        if prev_small is not None:
            m_n += 1
            alpha = max(0.02, 1.0 / m_n)
            delta = motion - m_mean
            m_mean += alpha * delta
            m_var = (1 - alpha) * (m_var + alpha * delta * delta)
        # "Motion" = clearly above THIS video's normal change level (camera noise, lighting flicker)
        motion_trigger = (m_mean + max(4.0 * m_var ** 0.5, 0.5 * motion_thr)) if m_n >= 10 else motion_thr

        sample = False
        # Trigger 1: uniform grid (always includes the chunk's first frame)
        if local_fi % interval == 0 and n_uniform < uniform_budget:
            sample = True
            n_uniform += 1
        # Triggers 2/3: motion or scene change, rate-limited
        elif (n_event < event_budget and prev_small is not None
              and fi - last_sampled_fi >= min_gap_frames
              and (local_fi % interval) <= interval - min_gap_frames):   # not right before a grid frame
            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if motion > motion_trigger or corr < 0.60:
                sample = True
                n_event += 1
                logger.debug(f"Event frame @ {fi/fps:.1f}s (motion={motion:.1f}, corr={corr:.2f})")

        if sample:
            ts      = round(fi / fps, 2)
            fh_, fw_ = frame.shape[:2]
            tw, th = _fit_size(fw_, fh_)
            resized = frame if (tw, th) == (fw_, fh_) else cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
            fname      = f"frame_{fi:07d}_{ts:.2f}s.jpg"
            fpath      = str(Path(output_dir) / fname)
            cv2.imwrite(fpath, resized, [cv2.IMWRITE_JPEG_QUALITY, 90])

            with open(fpath, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode("utf-8")

            extracted.append({
                "frame_index":   fi,
                "timestamp_sec": ts,
                "frame_path":    fpath,
                "base64":        b64,
            })
            last_sampled_fi = fi

        prev_small = small
        prev_hist  = hist
        fi += 1

    cap.release()
    return extracted


def get_video_info(video_path: str) -> dict:
    fps, total = probe_video(video_path)
    cap     = cv2.VideoCapture(str(video_path))
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
