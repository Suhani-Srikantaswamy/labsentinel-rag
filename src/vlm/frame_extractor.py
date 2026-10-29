"""
Frame Extractor — samples keyframes from an incident video.

Two complementary strategies:
  1. Uniform temporal sampling  — 1 frame every N seconds (ensures full coverage)
  2. Motion-based keyframes     — captures frames with significant activity
     (avoids missing the moment of an incident between sample points)
"""

from __future__ import annotations
import base64
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from config.settings import settings, OUTPUT_DIR


def extract_frames(
    video_path: str,
    output_dir: str | None = None,
    sample_rate: int | None = None,
    max_frames: int | None = None,
    motion_threshold: float | None = None,
) -> list[dict]:
    """
    Extract representative frames from a video file.

    Returns a list of dicts:
        {
            "frame_index": int,
            "timestamp_sec": float,
            "frame_path": str,   # absolute path to saved JPEG
            "base64": str,       # base64-encoded JPEG (for API calls)
        }
    """
    sample_rate      = sample_rate      or settings.video.frame_sample_rate
    max_frames       = max_frames       or settings.video.max_frames
    motion_threshold = motion_threshold or settings.video.motion_threshold
    video_path       = str(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    logger.info(
        f"Video: {Path(video_path).name} | "
        f"FPS={fps:.1f} | Duration={duration_sec:.1f}s | Frames={total_frames}"
    )

    # Output directory for extracted frames
    if output_dir is None:
        stem = Path(video_path).stem
        output_dir = str(OUTPUT_DIR / "frames" / stem)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    frame_interval = max(1, int(fps * sample_rate))
    extracted: list[dict] = []
    prev_gray: np.ndarray | None = None
    frame_idx = 0

    while cap.isOpened() and len(extracted) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        should_sample = (frame_idx % frame_interval == 0)

        # Also capture high-motion frames (catches the moment of an incident)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None and not should_sample:
            diff = cv2.absdiff(prev_gray, gray)
            if float(np.mean(diff)) > motion_threshold:
                should_sample = True
        prev_gray = gray

        if should_sample:
            timestamp = round(frame_idx / fps, 2)
            resized = cv2.resize(
                frame,
                (settings.video.resize_width, settings.video.resize_height),
                interpolation=cv2.INTER_AREA,
            )
            frame_filename = f"frame_{frame_idx:06d}_{timestamp:.2f}s.jpg"
            frame_path = str(Path(output_dir) / frame_filename)
            cv2.imwrite(frame_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 85])

            with open(frame_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")

            extracted.append({
                "frame_index": frame_idx,
                "timestamp_sec": timestamp,
                "frame_path": frame_path,
                "base64": b64,
            })
            logger.debug(f"  Sampled frame {frame_idx} @ {timestamp:.2f}s")

        frame_idx += 1

    cap.release()
    logger.info(f"Extracted {len(extracted)} frames from {Path(video_path).name}")
    return extracted


def encode_image_to_base64(image_path: str) -> str:
    """Encode any image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_video_duration(video_path: str) -> float:
    """Return video duration in seconds."""
    cap = cv2.VideoCapture(str(video_path))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return round(total / fps, 2)


def get_video_info(video_path: str) -> dict:
    """Return basic metadata about a video file."""
    cap = cv2.VideoCapture(str(video_path))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_sec": round(total_frames / fps, 2),
        "width": width,
        "height": height,
        "path": str(video_path),
    }
