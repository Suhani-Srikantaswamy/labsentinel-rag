# VideoAnalyzer and frame tools are imported lazily so that the
# openai package is not required at import time (e.g. during tests).
def _get_analyzer():
    from .video_analyzer import VideoAnalyzer
    return VideoAnalyzer

from .frame_extractor import extract_frames, get_video_duration, get_video_info
