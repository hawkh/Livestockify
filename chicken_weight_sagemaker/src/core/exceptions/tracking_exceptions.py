"""
Tracking related exceptions.
"""


class TrackingError(Exception):
    """Base exception for tracking errors."""
    pass


class TrackLostError(TrackingError):
    """Raised when a track is lost."""
    
    def __init__(self, chicken_id: str, frames_lost: int):
        self.chicken_id = chicken_id
        self.frames_lost = frames_lost
        super().__init__(f"Track lost for chicken {chicken_id} after {frames_lost} frames")


class TrackInitializationError(TrackingError):
    """Raised when track initialization fails."""
    pass


class ReidentificationError(TrackingError):
    """Raised when re-identification fails."""
    
    def __init__(self, chicken_id: str, message: str = "Re-identification failed"):
        self.chicken_id = chicken_id
        super().__init__(f"{message} for chicken {chicken_id}")


class OcclusionTrackingError(TrackingError):
    """Raised when occlusion tracking fails."""
    pass


class TemporalSmoothingError(TrackingError):
    """Raised when temporal smoothing fails."""
    
    def __init__(self, chicken_id: str, window_size: int):
        self.chicken_id = chicken_id
        self.window_size = window_size
        super().__init__(f"Temporal smoothing failed for chicken {chicken_id} with window size {window_size}")