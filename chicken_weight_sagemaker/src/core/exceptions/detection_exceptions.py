"""
Detection-related exceptions.
"""


class DetectionError(Exception):
    """Base exception for detection errors."""
    pass


class ModelLoadError(DetectionError):
    """Raised when model loading fails."""
    
    def __init__(self, model_path: str, message: str = "Failed to load model"):
        self.model_path = model_path
        super().__init__(f"{message}: {model_path}")


class InferenceError(DetectionError):
    """Raised when inference fails."""
    
    def __init__(self, message: str = "Inference failed", details: dict = None):
        self.details = details or {}
        super().__init__(message)


class OcclusionDetectionError(DetectionError):
    """Raised when occlusion detection fails."""
    pass


class TemporalConsistencyError(DetectionError):
    """Raised when temporal consistency check fails."""
    pass