"""
Core interfaces for the chicken weight estimation system.
"""

from .detection import Detection, DetectionResult
from .weight_estimation import WeightEstimate, WeightEstimationResult
from .tracking import TrackedChicken, TrackingResult
from .camera import CameraParameters, CameraCalibration
from .inference import InferenceRequest, InferenceResponse

__all__ = [
    'Detection',
    'DetectionResult', 
    'WeightEstimate',
    'WeightEstimationResult',
    'TrackedChicken',
    'TrackingResult',
    'CameraParameters',
    'CameraCalibration',
    'InferenceRequest',
    'InferenceResponse'
]