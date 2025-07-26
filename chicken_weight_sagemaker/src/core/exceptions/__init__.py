"""
Custom exceptions for the chicken weight estimation system.
"""

from .detection_exceptions import DetectionError, ModelLoadError, InferenceError
from .weight_estimation_exceptions import WeightEstimationError, FeatureExtractionError
from .tracking_exceptions import TrackingError, TrackLostError
from .camera_exceptions import CameraCalibrationError, DistanceEstimationError
from .inference_exceptions import InvalidInputError, ProcessingError, SageMakerError

__all__ = [
    'DetectionError',
    'ModelLoadError', 
    'InferenceError',
    'WeightEstimationError',
    'FeatureExtractionError',
    'TrackingError',
    'TrackLostError',
    'CameraCalibrationError',
    'DistanceEstimationError',
    'InvalidInputError',
    'ProcessingError',
    'SageMakerError'
]