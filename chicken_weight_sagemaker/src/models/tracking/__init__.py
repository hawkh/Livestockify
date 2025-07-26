"""
Multi-object tracking models for chicken identification.
"""

from .chicken_tracker import ChickenMultiObjectTracker
from .deepsort_tracker import DeepSORTChickenTracker
from .reid_features import ChickenReIDFeatureExtractor
from .kalman_filter import ChickenKalmanFilter

__all__ = [
    'ChickenMultiObjectTracker',
    'DeepSORTChickenTracker',
    'ChickenReIDFeatureExtractor',
    'ChickenKalmanFilter'
]