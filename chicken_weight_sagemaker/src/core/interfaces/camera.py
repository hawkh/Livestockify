"""
Camera calibration and distance estimation interfaces.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod
import numpy as np


@dataclass
class CameraParameters:
    """Camera intrinsic and extrinsic parameters."""
    focal_length: float = 1000.0  # in pixels
    sensor_width: float = 6.0     # in mm
    sensor_height: float = 4.5    # in mm
    image_width: int = 1920       # in pixels
    image_height: int = 1080      # in pixels
    camera_height: float = 3.0    # height above ground in meters
    tilt_angle: float = 0.0       # camera tilt in degrees
    known_object_width: float = 25.0  # average adult chicken width in cm


@dataclass
class CameraCalibration:
    """Camera calibration data and methods."""
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    rotation_matrix: Optional[np.ndarray] = None
    translation_vector: Optional[np.ndarray] = None
    parameters: Optional[CameraParameters] = None


class DistanceEstimator(ABC):
    """Abstract base class for distance estimation."""
    
    @abstractmethod
    def estimate_distance_to_chicken(
        self, 
        bbox: List[float], 
        frame_shape: Tuple[int, int]
    ) -> float:
        """
        Estimate distance from camera to chicken.
        
        Args:
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            frame_shape: Frame dimensions (height, width)
            
        Returns:
            Estimated distance in meters
        """
        pass
    
    @abstractmethod
    def validate_distance_estimate(
        self, 
        distance: float, 
        bbox_size: float
    ) -> bool:
        """
        Validate if distance estimate is reasonable.
        
        Args:
            distance: Estimated distance
            bbox_size: Size of bounding box
            
        Returns:
            True if distance estimate is valid
        """
        pass


class DistanceCompensator(ABC):
    """Abstract base class for distance compensation."""
    
    @abstractmethod
    def compensate_features_for_distance(
        self, 
        features: np.ndarray, 
        distance: float
    ) -> np.ndarray:
        """
        Compensate features for distance effects.
        
        Args:
            features: Original feature vector
            distance: Distance to chicken
            
        Returns:
            Distance-compensated features
        """
        pass
    
    @abstractmethod
    def compensate_bbox_for_perspective(
        self, 
        bbox: List[float], 
        distance: float,
        frame_shape: Tuple[int, int]
    ) -> List[float]:
        """
        Compensate bounding box for perspective distortion.
        
        Args:
            bbox: Original bounding box
            distance: Distance to chicken
            frame_shape: Frame dimensions
            
        Returns:
            Perspective-corrected bounding box
        """
        pass


class CameraCalibrator(ABC):
    """Abstract base class for camera calibration."""
    
    @abstractmethod
    def calibrate_camera(
        self, 
        calibration_images: List[np.ndarray],
        known_measurements: List[Tuple[float, float]]  # (distance, actual_size) pairs
    ) -> CameraCalibration:
        """
        Calibrate camera using calibration images and known measurements.
        
        Args:
            calibration_images: Images for calibration
            known_measurements: Known distance and size measurements
            
        Returns:
            CameraCalibration object
        """
        pass
    
    @abstractmethod
    def save_calibration(
        self, 
        calibration: CameraCalibration, 
        filepath: str
    ) -> None:
        """Save calibration to file."""
        pass
    
    @abstractmethod
    def load_calibration(self, filepath: str) -> CameraCalibration:
        """Load calibration from file."""
        pass