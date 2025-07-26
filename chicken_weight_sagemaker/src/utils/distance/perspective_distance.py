"""
Perspective-based distance estimation for chickens in farm environments.
"""

import math
from typing import List, Tuple, Optional
import numpy as np

from ...core.interfaces.camera import DistanceEstimator, CameraParameters
from ...core.exceptions.camera_exceptions import DistanceEstimationError, DistanceValidationError


class PerspectiveDistanceEstimator(DistanceEstimator):
    """Distance estimator using perspective projection principles."""
    
    def __init__(self, camera_params: CameraParameters):
        self.camera_params = camera_params
        self.validation_enabled = True
        
        # Distance estimation parameters
        self.min_distance = 1.0  # meters
        self.max_distance = 15.0  # meters
        self.confidence_threshold = 0.7
        
    def estimate_distance_to_chicken(
        self, 
        bbox: List[float], 
        frame_shape: Tuple[int, int]
    ) -> float:
        """
        Estimate distance from camera to chicken using perspective projection.
        
        Args:
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            frame_shape: Frame dimensions (height, width)
            
        Returns:
            Estimated distance in meters
        """
        try:
            x1, y1, x2, y2 = bbox
            frame_height, frame_width = frame_shape
            
            # Calculate bounding box dimensions
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            # Method 1: Using known object width (primary method)
            distance_from_width = self._estimate_from_object_width(bbox_width, frame_width)
            
            # Method 2: Using camera height and vertical position (secondary method)
            distance_from_height = self._estimate_from_vertical_position(
                y1, y2, frame_height
            )
            
            # Method 3: Using aspect ratio analysis (tertiary method)
            distance_from_aspect = self._estimate_from_aspect_ratio(
                bbox_width, bbox_height
            )
            
            # Combine estimates with weights
            distances = [distance_from_width, distance_from_height, distance_from_aspect]
            weights = [0.6, 0.3, 0.1]  # Prioritize width-based estimation
            
            # Filter out invalid distances
            valid_distances = []
            valid_weights = []
            
            for dist, weight in zip(distances, weights):
                if dist is not None and self.min_distance <= dist <= self.max_distance:
                    valid_distances.append(dist)
                    valid_weights.append(weight)
            
            if not valid_distances:
                # Fallback to width-based estimation even if outside normal range
                return max(self.min_distance, min(self.max_distance, distance_from_width))
            
            # Weighted average of valid distances
            weighted_distance = np.average(valid_distances, weights=valid_weights)
            
            # Apply validation if enabled
            if self.validation_enabled:
                bbox_size = math.sqrt(bbox_width * bbox_height)
                if not self.validate_distance_estimate(weighted_distance, bbox_size):
                    # Use fallback estimation
                    weighted_distance = distance_from_width
            
            return float(weighted_distance)
            
        except Exception as e:
            raise DistanceEstimationError(f"Distance estimation failed: {str(e)}")
    
    def _estimate_from_object_width(self, bbox_width: float, frame_width: int) -> float:
        """Estimate distance using known object width and pinhole camera model."""
        try:
            # Pinhole camera model: distance = (real_width * focal_length) / pixel_width
            real_width_cm = self.camera_params.known_object_width
            focal_length_pixels = self.camera_params.focal_length
            
            # Convert real width to meters
            real_width_m = real_width_cm / 100.0
            
            # Calculate distance
            distance = (real_width_m * focal_length_pixels) / bbox_width
            
            return distance
            
        except (ZeroDivisionError, ValueError):
            return None
    
    def _estimate_from_vertical_position(
        self, 
        y1: float, 
        y2: float, 
        frame_height: int
    ) -> Optional[float]:
        """Estimate distance using camera height and vertical position in frame."""
        try:
            # Calculate the bottom center of the bounding box
            bottom_y = y2
            
            # Convert to normalized coordinates (0 at center, -1 to +1)
            normalized_y = (bottom_y - frame_height / 2) / (frame_height / 2)
            
            # Calculate angle from camera optical axis
            # Assuming camera is tilted at tilt_angle degrees
            camera_tilt_rad = math.radians(self.camera_params.tilt_angle)
            
            # Vertical field of view calculation
            sensor_height_mm = self.camera_params.sensor_height
            focal_length_mm = (
                self.camera_params.focal_length * sensor_height_mm / 
                self.camera_params.image_height
            )
            
            vfov_rad = 2 * math.atan(sensor_height_mm / (2 * focal_length_mm))
            
            # Angle to the bottom of the chicken
            angle_to_chicken = camera_tilt_rad + (normalized_y * vfov_rad / 2)
            
            # Distance calculation using trigonometry
            if abs(angle_to_chicken) > 0.01:  # Avoid division by very small numbers
                distance = self.camera_params.camera_height / math.tan(abs(angle_to_chicken))
                return max(0.5, distance)  # Minimum reasonable distance
            
            return None
            
        except (ZeroDivisionError, ValueError, OverflowError):
            return None
    
    def _estimate_from_aspect_ratio(
        self, 
        bbox_width: float, 
        bbox_height: float
    ) -> Optional[float]:
        """Estimate distance using aspect ratio analysis."""
        try:
            aspect_ratio = bbox_width / bbox_height
            
            # Empirical relationship between aspect ratio and distance
            # Based on the observation that chickens appear more elongated when viewed from side
            # and more compact when viewed from above (at distance)
            
            # Typical aspect ratios:
            # Close (side view): 1.2 - 1.8
            # Medium distance: 0.8 - 1.2  
            # Far (top-down view): 0.6 - 1.0
            
            if aspect_ratio > 1.5:
                # Likely side view, closer distance
                estimated_distance = 2.0 + (aspect_ratio - 1.5) * 2.0
            elif aspect_ratio > 1.0:
                # Medium distance
                estimated_distance = 4.0 + (1.5 - aspect_ratio) * 4.0
            else:
                # More top-down view, farther distance
                estimated_distance = 6.0 + (1.0 - aspect_ratio) * 6.0
            
            return min(self.max_distance, max(self.min_distance, estimated_distance))
            
        except (ZeroDivisionError, ValueError):
            return None
    
    def validate_distance_estimate(self, distance: float, bbox_size: float) -> bool:
        """
        Validate if distance estimate is reasonable given bounding box size.
        
        Args:
            distance: Estimated distance
            bbox_size: Size of bounding box (diagonal or area-based)
            
        Returns:
            True if distance estimate is valid
        """
        try:
            # Expected bbox size at given distance
            expected_pixel_width = (
                self.camera_params.known_object_width * 
                self.camera_params.focal_length / (distance * 100)
            )
            
            # Allow for reasonable variation (±50%)
            min_expected_size = expected_pixel_width * 0.5
            max_expected_size = expected_pixel_width * 1.5
            
            # Check if actual bbox size is within expected range
            is_valid = min_expected_size <= bbox_size <= max_expected_size
            
            if not is_valid and distance < self.min_distance or distance > self.max_distance:
                raise DistanceValidationError(distance, bbox_size)
            
            return is_valid
            
        except Exception:
            # If validation fails, assume distance is valid
            return True
    
    def estimate_distance_confidence(
        self, 
        bbox: List[float], 
        frame_shape: Tuple[int, int]
    ) -> float:
        """
        Estimate confidence in distance measurement.
        
        Args:
            bbox: Bounding box coordinates
            frame_shape: Frame dimensions
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        try:
            x1, y1, x2, y2 = bbox
            
            # Factors affecting confidence:
            # 1. Bounding box size (larger = more confident)
            bbox_area = (x2 - x1) * (y2 - y1)
            frame_area = frame_shape[0] * frame_shape[1]
            size_ratio = bbox_area / frame_area
            size_confidence = min(1.0, size_ratio * 20)  # Normalize
            
            # 2. Position in frame (center = more confident)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            frame_center_x = frame_shape[1] / 2
            frame_center_y = frame_shape[0] / 2
            
            distance_from_center = math.sqrt(
                ((center_x - frame_center_x) / frame_center_x) ** 2 +
                ((center_y - frame_center_y) / frame_center_y) ** 2
            )
            position_confidence = max(0.3, 1.0 - distance_from_center)
            
            # 3. Aspect ratio reasonableness
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 1.0
            
            # Reasonable aspect ratios for chickens: 0.5 to 2.0
            if 0.5 <= aspect_ratio <= 2.0:
                aspect_confidence = 1.0
            else:
                aspect_confidence = max(0.2, 1.0 - abs(aspect_ratio - 1.0) * 0.5)
            
            # Combined confidence
            overall_confidence = (
                0.4 * size_confidence +
                0.3 * position_confidence +
                0.3 * aspect_confidence
            )
            
            return min(1.0, max(0.1, overall_confidence))
            
        except Exception:
            return 0.5  # Default moderate confidence
    
    def set_validation_enabled(self, enabled: bool) -> None:
        """Enable or disable distance validation."""
        self.validation_enabled = enabled
    
    def update_camera_parameters(self, camera_params: CameraParameters) -> None:
        """Update camera parameters."""
        self.camera_params = camera_params
    
    def get_estimation_info(self) -> dict:
        """Get information about the distance estimator."""
        return {
            "method": "perspective_projection",
            "camera_height": self.camera_params.camera_height,
            "focal_length": self.camera_params.focal_length,
            "known_object_width": self.camera_params.known_object_width,
            "distance_range": (self.min_distance, self.max_distance),
            "validation_enabled": self.validation_enabled
        }