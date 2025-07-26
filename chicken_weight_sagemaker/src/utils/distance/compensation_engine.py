"""
Distance compensation engine for adjusting features and measurements.
"""

import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from ...core.interfaces.camera import DistanceCompensator, CameraParameters
from ...core.exceptions.camera_exceptions import PerspectiveDistortionError


class DistanceCompensationEngine(DistanceCompensator):
    """Engine for compensating measurements and features for distance effects."""
    
    def __init__(self, camera_params: CameraParameters):
        self.camera_params = camera_params
        self.reference_distance = 3.0  # meters - reference distance for normalization
        
        # Compensation parameters
        self.perspective_correction_enabled = True
        self.feature_scaling_enabled = True
        self.bbox_correction_enabled = True
        
        # Empirical correction factors
        self.size_correction_factor = 1.0
        self.aspect_correction_factor = 0.8
        self.texture_correction_factor = 1.2
    
    def compensate_features_for_distance(
        self, 
        features: np.ndarray, 
        distance: float
    ) -> np.ndarray:
        """
        Compensate features for distance effects.
        
        Args:
            features: Original feature vector
            distance: Distance to chicken in meters
            
        Returns:
            Distance-compensated features
        """
        if not self.feature_scaling_enabled:
            return features
        
        try:
            compensated_features = features.copy()
            
            # Calculate distance scaling factor
            distance_ratio = distance / self.reference_distance
            
            # Compensate different types of features
            compensated_features = self._compensate_size_features(
                compensated_features, distance_ratio
            )
            compensated_features = self._compensate_texture_features(
                compensated_features, distance_ratio
            )
            compensated_features = self._compensate_color_features(
                compensated_features, distance_ratio
            )
            
            return compensated_features
            
        except Exception as e:
            raise PerspectiveDistortionError(f"Feature compensation failed: {str(e)}")
    
    def compensate_bbox_for_perspective(
        self, 
        bbox: List[float], 
        distance: float,
        frame_shape: Tuple[int, int]
    ) -> List[float]:
        """
        Compensate bounding box for perspective distortion.
        
        Args:
            bbox: Original bounding box [x1, y1, x2, y2]
            distance: Distance to chicken
            frame_shape: Frame dimensions (height, width)
            
        Returns:
            Perspective-corrected bounding box
        """
        if not self.bbox_correction_enabled:
            return bbox
        
        try:
            x1, y1, x2, y2 = bbox
            frame_height, frame_width = frame_shape
            
            # Calculate perspective correction factors
            correction_factors = self._calculate_perspective_correction(
                bbox, distance, frame_shape
            )
            
            # Apply corrections
            width_correction = correction_factors['width']
            height_correction = correction_factors['height']
            position_correction = correction_factors['position']
            
            # Adjust bounding box dimensions
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            current_width = x2 - x1
            current_height = y2 - y1
            
            # Apply corrections
            corrected_width = current_width * width_correction
            corrected_height = current_height * height_correction
            
            # Apply position correction (for camera tilt effects)
            corrected_center_x = center_x + position_correction['x']
            corrected_center_y = center_y + position_correction['y']
            
            # Calculate new bbox coordinates
            new_x1 = corrected_center_x - corrected_width / 2
            new_y1 = corrected_center_y - corrected_height / 2
            new_x2 = corrected_center_x + corrected_width / 2
            new_y2 = corrected_center_y + corrected_height / 2
            
            # Ensure bbox stays within frame bounds
            new_x1 = max(0, min(frame_width - 1, new_x1))
            new_y1 = max(0, min(frame_height - 1, new_y1))
            new_x2 = max(0, min(frame_width - 1, new_x2))
            new_y2 = max(0, min(frame_height - 1, new_y2))
            
            return [new_x1, new_y1, new_x2, new_y2]
            
        except Exception as e:
            raise PerspectiveDistortionError(f"Bbox compensation failed: {str(e)}")
    
    def _compensate_size_features(
        self, 
        features: np.ndarray, 
        distance_ratio: float
    ) -> np.ndarray:
        """Compensate size-related features (width, height, area)."""
        # Assuming first few features are size-related
        # This would need to be adjusted based on actual feature vector structure
        
        # Size features scale with square of distance ratio
        size_scaling = (distance_ratio ** 2) * self.size_correction_factor
        
        # Apply to size features (assuming indices 0-3 are size-related)
        if len(features) > 3:
            features[0:3] *= size_scaling  # width, height, area
        
        return features
    
    def _compensate_texture_features(
        self, 
        features: np.ndarray, 
        distance_ratio: float
    ) -> np.ndarray:
        """Compensate texture-related features."""
        # Texture features become less reliable at distance
        texture_scaling = (1.0 / distance_ratio) * self.texture_correction_factor
        
        # Apply to texture features (assuming indices 6-8 are texture-related)
        if len(features) > 8:
            features[6:8] *= texture_scaling
        
        return features
    
    def _compensate_color_features(
        self, 
        features: np.ndarray, 
        distance_ratio: float
    ) -> np.ndarray:
        """Compensate color-related features."""
        # Color features are relatively stable but may have atmospheric effects
        # at very long distances
        
        if distance_ratio > 3.0:  # Only apply for very distant objects
            color_scaling = 1.0 / (1.0 + 0.1 * (distance_ratio - 3.0))
            
            # Apply to color features (assuming indices 8+ are color-related)
            if len(features) > 8:
                features[8:] *= color_scaling
        
        return features
    
    def _calculate_perspective_correction(
        self,
        bbox: List[float],
        distance: float,
        frame_shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Calculate perspective correction factors."""
        x1, y1, x2, y2 = bbox
        frame_height, frame_width = frame_shape
        
        # Calculate position in frame
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Normalized coordinates (-1 to 1)
        norm_x = (center_x - frame_width / 2) / (frame_width / 2)
        norm_y = (center_y - frame_height / 2) / (frame_height / 2)
        
        # Calculate correction factors based on position and distance
        
        # Width correction: objects at edges appear more compressed
        width_correction = 1.0 + abs(norm_x) * 0.1 * (distance / self.reference_distance)
        
        # Height correction: objects higher/lower in frame have different perspective
        height_correction = 1.0 + abs(norm_y) * 0.15 * (distance / self.reference_distance)
        
        # Position correction for camera tilt
        tilt_rad = math.radians(self.camera_params.tilt_angle)
        position_x_correction = -norm_y * math.sin(tilt_rad) * distance * 2
        position_y_correction = norm_y * (1 - math.cos(tilt_rad)) * distance * 2
        
        return {
            'width': width_correction,
            'height': height_correction,
            'position': {
                'x': position_x_correction,
                'y': position_y_correction
            }
        }
    
    def calculate_real_world_dimensions(
        self,
        bbox: List[float],
        distance: float,
        frame_shape: Tuple[int, int]
    ) -> Dict[str, float]:
        """
        Calculate real-world dimensions of the chicken.
        
        Args:
            bbox: Bounding box coordinates
            distance: Distance to chicken
            frame_shape: Frame dimensions
            
        Returns:
            Dictionary with real-world dimensions
        """
        try:
            x1, y1, x2, y2 = bbox
            
            # Pixel dimensions
            pixel_width = x2 - x1
            pixel_height = y2 - y1
            
            # Convert to real-world dimensions using pinhole camera model
            # real_size = (pixel_size * distance) / focal_length
            
            real_width = (pixel_width * distance * 100) / self.camera_params.focal_length  # cm
            real_height = (pixel_height * distance * 100) / self.camera_params.focal_length  # cm
            
            # Estimate depth using aspect ratio and viewing angle
            aspect_ratio = pixel_width / pixel_height if pixel_height > 0 else 1.0
            
            if aspect_ratio > 1.2:  # Side view
                estimated_depth = real_width * 0.4
            elif aspect_ratio < 0.8:  # Front/back view
                estimated_depth = real_height * 0.3
            else:  # Angled view
                estimated_depth = min(real_width, real_height) * 0.35
            
            return {
                'width_cm': real_width,
                'height_cm': real_height,
                'depth_cm': estimated_depth,
                'aspect_ratio': aspect_ratio,
                'distance_m': distance
            }
            
        except Exception as e:
            raise PerspectiveDistortionError(f"Real-world dimension calculation failed: {str(e)}")
    
    def estimate_volume_from_dimensions(
        self,
        dimensions: Dict[str, float]
    ) -> float:
        """
        Estimate chicken volume from real-world dimensions.
        
        Args:
            dimensions: Real-world dimensions dictionary
            
        Returns:
            Estimated volume in cubic centimeters
        """
        try:
            width = dimensions['width_cm']
            height = dimensions['height_cm']
            depth = dimensions['depth_cm']
            
            # Use ellipsoid approximation for chicken body
            # Volume = (4/3) * π * a * b * c
            # where a, b, c are semi-axes
            
            semi_width = width / 2
            semi_height = height / 2
            semi_depth = depth / 2
            
            volume = (4/3) * math.pi * semi_width * semi_height * semi_depth
            
            # Apply correction factor for chicken body shape
            # (chickens are not perfect ellipsoids)
            shape_correction_factor = 0.7
            
            return volume * shape_correction_factor
            
        except Exception as e:
            return 0.0
    
    def set_compensation_parameters(
        self,
        reference_distance: Optional[float] = None,
        size_correction_factor: Optional[float] = None,
        aspect_correction_factor: Optional[float] = None,
        texture_correction_factor: Optional[float] = None
    ) -> None:
        """Update compensation parameters."""
        if reference_distance is not None:
            self.reference_distance = reference_distance
        
        if size_correction_factor is not None:
            self.size_correction_factor = size_correction_factor
        
        if aspect_correction_factor is not None:
            self.aspect_correction_factor = aspect_correction_factor
        
        if texture_correction_factor is not None:
            self.texture_correction_factor = texture_correction_factor
    
    def enable_compensation_features(
        self,
        perspective_correction: bool = True,
        feature_scaling: bool = True,
        bbox_correction: bool = True
    ) -> None:
        """Enable or disable compensation features."""
        self.perspective_correction_enabled = perspective_correction
        self.feature_scaling_enabled = feature_scaling
        self.bbox_correction_enabled = bbox_correction
    
    def get_compensation_info(self) -> Dict[str, Any]:
        """Get information about compensation settings."""
        return {
            'reference_distance': self.reference_distance,
            'perspective_correction_enabled': self.perspective_correction_enabled,
            'feature_scaling_enabled': self.feature_scaling_enabled,
            'bbox_correction_enabled': self.bbox_correction_enabled,
            'correction_factors': {
                'size': self.size_correction_factor,
                'aspect': self.aspect_correction_factor,
                'texture': self.texture_correction_factor
            },
            'camera_parameters': {
                'focal_length': self.camera_params.focal_length,
                'camera_height': self.camera_params.camera_height,
                'tilt_angle': self.camera_params.tilt_angle
            }
        }