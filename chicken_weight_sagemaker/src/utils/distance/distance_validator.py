"""
Distance estimation validation utilities.
"""

import math
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from collections import deque

from ...core.interfaces.camera import CameraParameters
from ...core.exceptions.camera_exceptions import DistanceValidationError


class DistanceValidator:
    """Validates distance estimates for consistency and reasonableness."""
    
    def __init__(
        self,
        camera_params: CameraParameters,
        history_size: int = 10,
        max_change_rate: float = 2.0,  # meters per frame
        outlier_threshold: float = 2.0  # standard deviations
    ):
        self.camera_params = camera_params
        self.history_size = history_size
        self.max_change_rate = max_change_rate
        self.outlier_threshold = outlier_threshold
        
        # Distance history for temporal validation
        self.distance_history = deque(maxlen=history_size)
        self.bbox_history = deque(maxlen=history_size)
        
        # Validation statistics
        self.validation_stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_range_check': 0,
            'failed_temporal_check': 0,
            'failed_consistency_check': 0
        }
    
    def validate_distance(
        self,
        distance: float,
        bbox: List[float],
        frame_shape: Tuple[int, int],
        previous_distance: Optional[float] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Comprehensive distance validation.
        
        Args:
            distance: Estimated distance to validate
            bbox: Bounding box coordinates
            frame_shape: Frame dimensions
            previous_distance: Previous frame distance for temporal validation
            
        Returns:
            Tuple of (is_valid, validation_details)
        """
        self.validation_stats['total_validations'] += 1
        
        validation_results = {
            'range_check': True,
            'temporal_check': True,
            'consistency_check': True,
            'outlier_check': True,
            'overall_valid': True,
            'confidence': 1.0,
            'issues': []
        }
        
        # 1. Range validation
        if not self._validate_distance_range(distance):
            validation_results['range_check'] = False
            validation_results['issues'].append('distance_out_of_range')
            self.validation_stats['failed_range_check'] += 1
        
        # 2. Temporal consistency validation
        if previous_distance is not None:
            if not self._validate_temporal_consistency(distance, previous_distance):
                validation_results['temporal_check'] = False
                validation_results['issues'].append('temporal_inconsistency')
                self.validation_stats['failed_temporal_check'] += 1
        
        # 3. Bbox-distance consistency validation
        if not self._validate_bbox_consistency(distance, bbox, frame_shape):
            validation_results['consistency_check'] = False
            validation_results['issues'].append('bbox_distance_inconsistency')
            self.validation_stats['failed_consistency_check'] += 1
        
        # 4. Outlier detection using history
        if len(self.distance_history) >= 3:
            if not self._validate_outlier_detection(distance):
                validation_results['outlier_check'] = False
                validation_results['issues'].append('statistical_outlier')
        
        # Calculate overall validity and confidence
        validation_results['overall_valid'] = (
            validation_results['range_check'] and
            validation_results['temporal_check'] and
            validation_results['consistency_check'] and
            validation_results['outlier_check']
        )
        
        if validation_results['overall_valid']:
            self.validation_stats['passed_validations'] += 1
        
        # Calculate confidence based on validation results
        validation_results['confidence'] = self._calculate_validation_confidence(
            validation_results
        )
        
        # Update history
        self.distance_history.append(distance)
        self.bbox_history.append(bbox)
        
        return validation_results['overall_valid'], validation_results
    
    def _validate_distance_range(self, distance: float) -> bool:
        """Validate that distance is within reasonable range."""
        min_distance = 0.5  # 50cm minimum
        max_distance = 20.0  # 20m maximum for farm environments
        
        return min_distance <= distance <= max_distance
    
    def _validate_temporal_consistency(
        self, 
        current_distance: float, 
        previous_distance: float
    ) -> bool:
        """Validate temporal consistency between consecutive frames."""
        distance_change = abs(current_distance - previous_distance)
        
        # Allow for reasonable movement between frames
        # Assuming 30 FPS, chickens can move at most ~1 m/s
        return distance_change <= self.max_change_rate
    
    def _validate_bbox_consistency(
        self,
        distance: float,
        bbox: List[float],
        frame_shape: Tuple[int, int]
    ) -> bool:
        """Validate consistency between distance and bounding box size."""
        try:
            x1, y1, x2, y2 = bbox
            bbox_width = x2 - x1
            
            # Expected pixel width at given distance
            expected_pixel_width = (
                self.camera_params.known_object_width * 
                self.camera_params.focal_length / (distance * 100)
            )
            
            # Allow for ±60% variation to account for chicken size variation
            # and perspective effects
            min_expected = expected_pixel_width * 0.4
            max_expected = expected_pixel_width * 1.6
            
            return min_expected <= bbox_width <= max_expected
            
        except (ZeroDivisionError, ValueError):
            return True  # If calculation fails, assume valid
    
    def _validate_outlier_detection(self, distance: float) -> bool:
        """Detect statistical outliers in distance measurements."""
        if len(self.distance_history) < 3:
            return True
        
        # Calculate statistics from recent history
        recent_distances = list(self.distance_history)
        mean_distance = np.mean(recent_distances)
        std_distance = np.std(recent_distances)
        
        if std_distance == 0:
            return True  # No variation, can't detect outliers
        
        # Z-score based outlier detection
        z_score = abs(distance - mean_distance) / std_distance
        
        return z_score <= self.outlier_threshold
    
    def _calculate_validation_confidence(self, validation_results: Dict[str, Any]) -> float:
        """Calculate confidence score based on validation results."""
        base_confidence = 1.0
        
        # Reduce confidence for each failed check
        if not validation_results['range_check']:
            base_confidence *= 0.3
        
        if not validation_results['temporal_check']:
            base_confidence *= 0.7
        
        if not validation_results['consistency_check']:
            base_confidence *= 0.5
        
        if not validation_results['outlier_check']:
            base_confidence *= 0.8
        
        return max(0.1, base_confidence)
    
    def suggest_corrected_distance(
        self,
        invalid_distance: float,
        bbox: List[float],
        frame_shape: Tuple[int, int]
    ) -> Optional[float]:
        """Suggest a corrected distance for invalid measurements."""
        try:
            # Method 1: Use median of recent valid distances
            if len(self.distance_history) >= 3:
                recent_distances = list(self.distance_history)
                median_distance = np.median(recent_distances)
                
                # Check if median is reasonable
                if self._validate_distance_range(median_distance):
                    return median_distance
            
            # Method 2: Recalculate based on bbox size
            x1, y1, x2, y2 = bbox
            bbox_width = x2 - x1
            
            if bbox_width > 0:
                corrected_distance = (
                    self.camera_params.known_object_width * 
                    self.camera_params.focal_length / (bbox_width * 100)
                )
                
                # Clamp to reasonable range
                corrected_distance = max(0.5, min(20.0, corrected_distance))
                return corrected_distance
            
            # Method 3: Use default distance based on camera setup
            return self.camera_params.camera_height * 1.5  # Reasonable default
            
        except Exception:
            return None
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total = self.validation_stats['total_validations']
        
        if total == 0:
            return {'status': 'no_validations'}
        
        return {
            'total_validations': total,
            'success_rate': self.validation_stats['passed_validations'] / total,
            'range_failure_rate': self.validation_stats['failed_range_check'] / total,
            'temporal_failure_rate': self.validation_stats['failed_temporal_check'] / total,
            'consistency_failure_rate': self.validation_stats['failed_consistency_check'] / total,
            'recent_distances': list(self.distance_history),
            'distance_stability': np.std(list(self.distance_history)) if len(self.distance_history) > 1 else 0.0
        }
    
    def reset_history(self) -> None:
        """Reset validation history."""
        self.distance_history.clear()
        self.bbox_history.clear()
        
        # Reset statistics
        for key in self.validation_stats:
            self.validation_stats[key] = 0
    
    def update_parameters(
        self,
        max_change_rate: Optional[float] = None,
        outlier_threshold: Optional[float] = None
    ) -> None:
        """Update validation parameters."""
        if max_change_rate is not None:
            self.max_change_rate = max_change_rate
        
        if outlier_threshold is not None:
            self.outlier_threshold = outlier_threshold