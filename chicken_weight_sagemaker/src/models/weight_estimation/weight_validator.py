"""
Weight validation utilities for chicken weight estimates.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
from dataclasses import dataclass

from ...core.interfaces.weight_estimation import WeightEstimate
from ...core.exceptions.weight_estimation_exceptions import WeightValidationError
from .age_classifier import ChickenAgeCategory


@dataclass
class ValidationResult:
    """Result of weight validation."""
    is_valid: bool
    confidence: float
    issues: List[str]
    corrected_weight: Optional[float] = None
    validation_details: Dict[str, Any] = None


class WeightValidator:
    """Validates weight estimates for consistency and reasonableness."""
    
    def __init__(
        self,
        history_size: int = 10,
        outlier_threshold: float = 2.0,  # standard deviations
        max_change_rate: float = 0.5,    # kg per frame
        age_tolerance: float = 0.3       # ±30% tolerance for age ranges
    ):
        self.history_size = history_size
        self.outlier_threshold = outlier_threshold
        self.max_change_rate = max_change_rate
        self.age_tolerance = age_tolerance
        
        # Weight history for temporal validation
        self.weight_history = deque(maxlen=history_size)
        self.confidence_history = deque(maxlen=history_size)
        
        # Validation statistics
        self.validation_stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_range_check': 0,
            'failed_temporal_check': 0,
            'failed_age_check': 0,
            'failed_outlier_check': 0
        }
        
        # Age-based weight ranges
        self.age_weight_ranges = {
            'DAY_OLD': (0.035, 0.045),
            'WEEK_1': (0.150, 0.200),
            'WEEK_2': (0.400, 0.500),
            'WEEK_3': (0.800, 1.000),
            'WEEK_4': (1.200, 1.500),
            'WEEK_5': (1.800, 2.200),
            'WEEK_6': (2.500, 3.000),
            'ADULT': (3.000, 5.000)
        }
    
    def validate_weight_estimate(
        self,
        weight_estimate: WeightEstimate,
        previous_weight: Optional[float] = None,
        chicken_id: Optional[str] = None
    ) -> ValidationResult:
        """
        Comprehensive weight estimate validation.
        
        Args:
            weight_estimate: Weight estimate to validate
            previous_weight: Previous weight for temporal validation
            chicken_id: Optional chicken identifier for tracking
            
        Returns:
            ValidationResult with validation details
        """
        self.validation_stats['total_validations'] += 1
        
        issues = []
        validation_details = {}
        
        # 1. Basic range validation
        range_valid = self._validate_weight_range(weight_estimate.value)
        if not range_valid:
            issues.append('weight_out_of_range')
            self.validation_stats['failed_range_check'] += 1
        
        validation_details['range_check'] = range_valid
        
        # 2. Age-based validation
        age_valid = True
        if weight_estimate.age_category:
            age_valid = self._validate_age_consistency(
                weight_estimate.value, 
                weight_estimate.age_category
            )
            if not age_valid:
                issues.append('age_weight_inconsistency')
                self.validation_stats['failed_age_check'] += 1
        
        validation_details['age_check'] = age_valid
        
        # 3. Temporal consistency validation
        temporal_valid = True
        if previous_weight is not None:
            temporal_valid = self._validate_temporal_consistency(
                weight_estimate.value, 
                previous_weight
            )
            if not temporal_valid:
                issues.append('temporal_inconsistency')
                self.validation_stats['failed_temporal_check'] += 1
        
        validation_details['temporal_check'] = temporal_valid
        
        # 4. Statistical outlier detection
        outlier_valid = True
        if len(self.weight_history) >= 3:
            outlier_valid = self._validate_outlier_detection(weight_estimate.value)
            if not outlier_valid:
                issues.append('statistical_outlier')
                self.validation_stats['failed_outlier_check'] += 1
        
        validation_details['outlier_check'] = outlier_valid
        
        # 5. Confidence-based validation
        confidence_valid = weight_estimate.confidence >= 0.3  # Minimum confidence threshold
        if not confidence_valid:
            issues.append('low_confidence')
        
        validation_details['confidence_check'] = confidence_valid
        
        # Overall validity
        is_valid = (
            range_valid and 
            age_valid and 
            temporal_valid and 
            outlier_valid and 
            confidence_valid
        )
        
        if is_valid:
            self.validation_stats['passed_validations'] += 1
        
        # Calculate overall validation confidence
        validation_confidence = self._calculate_validation_confidence(
            weight_estimate, validation_details
        )
        
        # Suggest correction if needed
        corrected_weight = None
        if not is_valid:
            corrected_weight = self._suggest_weight_correction(
                weight_estimate, validation_details
            )
        
        # Update history
        self.weight_history.append(weight_estimate.value)
        self.confidence_history.append(weight_estimate.confidence)
        
        return ValidationResult(
            is_valid=is_valid,
            confidence=validation_confidence,
            issues=issues,
            corrected_weight=corrected_weight,
            validation_details=validation_details
        )
    
    def _validate_weight_range(self, weight: float) -> bool:
        """Validate that weight is within reasonable range for chickens."""
        min_weight = 0.01  # 10 grams minimum
        max_weight = 6.0   # 6 kg maximum (very large chicken)
        
        return min_weight <= weight <= max_weight
    
    def _validate_age_consistency(self, weight: float, age_category: str) -> bool:
        """Validate weight consistency with age category."""
        if age_category not in self.age_weight_ranges:
            return True  # Can't validate unknown age category
        
        min_weight, max_weight = self.age_weight_ranges[age_category]
        
        # Apply tolerance
        tolerance_range = (max_weight - min_weight) * self.age_tolerance
        adjusted_min = max(0.01, min_weight - tolerance_range)
        adjusted_max = max_weight + tolerance_range
        
        return adjusted_min <= weight <= adjusted_max
    
    def _validate_temporal_consistency(
        self, 
        current_weight: float, 
        previous_weight: float
    ) -> bool:
        """Validate temporal consistency between consecutive measurements."""
        weight_change = abs(current_weight - previous_weight)
        
        # Allow for reasonable weight change between measurements
        # Chickens can gain/lose weight, but not dramatically in short time
        return weight_change <= self.max_change_rate
    
    def _validate_outlier_detection(self, weight: float) -> bool:
        """Detect statistical outliers in weight measurements."""
        if len(self.weight_history) < 3:
            return True
        
        # Calculate statistics from recent history
        recent_weights = list(self.weight_history)
        mean_weight = np.mean(recent_weights)
        std_weight = np.std(recent_weights)
        
        if std_weight == 0:
            return True  # No variation, can't detect outliers
        
        # Z-score based outlier detection
        z_score = abs(weight - mean_weight) / std_weight
        
        return z_score <= self.outlier_threshold
    
    def _calculate_validation_confidence(
        self,
        weight_estimate: WeightEstimate,
        validation_details: Dict[str, bool]
    ) -> float:
        """Calculate overall validation confidence."""
        base_confidence = weight_estimate.confidence
        
        # Adjust confidence based on validation results
        validation_multiplier = 1.0
        
        if not validation_details.get('range_check', True):
            validation_multiplier *= 0.2
        
        if not validation_details.get('age_check', True):
            validation_multiplier *= 0.6
        
        if not validation_details.get('temporal_check', True):
            validation_multiplier *= 0.7
        
        if not validation_details.get('outlier_check', True):
            validation_multiplier *= 0.8
        
        if not validation_details.get('confidence_check', True):
            validation_multiplier *= 0.5
        
        # Consider distance compensation and occlusion adjustment
        if weight_estimate.distance_compensated:
            validation_multiplier *= 1.1  # Slight boost for distance compensation
        
        if weight_estimate.occlusion_adjusted:
            validation_multiplier *= 0.9  # Slight penalty for occlusion
        
        final_confidence = base_confidence * validation_multiplier
        
        return min(1.0, max(0.0, final_confidence))
    
    def _suggest_weight_correction(
        self,
        weight_estimate: WeightEstimate,
        validation_details: Dict[str, bool]
    ) -> Optional[float]:
        """Suggest a corrected weight for invalid estimates."""
        try:
            current_weight = weight_estimate.value
            
            # Method 1: Use median of recent valid weights
            if len(self.weight_history) >= 3:
                recent_weights = list(self.weight_history)
                median_weight = np.median(recent_weights)
                
                if self._validate_weight_range(median_weight):
                    return median_weight
            
            # Method 2: Adjust to age category range
            if weight_estimate.age_category and not validation_details.get('age_check', True):
                if weight_estimate.age_category in self.age_weight_ranges:
                    min_weight, max_weight = self.age_weight_ranges[weight_estimate.age_category]
                    
                    # Clamp to age range
                    if current_weight < min_weight:
                        return min_weight
                    elif current_weight > max_weight:
                        return max_weight
            
            # Method 3: Use previous weight if temporal consistency failed
            if (not validation_details.get('temporal_check', True) and 
                len(self.weight_history) > 0):
                return self.weight_history[-1]
            
            # Method 4: Clamp to reasonable range
            if not validation_details.get('range_check', True):
                return max(0.5, min(5.0, current_weight))
            
            return None
            
        except Exception:
            return None
    
    def validate_batch_weights(
        self,
        weight_estimates: List[WeightEstimate],
        previous_weights: Optional[List[float]] = None
    ) -> List[ValidationResult]:
        """Validate multiple weight estimates."""
        results = []
        
        for i, estimate in enumerate(weight_estimates):
            prev_weight = previous_weights[i] if previous_weights and i < len(previous_weights) else None
            
            result = self.validate_weight_estimate(estimate, prev_weight)
            results.append(result)
        
        return results
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total = self.validation_stats['total_validations']
        
        if total == 0:
            return {'status': 'no_validations'}
        
        return {
            'total_validations': total,
            'success_rate': self.validation_stats['passed_validations'] / total,
            'failure_rates': {
                'range_check': self.validation_stats['failed_range_check'] / total,
                'age_check': self.validation_stats['failed_age_check'] / total,
                'temporal_check': self.validation_stats['failed_temporal_check'] / total,
                'outlier_check': self.validation_stats['failed_outlier_check'] / total
            },
            'recent_weights': list(self.weight_history),
            'weight_stability': np.std(list(self.weight_history)) if len(self.weight_history) > 1 else 0.0,
            'average_confidence': np.mean(list(self.confidence_history)) if self.confidence_history else 0.0
        }
    
    def reset_history(self) -> None:
        """Reset validation history."""
        self.weight_history.clear()
        self.confidence_history.clear()
        
        # Reset statistics
        for key in self.validation_stats:
            self.validation_stats[key] = 0
    
    def update_validation_parameters(
        self,
        outlier_threshold: Optional[float] = None,
        max_change_rate: Optional[float] = None,
        age_tolerance: Optional[float] = None
    ) -> None:
        """Update validation parameters."""
        if outlier_threshold is not None:
            self.outlier_threshold = outlier_threshold
        
        if max_change_rate is not None:
            self.max_change_rate = max_change_rate
        
        if age_tolerance is not None:
            self.age_tolerance = age_tolerance
    
    def get_weight_trend_analysis(self) -> Dict[str, Any]:
        """Analyze weight trends from history."""
        if len(self.weight_history) < 3:
            return {'status': 'insufficient_data'}
        
        weights = list(self.weight_history)
        
        # Calculate trend
        x = np.arange(len(weights))
        coeffs = np.polyfit(x, weights, 1)
        trend_slope = coeffs[0]
        
        # Determine trend direction
        if abs(trend_slope) < 0.01:
            trend_direction = 'stable'
        elif trend_slope > 0:
            trend_direction = 'increasing'
        else:
            trend_direction = 'decreasing'
        
        # Calculate variability
        weight_std = np.std(weights)
        weight_cv = weight_std / np.mean(weights) if np.mean(weights) > 0 else 0
        
        return {
            'status': 'analyzed',
            'trend_direction': trend_direction,
            'trend_slope': trend_slope,
            'weight_variability': weight_std,
            'coefficient_of_variation': weight_cv,
            'recent_average': np.mean(weights[-5:]) if len(weights) >= 5 else np.mean(weights),
            'min_weight': min(weights),
            'max_weight': max(weights)
        }