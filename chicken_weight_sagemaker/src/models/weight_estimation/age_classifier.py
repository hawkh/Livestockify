"""
Age classification for chickens to improve weight estimation accuracy.
"""

import numpy as np
import math
from typing import Optional, Dict, Any
from enum import Enum

from ...core.exceptions.weight_estimation_exceptions import WeightValidationError


class ChickenAgeCategory(Enum):
    """Chicken age categories with typical weight ranges."""
    DAY_OLD = (0.035, 0.045)  # 35-45 grams
    WEEK_1 = (0.150, 0.200)   # 150-200 grams
    WEEK_2 = (0.400, 0.500)   # 400-500 grams
    WEEK_3 = (0.800, 1.000)   # 800g-1kg
    WEEK_4 = (1.200, 1.500)   # 1.2-1.5kg
    WEEK_5 = (1.800, 2.200)   # 1.8-2.2kg
    WEEK_6 = (2.500, 3.000)   # 2.5-3kg
    ADULT = (3.000, 5.000)    # 3-5kg


class ChickenAgeClassifier:
    """Classifies chicken age based on visual features and size."""
    
    def __init__(self):
        self.age_categories = ChickenAgeCategory
        
        # Size thresholds for age classification (in pixels at reference distance)
        self.size_thresholds = {
            'area': {
                ChickenAgeCategory.DAY_OLD: (0, 800),
                ChickenAgeCategory.WEEK_1: (800, 2000),
                ChickenAgeCategory.WEEK_2: (2000, 4000),
                ChickenAgeCategory.WEEK_3: (4000, 7000),
                ChickenAgeCategory.WEEK_4: (7000, 10000),
                ChickenAgeCategory.WEEK_5: (10000, 15000),
                ChickenAgeCategory.WEEK_6: (15000, 20000),
                ChickenAgeCategory.ADULT: (20000, float('inf'))
            },
            'width': {
                ChickenAgeCategory.DAY_OLD: (0, 20),
                ChickenAgeCategory.WEEK_1: (20, 40),
                ChickenAgeCategory.WEEK_2: (40, 60),
                ChickenAgeCategory.WEEK_3: (60, 80),
                ChickenAgeCategory.WEEK_4: (80, 100),
                ChickenAgeCategory.WEEK_5: (100, 120),
                ChickenAgeCategory.WEEK_6: (120, 150),
                ChickenAgeCategory.ADULT: (150, float('inf'))
            }
        }
        
        # Feature-based classification parameters
        self.feature_weights = {
            'size': 0.6,
            'aspect_ratio': 0.2,
            'texture': 0.1,
            'color': 0.1
        }
    
    def classify_age_from_features(
        self, 
        features: np.ndarray,
        distance: Optional[float] = None
    ) -> ChickenAgeCategory:
        """
        Classify chicken age based on extracted features.
        
        Args:
            features: Feature vector from ChickenFeatureExtractor
            distance: Distance to chicken for size compensation
            
        Returns:
            Estimated age category
        """
        try:
            # Extract relevant features (assuming standard feature order)
            width = features[0] if len(features) > 0 else 50
            height = features[1] if len(features) > 1 else 50
            area = features[2] if len(features) > 2 else 2500
            aspect_ratio = features[3] if len(features) > 3 else 1.0
            
            # Compensate for distance if provided
            if distance is not None:
                reference_distance = 3.0  # meters
                distance_factor = (distance / reference_distance) ** 2
                area = area / distance_factor
                width = width / math.sqrt(distance_factor)
            
            # Size-based classification (primary method)
            size_age = self._classify_by_size(area, width)
            
            # Aspect ratio analysis (secondary method)
            aspect_age = self._classify_by_aspect_ratio(aspect_ratio)
            
            # Texture analysis (tertiary method)
            texture_age = self._classify_by_texture(features)
            
            # Combine classifications with weights
            age_scores = self._combine_age_classifications(
                size_age, aspect_age, texture_age
            )
            
            # Return most likely age category
            best_age = max(age_scores.items(), key=lambda x: x[1])[0]
            
            return best_age
            
        except Exception:
            # Default to adult if classification fails
            return ChickenAgeCategory.ADULT
    
    def _classify_by_size(
        self, 
        area: float, 
        width: float
    ) -> Dict[ChickenAgeCategory, float]:
        """Classify age based on size features."""
        age_scores = {}
        
        # Area-based classification
        for age_cat, (min_area, max_area) in self.size_thresholds['area'].items():
            if min_area <= area <= max_area:
                # Score based on how well the area fits the range
                range_center = (min_area + max_area) / 2
                range_width = max_area - min_area
                
                if range_width > 0:
                    distance_from_center = abs(area - range_center)
                    score = max(0.0, 1.0 - (distance_from_center / (range_width / 2)))
                else:
                    score = 1.0
                
                age_scores[age_cat] = score
            else:
                age_scores[age_cat] = 0.0
        
        # Width-based classification (secondary validation)
        width_scores = {}
        for age_cat, (min_width, max_width) in self.size_thresholds['width'].items():
            if min_width <= width <= max_width:
                range_center = (min_width + max_width) / 2
                range_width = max_width - min_width
                
                if range_width > 0:
                    distance_from_center = abs(width - range_center)
                    score = max(0.0, 1.0 - (distance_from_center / (range_width / 2)))
                else:
                    score = 1.0
                
                width_scores[age_cat] = score
            else:
                width_scores[age_cat] = 0.0
        
        # Combine area and width scores
        combined_scores = {}
        for age_cat in ChickenAgeCategory:
            area_score = age_scores.get(age_cat, 0.0)
            width_score = width_scores.get(age_cat, 0.0)
            combined_scores[age_cat] = 0.7 * area_score + 0.3 * width_score
        
        return combined_scores
    
    def _classify_by_aspect_ratio(self, aspect_ratio: float) -> Dict[ChickenAgeCategory, float]:
        """Classify age based on aspect ratio (body proportions)."""
        age_scores = {}
        
        # Aspect ratio patterns by age:
        # Young chickens: more compact (lower aspect ratio)
        # Adult chickens: more elongated (higher aspect ratio)
        
        aspect_preferences = {
            ChickenAgeCategory.DAY_OLD: (0.8, 1.2),
            ChickenAgeCategory.WEEK_1: (0.9, 1.3),
            ChickenAgeCategory.WEEK_2: (1.0, 1.4),
            ChickenAgeCategory.WEEK_3: (1.1, 1.5),
            ChickenAgeCategory.WEEK_4: (1.2, 1.6),
            ChickenAgeCategory.WEEK_5: (1.3, 1.7),
            ChickenAgeCategory.WEEK_6: (1.4, 1.8),
            ChickenAgeCategory.ADULT: (1.5, 2.0)
        }
        
        for age_cat, (min_ratio, max_ratio) in aspect_preferences.items():
            if min_ratio <= aspect_ratio <= max_ratio:
                # Score based on how well the ratio fits the preferred range
                range_center = (min_ratio + max_ratio) / 2
                range_width = max_ratio - min_ratio
                
                distance_from_center = abs(aspect_ratio - range_center)
                score = max(0.0, 1.0 - (distance_from_center / (range_width / 2)))
                
                age_scores[age_cat] = score
            else:
                # Penalty for being outside preferred range
                if aspect_ratio < min_ratio:
                    penalty = (min_ratio - aspect_ratio) / min_ratio
                else:
                    penalty = (aspect_ratio - max_ratio) / max_ratio
                
                age_scores[age_cat] = max(0.0, 1.0 - penalty)
        
        return age_scores
    
    def _classify_by_texture(self, features: np.ndarray) -> Dict[ChickenAgeCategory, float]:
        """Classify age based on texture features (feather development)."""
        age_scores = {}
        
        # Extract texture features (assuming they're at specific indices)
        texture_energy = features[10] if len(features) > 10 else 0.5
        edge_density = features[12] if len(features) > 12 else 0.5
        
        # Texture patterns by age:
        # Young chickens: smoother texture (lower texture energy)
        # Adult chickens: more detailed feathers (higher texture energy)
        
        texture_preferences = {
            ChickenAgeCategory.DAY_OLD: (0.0, 0.3),
            ChickenAgeCategory.WEEK_1: (0.1, 0.4),
            ChickenAgeCategory.WEEK_2: (0.2, 0.5),
            ChickenAgeCategory.WEEK_3: (0.3, 0.6),
            ChickenAgeCategory.WEEK_4: (0.4, 0.7),
            ChickenAgeCategory.WEEK_5: (0.5, 0.8),
            ChickenAgeCategory.WEEK_6: (0.6, 0.9),
            ChickenAgeCategory.ADULT: (0.7, 1.0)
        }
        
        for age_cat, (min_texture, max_texture) in texture_preferences.items():
            # Score based on texture energy
            if min_texture <= texture_energy <= max_texture:
                range_center = (min_texture + max_texture) / 2
                range_width = max_texture - min_texture
                
                if range_width > 0:
                    distance_from_center = abs(texture_energy - range_center)
                    score = max(0.0, 1.0 - (distance_from_center / (range_width / 2)))
                else:
                    score = 1.0
                
                age_scores[age_cat] = score
            else:
                age_scores[age_cat] = 0.1  # Small baseline score
        
        return age_scores
    
    def _combine_age_classifications(
        self,
        size_scores: Dict[ChickenAgeCategory, float],
        aspect_scores: Dict[ChickenAgeCategory, float],
        texture_scores: Dict[ChickenAgeCategory, float]
    ) -> Dict[ChickenAgeCategory, float]:
        """Combine different age classification methods."""
        combined_scores = {}
        
        for age_cat in ChickenAgeCategory:
            size_score = size_scores.get(age_cat, 0.0)
            aspect_score = aspect_scores.get(age_cat, 0.0)
            texture_score = texture_scores.get(age_cat, 0.0)
            
            # Weighted combination
            combined_score = (
                self.feature_weights['size'] * size_score +
                self.feature_weights['aspect_ratio'] * aspect_score +
                self.feature_weights['texture'] * texture_score
            )
            
            combined_scores[age_cat] = combined_score
        
        return combined_scores
    
    def get_weight_range_for_age(self, age_category: ChickenAgeCategory) -> tuple:
        """Get expected weight range for age category."""
        return age_category.value
    
    def validate_weight_for_age(
        self, 
        weight: float, 
        age_category: ChickenAgeCategory,
        tolerance: float = 0.3
    ) -> bool:
        """
        Validate if weight is reasonable for the estimated age.
        
        Args:
            weight: Estimated weight in kg
            age_category: Estimated age category
            tolerance: Tolerance factor (±30% by default)
            
        Returns:
            True if weight is within expected range
        """
        min_weight, max_weight = age_category.value
        
        # Apply tolerance
        tolerance_range = (max_weight - min_weight) * tolerance
        adjusted_min = max(0.01, min_weight - tolerance_range)
        adjusted_max = max_weight + tolerance_range
        
        is_valid = adjusted_min <= weight <= adjusted_max
        
        if not is_valid:
            raise WeightValidationError(weight, age_category.name)
        
        return is_valid
    
    def get_age_classification_confidence(
        self, 
        age_scores: Dict[ChickenAgeCategory, float]
    ) -> float:
        """Calculate confidence in age classification."""
        if not age_scores:
            return 0.0
        
        scores = list(age_scores.values())
        max_score = max(scores)
        
        if max_score == 0:
            return 0.0
        
        # Calculate confidence based on separation between top scores
        sorted_scores = sorted(scores, reverse=True)
        
        if len(sorted_scores) > 1:
            separation = sorted_scores[0] - sorted_scores[1]
            confidence = max_score * (1.0 + separation)
        else:
            confidence = max_score
        
        return min(1.0, confidence)
    
    def classify_age_from_weight(self, weight: float) -> ChickenAgeCategory:
        """Classify age based on weight (fallback method)."""
        for age_cat in ChickenAgeCategory:
            min_weight, max_weight = age_cat.value
            if min_weight <= weight <= max_weight:
                return age_cat
        
        # If weight doesn't fit any category, return closest
        if weight < ChickenAgeCategory.DAY_OLD.value[0]:
            return ChickenAgeCategory.DAY_OLD
        else:
            return ChickenAgeCategory.ADULT