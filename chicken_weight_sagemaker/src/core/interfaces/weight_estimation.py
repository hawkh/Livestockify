"""
Weight estimation interfaces and data models.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
import numpy as np
from .detection import Detection


@dataclass
class WeightEstimate:
    """Represents a weight estimate for a chicken."""
    value: float  # Weight in kg
    unit: str = "kg"
    confidence: float = 0.0  # Confidence score 0-1
    error_range: str = "±0.5kg"  # Expected error range
    distance_compensated: bool = False
    occlusion_adjusted: bool = False
    age_category: Optional[str] = None
    method: str = "distance_adaptive_nn"
    features: Optional[Dict[str, float]] = None


@dataclass
class WeightEstimationResult:
    """Result of weight estimation for multiple chickens."""
    estimates: List[WeightEstimate]
    processing_time_ms: float
    average_weight: Optional[float] = None
    total_chickens: int = 0


class WeightEstimationModel(ABC):
    """Abstract base class for weight estimation models."""
    
    @abstractmethod
    def estimate_weight(
        self, 
        frame: np.ndarray, 
        detection: Detection
    ) -> WeightEstimate:
        """
        Estimate weight for a single chicken detection.
        
        Args:
            frame: Input frame containing the chicken
            detection: Detection information
            
        Returns:
            WeightEstimate for the chicken
        """
        pass
    
    @abstractmethod
    def estimate_batch_weights(
        self, 
        frame: np.ndarray, 
        detections: List[Detection]
    ) -> WeightEstimationResult:
        """
        Estimate weights for multiple chicken detections.
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            WeightEstimationResult for all chickens
        """
        pass
    
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """Load the weight estimation model."""
        pass


class DistanceAdaptiveWeightModel(WeightEstimationModel):
    """Interface for distance-adaptive weight estimation."""
    
    @abstractmethod
    def estimate_weight_with_distance(
        self,
        frame: np.ndarray,
        detection: Detection,
        distance: float,
        occlusion_level: float = 0.0
    ) -> WeightEstimate:
        """
        Estimate weight with distance compensation.
        
        Args:
            frame: Input frame
            detection: Detection information
            distance: Estimated distance to chicken in meters
            occlusion_level: Level of occlusion (0-1)
            
        Returns:
            Distance-compensated WeightEstimate
        """
        pass
    
    @abstractmethod
    def extract_distance_compensated_features(
        self,
        frame: np.ndarray,
        bbox: List[float],
        distance: float
    ) -> np.ndarray:
        """
        Extract features compensated for distance.
        
        Args:
            frame: Input frame
            bbox: Bounding box coordinates
            distance: Distance to chicken
            
        Returns:
            Feature vector as numpy array
        """
        pass
    
    @abstractmethod
    def validate_weight_range(
        self, 
        weight: float, 
        estimated_age: Optional[str] = None
    ) -> bool:
        """
        Validate if weight is within expected range for age.
        
        Args:
            weight: Estimated weight in kg
            estimated_age: Estimated age category
            
        Returns:
            True if weight is within expected range
        """
        pass