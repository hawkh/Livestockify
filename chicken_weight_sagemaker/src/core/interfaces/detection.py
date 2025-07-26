"""
Detection interfaces and data models.
"""

from dataclasses import dataclass
from typing import List, Optional
from abc import ABC, abstractmethod
import numpy as np


@dataclass
class Detection:
    """Represents a single chicken detection."""
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    occlusion_level: Optional[float] = None
    visibility_score: Optional[float] = None


@dataclass
class DetectionResult:
    """Result of detection inference on a frame."""
    detections: List[Detection]
    processing_time_ms: float
    frame_id: Optional[str] = None
    timestamp: Optional[str] = None


class DetectionModel(ABC):
    """Abstract base class for detection models."""
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Detect chickens in a frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            DetectionResult containing all detections
        """
        pass
    
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """Load the detection model from path."""
        pass
    
    @abstractmethod
    def set_confidence_threshold(self, threshold: float) -> None:
        """Set minimum confidence threshold for detections."""
        pass


class OcclusionAwareDetector(DetectionModel):
    """Interface for occlusion-aware detection models."""
    
    @abstractmethod
    def detect_with_occlusion_handling(
        self, 
        frame: np.ndarray, 
        previous_detections: Optional[List[Detection]] = None
    ) -> DetectionResult:
        """
        Detect chickens with occlusion handling.
        
        Args:
            frame: Current frame
            previous_detections: Detections from previous frame for temporal consistency
            
        Returns:
            DetectionResult with occlusion information
        """
        pass
    
    @abstractmethod
    def estimate_occlusion_level(self, detection: Detection, frame: np.ndarray) -> float:
        """
        Estimate occlusion level for a detection.
        
        Args:
            detection: Detection to analyze
            frame: Frame containing the detection
            
        Returns:
            Occlusion level between 0.0 (no occlusion) and 1.0 (fully occluded)
        """
        pass