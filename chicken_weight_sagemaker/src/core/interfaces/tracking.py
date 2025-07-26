"""
Multi-object tracking interfaces and data models.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
import numpy as np
from datetime import datetime
from .detection import Detection
from .weight_estimation import WeightEstimate


@dataclass
class TrackedChicken:
    """Represents a tracked chicken across multiple frames."""
    chicken_id: str
    current_detection: Detection
    current_weight: Optional[WeightEstimate] = None
    tracking_status: str = "active"  # active, lost, stable
    confidence: float = 0.0
    frames_tracked: int = 0
    frames_lost: int = 0
    last_seen_timestamp: Optional[datetime] = None
    weight_history: List[WeightEstimate] = field(default_factory=list)
    detection_history: List[Detection] = field(default_factory=list)
    stable_weight: Optional[WeightEstimate] = None


@dataclass
class TrackingResult:
    """Result of tracking update for a frame."""
    tracked_chickens: List[TrackedChicken]
    new_tracks: List[TrackedChicken]
    lost_tracks: List[str]  # IDs of lost tracks
    processing_time_ms: float
    frame_id: Optional[str] = None
    total_active_tracks: int = 0


class ChickenTracker(ABC):
    """Abstract base class for chicken tracking."""
    
    @abstractmethod
    def update_tracks(
        self, 
        detections: List[Detection], 
        weights: Optional[List[WeightEstimate]] = None,
        frame_id: Optional[str] = None
    ) -> TrackingResult:
        """
        Update tracks with new detections and weights.
        
        Args:
            detections: Current frame detections
            weights: Corresponding weight estimates
            frame_id: Optional frame identifier
            
        Returns:
            TrackingResult with updated tracks
        """
        pass
    
    @abstractmethod
    def get_stable_weight_estimates(
        self, 
        chicken_id: str, 
        window_size: int = 10
    ) -> Optional[WeightEstimate]:
        """
        Get stable weight estimate for a chicken using temporal smoothing.
        
        Args:
            chicken_id: ID of the tracked chicken
            window_size: Number of recent estimates to consider
            
        Returns:
            Smoothed weight estimate or None if insufficient data
        """
        pass
    
    @abstractmethod
    def get_active_tracks(self) -> List[TrackedChicken]:
        """Get all currently active tracks."""
        pass
    
    @abstractmethod
    def cleanup_lost_tracks(self, max_frames_lost: int = 30) -> List[str]:
        """
        Remove tracks that have been lost for too long.
        
        Args:
            max_frames_lost: Maximum frames a track can be lost before removal
            
        Returns:
            List of removed track IDs
        """
        pass


class OcclusionAwareTracker(ChickenTracker):
    """Interface for occlusion-aware tracking."""
    
    @abstractmethod
    def handle_occlusion_tracking(
        self, 
        tracks: List[TrackedChicken],
        occlusion_threshold: float = 0.7
    ) -> List[TrackedChicken]:
        """
        Handle tracking through occlusions.
        
        Args:
            tracks: Current tracks
            occlusion_threshold: Threshold above which to use occlusion handling
            
        Returns:
            Updated tracks with occlusion handling
        """
        pass
    
    @abstractmethod
    def predict_occluded_position(
        self, 
        track: TrackedChicken, 
        frames_occluded: int
    ) -> Optional[List[float]]:
        """
        Predict position of occluded chicken.
        
        Args:
            track: Tracked chicken
            frames_occluded: Number of frames the chicken has been occluded
            
        Returns:
            Predicted bounding box or None if prediction not possible
        """
        pass
    
    @abstractmethod
    def calculate_reidentification_features(
        self, 
        frame: np.ndarray, 
        detection: Detection
    ) -> np.ndarray:
        """
        Calculate features for chicken re-identification.
        
        Args:
            frame: Input frame
            detection: Detection to extract features from
            
        Returns:
            Feature vector for re-identification
        """
        pass