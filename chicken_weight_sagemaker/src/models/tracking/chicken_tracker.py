"""
Main chicken tracker implementation with DeepSORT-based tracking.
"""

import time
from typing import List, Optional, Dict, Any
import numpy as np
from collections import defaultdict

from ...core.interfaces.detection import Detection
from ...core.interfaces.tracking import ChickenTracker, TrackedChicken, TrackingResult, OcclusionAwareTracker
from ...core.interfaces.weight_estimation import WeightEstimate
from ...core.exceptions.tracking_exceptions import TrackingError, TrackLostError
from .deepsort_tracker import DeepSORTChickenTracker
from .reid_features import ChickenReIDFeatureExtractor


class ChickenMultiObjectTracker(OcclusionAwareTracker):
    """Enhanced chicken tracker with weight history and occlusion handling."""
    
    def __init__(
        self,
        max_disappeared: int = 30,
        max_distance: float = 100.0,
        min_track_length: int = 5,
        reid_threshold: float = 0.7
    ):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.min_track_length = min_track_length
        self.reid_threshold = reid_threshold
        
        # Initialize components
        self.deepsort_tracker = DeepSORTChickenTracker(
            max_age=max_disappeared,
            min_hits=min_track_length
        )
        self.reid_extractor = ChickenReIDFeatureExtractor()
        
        # Track management
        self.active_tracks: Dict[str, TrackedChicken] = {}
        self.lost_tracks: Dict[str, TrackedChicken] = {}
        self.frame_count = 0
        
        # Statistics
        self.tracking_stats = {
            'total_tracks_created': 0,
            'total_tracks_lost': 0,
            'successful_reidentifications': 0,
            'average_track_length': 0.0
        }
    
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
        start_time = time.time()
        self.frame_count += 1
        current_frame_id = frame_id or str(self.frame_count)
        
        try:
            # Extract ReID features for detections
            self._extract_reid_features(detections)
            
            # Update DeepSORT tracker
            deepsort_tracks = self.deepsort_tracker.update(detections, self.frame_count)
            
            # Convert to TrackedChicken objects and update weights
            updated_tracks = self._convert_to_tracked_chickens(
                deepsort_tracks, weights, current_frame_id
            )
            
            # Handle occlusion tracking
            updated_tracks = self.handle_occlusion_tracking(updated_tracks)
            
            # Update active tracks
            new_tracks = []
            for track in updated_tracks:
                if track.chicken_id not in self.active_tracks:
                    new_tracks.append(track)
                    self.tracking_stats['total_tracks_created'] += 1
                
                self.active_tracks[track.chicken_id] = track
            
            # Clean up lost tracks
            lost_track_ids = self.cleanup_lost_tracks()
            
            processing_time = (time.time() - start_time) * 1000
            
            return TrackingResult(
                tracked_chickens=list(self.active_tracks.values()),
                new_tracks=new_tracks,
                lost_tracks=lost_track_ids,
                processing_time_ms=processing_time,
                frame_id=current_frame_id,
                total_active_tracks=len(self.active_tracks)
            )
            
        except Exception as e:
            raise TrackingError(f"Track update failed: {str(e)}")
    
    def _extract_reid_features(self, detections: List[Detection]) -> None:
        """Extract ReID features for detections that don't have them."""
        for detection in detections:
            if not hasattr(detection, 'reid_features') or detection.reid_features is None:
                # This would typically extract features from the detection's image crop
                # For now, we'll use a placeholder
                detection.reid_features = np.random.rand(128)  # Placeholder
    
    def _convert_to_tracked_chickens(
        self,
        deepsort_tracks: List[Any],
        weights: Optional[List[WeightEstimate]],
        frame_id: str
    ) -> List[TrackedChicken]:
        """Convert DeepSORT tracks to TrackedChicken objects."""
        tracked_chickens = []
        
        for track in deepsort_tracks:
            # Find corresponding weight estimate
            weight_estimate = None
            if weights and len(weights) > 0:
                # Simple matching by index - in practice, would use more sophisticated matching
                track_idx = min(len(weights) - 1, len(tracked_chickens))
                weight_estimate = weights[track_idx]
            
            # Create or update TrackedChicken
            chicken_id = f"chicken_{track.track_id}"
            
            if chicken_id in self.active_tracks:
                # Update existing track
                tracked_chicken = self.active_tracks[chicken_id]
                tracked_chicken.current_detection = track.current_detection
                tracked_chicken.frames_tracked += 1
                tracked_chicken.last_seen_timestamp = time.time()
                
                # Update weight history
                if weight_estimate:
                    tracked_chicken.weight_history.append(weight_estimate)
                    tracked_chicken.current_weight = weight_estimate
                    
                    # Update stable weight using temporal smoothing
                    tracked_chicken.stable_weight = self.get_stable_weight_estimates(
                        chicken_id, window_size=10
                    )
            else:
                # Create new track
                tracked_chicken = TrackedChicken(
                    chicken_id=chicken_id,
                    current_detection=track.current_detection,
                    current_weight=weight_estimate,
                    tracking_status="active",
                    confidence=track.current_detection.confidence,
                    frames_tracked=1,
                    frames_lost=0,
                    last_seen_timestamp=time.time(),
                    weight_history=[weight_estimate] if weight_estimate else [],
                    detection_history=[track.current_detection]
                )
            
            tracked_chickens.append(tracked_chicken)
        
        return tracked_chickens
    
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
        if chicken_id not in self.active_tracks:
            return None
        
        track = self.active_tracks[chicken_id]
        
        if len(track.weight_history) < 3:
            return track.current_weight
        
        # Get recent weight estimates
        recent_weights = track.weight_history[-window_size:]
        
        # Calculate median weight (more robust than mean)
        weights_values = [w.value for w in recent_weights if w is not None]
        
        if not weights_values:
            return track.current_weight
        
        median_weight = np.median(weights_values)
        
        # Calculate confidence based on weight consistency
        weight_std = np.std(weights_values)
        consistency_confidence = max(0.3, 1.0 - (weight_std / median_weight))
        
        # Use the most recent weight estimate as template
        template_weight = recent_weights[-1]
        
        return WeightEstimate(
            value=float(median_weight),
            unit=template_weight.unit,
            confidence=consistency_confidence,
            error_range=f"±{median_weight * 0.15:.2f}kg",  # 15% error range for stable estimates
            distance_compensated=template_weight.distance_compensated,
            occlusion_adjusted=template_weight.occlusion_adjusted,
            age_category=template_weight.age_category,
            method="temporal_smoothing"
        )
    
    def get_active_tracks(self) -> List[TrackedChicken]:
        """Get all currently active tracks."""
        return list(self.active_tracks.values())
    
    def cleanup_lost_tracks(self, max_frames_lost: int = None) -> List[str]:
        """
        Remove tracks that have been lost for too long.
        
        Args:
            max_frames_lost: Maximum frames a track can be lost before removal
            
        Returns:
            List of removed track IDs
        """
        max_frames = max_frames_lost or self.max_disappeared
        current_time = time.time()
        
        lost_track_ids = []
        tracks_to_remove = []
        
        for chicken_id, track in self.active_tracks.items():
            # Check if track has been lost for too long
            time_since_last_seen = current_time - track.last_seen_timestamp
            
            if time_since_last_seen > max_frames:  # Assuming 1 second = lost for too long
                tracks_to_remove.append(chicken_id)
                lost_track_ids.append(chicken_id)
                
                # Move to lost tracks for potential re-identification
                self.lost_tracks[chicken_id] = track
                self.tracking_stats['total_tracks_lost'] += 1
        
        # Remove lost tracks from active tracks
        for chicken_id in tracks_to_remove:
            del self.active_tracks[chicken_id]
        
        return lost_track_ids
    
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
        for track in tracks:
            if (track.current_detection.occlusion_level and 
                track.current_detection.occlusion_level > occlusion_threshold):
                
                # Mark as occluded
                track.tracking_status = "occluded"
                track.frames_lost += 1
                
                # Predict position if heavily occluded
                predicted_bbox = self.predict_occluded_position(track, track.frames_lost)
                
                if predicted_bbox:
                    # Update detection with predicted position
                    track.current_detection.bbox = predicted_bbox
                    track.confidence *= 0.9  # Reduce confidence for predicted positions
            else:
                # Reset occlusion status
                if track.tracking_status == "occluded":
                    track.tracking_status = "active"
                    track.frames_lost = 0
        
        return tracks
    
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
        if len(track.detection_history) < 2:
            return None
        
        try:
            # Simple linear prediction based on recent movement
            recent_detections = track.detection_history[-3:]  # Last 3 detections
            
            if len(recent_detections) < 2:
                return track.current_detection.bbox
            
            # Calculate average movement
            movements = []
            for i in range(1, len(recent_detections)):
                prev_bbox = recent_detections[i-1].bbox
                curr_bbox = recent_detections[i].bbox
                
                # Calculate center movement
                prev_center = [(prev_bbox[0] + prev_bbox[2])/2, (prev_bbox[1] + prev_bbox[3])/2]
                curr_center = [(curr_bbox[0] + curr_bbox[2])/2, (curr_bbox[1] + curr_bbox[3])/2]
                
                movement = [curr_center[0] - prev_center[0], curr_center[1] - prev_center[1]]
                movements.append(movement)
            
            # Average movement per frame
            avg_movement = [np.mean([m[0] for m in movements]), np.mean([m[1] for m in movements])]
            
            # Predict position
            last_bbox = track.current_detection.bbox
            last_center = [(last_bbox[0] + last_bbox[2])/2, (last_bbox[1] + last_bbox[3])/2]
            
            predicted_center = [
                last_center[0] + avg_movement[0] * frames_occluded,
                last_center[1] + avg_movement[1] * frames_occluded
            ]
            
            # Maintain same size
            width = last_bbox[2] - last_bbox[0]
            height = last_bbox[3] - last_bbox[1]
            
            predicted_bbox = [
                predicted_center[0] - width/2,
                predicted_center[1] - height/2,
                predicted_center[0] + width/2,
                predicted_center[1] + height/2
            ]
            
            return predicted_bbox
            
        except Exception:
            return None
    
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
        return self.reid_extractor.extract_features(frame, detection)
    
    def get_tracking_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tracking statistics."""
        active_track_lengths = [track.frames_tracked for track in self.active_tracks.values()]
        
        return {
            'total_active_tracks': len(self.active_tracks),
            'total_lost_tracks': len(self.lost_tracks),
            'total_tracks_created': self.tracking_stats['total_tracks_created'],
            'total_tracks_lost': self.tracking_stats['total_tracks_lost'],
            'successful_reidentifications': self.tracking_stats['successful_reidentifications'],
            'average_track_length': np.mean(active_track_lengths) if active_track_lengths else 0.0,
            'max_track_length': max(active_track_lengths) if active_track_lengths else 0,
            'frames_processed': self.frame_count,
            'track_retention_rate': (
                self.tracking_stats['total_tracks_created'] - self.tracking_stats['total_tracks_lost']
            ) / max(1, self.tracking_stats['total_tracks_created'])
        }
    
    def reset_tracker(self) -> None:
        """Reset the tracker state."""
        self.active_tracks.clear()
        self.lost_tracks.clear()
        self.frame_count = 0
        self.deepsort_tracker.reset()
        
        # Reset statistics
        for key in self.tracking_stats:
            if isinstance(self.tracking_stats[key], (int, float)):
                self.tracking_stats[key] = 0