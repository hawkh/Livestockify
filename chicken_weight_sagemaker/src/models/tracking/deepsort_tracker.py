"""
DeepSORT-based tracker implementation for chicken tracking.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from scipy.optimize import linear_sum_assignment
from collections import deque
import time

from ...core.interfaces.detection import Detection
from ...core.exceptions.tracking_exceptions import TrackingError, TrackInitializationError
from .kalman_filter import ChickenKalmanFilter


class DeepSORTTrack:
    """Individual track in DeepSORT tracker."""
    
    def __init__(self, track_id: int, detection: Detection, frame_id: int):
        self.track_id = track_id
        self.current_detection = detection
        self.detection_history = deque([detection], maxlen=30)
        self.frame_ids = deque([frame_id], maxlen=30)
        
        # Initialize Kalman filter
        self.kalman_filter = ChickenKalmanFilter()
        bbox = detection.bbox
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        self.kalman_filter.initiate([center_x, center_y, bbox[2] - bbox[0], bbox[3] - bbox[1]])
        
        # Track state
        self.state = "tentative"  # tentative, confirmed, deleted
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        
        # Feature history for re-identification
        self.feature_history = deque(maxlen=10)
        if hasattr(detection, 'reid_features') and detection.reid_features is not None:
            self.feature_history.append(detection.reid_features)
        
        # Confidence tracking
        self.confidence_history = deque([detection.confidence], maxlen=10)
        
    def predict(self):
        """Predict next state using Kalman filter."""
        self.kalman_filter.predict()
        self.age += 1
        self.time_since_update += 1
    
    def update(self, detection: Detection, frame_id: int):
        """Update track with new detection."""
        self.current_detection = detection
        self.detection_history.append(detection)
        self.frame_ids.append(frame_id)
        
        # Update Kalman filter
        bbox = detection.bbox
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        measurement = [center_x, center_y, bbox[2] - bbox[0], bbox[3] - bbox[1]]
        self.kalman_filter.update(measurement)
        
        # Update features
        if hasattr(detection, 'reid_features') and detection.reid_features is not None:
            self.feature_history.append(detection.reid_features)
        
        # Update confidence
        self.confidence_history.append(detection.confidence)
        
        # Update counters
        self.hits += 1
        self.time_since_update = 0
        
        # Update state
        if self.state == "tentative" and self.hits >= 3:
            self.state = "confirmed"
    
    def mark_missed(self):
        """Mark track as missed in current frame."""
        self.time_since_update += 1
        if self.time_since_update > 30:  # Max age
            self.state = "deleted"
    
    def get_predicted_bbox(self) -> List[float]:
        """Get predicted bounding box from Kalman filter."""
        mean, _ = self.kalman_filter.get_state()
        center_x, center_y, width, height = mean
        
        return [
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2
        ]
    
    @property
    def average_feature(self) -> Optional[np.ndarray]:
        """Get average feature vector for re-identification."""
        if not self.feature_history:
            return None
        return np.mean(list(self.feature_history), axis=0)
    
    @property
    def average_confidence(self) -> float:
        """Get average confidence score."""
        if not self.confidence_history:
            return 0.0
        return np.mean(list(self.confidence_history))


class DeepSORTChickenTracker:
    """DeepSORT tracker adapted for chicken tracking."""
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        feature_threshold: float = 0.7
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.feature_threshold = feature_threshold
        
        self.tracks: List[DeepSORTTrack] = []
        self.track_id_count = 0
        self.frame_count = 0
        
        # Tracking statistics
        self.stats = {
            'total_tracks': 0,
            'active_tracks': 0,
            'confirmed_tracks': 0,
            'lost_tracks': 0
        }
    
    def update(self, detections: List[Detection], frame_id: int) -> List[DeepSORTTrack]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detections in current frame
            frame_id: Current frame identifier
            
        Returns:
            List of confirmed tracks
        """
        self.frame_count += 1
        
        try:
            # Predict new locations for existing tracks
            for track in self.tracks:
                track.predict()
            
            # Associate detections to tracks
            matched_pairs, unmatched_detections, unmatched_tracks = self._associate_detections_to_tracks(
                detections, self.tracks
            )
            
            # Update matched tracks
            for detection_idx, track_idx in matched_pairs:
                self.tracks[track_idx].update(detections[detection_idx], frame_id)
            
            # Mark unmatched tracks as missed
            for track_idx in unmatched_tracks:
                self.tracks[track_idx].mark_missed()
            
            # Create new tracks for unmatched detections
            for detection_idx in unmatched_detections:
                self._create_new_track(detections[detection_idx], frame_id)
            
            # Remove deleted tracks
            self.tracks = [track for track in self.tracks if track.state != "deleted"]
            
            # Update statistics
            self._update_statistics()
            
            # Return confirmed tracks
            confirmed_tracks = [track for track in self.tracks if track.state == "confirmed"]
            
            return confirmed_tracks
            
        except Exception as e:
            raise TrackingError(f"DeepSORT update failed: {str(e)}")
    
    def _associate_detections_to_tracks(
        self, 
        detections: List[Detection], 
        tracks: List[DeepSORTTrack]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections to existing tracks using Hungarian algorithm.
        
        Returns:
            Tuple of (matched_pairs, unmatched_detections, unmatched_tracks)
        """
        if not tracks or not detections:
            return [], list(range(len(detections))), list(range(len(tracks)))
        
        # Calculate cost matrix
        cost_matrix = self._calculate_cost_matrix(detections, tracks)
        
        # Apply Hungarian algorithm
        detection_indices, track_indices = linear_sum_assignment(cost_matrix)
        
        # Filter matches based on cost threshold
        matched_pairs = []
        for det_idx, track_idx in zip(detection_indices, track_indices):
            if cost_matrix[det_idx, track_idx] < 0.8:  # Cost threshold
                matched_pairs.append((det_idx, track_idx))
        
        # Find unmatched detections and tracks
        matched_detection_indices = [pair[0] for pair in matched_pairs]
        matched_track_indices = [pair[1] for pair in matched_pairs]
        
        unmatched_detections = [
            i for i in range(len(detections)) 
            if i not in matched_detection_indices
        ]
        
        unmatched_tracks = [
            i for i in range(len(tracks)) 
            if i not in matched_track_indices
        ]
        
        return matched_pairs, unmatched_detections, unmatched_tracks
    
    def _calculate_cost_matrix(
        self, 
        detections: List[Detection], 
        tracks: List[DeepSORTTrack]
    ) -> np.ndarray:
        """Calculate cost matrix for detection-track association."""
        cost_matrix = np.zeros((len(detections), len(tracks)))
        
        for det_idx, detection in enumerate(detections):
            for track_idx, track in enumerate(tracks):
                # IoU-based cost
                predicted_bbox = track.get_predicted_bbox()
                iou = self._calculate_iou(detection.bbox, predicted_bbox)
                iou_cost = 1.0 - iou
                
                # Feature-based cost
                feature_cost = 0.5  # Default moderate cost
                if (hasattr(detection, 'reid_features') and 
                    detection.reid_features is not None and 
                    track.average_feature is not None):
                    
                    similarity = self._cosine_similarity(
                        detection.reid_features, 
                        track.average_feature
                    )
                    feature_cost = 1.0 - similarity
                
                # Confidence-based cost adjustment
                confidence_factor = (detection.confidence + track.average_confidence) / 2
                confidence_adjustment = 1.0 - confidence_factor * 0.2
                
                # Combined cost
                combined_cost = (
                    0.6 * iou_cost + 
                    0.3 * feature_cost + 
                    0.1 * confidence_adjustment
                )
                
                cost_matrix[det_idx, track_idx] = combined_cost
        
        return cost_matrix
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _cosine_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Calculate cosine similarity between feature vectors."""
        try:
            dot_product = np.dot(feat1, feat2)
            norm1 = np.linalg.norm(feat1)
            norm2 = np.linalg.norm(feat2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        except Exception:
            return 0.0
    
    def _create_new_track(self, detection: Detection, frame_id: int):
        """Create new track from unmatched detection."""
        try:
            new_track = DeepSORTTrack(self.track_id_count, detection, frame_id)
            self.tracks.append(new_track)
            self.track_id_count += 1
            self.stats['total_tracks'] += 1
            
        except Exception as e:
            raise TrackInitializationError(f"Failed to create new track: {str(e)}")
    
    def _update_statistics(self):
        """Update tracking statistics."""
        self.stats['active_tracks'] = len([t for t in self.tracks if t.state != "deleted"])
        self.stats['confirmed_tracks'] = len([t for t in self.tracks if t.state == "confirmed"])
        self.stats['lost_tracks'] = len([t for t in self.tracks if t.state == "deleted"])
    
    def get_track_by_id(self, track_id: int) -> Optional[DeepSORTTrack]:
        """Get track by ID."""
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics."""
        active_tracks = [t for t in self.tracks if t.state != "deleted"]
        
        return {
            **self.stats,
            'frames_processed': self.frame_count,
            'average_track_age': np.mean([t.age for t in active_tracks]) if active_tracks else 0.0,
            'average_track_hits': np.mean([t.hits for t in active_tracks]) if active_tracks else 0.0,
            'track_id_counter': self.track_id_count
        }
    
    def reset(self):
        """Reset tracker state."""
        self.tracks.clear()
        self.track_id_count = 0
        self.frame_count = 0
        
        # Reset statistics
        for key in self.stats:
            self.stats[key] = 0