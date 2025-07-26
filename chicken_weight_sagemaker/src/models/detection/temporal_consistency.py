"""
Temporal consistency filter for detection smoothing across frames.
"""

from typing import List, Optional, Dict, Tuple
import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass

from ...core.interfaces.detection import Detection


@dataclass
class TemporalTrack:
    """Represents a temporal track for a detection."""
    track_id: int
    detections: deque
    last_seen_frame: int
    confidence_history: deque
    bbox_history: deque
    is_stable: bool = False


class TemporalConsistencyFilter:
    """Filters detections for temporal consistency across frames."""
    
    def __init__(
        self,
        max_history: int = 10,
        iou_threshold: float = 0.5,
        confidence_smoothing_weight: float = 0.3,
        bbox_smoothing_weight: float = 0.2,
        stability_threshold: int = 5
    ):
        self.max_history = max_history
        self.iou_threshold = iou_threshold
        self.confidence_smoothing_weight = confidence_smoothing_weight
        self.bbox_smoothing_weight = bbox_smoothing_weight
        self.stability_threshold = stability_threshold
        
        self.tracks: Dict[int, TemporalTrack] = {}
        self.next_track_id = 0
        self.current_frame = 0
        
    def filter_detections(
        self, 
        detections: List[Detection],
        frame_id: Optional[int] = None
    ) -> List[Detection]:
        """
        Apply temporal consistency filtering to detections.
        
        Args:
            detections: Current frame detections
            frame_id: Optional frame identifier
            
        Returns:
            Temporally filtered detections
        """
        if frame_id is not None:
            self.current_frame = frame_id
        else:
            self.current_frame += 1
        
        # Match detections to existing tracks
        matched_tracks, unmatched_detections = self._match_detections_to_tracks(detections)
        
        # Update existing tracks
        for track_id, detection in matched_tracks.items():
            self._update_track(track_id, detection)
        
        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            self._create_new_track(detection)
        
        # Remove old tracks
        self._cleanup_old_tracks()
        
        # Generate filtered detections
        filtered_detections = self._generate_filtered_detections()
        
        return filtered_detections
    
    def _match_detections_to_tracks(
        self, 
        detections: List[Detection]
    ) -> Tuple[Dict[int, Detection], List[Detection]]:
        """Match current detections to existing tracks."""
        matched_tracks = {}
        unmatched_detections = list(detections)
        
        # Calculate IoU matrix between detections and tracks
        for track_id, track in self.tracks.items():
            if not track.bbox_history:
                continue
                
            last_bbox = track.bbox_history[-1]
            best_match_idx = -1
            best_iou = 0.0
            
            for i, detection in enumerate(unmatched_detections):
                iou = self._calculate_iou(detection.bbox, last_bbox)
                
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_match_idx = i
            
            # If good match found, assign detection to track
            if best_match_idx >= 0:
                matched_tracks[track_id] = unmatched_detections.pop(best_match_idx)
        
        return matched_tracks, unmatched_detections
    
    def _update_track(self, track_id: int, detection: Detection) -> None:
        """Update an existing track with new detection."""
        track = self.tracks[track_id]
        
        # Add detection to history
        track.detections.append(detection)
        track.confidence_history.append(detection.confidence)
        track.bbox_history.append(detection.bbox)
        track.last_seen_frame = self.current_frame
        
        # Check if track is stable
        if len(track.detections) >= self.stability_threshold:
            track.is_stable = True
    
    def _create_new_track(self, detection: Detection) -> None:
        """Create a new track for unmatched detection."""
        track = TemporalTrack(
            track_id=self.next_track_id,
            detections=deque([detection], maxlen=self.max_history),
            last_seen_frame=self.current_frame,
            confidence_history=deque([detection.confidence], maxlen=self.max_history),
            bbox_history=deque([detection.bbox], maxlen=self.max_history)
        )
        
        self.tracks[self.next_track_id] = track
        self.next_track_id += 1
    
    def _cleanup_old_tracks(self, max_frames_missing: int = 5) -> None:
        """Remove tracks that haven't been seen for too long."""
        tracks_to_remove = []
        
        for track_id, track in self.tracks.items():
            frames_missing = self.current_frame - track.last_seen_frame
            if frames_missing > max_frames_missing:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
    
    def _generate_filtered_detections(self) -> List[Detection]:
        """Generate temporally filtered detections from active tracks."""
        filtered_detections = []
        
        for track in self.tracks.values():
            # Only include tracks seen in current frame
            if track.last_seen_frame != self.current_frame:
                continue
            
            # Get the most recent detection
            current_detection = track.detections[-1]
            
            # Apply temporal smoothing if track is stable
            if track.is_stable and len(track.detections) > 1:
                smoothed_detection = self._apply_temporal_smoothing(track, current_detection)
                filtered_detections.append(smoothed_detection)
            else:
                # For new/unstable tracks, use original detection
                filtered_detections.append(current_detection)
        
        return filtered_detections
    
    def _apply_temporal_smoothing(
        self, 
        track: TemporalTrack, 
        current_detection: Detection
    ) -> Detection:
        """Apply temporal smoothing to a detection based on track history."""
        # Smooth confidence
        confidence_history = list(track.confidence_history)
        smoothed_confidence = (
            (1 - self.confidence_smoothing_weight) * current_detection.confidence +
            self.confidence_smoothing_weight * np.mean(confidence_history[:-1])
        )
        
        # Smooth bounding box
        bbox_history = list(track.bbox_history)
        if len(bbox_history) > 1:
            prev_bbox = np.mean(bbox_history[:-1], axis=0)
            current_bbox = np.array(current_detection.bbox)
            
            smoothed_bbox = (
                (1 - self.bbox_smoothing_weight) * current_bbox +
                self.bbox_smoothing_weight * prev_bbox
            ).tolist()
        else:
            smoothed_bbox = current_detection.bbox
        
        # Create smoothed detection
        smoothed_detection = Detection(
            bbox=smoothed_bbox,
            confidence=smoothed_confidence,
            class_id=current_detection.class_id,
            class_name=current_detection.class_name,
            occlusion_level=current_detection.occlusion_level,
            visibility_score=current_detection.visibility_score
        )
        
        return smoothed_detection
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
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
    
    def get_track_statistics(self) -> Dict[str, any]:
        """Get statistics about current tracks."""
        if not self.tracks:
            return {"status": "no_tracks"}
        
        stable_tracks = sum(1 for track in self.tracks.values() if track.is_stable)
        active_tracks = sum(
            1 for track in self.tracks.values() 
            if track.last_seen_frame == self.current_frame
        )
        
        avg_track_length = np.mean([len(track.detections) for track in self.tracks.values()])
        
        return {
            "status": "active",
            "total_tracks": len(self.tracks),
            "stable_tracks": stable_tracks,
            "active_tracks": active_tracks,
            "avg_track_length": avg_track_length,
            "current_frame": self.current_frame
        }
    
    def reset(self) -> None:
        """Reset the temporal filter state."""
        self.tracks.clear()
        self.next_track_id = 0
        self.current_frame = 0