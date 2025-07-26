"""
Occlusion-robust YOLO detector for handling partially visible chickens.
"""

import time
from typing import List, Optional, Tuple
import numpy as np
import cv2
from collections import deque

from .yolo_detector import YOLOChickenDetector
from ...core.interfaces.detection import Detection, DetectionResult, OcclusionAwareDetector
from ...core.exceptions.detection_exceptions import OcclusionDetectionError, TemporalConsistencyError


class OcclusionRobustYOLODetector(YOLOChickenDetector, OcclusionAwareDetector):
    """Enhanced YOLO detector with occlusion handling capabilities."""
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.4,  # Lower threshold for occluded chickens
        iou_threshold: float = 0.45,
        min_visibility_threshold: float = 0.3,
        temporal_window: int = 5,
        device: str = "auto"
    ):
        super().__init__(model_path, confidence_threshold, iou_threshold, device)
        self.min_visibility_threshold = min_visibility_threshold
        self.temporal_window = temporal_window
        
        # Temporal consistency tracking
        self.detection_history = deque(maxlen=temporal_window)
        self.frame_count = 0
        
        # Occlusion detection parameters
        self.edge_threshold = 50
        self.contour_area_threshold = 500
        
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
        start_time = time.time()
        
        try:
            # Get base YOLO detections
            base_result = self.detect(frame)
            detections = base_result.detections
            
            # Enhance detections with occlusion information
            enhanced_detections = []
            for detection in detections:
                # Estimate occlusion level
                occlusion_level = self.estimate_occlusion_level(detection, frame)
                
                # Calculate visibility score
                visibility_score = 1.0 - occlusion_level
                
                # Filter by minimum visibility
                if visibility_score >= self.min_visibility_threshold:
                    detection.occlusion_level = occlusion_level
                    detection.visibility_score = visibility_score
                    enhanced_detections.append(detection)
            
            # Apply temporal consistency if previous detections available
            if previous_detections is not None:
                enhanced_detections = self.apply_temporal_smoothing(
                    enhanced_detections, 
                    [previous_detections]
                )
            
            # Update detection history
            self.detection_history.append(enhanced_detections)
            self.frame_count += 1
            
            processing_time = (time.time() - start_time) * 1000
            
            return DetectionResult(
                detections=enhanced_detections,
                processing_time_ms=processing_time,
                frame_id=str(self.frame_count)
            )
            
        except Exception as e:
            raise OcclusionDetectionError(f"Occlusion-aware detection failed: {str(e)}")
    
    def estimate_occlusion_level(self, detection: Detection, frame: np.ndarray) -> float:
        """
        Estimate occlusion level for a detection.
        
        Args:
            detection: Detection to analyze
            frame: Frame containing the detection
            
        Returns:
            Occlusion level between 0.0 (no occlusion) and 1.0 (fully occluded)
        """
        try:
            x1, y1, x2, y2 = [int(coord) for coord in detection.bbox]
            
            # Extract chicken region
            chicken_region = frame[y1:y2, x1:x2]
            
            if chicken_region.size == 0:
                return 1.0  # Fully occluded if no region
            
            # Convert to grayscale for analysis
            gray_region = cv2.cvtColor(chicken_region, cv2.COLOR_BGR2GRAY)
            
            # Method 1: Edge density analysis
            edges = cv2.Canny(gray_region, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Method 2: Contour completeness
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find largest contour (assumed to be the chicken)
                largest_contour = max(contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(largest_contour)
                
                # Calculate contour completeness
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                
                if hull_area > 0:
                    completeness = contour_area / hull_area
                else:
                    completeness = 0.0
            else:
                completeness = 0.0
            
            # Method 3: Bounding box fill ratio
            bbox_area = (x2 - x1) * (y2 - y1)
            non_zero_pixels = np.sum(gray_region > 20)  # Threshold for background
            fill_ratio = non_zero_pixels / bbox_area if bbox_area > 0 else 0.0
            
            # Combine metrics to estimate occlusion
            # Lower edge density, completeness, and fill ratio indicate higher occlusion
            edge_score = min(edge_density * 10, 1.0)  # Normalize edge density
            completeness_score = completeness
            fill_score = min(fill_ratio, 1.0)
            
            # Weighted combination
            visibility_score = (0.3 * edge_score + 0.4 * completeness_score + 0.3 * fill_score)
            occlusion_level = 1.0 - visibility_score
            
            return max(0.0, min(1.0, occlusion_level))
            
        except Exception as e:
            # If occlusion estimation fails, assume moderate occlusion
            return 0.5
    
    def apply_temporal_smoothing(
        self, 
        current_detections: List[Detection], 
        history: List[List[Detection]]
    ) -> List[Detection]:
        """
        Apply temporal smoothing to reduce detection noise.
        
        Args:
            current_detections: Current frame detections
            history: Previous frame detections
            
        Returns:
            Temporally smoothed detections
        """
        try:
            if not history or not current_detections:
                return current_detections
            
            smoothed_detections = []
            
            for current_det in current_detections:
                # Find matching detection in previous frame
                best_match = self._find_best_temporal_match(current_det, history[-1])
                
                if best_match is not None:
                    # Smooth confidence and occlusion level
                    smoothed_confidence = (
                        0.7 * current_det.confidence + 
                        0.3 * best_match.confidence
                    )
                    
                    smoothed_occlusion = (
                        0.7 * (current_det.occlusion_level or 0.0) + 
                        0.3 * (best_match.occlusion_level or 0.0)
                    )
                    
                    # Create smoothed detection
                    smoothed_det = Detection(
                        bbox=current_det.bbox,
                        confidence=smoothed_confidence,
                        class_id=current_det.class_id,
                        class_name=current_det.class_name,
                        occlusion_level=smoothed_occlusion,
                        visibility_score=1.0 - smoothed_occlusion
                    )
                    
                    smoothed_detections.append(smoothed_det)
                else:
                    # No match found, keep original detection
                    smoothed_detections.append(current_det)
            
            return smoothed_detections
            
        except Exception as e:
            raise TemporalConsistencyError(f"Temporal smoothing failed: {str(e)}")
    
    def _find_best_temporal_match(
        self, 
        current_detection: Detection, 
        previous_detections: List[Detection]
    ) -> Optional[Detection]:
        """Find the best matching detection from previous frame."""
        if not previous_detections:
            return None
        
        current_bbox = current_detection.bbox
        best_match = None
        best_iou = 0.0
        
        for prev_det in previous_detections:
            iou = self._calculate_iou(current_bbox, prev_det.bbox)
            
            if iou > best_iou and iou > 0.3:  # Minimum IoU threshold
                best_iou = iou
                best_match = prev_det
        
        return best_match
    
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
    
    def get_detection_statistics(self) -> dict:
        """Get statistics about recent detections."""
        if not self.detection_history:
            return {"status": "no_history"}
        
        recent_detections = list(self.detection_history)
        total_detections = sum(len(dets) for dets in recent_detections)
        
        if total_detections == 0:
            return {"status": "no_detections"}
        
        # Calculate average occlusion level
        occlusion_levels = []
        confidence_scores = []
        
        for frame_detections in recent_detections:
            for det in frame_detections:
                if det.occlusion_level is not None:
                    occlusion_levels.append(det.occlusion_level)
                confidence_scores.append(det.confidence)
        
        avg_occlusion = np.mean(occlusion_levels) if occlusion_levels else 0.0
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        return {
            "status": "active",
            "frames_processed": len(recent_detections),
            "total_detections": total_detections,
            "avg_detections_per_frame": total_detections / len(recent_detections),
            "avg_occlusion_level": avg_occlusion,
            "avg_confidence": avg_confidence,
            "highly_occluded_ratio": sum(1 for ol in occlusion_levels if ol > 0.7) / len(occlusion_levels) if occlusion_levels else 0.0
        }