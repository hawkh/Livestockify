"""
Base YOLO detector for chicken detection.
"""

import time
from typing import List, Optional
import numpy as np
import torch
from ultralytics import YOLO

from ...core.interfaces.detection import Detection, DetectionResult, DetectionModel
from ...core.exceptions.detection_exceptions import ModelLoadError, InferenceError


class YOLOChickenDetector(DetectionModel):
    """Base YOLO detector for chicken detection."""
    
    def __init__(
        self, 
        model_path: str,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "auto"
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None
        self._class_names = {0: "chicken"}  # Assuming chicken is class 0
        
    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load the YOLO model from path."""
        path = model_path or self.model_path
        
        try:
            self.model = YOLO(path)
            
            # Set device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Move model to device
            self.model.to(self.device)
            
            print(f"YOLO model loaded successfully from {path} on {self.device}")
            
        except Exception as e:
            raise ModelLoadError(path, f"Failed to load YOLO model: {str(e)}")
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """Set minimum confidence threshold for detections."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        self.confidence_threshold = threshold
    
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Detect chickens in a frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            DetectionResult containing all detections
        """
        if self.model is None:
            self.load_model()
        
        start_time = time.time()
        
        try:
            # Run YOLO inference
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            # Parse results
            detections = self._parse_yolo_results(results[0])
            
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return DetectionResult(
                detections=detections,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            raise InferenceError(f"YOLO inference failed: {str(e)}")
    
    def _parse_yolo_results(self, result) -> List[Detection]:
        """Parse YOLO results into Detection objects."""
        detections = []
        
        if result.boxes is None:
            return detections
        
        boxes = result.boxes.data.cpu().numpy()
        
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            
            # Filter by class (only chickens)
            if int(cls) in self._class_names:
                detection = Detection(
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    confidence=float(conf),
                    class_id=int(cls),
                    class_name=self._class_names[int(cls)]
                )
                detections.append(detection)
        
        return detections
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[DetectionResult]:
        """
        Detect chickens in multiple frames.
        
        Args:
            frames: List of input frames
            
        Returns:
            List of DetectionResult objects
        """
        results = []
        for frame in frames:
            result = self.detect(frame)
            results.append(result)
        return results
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_path": self.model_path,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "class_names": self._class_names
        }