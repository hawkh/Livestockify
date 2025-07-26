"""
Inference request/response interfaces for SageMaker deployment.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from .detection import Detection
from .weight_estimation import WeightEstimate
from .tracking import TrackedChicken


@dataclass
class InferenceRequest:
    """Request format for SageMaker inference."""
    stream_data: Dict[str, Any]
    
    @classmethod
    def from_frame_data(
        cls,
        frame_data: str,  # base64 encoded
        camera_id: str,
        timestamp: Optional[str] = None,
        frame_sequence: Optional[int] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> 'InferenceRequest':
        """Create request from frame data."""
        return cls(
            stream_data={
                "frame": frame_data,
                "timestamp": timestamp or datetime.now().isoformat(),
                "camera_id": camera_id,
                "frame_sequence": frame_sequence or 0,
                "parameters": parameters or {}
            }
        )


@dataclass
class ChickenDetectionResponse:
    """Individual chicken detection in response."""
    chicken_id: str
    bbox: List[float]
    confidence: float
    occlusion_level: float
    distance_estimate: float
    weight_estimate: WeightEstimate
    age_category: Optional[str] = None
    tracking_status: str = "active"


@dataclass
class InferenceResponse:
    """Response format for SageMaker inference."""
    frame_results: Dict[str, Any]
    
    @classmethod
    def from_processing_results(
        cls,
        camera_id: str,
        timestamp: str,
        frame_sequence: int,
        detections: List[Detection],
        weight_estimates: List[WeightEstimate],
        tracked_chickens: Optional[List[TrackedChicken]] = None,
        processing_time_ms: float = 0.0,
        status: str = "success",
        error_message: Optional[str] = None
    ) -> 'InferenceResponse':
        """Create response from processing results."""
        
        # Combine detections with weight estimates and tracking info
        detection_responses = []
        for i, detection in enumerate(detections):
            weight_est = weight_estimates[i] if i < len(weight_estimates) else None
            tracked_chicken = None
            
            # Find corresponding tracked chicken
            if tracked_chickens:
                for tc in tracked_chickens:
                    if (tc.current_detection.bbox == detection.bbox and 
                        tc.current_detection.confidence == detection.confidence):
                        tracked_chicken = tc
                        break
            
            chicken_response = ChickenDetectionResponse(
                chicken_id=tracked_chicken.chicken_id if tracked_chicken else f"chicken_{i}",
                bbox=detection.bbox,
                confidence=detection.confidence,
                occlusion_level=detection.occlusion_level or 0.0,
                distance_estimate=0.0,  # Will be filled by distance estimator
                weight_estimate=weight_est or WeightEstimate(value=0.0),
                age_category=weight_est.age_category if weight_est else None,
                tracking_status=tracked_chicken.tracking_status if tracked_chicken else "new"
            )
            detection_responses.append(chicken_response)
        
        # Calculate summary statistics
        total_chickens = len(detections)
        average_weight = (
            sum(est.value for est in weight_estimates if est) / len(weight_estimates)
            if weight_estimates else 0.0
        )
        
        return cls(
            frame_results={
                "camera_id": camera_id,
                "timestamp": timestamp,
                "frame_sequence": frame_sequence,
                "detections": [
                    {
                        "chicken_id": dr.chicken_id,
                        "bbox": dr.bbox,
                        "confidence": dr.confidence,
                        "occlusion_level": dr.occlusion_level,
                        "distance_estimate": dr.distance_estimate,
                        "weight_estimate": {
                            "value": dr.weight_estimate.value,
                            "unit": dr.weight_estimate.unit,
                            "confidence": dr.weight_estimate.confidence,
                            "error_range": dr.weight_estimate.error_range,
                            "method": dr.weight_estimate.method
                        },
                        "age_category": dr.age_category,
                        "tracking_status": dr.tracking_status
                    }
                    for dr in detection_responses
                ],
                "processing_time_ms": processing_time_ms,
                "total_chickens_detected": total_chickens,
                "average_weight": average_weight,
                "status": status,
                "error_message": error_message
            }
        )


@dataclass
class ErrorResponse:
    """Error response format."""
    error: Dict[str, Any]
    request_id: str
    timestamp: str
    
    @classmethod
    def from_exception(
        cls,
        error_code: str,
        error_message: str,
        request_id: str,
        details: Optional[Dict[str, Any]] = None
    ) -> 'ErrorResponse':
        """Create error response from exception."""
        return cls(
            error={
                "code": error_code,
                "message": error_message,
                "details": details or {}
            },
            request_id=request_id,
            timestamp=datetime.now().isoformat()
        )