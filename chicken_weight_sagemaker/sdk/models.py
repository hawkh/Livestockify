"""
Data models for the Chicken Weight Estimation SDK
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class ChickenDetection:
    """Represents a detected chicken in an image."""
    
    bbox: Dict[str, float]  # Bounding box coordinates
    confidence: float  # Detection confidence score
    class_name: str = 'chicken'  # Object class name
    weight_estimate: Optional[float] = None  # Estimated weight in kg
    occlusion_score: float = 0.0  # Occlusion score (0-1)
    distance_estimate: Optional[float] = None  # Distance from camera in meters
    age_estimate: Optional[int] = None  # Estimated age in days
    
    @property
    def center_x(self) -> float:
        """Get center X coordinate of bounding box."""
        return (self.bbox['x1'] + self.bbox['x2']) / 2
    
    @property
    def center_y(self) -> float:
        """Get center Y coordinate of bounding box."""
        return (self.bbox['y1'] + self.bbox['y2']) / 2
    
    @property
    def width(self) -> float:
        """Get width of bounding box."""
        return self.bbox['x2'] - self.bbox['x1']
    
    @property
    def height(self) -> float:
        """Get height of bounding box."""
        return self.bbox['y2'] - self.bbox['y1']
    
    @property
    def area(self) -> float:
        """Get area of bounding box."""
        return self.width * self.height
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'bbox': self.bbox,
            'confidence': self.confidence,
            'class_name': self.class_name,
            'weight_estimate': self.weight_estimate,
            'occlusion_score': self.occlusion_score,
            'distance_estimate': self.distance_estimate,
            'age_estimate': self.age_estimate,
            'center': {'x': self.center_x, 'y': self.center_y},
            'dimensions': {'width': self.width, 'height': self.height, 'area': self.area}
        }


@dataclass
class TrackingInfo:
    """Represents tracking information for a chicken."""
    
    track_id: int  # Unique track identifier
    bbox: Dict[str, float]  # Current bounding box
    weight_history: List[float]  # Historical weight measurements
    average_weight: Optional[float] = None  # Average weight over time
    confidence: float = 1.0  # Tracking confidence
    first_seen: Optional[str] = None  # Timestamp when first detected
    last_seen: Optional[str] = None  # Timestamp when last detected
    total_frames: int = 0  # Total frames this track has been active
    
    @property
    def weight_trend(self) -> Optional[str]:
        """Get weight trend (increasing, decreasing, stable)."""
        if len(self.weight_history) < 2:
            return None
        
        recent_weights = self.weight_history[-5:]  # Last 5 measurements
        if len(recent_weights) < 2:
            return None
        
        trend = recent_weights[-1] - recent_weights[0]
        if abs(trend) < 0.1:  # Less than 100g change
            return 'stable'
        elif trend > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    @property
    def weight_variance(self) -> Optional[float]:
        """Get variance in weight measurements."""
        if len(self.weight_history) < 2:
            return None
        
        import statistics
        return statistics.variance(self.weight_history)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'track_id': self.track_id,
            'bbox': self.bbox,
            'weight_history': self.weight_history,
            'average_weight': self.average_weight,
            'confidence': self.confidence,
            'first_seen': self.first_seen,
            'last_seen': self.last_seen,
            'total_frames': self.total_frames,
            'weight_trend': self.weight_trend,
            'weight_variance': self.weight_variance
        }


@dataclass
class ProcessingResult:
    """Represents the result of processing a single frame."""
    
    frame_id: int  # Frame identifier
    timestamp: str  # Processing timestamp
    detections: List[ChickenDetection]  # Detected chickens
    tracks: List[TrackingInfo]  # Tracking information
    processing_time: float  # Processing time in seconds
    camera_id: str = 'unknown'  # Camera identifier
    error: Optional[str] = None  # Error message if processing failed
    
    @property
    def detection_count(self) -> int:
        """Get number of detections."""
        return len(self.detections)
    
    @property
    def track_count(self) -> int:
        """Get number of active tracks."""
        return len(self.tracks)
    
    @property
    def average_confidence(self) -> float:
        """Get average detection confidence."""
        if not self.detections:
            return 0.0
        return sum(d.confidence for d in self.detections) / len(self.detections)
    
    @property
    def average_weight(self) -> Optional[float]:
        """Get average estimated weight."""
        weights = [d.weight_estimate for d in self.detections if d.weight_estimate is not None]
        if not weights:
            return None
        return sum(weights) / len(weights)
    
    @property
    def fps(self) -> float:
        """Get processing FPS (frames per second)."""
        if self.processing_time <= 0:
            return 0.0
        return 1.0 / self.processing_time
    
    def get_detections_by_confidence(self, min_confidence: float = 0.5) -> List[ChickenDetection]:
        """Get detections above confidence threshold."""
        return [d for d in self.detections if d.confidence >= min_confidence]
    
    def get_detections_with_weight(self) -> List[ChickenDetection]:
        """Get detections that have weight estimates."""
        return [d for d in self.detections if d.weight_estimate is not None]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'frame_id': self.frame_id,
            'timestamp': self.timestamp,
            'camera_id': self.camera_id,
            'processing_time': self.processing_time,
            'fps': self.fps,
            'detection_count': self.detection_count,
            'track_count': self.track_count,
            'average_confidence': self.average_confidence,
            'average_weight': self.average_weight,
            'detections': [d.to_dict() for d in self.detections],
            'tracks': [t.to_dict() for t in self.tracks],
            'error': self.error
        }


@dataclass
class BatchProcessingResult:
    """Represents the result of batch processing multiple frames."""
    
    total_frames: int  # Total number of frames processed
    successful_frames: int  # Number of successfully processed frames
    failed_frames: int  # Number of failed frames
    total_processing_time: float  # Total processing time
    results: List[ProcessingResult]  # Individual frame results
    start_time: str  # Batch processing start time
    end_time: str  # Batch processing end time
    
    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.total_frames == 0:
            return 0.0
        return (self.successful_frames / self.total_frames) * 100
    
    @property
    def average_fps(self) -> float:
        """Get average processing FPS."""
        if self.total_processing_time <= 0:
            return 0.0
        return self.successful_frames / self.total_processing_time
    
    @property
    def total_detections(self) -> int:
        """Get total number of detections across all frames."""
        return sum(r.detection_count for r in self.results)
    
    @property
    def average_detections_per_frame(self) -> float:
        """Get average detections per frame."""
        if self.successful_frames == 0:
            return 0.0
        return self.total_detections / self.successful_frames
    
    def get_weight_statistics(self) -> Dict[str, float]:
        """Get weight statistics across all detections."""
        all_weights = []
        for result in self.results:
            for detection in result.detections:
                if detection.weight_estimate is not None:
                    all_weights.append(detection.weight_estimate)
        
        if not all_weights:
            return {
                'count': 0,
                'mean': 0.0,
                'min': 0.0,
                'max': 0.0,
                'std': 0.0
            }
        
        import statistics
        return {
            'count': len(all_weights),
            'mean': statistics.mean(all_weights),
            'min': min(all_weights),
            'max': max(all_weights),
            'std': statistics.stdev(all_weights) if len(all_weights) > 1 else 0.0
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'total_frames': self.total_frames,
            'successful_frames': self.successful_frames,
            'failed_frames': self.failed_frames,
            'success_rate': self.success_rate,
            'total_processing_time': self.total_processing_time,
            'average_fps': self.average_fps,
            'total_detections': self.total_detections,
            'average_detections_per_frame': self.average_detections_per_frame,
            'weight_statistics': self.get_weight_statistics(),
            'start_time': self.start_time,
            'end_time': self.end_time,
            'results': [r.to_dict() for r in self.results]
        }


@dataclass
class CameraConfig:
    """Configuration for a camera feed."""
    
    camera_id: str  # Unique camera identifier
    name: str  # Human-readable camera name
    location: str  # Camera location description
    rtsp_url: Optional[str] = None  # RTSP stream URL
    http_url: Optional[str] = None  # HTTP stream URL
    resolution: Optional[Dict[str, int]] = None  # Camera resolution
    fps: Optional[int] = None  # Camera FPS
    calibration_data: Optional[Dict[str, Any]] = None  # Camera calibration parameters
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'camera_id': self.camera_id,
            'name': self.name,
            'location': self.location,
            'rtsp_url': self.rtsp_url,
            'http_url': self.http_url,
            'resolution': self.resolution,
            'fps': self.fps,
            'calibration_data': self.calibration_data
        }


@dataclass
class FarmConfig:
    """Configuration for a farm with multiple cameras."""
    
    farm_id: str  # Unique farm identifier
    name: str  # Farm name
    location: str  # Farm location
    cameras: List[CameraConfig]  # List of cameras
    contact_info: Optional[Dict[str, str]] = None  # Contact information
    
    def get_camera(self, camera_id: str) -> Optional[CameraConfig]:
        """Get camera configuration by ID."""
        for camera in self.cameras:
            if camera.camera_id == camera_id:
                return camera
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'farm_id': self.farm_id,
            'name': self.name,
            'location': self.location,
            'cameras': [c.to_dict() for c in self.cameras],
            'contact_info': self.contact_info
        }