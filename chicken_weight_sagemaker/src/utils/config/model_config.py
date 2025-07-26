"""
Model configuration management.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List


@dataclass
class YOLOConfig:
    """YOLO model configuration."""
    model_path: str = "model_artifacts/yolo_best.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 100
    input_size: int = 640
    device: str = "cuda"  # cuda, cpu, auto
    
    # Occlusion handling
    min_visibility_threshold: float = 0.3
    occlusion_detection: bool = True
    temporal_consistency: bool = True
    temporal_window: int = 5


@dataclass
class WeightEstimationConfig:
    """Weight estimation model configuration."""
    model_path: str = "model_artifacts/weight_nn.pt"
    feature_size: int = 20
    device: str = "cuda"
    
    # Distance compensation
    distance_compensation: bool = True
    perspective_correction: bool = True
    
    # Age-based validation
    age_validation: bool = True
    weight_tolerance: float = 0.25  # ±25% tolerance
    
    # Weight ranges by age (in kg)
    age_weight_ranges: Dict[str, tuple] = None
    
    def __post_init__(self):
        if self.age_weight_ranges is None:
            self.age_weight_ranges = {
                "DAY_OLD": (0.035, 0.045),
                "WEEK_1": (0.150, 0.200),
                "WEEK_2": (0.400, 0.500),
                "WEEK_3": (0.800, 1.000),
                "WEEK_4": (1.200, 1.500),
                "WEEK_5": (1.800, 2.200),
                "WEEK_6": (2.500, 3.000),
                "ADULT": (3.000, 5.000)
            }


@dataclass
class TrackingConfig:
    """Tracking configuration."""
    max_disappeared: int = 30
    max_distance: float = 100.0
    min_track_length: int = 5
    
    # Re-identification
    reid_threshold: float = 0.7
    reid_feature_size: int = 128
    
    # Occlusion tracking
    occlusion_tracking: bool = True
    max_occlusion_frames: int = 15
    occlusion_threshold: float = 0.7
    
    # Temporal smoothing
    weight_smoothing: bool = True
    smoothing_window: int = 10
    smoothing_method: str = "median"  # mean, median, weighted


@dataclass
class ModelConfig:
    """Complete model configuration."""
    yolo: YOLOConfig
    weight_estimation: WeightEstimationConfig
    tracking: TrackingConfig
    
    # Processing parameters
    batch_size: int = 1
    max_processing_time_ms: float = 100.0
    enable_gpu: bool = True
    gpu_memory_fraction: float = 0.8
    
    # Model artifacts
    model_artifacts_dir: str = "model_artifacts"
    
    def __init__(
        self,
        yolo_config: Optional[Dict[str, Any]] = None,
        weight_config: Optional[Dict[str, Any]] = None,
        tracking_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        # Initialize sub-configs
        self.yolo = YOLOConfig(**(yolo_config or {}))
        self.weight_estimation = WeightEstimationConfig(**(weight_config or {}))
        self.tracking = TrackingConfig(**(tracking_config or {}))
        
        # Set other attributes
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create ModelConfig from dictionary."""
        return cls(
            yolo_config=config_dict.get('yolo', {}),
            weight_config=config_dict.get('weight_estimation', {}),
            tracking_config=config_dict.get('tracking', {}),
            **{k: v for k, v in config_dict.items() 
               if k not in ['yolo', 'weight_estimation', 'tracking']}
        )
    
    def validate(self) -> bool:
        """Validate model configuration."""
        # Validate YOLO config
        if self.yolo.confidence_threshold < 0 or self.yolo.confidence_threshold > 1:
            raise ValueError("YOLO confidence threshold must be between 0 and 1")
        
        if self.yolo.min_visibility_threshold < 0 or self.yolo.min_visibility_threshold > 1:
            raise ValueError("Minimum visibility threshold must be between 0 and 1")
        
        # Validate weight estimation config
        if self.weight_estimation.weight_tolerance < 0 or self.weight_estimation.weight_tolerance > 1:
            raise ValueError("Weight tolerance must be between 0 and 1")
        
        # Validate tracking config
        if self.tracking.max_disappeared <= 0:
            raise ValueError("Max disappeared frames must be positive")
        
        if self.tracking.reid_threshold < 0 or self.tracking.reid_threshold > 1:
            raise ValueError("Re-identification threshold must be between 0 and 1")
        
        return True