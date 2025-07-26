"""
Camera configuration management.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from ...core.interfaces.camera import CameraParameters


@dataclass
class CameraConfig:
    """Camera configuration settings."""
    
    # Camera intrinsic parameters
    focal_length: float = 1000.0
    sensor_width: float = 6.0
    sensor_height: float = 4.5
    image_width: int = 1920
    image_height: int = 1080
    
    # Camera extrinsic parameters
    camera_height: float = 3.0
    tilt_angle: float = 0.0
    
    # Calibration parameters
    known_object_width: float = 25.0  # Average adult chicken width in cm
    distance_range_min: float = 2.0   # Minimum detection distance in meters
    distance_range_max: float = 10.0  # Maximum detection distance in meters
    
    # Distance estimation parameters
    perspective_correction: bool = True
    distance_validation: bool = True
    distance_smoothing: bool = True
    smoothing_window: int = 5
    
    # Multi-camera support
    camera_id: str = "default"
    camera_name: str = "Default Camera"
    calibration_file: Optional[str] = None
    
    def to_camera_parameters(self) -> CameraParameters:
        """Convert to CameraParameters object."""
        return CameraParameters(
            focal_length=self.focal_length,
            sensor_width=self.sensor_width,
            sensor_height=self.sensor_height,
            image_width=self.image_width,
            image_height=self.image_height,
            camera_height=self.camera_height,
            tilt_angle=self.tilt_angle,
            known_object_width=self.known_object_width
        )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CameraConfig':
        """Create CameraConfig from dictionary."""
        return cls(**config_dict)
    
    def validate(self) -> bool:
        """Validate camera configuration."""
        if self.focal_length <= 0:
            raise ValueError("Focal length must be positive")
        
        if self.sensor_width <= 0 or self.sensor_height <= 0:
            raise ValueError("Sensor dimensions must be positive")
        
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("Image dimensions must be positive")
        
        if self.camera_height <= 0:
            raise ValueError("Camera height must be positive")
        
        if self.known_object_width <= 0:
            raise ValueError("Known object width must be positive")
        
        if self.distance_range_min >= self.distance_range_max:
            raise ValueError("Distance range minimum must be less than maximum")
        
        return True