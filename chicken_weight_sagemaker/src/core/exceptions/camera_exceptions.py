"""
Camera calibration and distance estimation exceptions.
"""


class CameraCalibrationError(Exception):
    """Base exception for camera calibration errors."""
    pass


class DistanceEstimationError(Exception):
    """Base exception for distance estimation errors."""
    pass


class CalibrationFileError(CameraCalibrationError):
    """Raised when calibration file operations fail."""
    
    def __init__(self, filepath: str, message: str = "Calibration file error"):
        self.filepath = filepath
        super().__init__(f"{message}: {filepath}")


class InvalidCameraParametersError(CameraCalibrationError):
    """Raised when camera parameters are invalid."""
    
    def __init__(self, parameter_name: str, value: float):
        self.parameter_name = parameter_name
        self.value = value
        super().__init__(f"Invalid camera parameter {parameter_name}: {value}")


class PerspectiveDistortionError(DistanceEstimationError):
    """Raised when perspective distortion correction fails."""
    pass


class DistanceValidationError(DistanceEstimationError):
    """Raised when distance validation fails."""
    
    def __init__(self, distance: float, bbox_size: float):
        self.distance = distance
        self.bbox_size = bbox_size
        super().__init__(f"Invalid distance {distance}m for bbox size {bbox_size}")


class InsufficientCalibrationDataError(CameraCalibrationError):
    """Raised when insufficient calibration data is provided."""
    
    def __init__(self, required_points: int, provided_points: int):
        self.required_points = required_points
        self.provided_points = provided_points
        super().__init__(f"Insufficient calibration data: need {required_points}, got {provided_points}")