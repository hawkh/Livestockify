"""
Camera calibration utilities for distance estimation.
"""

import numpy as np
import cv2
import yaml
import json
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

from ...core.interfaces.camera import CameraCalibrator, CameraCalibration, CameraParameters
from ...core.exceptions.camera_exceptions import (
    CameraCalibrationError, 
    CalibrationFileError,
    InsufficientCalibrationDataError
)


class ChickenFarmCameraCalibrator(CameraCalibrator):
    """Camera calibrator specifically designed for poultry farm environments."""
    
    def __init__(self):
        self.calibration_points = []
        self.known_measurements = []
        
    def calibrate_camera(
        self, 
        calibration_images: List[np.ndarray],
        known_measurements: List[Tuple[float, float]]  # (distance, actual_size) pairs
    ) -> CameraCalibration:
        """
        Calibrate camera using calibration images and known measurements.
        
        Args:
            calibration_images: Images for calibration
            known_measurements: Known distance and size measurements
            
        Returns:
            CameraCalibration object
        """
        if len(calibration_images) < 3:
            raise InsufficientCalibrationDataError(3, len(calibration_images))
        
        if len(known_measurements) < 3:
            raise InsufficientCalibrationDataError(3, len(known_measurements))
        
        try:
            # Method 1: Traditional checkerboard calibration (if available)
            camera_matrix, dist_coeffs = self._calibrate_with_checkerboard(calibration_images)
            
            # Method 2: Known object size calibration
            if camera_matrix is None:
                camera_matrix, dist_coeffs = self._calibrate_with_known_objects(
                    calibration_images, known_measurements
                )
            
            # Estimate camera parameters from calibration
            parameters = self._estimate_camera_parameters(
                camera_matrix, calibration_images[0].shape, known_measurements
            )
            
            return CameraCalibration(
                camera_matrix=camera_matrix,
                distortion_coefficients=dist_coeffs,
                parameters=parameters
            )
            
        except Exception as e:
            raise CameraCalibrationError(f"Camera calibration failed: {str(e)}")
    
    def _calibrate_with_checkerboard(
        self, 
        images: List[np.ndarray],
        pattern_size: Tuple[int, int] = (9, 6)
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Calibrate camera using checkerboard pattern."""
        try:
            # Prepare object points
            objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
            
            # Arrays to store object points and image points
            objpoints = []  # 3d points in real world space
            imgpoints = []  # 2d points in image plane
            
            for img in images:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Find the chess board corners
                ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
                
                if ret:
                    objpoints.append(objp)
                    
                    # Refine corner positions
                    corners2 = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1),
                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    )
                    imgpoints.append(corners2)
            
            if len(objpoints) < 3:
                return None, None
            
            # Calibrate camera
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None
            )
            
            if ret:
                return camera_matrix, dist_coeffs
            else:
                return None, None
                
        except Exception:
            return None, None
    
    def _calibrate_with_known_objects(
        self,
        images: List[np.ndarray],
        known_measurements: List[Tuple[float, float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calibrate camera using known object sizes."""
        # This is a simplified calibration method for farm environments
        # where traditional checkerboard patterns may not be practical
        
        if len(images) != len(known_measurements):
            raise ValueError("Number of images must match number of measurements")
        
        # Estimate focal length from known measurements
        focal_lengths = []
        
        for img, (distance, actual_size) in zip(images, known_measurements):
            # Detect largest object in image (assumed to be the reference object)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Simple edge detection to find object
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Use width as the measured dimension
                pixel_size = w
                
                # Calculate focal length: f = (pixel_size * distance) / actual_size
                focal_length = (pixel_size * distance * 100) / actual_size  # Convert to cm
                focal_lengths.append(focal_length)
        
        # Average focal length
        avg_focal_length = np.mean(focal_lengths)
        
        # Create camera matrix
        h, w = images[0].shape[:2]
        camera_matrix = np.array([
            [avg_focal_length, 0, w/2],
            [0, avg_focal_length, h/2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Assume minimal distortion for farm cameras
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        
        return camera_matrix, dist_coeffs
    
    def _estimate_camera_parameters(
        self,
        camera_matrix: np.ndarray,
        image_shape: Tuple[int, int],
        known_measurements: List[Tuple[float, float]]
    ) -> CameraParameters:
        """Estimate camera parameters from calibration data."""
        h, w = image_shape[:2]
        focal_length = camera_matrix[0, 0]  # Assuming fx = fy
        
        # Estimate sensor size (assuming standard camera sensor)
        # This is an approximation - ideally should be provided
        sensor_width = 6.0  # mm (typical for many cameras)
        sensor_height = sensor_width * (h / w)
        
        # Estimate camera height from known measurements
        # This is a rough estimate - should be measured in practice
        camera_height = 3.0  # meters (typical mounting height)
        
        # Estimate known object width from measurements
        if known_measurements:
            known_object_width = np.mean([size for _, size in known_measurements])
        else:
            known_object_width = 25.0  # cm (average adult chicken width)
        
        return CameraParameters(
            focal_length=focal_length,
            sensor_width=sensor_width,
            sensor_height=sensor_height,
            image_width=w,
            image_height=h,
            camera_height=camera_height,
            tilt_angle=0.0,  # Assume level camera
            known_object_width=known_object_width
        )
    
    def save_calibration(self, calibration: CameraCalibration, filepath: str) -> None:
        """Save calibration to file."""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare calibration data for serialization
            calibration_data = {
                'camera_matrix': calibration.camera_matrix.tolist(),
                'distortion_coefficients': calibration.distortion_coefficients.tolist(),
                'parameters': {
                    'focal_length': calibration.parameters.focal_length,
                    'sensor_width': calibration.parameters.sensor_width,
                    'sensor_height': calibration.parameters.sensor_height,
                    'image_width': calibration.parameters.image_width,
                    'image_height': calibration.parameters.image_height,
                    'camera_height': calibration.parameters.camera_height,
                    'tilt_angle': calibration.parameters.tilt_angle,
                    'known_object_width': calibration.parameters.known_object_width
                } if calibration.parameters else None
            }
            
            # Save based on file extension
            if filepath.suffix.lower() == '.yaml':
                with open(filepath, 'w') as f:
                    yaml.dump(calibration_data, f, default_flow_style=False)
            elif filepath.suffix.lower() == '.json':
                with open(filepath, 'w') as f:
                    json.dump(calibration_data, f, indent=2)
            else:
                # Default to numpy format
                np.savez(filepath, **calibration_data)
                
        except Exception as e:
            raise CalibrationFileError(str(filepath), f"Failed to save calibration: {str(e)}")
    
    def load_calibration(self, filepath: str) -> CameraCalibration:
        """Load calibration from file."""
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                raise CalibrationFileError(str(filepath), "Calibration file not found")
            
            # Load based on file extension
            if filepath.suffix.lower() == '.yaml':
                with open(filepath, 'r') as f:
                    data = yaml.safe_load(f)
            elif filepath.suffix.lower() == '.json':
                with open(filepath, 'r') as f:
                    data = json.load(f)
            else:
                # Assume numpy format
                data = dict(np.load(filepath, allow_pickle=True))
            
            # Reconstruct calibration object
            camera_matrix = np.array(data['camera_matrix'], dtype=np.float32)
            dist_coeffs = np.array(data['distortion_coefficients'], dtype=np.float32)
            
            parameters = None
            if data.get('parameters'):
                param_data = data['parameters']
                parameters = CameraParameters(**param_data)
            
            return CameraCalibration(
                camera_matrix=camera_matrix,
                distortion_coefficients=dist_coeffs,
                parameters=parameters
            )
            
        except Exception as e:
            raise CalibrationFileError(str(filepath), f"Failed to load calibration: {str(e)}")
    
    def validate_calibration(self, calibration: CameraCalibration) -> bool:
        """Validate calibration data."""
        try:
            # Check camera matrix
            if calibration.camera_matrix.shape != (3, 3):
                raise ValueError("Camera matrix must be 3x3")
            
            # Check that focal lengths are positive
            fx = calibration.camera_matrix[0, 0]
            fy = calibration.camera_matrix[1, 1]
            
            if fx <= 0 or fy <= 0:
                raise ValueError("Focal lengths must be positive")
            
            # Check distortion coefficients
            if calibration.distortion_coefficients.size < 4:
                raise ValueError("Distortion coefficients must have at least 4 elements")
            
            # Validate parameters if present
            if calibration.parameters:
                if calibration.parameters.focal_length <= 0:
                    raise ValueError("Focal length must be positive")
                
                if calibration.parameters.sensor_width <= 0 or calibration.parameters.sensor_height <= 0:
                    raise ValueError("Sensor dimensions must be positive")
            
            return True
            
        except Exception as e:
            raise CameraCalibrationError(f"Calibration validation failed: {str(e)}")
    
    def create_test_calibration(
        self, 
        image_width: int = 1920, 
        image_height: int = 1080
    ) -> CameraCalibration:
        """Create a test calibration for development/testing purposes."""
        # Create default camera matrix
        focal_length = 1000.0
        camera_matrix = np.array([
            [focal_length, 0, image_width/2],
            [0, focal_length, image_height/2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Minimal distortion
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        
        # Default parameters
        parameters = CameraParameters(
            focal_length=focal_length,
            sensor_width=6.0,
            sensor_height=4.5,
            image_width=image_width,
            image_height=image_height,
            camera_height=3.0,
            tilt_angle=0.0,
            known_object_width=25.0
        )
        
        return CameraCalibration(
            camera_matrix=camera_matrix,
            distortion_coefficients=dist_coeffs,
            parameters=parameters
        )