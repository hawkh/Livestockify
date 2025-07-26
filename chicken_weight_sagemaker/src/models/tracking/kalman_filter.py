"""
Kalman filter implementation for chicken movement prediction.
"""

import numpy as np
from typing import Tuple, Optional

from ...core.exceptions.tracking_exceptions import TrackingError


class ChickenKalmanFilter:
    """
    Kalman filter for tracking chicken movement in 2D space.
    
    State vector: [center_x, center_y, width, height, velocity_x, velocity_y, delta_width, delta_height]
    """
    
    def __init__(self, dt: float = 1.0):
        """
        Initialize Kalman filter for chicken tracking.
        
        Args:
            dt: Time step between frames (default: 1.0)
        """
        self.dt = dt
        self.ndim = 4  # [x, y, w, h]
        self.motion_mat = None
        self.update_mat = None
        self.std_weight_position = 1.0 / 20
        self.std_weight_velocity = 1.0 / 160
        
        # State variables
        self.mean = None
        self.covariance = None
        self._is_initialized = False
    
    def initiate(self, measurement: list) -> None:
        """
        Initialize filter with first measurement.
        
        Args:
            measurement: [center_x, center_y, width, height]
        """
        try:
            mean_pos = measurement
            mean_vel = [0.0] * self.ndim
            self.mean = np.r_[mean_pos, mean_vel]
            
            # Initialize covariance matrix
            std = [
                2 * self.std_weight_position * measurement[2],  # x std
                2 * self.std_weight_position * measurement[3],  # y std
                1e-2,  # width std
                1e-2,  # height std
                10 * self.std_weight_velocity * measurement[2],  # vx std
                10 * self.std_weight_velocity * measurement[3],  # vy std
                1e-5,  # width velocity std
                1e-5   # height velocity std
            ]
            
            self.covariance = np.diag(np.square(std))
            self._is_initialized = True
            
        except Exception as e:
            raise TrackingError(f"Kalman filter initialization failed: {str(e)}")
    
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict next state.
        
        Returns:
            Tuple of (predicted_mean, predicted_covariance)
        """
        if not self._is_initialized:
            raise TrackingError("Kalman filter not initialized")
        
        try:
            # Motion model matrix
            if self.motion_mat is None:
                self.motion_mat = np.eye(2 * self.ndim, 2 * self.ndim)
                for i in range(self.ndim):
                    self.motion_mat[i, self.ndim + i] = self.dt
            
            # Process noise
            std_pos = [
                self.std_weight_position * self.mean[2],
                self.std_weight_position * self.mean[3],
                1e-2,
                1e-2
            ]
            std_vel = [
                self.std_weight_velocity * self.mean[2],
                self.std_weight_velocity * self.mean[3],
                1e-5,
                1e-5
            ]
            
            motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
            
            # Predict
            self.mean = np.dot(self.motion_mat, self.mean)
            self.covariance = np.linalg.multi_dot((
                self.motion_mat, self.covariance, self.motion_mat.T
            )) + motion_cov
            
            return self.mean.copy(), self.covariance.copy()
            
        except Exception as e:
            raise TrackingError(f"Kalman filter prediction failed: {str(e)}")
    
    def update(self, measurement: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update filter with new measurement.
        
        Args:
            measurement: [center_x, center_y, width, height]
            
        Returns:
            Tuple of (updated_mean, updated_covariance)
        """
        if not self._is_initialized:
            raise TrackingError("Kalman filter not initialized")
        
        try:
            # Measurement model matrix
            if self.update_mat is None:
                self.update_mat = np.eye(self.ndim, 2 * self.ndim)
            
            # Measurement noise
            std = [
                self.std_weight_position * self.mean[2],
                self.std_weight_position * self.mean[3],
                1e-1,
                1e-1
            ]
            innovation_cov = np.diag(np.square(std))
            
            # Innovation (residual)
            mean_pred = np.dot(self.update_mat, self.mean)
            innovation = np.array(measurement) - mean_pred
            
            # Innovation covariance
            projected_cov = np.linalg.multi_dot((
                self.update_mat, self.covariance, self.update_mat.T
            ))
            chol_factor, lower = self._cholesky_decomposition(
                projected_cov + innovation_cov
            )
            
            # Kalman gain
            kalman_gain = np.linalg.multi_dot((
                self.covariance, self.update_mat.T, 
                np.linalg.inv(projected_cov + innovation_cov)
            ))
            
            # Update state
            self.mean = self.mean + np.dot(kalman_gain, innovation)
            self.covariance = self.covariance - np.linalg.multi_dot((
                kalman_gain, self.update_mat, self.covariance
            ))
            
            return self.mean.copy(), self.covariance.copy()
            
        except Exception as e:
            raise TrackingError(f"Kalman filter update failed: {str(e)}")
    
    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current state estimate.
        
        Returns:
            Tuple of (mean, covariance)
        """
        if not self._is_initialized:
            raise TrackingError("Kalman filter not initialized")
        
        return self.mean.copy(), self.covariance.copy()
    
    def get_position_estimate(self) -> np.ndarray:
        """
        Get position estimate [center_x, center_y, width, height].
        
        Returns:
            Position estimate
        """
        if not self._is_initialized:
            raise TrackingError("Kalman filter not initialized")
        
        return self.mean[:self.ndim].copy()
    
    def get_velocity_estimate(self) -> np.ndarray:
        """
        Get velocity estimate [velocity_x, velocity_y, delta_width, delta_height].
        
        Returns:
            Velocity estimate
        """
        if not self._is_initialized:
            raise TrackingError("Kalman filter not initialized")
        
        return self.mean[self.ndim:].copy()
    
    def _cholesky_decomposition(self, matrix: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Perform Cholesky decomposition with fallback to LU decomposition.
        
        Args:
            matrix: Matrix to decompose
            
        Returns:
            Tuple of (decomposed_matrix, is_lower_triangular)
        """
        try:
            # Try Cholesky decomposition
            chol_factor = np.linalg.cholesky(matrix)
            return chol_factor, True
        except np.linalg.LinAlgError:
            # Fallback to LU decomposition
            try:
                from scipy.linalg import lu
                _, l_matrix, u_matrix = lu(matrix)
                return l_matrix, True
            except ImportError:
                # Simple fallback - use identity matrix
                return np.eye(matrix.shape[0]), True
    
    def predict_position(self, n_steps: int = 1) -> np.ndarray:
        """
        Predict position after n steps without updating internal state.
        
        Args:
            n_steps: Number of time steps to predict ahead
            
        Returns:
            Predicted position [center_x, center_y, width, height]
        """
        if not self._is_initialized:
            raise TrackingError("Kalman filter not initialized")
        
        # Create temporary motion matrix for n steps
        temp_motion_mat = np.eye(2 * self.ndim, 2 * self.ndim)
        for i in range(self.ndim):
            temp_motion_mat[i, self.ndim + i] = self.dt * n_steps
        
        # Predict without updating internal state
        predicted_mean = np.dot(temp_motion_mat, self.mean)
        
        return predicted_mean[:self.ndim]
    
    def get_uncertainty(self) -> float:
        """
        Get current state uncertainty as a scalar value.
        
        Returns:
            Uncertainty measure (trace of covariance matrix)
        """
        if not self._is_initialized:
            return float('inf')
        
        return np.trace(self.covariance)
    
    def is_initialized(self) -> bool:
        """Check if filter is initialized."""
        return self._is_initialized
    
    def reset(self) -> None:
        """Reset filter state."""
        self.mean = None
        self.covariance = None
        self.motion_mat = None
        self.update_mat = None
        self._is_initialized = False