"""
Feature extraction for chicken weight estimation.
"""

import cv2
import numpy as np
import math
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler

from ...core.interfaces.detection import Detection
from ...core.exceptions.weight_estimation_exceptions import FeatureExtractionError


class ChickenFeatureExtractor:
    """Extracts comprehensive features from chicken detections for weight estimation."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Feature extraction parameters
        self.texture_window_size = 5
        self.color_bins = 16
        self.gradient_threshold = 50
        
    def extract_features(
        self,
        frame: np.ndarray,
        detection: Detection,
        distance: Optional[float] = None,
        occlusion_level: Optional[float] = None
    ) -> np.ndarray:
        """
        Extract comprehensive features from chicken detection.
        
        Args:
            frame: Input frame containing the chicken
            detection: Detection information
            distance: Estimated distance to chicken
            occlusion_level: Level of occlusion (0-1)
            
        Returns:
            Feature vector as numpy array
        """
        try:
            x1, y1, x2, y2 = [int(coord) for coord in detection.bbox]
            
            # Extract chicken region
            chicken_region = frame[y1:y2, x1:x2]
            
            if chicken_region.size == 0:
                return self._get_default_features()
            
            # Extract different types of features
            dimensional_features = self._extract_dimensional_features(detection.bbox)
            color_features = self._extract_color_features(chicken_region)
            texture_features = self._extract_texture_features(chicken_region)
            shape_features = self._extract_shape_features(chicken_region, detection.bbox)
            context_features = self._extract_context_features(
                frame, detection.bbox, distance, occlusion_level
            )
            
            # Combine all features
            all_features = np.concatenate([
                dimensional_features,
                color_features,
                texture_features,
                shape_features,
                context_features
            ])
            
            # Ensure consistent feature vector size
            target_size = 25  # Increased from 20 to accommodate more features
            if len(all_features) < target_size:
                all_features = np.pad(all_features, (0, target_size - len(all_features)), 'constant')
            elif len(all_features) > target_size:
                all_features = all_features[:target_size]
            
            return all_features
            
        except Exception as e:
            raise FeatureExtractionError(f"Feature extraction failed: {str(e)}", detection.bbox)
    
    def _extract_dimensional_features(self, bbox: List[float]) -> np.ndarray:
        """Extract dimensional features from bounding box."""
        x1, y1, x2, y2 = bbox
        
        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = width / height if height > 0 else 1.0
        perimeter = 2 * (width + height)
        compactness = (4 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
        
        return np.array([width, height, area, aspect_ratio, perimeter, compactness])
    
    def _extract_color_features(self, chicken_region: np.ndarray) -> np.ndarray:
        """Extract color-based features."""
        if len(chicken_region.shape) == 3:
            # Convert to different color spaces
            hsv = cv2.cvtColor(chicken_region, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(chicken_region, cv2.COLOR_BGR2LAB)
            
            # Mean and std in different color spaces
            bgr_mean = np.mean(chicken_region, axis=(0, 1))
            bgr_std = np.std(chicken_region, axis=(0, 1))
            
            hsv_mean = np.mean(hsv, axis=(0, 1))
            lab_mean = np.mean(lab, axis=(0, 1))
            
            # Color histogram features
            hist_features = self._extract_color_histogram_features(chicken_region)
            
            color_features = np.concatenate([
                bgr_mean, bgr_std, hsv_mean[:2], lab_mean[:2], hist_features
            ])
        else:
            # Grayscale image
            mean_intensity = np.mean(chicken_region)
            std_intensity = np.std(chicken_region)
            color_features = np.array([mean_intensity, std_intensity] + [0] * 8)
        
        return color_features[:10]  # Limit to 10 color features
    
    def _extract_color_histogram_features(self, chicken_region: np.ndarray) -> np.ndarray:
        """Extract color histogram features."""
        # Calculate histogram for each channel
        hist_features = []
        
        for i in range(min(3, chicken_region.shape[2])):
            hist = cv2.calcHist([chicken_region], [i], None, [self.color_bins], [0, 256])
            hist = hist.flatten() / hist.sum()  # Normalize
            
            # Use only first few bins to keep feature size manageable
            hist_features.extend(hist[:4])
        
        return np.array(hist_features)
    
    def _extract_texture_features(self, chicken_region: np.ndarray) -> np.ndarray:
        """Extract texture-based features."""
        # Convert to grayscale for texture analysis
        if len(chicken_region.shape) == 3:
            gray = cv2.cvtColor(chicken_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = chicken_region
        
        # Gradient-based texture features
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        texture_energy = np.mean(gradient_magnitude)
        texture_variance = np.var(gradient_magnitude)
        
        # Local Binary Pattern approximation
        lbp_features = self._extract_lbp_features(gray)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        return np.array([texture_energy, texture_variance, edge_density] + lbp_features)
    
    def _extract_lbp_features(self, gray_region: np.ndarray) -> List[float]:
        """Extract simplified Local Binary Pattern features."""
        if gray_region.shape[0] < 3 or gray_region.shape[1] < 3:
            return [0.0, 0.0]
        
        # Simplified LBP calculation
        h, w = gray_region.shape
        lbp_values = []
        
        # Sample a few points for LBP calculation
        for i in range(1, min(h-1, 10)):
            for j in range(1, min(w-1, 10)):
                center = gray_region[i, j]
                
                # 8-neighbor LBP
                neighbors = [
                    gray_region[i-1, j-1], gray_region[i-1, j], gray_region[i-1, j+1],
                    gray_region[i, j+1], gray_region[i+1, j+1], gray_region[i+1, j],
                    gray_region[i+1, j-1], gray_region[i, j-1]
                ]
                
                lbp_code = sum([(n >= center) * (2**idx) for idx, n in enumerate(neighbors)])
                lbp_values.append(lbp_code)
        
        if lbp_values:
            lbp_mean = np.mean(lbp_values)
            lbp_std = np.std(lbp_values)
        else:
            lbp_mean = lbp_std = 0.0
        
        return [lbp_mean / 255.0, lbp_std / 255.0]  # Normalize
    
    def _extract_shape_features(
        self, 
        chicken_region: np.ndarray, 
        bbox: List[float]
    ) -> np.ndarray:
        """Extract shape-based features."""
        # Convert to grayscale for contour detection
        if len(chicken_region.shape) == 3:
            gray = cv2.cvtColor(chicken_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = chicken_region
        
        # Find contours
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (assumed to be the chicken)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Shape features
            contour_area = cv2.contourArea(largest_contour)
            contour_perimeter = cv2.arcLength(largest_contour, True)
            
            # Convex hull features
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            
            # Shape ratios
            solidity = contour_area / hull_area if hull_area > 0 else 0.0
            
            # Extent (ratio of contour area to bounding rectangle area)
            x, y, w, h = cv2.boundingRect(largest_contour)
            rect_area = w * h
            extent = contour_area / rect_area if rect_area > 0 else 0.0
            
            shape_features = [solidity, extent, contour_area / 1000.0]  # Normalize area
        else:
            shape_features = [0.0, 0.0, 0.0]
        
        return np.array(shape_features)
    
    def _extract_context_features(
        self,
        frame: np.ndarray,
        bbox: List[float],
        distance: Optional[float] = None,
        occlusion_level: Optional[float] = None
    ) -> np.ndarray:
        """Extract contextual features."""
        x1, y1, x2, y2 = bbox
        frame_height, frame_width = frame.shape[:2]
        
        # Position features
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Normalized position (0-1)
        norm_x = center_x / frame_width
        norm_y = center_y / frame_height
        
        # Distance from frame center
        center_distance = math.sqrt(
            ((center_x - frame_width/2) / frame_width) ** 2 +
            ((center_y - frame_height/2) / frame_height) ** 2
        )
        
        # Size relative to frame
        bbox_area = (x2 - x1) * (y2 - y1)
        frame_area = frame_width * frame_height
        relative_size = bbox_area / frame_area
        
        # Distance and occlusion features
        distance_feature = distance / 10.0 if distance is not None else 0.3  # Normalize to ~0-1
        occlusion_feature = occlusion_level if occlusion_level is not None else 0.0
        
        context_features = [
            norm_x, norm_y, center_distance, relative_size, 
            distance_feature, occlusion_feature
        ]
        
        return np.array(context_features)
    
    def _get_default_features(self) -> np.ndarray:
        """Return default feature vector when extraction fails."""
        return np.zeros(25)
    
    def extract_batch_features(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        distances: Optional[List[float]] = None,
        occlusion_levels: Optional[List[float]] = None
    ) -> np.ndarray:
        """Extract features for multiple detections."""
        features_list = []
        
        for i, detection in enumerate(detections):
            distance = distances[i] if distances and i < len(distances) else None
            occlusion = occlusion_levels[i] if occlusion_levels and i < len(occlusion_levels) else None
            
            features = self.extract_features(frame, detection, distance, occlusion)
            features_list.append(features)
        
        return np.array(features_list)
    
    def fit_scaler(self, features: np.ndarray) -> None:
        """Fit the feature scaler on training data."""
        self.scaler.fit(features)
        self.is_fitted = True
    
    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler."""
        if not self.is_fitted:
            return features
        
        return self.scaler.transform(features.reshape(1, -1) if features.ndim == 1 else features)
    
    def get_feature_names(self) -> List[str]:
        """Get names of extracted features."""
        return [
            # Dimensional features (6)
            'width', 'height', 'area', 'aspect_ratio', 'perimeter', 'compactness',
            # Color features (10)
            'bgr_mean_b', 'bgr_mean_g', 'bgr_mean_r',
            'bgr_std_b', 'bgr_std_g', 'bgr_std_r',
            'hsv_mean_h', 'hsv_mean_s', 'lab_mean_l', 'lab_mean_a',
            # Texture features (5)
            'texture_energy', 'texture_variance', 'edge_density', 'lbp_mean', 'lbp_std',
            # Shape features (3)
            'solidity', 'extent', 'contour_area_norm',
            # Context features (6)
            'norm_x', 'norm_y', 'center_distance', 'relative_size', 'distance_norm', 'occlusion_level'
        ][:25]  # Ensure we have exactly 25 feature names