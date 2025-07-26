"""
Re-identification feature extraction for chicken tracking.
"""

import cv2
import numpy as np
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.interfaces.detection import Detection
from ...core.exceptions.tracking_exceptions import ReidentificationError


class ChickenReIDFeatureExtractor:
    """Extract re-identification features for chicken tracking."""
    
    def __init__(self, feature_dim: int = 128):
        self.feature_dim = feature_dim
        
        # Simple CNN-based feature extractor
        self.feature_extractor = self._build_feature_extractor()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        
        # Image preprocessing
        self.input_size = (64, 128)  # Height, Width for person ReID standard
        
    def _build_feature_extractor(self) -> nn.Module:
        """Build a simple CNN for feature extraction."""
        return nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            
            # FC layers
            nn.Linear(128, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, self.feature_dim),
            # L2 normalization will be applied in forward pass
        )
    
    def extract_features(self, frame: np.ndarray, detection: Detection) -> np.ndarray:
        """
        Extract ReID features from chicken detection.
        
        Args:
            frame: Input frame
            detection: Detection to extract features from
            
        Returns:
            Feature vector for re-identification
        """
        try:
            # Extract chicken region
            x1, y1, x2, y2 = [int(coord) for coord in detection.bbox]
            chicken_crop = frame[y1:y2, x1:x2]
            
            if chicken_crop.size == 0:
                return np.zeros(self.feature_dim)
            
            # Preprocess image
            processed_crop = self._preprocess_image(chicken_crop)
            
            # Extract features using CNN
            with torch.no_grad():
                input_tensor = torch.FloatTensor(processed_crop).unsqueeze(0).to(self.device)
                features = self.feature_extractor(input_tensor)
                # Apply L2 normalization
                features = torch.nn.functional.normalize(features, p=2, dim=1)
                features = features.cpu().numpy().flatten()
            
            return features
            
        except Exception as e:
            raise ReidentificationError(
                detection.bbox[0] if detection.bbox else "unknown",
                f"Feature extraction failed: {str(e)}"
            )
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for feature extraction."""
        # Resize to standard size
        resized = cv2.resize(image, (self.input_size[1], self.input_size[0]))
        
        # Convert BGR to RGB
        if len(resized.shape) == 3:
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb_image.astype(np.float32) / 255.0
        
        # Transpose to CHW format
        transposed = np.transpose(normalized, (2, 0, 1))
        
        return transposed
    
    def extract_batch_features(
        self, 
        frame: np.ndarray, 
        detections: List[Detection]
    ) -> List[np.ndarray]:
        """Extract features for multiple detections."""
        features_list = []
        
        for detection in detections:
            features = self.extract_features(frame, detection)
            features_list.append(features)
        
        return features_list
    
    def calculate_similarity(
        self, 
        features1: np.ndarray, 
        features2: np.ndarray
    ) -> float:
        """Calculate cosine similarity between two feature vectors."""
        try:
            # Ensure features are normalized
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Cosine similarity
            similarity = np.dot(features1, features2) / (norm1 * norm2)
            
            return float(similarity)
            
        except Exception:
            return 0.0
    
    def calculate_distance(
        self, 
        features1: np.ndarray, 
        features2: np.ndarray
    ) -> float:
        """Calculate Euclidean distance between feature vectors."""
        try:
            distance = np.linalg.norm(features1 - features2)
            return float(distance)
        except Exception:
            return float('inf')


class L2Norm(nn.Module):
    """L2 normalization layer."""
    
    def __init__(self, dim: int = 1):
        super(L2Norm, self).__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=self.dim)


class SimpleReIDFeatureExtractor:
    """Simple hand-crafted feature extractor as fallback."""
    
    def __init__(self, feature_dim: int = 64):
        self.feature_dim = feature_dim
    
    def extract_features(self, frame: np.ndarray, detection: Detection) -> np.ndarray:
        """Extract simple hand-crafted features."""
        try:
            x1, y1, x2, y2 = [int(coord) for coord in detection.bbox]
            chicken_crop = frame[y1:y2, x1:x2]
            
            if chicken_crop.size == 0:
                return np.zeros(self.feature_dim)
            
            # Extract various features
            color_features = self._extract_color_features(chicken_crop)
            texture_features = self._extract_texture_features(chicken_crop)
            shape_features = self._extract_shape_features(chicken_crop)
            
            # Combine features
            all_features = np.concatenate([
                color_features,
                texture_features,
                shape_features
            ])
            
            # Pad or truncate to desired dimension
            if len(all_features) < self.feature_dim:
                all_features = np.pad(
                    all_features, 
                    (0, self.feature_dim - len(all_features)), 
                    'constant'
                )
            elif len(all_features) > self.feature_dim:
                all_features = all_features[:self.feature_dim]
            
            # Normalize
            norm = np.linalg.norm(all_features)
            if norm > 0:
                all_features = all_features / norm
            
            return all_features
            
        except Exception:
            return np.zeros(self.feature_dim)
    
    def _extract_color_features(self, image: np.ndarray) -> np.ndarray:
        """Extract color histogram features."""
        if len(image.shape) == 3:
            # Convert to HSV for better color representation
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Calculate histograms for each channel
            h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256])
            
            # Normalize histograms
            h_hist = h_hist.flatten() / (h_hist.sum() + 1e-7)
            s_hist = s_hist.flatten() / (s_hist.sum() + 1e-7)
            v_hist = v_hist.flatten() / (v_hist.sum() + 1e-7)
            
            return np.concatenate([h_hist, s_hist, v_hist])
        else:
            # Grayscale histogram
            hist = cv2.calcHist([image], [0], None, [16], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-7)
            return np.tile(hist, 3)  # Repeat to match color feature size
    
    def _extract_texture_features(self, image: np.ndarray) -> np.ndarray:
        """Extract texture features using LBP-like approach."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Simple texture measures
        # 1. Standard deviation (texture roughness)
        texture_std = np.std(gray)
        
        # 2. Gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        gradient_mean = np.mean(gradient_mag)
        
        # 3. Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        return np.array([texture_std / 255.0, gradient_mean / 255.0, edge_density])
    
    def _extract_shape_features(self, image: np.ndarray) -> np.ndarray:
        """Extract shape-based features."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Find contours
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Shape features
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Aspect ratio
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = w / h if h > 0 else 1.0
            
            # Extent (ratio of contour area to bounding rectangle area)
            rect_area = w * h
            extent = area / rect_area if rect_area > 0 else 0.0
            
            # Solidity (ratio of contour area to convex hull area)
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0.0
            
            # Normalize features
            normalized_area = area / (image.shape[0] * image.shape[1])
            normalized_perimeter = perimeter / (2 * (image.shape[0] + image.shape[1]))
            
            return np.array([
                normalized_area, normalized_perimeter, aspect_ratio, extent, solidity
            ])
        else:
            return np.zeros(5)


class FeatureBank:
    """Bank for storing and retrieving ReID features."""
    
    def __init__(self, max_features_per_track: int = 10):
        self.max_features_per_track = max_features_per_track
        self.feature_bank = {}  # track_id -> list of features
    
    def add_features(self, track_id: str, features: np.ndarray):
        """Add features for a track."""
        if track_id not in self.feature_bank:
            self.feature_bank[track_id] = []
        
        self.feature_bank[track_id].append(features)
        
        # Keep only recent features
        if len(self.feature_bank[track_id]) > self.max_features_per_track:
            self.feature_bank[track_id] = self.feature_bank[track_id][-self.max_features_per_track:]
    
    def get_average_features(self, track_id: str) -> Optional[np.ndarray]:
        """Get average features for a track."""
        if track_id not in self.feature_bank or not self.feature_bank[track_id]:
            return None
        
        features_array = np.array(self.feature_bank[track_id])
        return np.mean(features_array, axis=0)
    
    def calculate_similarity_to_bank(
        self, 
        features: np.ndarray, 
        track_id: str,
        similarity_func: callable = None
    ) -> float:
        """Calculate similarity between features and track's feature bank."""
        if similarity_func is None:
            similarity_func = self._cosine_similarity
        
        track_features = self.get_average_features(track_id)
        if track_features is None:
            return 0.0
        
        return similarity_func(features, track_features)
    
    def _cosine_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Calculate cosine similarity."""
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(feat1, feat2) / (norm1 * norm2)
    
    def remove_track(self, track_id: str):
        """Remove features for a track."""
        if track_id in self.feature_bank:
            del self.feature_bank[track_id]
    
    def get_all_track_ids(self) -> List[str]:
        """Get all track IDs in the bank."""
        return list(self.feature_bank.keys())
    
    def clear(self):
        """Clear all features."""
        self.feature_bank.clear()