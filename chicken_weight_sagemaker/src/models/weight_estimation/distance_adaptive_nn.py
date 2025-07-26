"""
Distance-adaptive neural network for chicken weight estimation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
import joblib
from pathlib import Path

from ...core.interfaces.detection import Detection
from ...core.interfaces.weight_estimation import (
    WeightEstimate, WeightEstimationResult, DistanceAdaptiveWeightModel
)
from ...core.exceptions.weight_estimation_exceptions import (
    ModelPredictionError, WeightValidationError, FeatureExtractionError
)
from .feature_extractor import ChickenFeatureExtractor
from .age_classifier import ChickenAgeClassifier, ChickenAgeCategory


class DistanceAdaptiveWeightNetwork(nn.Module):
    """Neural network architecture for distance-adaptive weight estimation."""
    
    def __init__(
        self, 
        input_size: int = 25,
        hidden_sizes: List[int] = [128, 256, 128, 64],
        dropout_rate: float = 0.3
    ):
        super(DistanceAdaptiveWeightNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            
            # Add dropout except for the last layer
            if i < len(hidden_sizes) - 1:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_size = hidden_size
        
        # Output layer (single neuron for weight prediction)
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.ReLU())  # Ensure positive weights
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)


class DistanceAdaptiveWeightNN(DistanceAdaptiveWeightModel):
    """Distance-adaptive weight estimation using neural networks."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        camera_params: Optional[Any] = None,
        device: str = "auto"
    ):
        self.model_path = model_path
        self.camera_params = camera_params
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize components
        self.feature_extractor = ChickenFeatureExtractor()
        self.age_classifier = ChickenAgeClassifier()
        self.model = None
        self.is_loaded = False
        
        # Model parameters
        self.input_size = 25
        self.weight_tolerance = 0.25  # ±25% tolerance
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """Load the weight estimation model."""
        try:
            model_path = Path(model_path)
            
            if not model_path.exists():
                # Create a new model for training
                self.model = DistanceAdaptiveWeightNetwork(
                    input_size=self.input_size
                ).to(self.device)
                print(f"Created new model (file not found: {model_path})")
                return
            
            # Load saved model
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Create model with saved architecture
            if 'model_config' in checkpoint:
                config = checkpoint['model_config']
                self.model = DistanceAdaptiveWeightNetwork(**config).to(self.device)
            else:
                self.model = DistanceAdaptiveWeightNetwork(
                    input_size=self.input_size
                ).to(self.device)
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load feature scaler if available
            if 'scaler' in checkpoint:
                self.feature_extractor.scaler = checkpoint['scaler']
                self.feature_extractor.is_fitted = True
            
            self.model.eval()
            self.is_loaded = True
            
            print(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            raise ModelPredictionError(f"Failed to load model: {str(e)}")
    
    def save_model(self, model_path: str) -> None:
        """Save the weight estimation model."""
        try:
            if self.model is None:
                raise ModelPredictionError("No model to save")
            
            model_path = Path(model_path)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare checkpoint
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'model_config': {
                    'input_size': self.input_size,
                    'hidden_sizes': self.model.hidden_sizes,
                    'dropout_rate': self.model.dropout_rate
                },
                'scaler': self.feature_extractor.scaler if self.feature_extractor.is_fitted else None
            }
            
            torch.save(checkpoint, model_path)
            print(f"Model saved to {model_path}")
            
        except Exception as e:
            raise ModelPredictionError(f"Failed to save model: {str(e)}")
    
    def estimate_weight(
        self, 
        frame: np.ndarray, 
        detection: Detection
    ) -> WeightEstimate:
        """
        Estimate weight for a single chicken detection.
        
        Args:
            frame: Input frame containing the chicken
            detection: Detection information
            
        Returns:
            WeightEstimate for the chicken
        """
        # Use distance-aware estimation with default distance
        return self.estimate_weight_with_distance(
            frame, detection, distance=3.0, occlusion_level=0.0
        )
    
    def estimate_weight_with_distance(
        self,
        frame: np.ndarray,
        detection: Detection,
        distance: float,
        occlusion_level: float = 0.0
    ) -> WeightEstimate:
        """
        Estimate weight with distance compensation.
        
        Args:
            frame: Input frame
            detection: Detection information
            distance: Estimated distance to chicken in meters
            occlusion_level: Level of occlusion (0-1)
            
        Returns:
            Distance-compensated WeightEstimate
        """
        try:
            if self.model is None:
                raise ModelPredictionError("Model not loaded")
            
            # Extract distance-compensated features
            features = self.extract_distance_compensated_features(
                frame, detection.bbox, distance
            )
            
            # Add occlusion information to features
            features_with_occlusion = np.append(features, occlusion_level)
            
            # Ensure correct feature size
            if len(features_with_occlusion) < self.input_size:
                features_with_occlusion = np.pad(
                    features_with_occlusion, 
                    (0, self.input_size - len(features_with_occlusion)), 
                    'constant'
                )
            elif len(features_with_occlusion) > self.input_size:
                features_with_occlusion = features_with_occlusion[:self.input_size]
            
            # Transform features if scaler is fitted
            if self.feature_extractor.is_fitted:
                features_with_occlusion = self.feature_extractor.transform_features(
                    features_with_occlusion
                )
            
            # Convert to tensor and predict
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features_with_occlusion).unsqueeze(0).to(self.device)
                weight_pred = self.model(features_tensor).cpu().numpy()[0, 0]
            
            # Classify age for validation
            age_category = self.age_classifier.classify_age_from_features(
                features, distance
            )
            
            # Validate weight range
            is_valid = self.validate_weight_range(weight_pred, age_category.name)
            
            # Calculate confidence
            confidence = self._calculate_weight_confidence(
                weight_pred, age_category, distance, occlusion_level
            )
            
            # Adjust confidence for occlusion
            confidence *= (1.0 - occlusion_level * 0.3)
            
            # Calculate error range
            error_range = f"±{weight_pred * self.weight_tolerance:.2f}kg"
            
            return WeightEstimate(
                value=float(weight_pred),
                unit="kg",
                confidence=confidence,
                error_range=error_range,
                distance_compensated=True,
                occlusion_adjusted=occlusion_level > 0.1,
                age_category=age_category.name,
                method="distance_adaptive_nn"
            )
            
        except Exception as e:
            raise ModelPredictionError(f"Weight estimation failed: {str(e)}")
    
    def extract_distance_compensated_features(
        self,
        frame: np.ndarray,
        bbox: List[float],
        distance: float
    ) -> np.ndarray:
        """
        Extract features compensated for distance.
        
        Args:
            frame: Input frame
            bbox: Bounding box coordinates
            distance: Distance to chicken
            
        Returns:
            Feature vector as numpy array
        """
        try:
            # Create a temporary detection object for feature extraction
            temp_detection = Detection(
                bbox=bbox,
                confidence=1.0,
                class_id=0,
                class_name="chicken"
            )
            
            # Extract features with distance information
            features = self.feature_extractor.extract_features(
                frame, temp_detection, distance, None
            )
            
            return features
            
        except Exception as e:
            raise FeatureExtractionError(f"Distance-compensated feature extraction failed: {str(e)}")
    
    def validate_weight_range(
        self, 
        weight: float, 
        estimated_age: Optional[str] = None
    ) -> bool:
        """
        Validate if weight is within expected range for age.
        
        Args:
            weight: Estimated weight in kg
            estimated_age: Estimated age category
            
        Returns:
            True if weight is within expected range
        """
        try:
            if estimated_age:
                age_category = ChickenAgeCategory[estimated_age]
                return self.age_classifier.validate_weight_for_age(
                    weight, age_category, self.weight_tolerance
                )
            else:
                # General validation - reasonable weight range for chickens
                return 0.01 <= weight <= 6.0
                
        except (KeyError, WeightValidationError):
            return False
    
    def estimate_batch_weights(
        self, 
        frame: np.ndarray, 
        detections: List[Detection]
    ) -> WeightEstimationResult:
        """
        Estimate weights for multiple chicken detections.
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            WeightEstimationResult for all chickens
        """
        estimates = []
        processing_times = []
        
        for detection in detections:
            import time
            start_time = time.time()
            
            estimate = self.estimate_weight(frame, detection)
            estimates.append(estimate)
            
            processing_time = (time.time() - start_time) * 1000
            processing_times.append(processing_time)
        
        # Calculate summary statistics
        total_processing_time = sum(processing_times)
        weights = [est.value for est in estimates]
        average_weight = np.mean(weights) if weights else 0.0
        
        return WeightEstimationResult(
            estimates=estimates,
            processing_time_ms=total_processing_time,
            average_weight=average_weight,
            total_chickens=len(detections)
        )
    
    def _calculate_weight_confidence(
        self,
        weight: float,
        age_category: ChickenAgeCategory,
        distance: float,
        occlusion_level: float
    ) -> float:
        """Calculate confidence in weight estimation."""
        base_confidence = 0.8
        
        # Age-based confidence
        min_weight, max_weight = age_category.value
        weight_range = max_weight - min_weight
        
        if weight_range > 0:
            # Higher confidence if weight is near the center of expected range
            range_center = (min_weight + max_weight) / 2
            distance_from_center = abs(weight - range_center)
            age_confidence = max(0.3, 1.0 - (distance_from_center / (weight_range / 2)))
        else:
            age_confidence = 0.5
        
        # Distance-based confidence (closer = more confident)
        distance_confidence = max(0.4, 1.0 - (distance - 2.0) / 8.0)  # Optimal at 2m
        
        # Occlusion-based confidence
        occlusion_confidence = 1.0 - occlusion_level
        
        # Combined confidence
        overall_confidence = (
            0.4 * base_confidence +
            0.3 * age_confidence +
            0.2 * distance_confidence +
            0.1 * occlusion_confidence
        )
        
        return min(1.0, max(0.1, overall_confidence))
    
    def train_model(
        self,
        training_data: List[Dict[str, Any]],
        validation_data: Optional[List[Dict[str, Any]]] = None,
        epochs: int = 100,
        learning_rate: float = 0.001,
        batch_size: int = 32
    ) -> Dict[str, List[float]]:
        """
        Train the weight estimation model.
        
        Args:
            training_data: List of training samples
            validation_data: Optional validation data
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            self.model = DistanceAdaptiveWeightNetwork(
                input_size=self.input_size
            ).to(self.device)
        
        # Prepare training data
        X_train, y_train = self._prepare_training_data(training_data)
        
        # Fit feature scaler
        self.feature_extractor.fit_scaler(X_train)
        X_train = self.feature_extractor.transform_features(X_train)
        
        # Prepare validation data if provided
        X_val, y_val = None, None
        if validation_data:
            X_val, y_val = self._prepare_training_data(validation_data)
            X_val = self.feature_extractor.transform_features(X_val)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        
        # Setup training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training history
        history = {'train_loss': [], 'val_loss': []}
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / (len(X_train) // batch_size + 1)
            history['train_loss'].append(avg_train_loss)
            
            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.FloatTensor(X_val).to(self.device)
                    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
                    
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                    history['val_loss'].append(val_loss)
                    
                    scheduler.step(val_loss)
                
                self.model.train()
            
            # Print progress
            if epoch % 20 == 0:
                if X_val is not None:
                    print(f'Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')
                else:
                    print(f'Epoch {epoch}, Train Loss: {avg_train_loss:.4f}')
        
        self.model.eval()
        self.is_loaded = True
        
        return history
    
    def _prepare_training_data(
        self, 
        training_data: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for model training."""
        features_list = []
        weights_list = []
        
        for sample in training_data:
            frame = sample['frame']
            bbox = sample['bbox']
            weight = sample['weight']
            distance = sample.get('distance', 3.0)
            
            # Extract features
            features = self.extract_distance_compensated_features(frame, bbox, distance)
            
            features_list.append(features)
            weights_list.append(weight)
        
        return np.array(features_list), np.array(weights_list)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            'model_loaded': self.is_loaded,
            'model_path': str(self.model_path) if self.model_path else None,
            'device': str(self.device),
            'input_size': self.input_size,
            'weight_tolerance': self.weight_tolerance,
            'feature_scaler_fitted': self.feature_extractor.is_fitted
        }