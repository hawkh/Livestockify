"""
Weight estimation related exceptions.
"""


class WeightEstimationError(Exception):
    """Base exception for weight estimation errors."""
    pass


class FeatureExtractionError(WeightEstimationError):
    """Raised when feature extraction fails."""
    
    def __init__(self, message: str = "Feature extraction failed", bbox: list = None):
        self.bbox = bbox
        super().__init__(message)


class DistanceCompensationError(WeightEstimationError):
    """Raised when distance compensation fails."""
    
    def __init__(self, distance: float, message: str = "Distance compensation failed"):
        self.distance = distance
        super().__init__(f"{message} for distance: {distance}m")


class WeightValidationError(WeightEstimationError):
    """Raised when weight validation fails."""
    
    def __init__(self, weight: float, age_category: str = None):
        self.weight = weight
        self.age_category = age_category
        message = f"Invalid weight: {weight}kg"
        if age_category:
            message += f" for age category: {age_category}"
        super().__init__(message)


class ModelPredictionError(WeightEstimationError):
    """Raised when neural network prediction fails."""
    pass