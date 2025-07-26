"""
Inference and SageMaker deployment exceptions.
"""


class InvalidInputError(Exception):
    """Raised when input validation fails."""
    
    def __init__(self, message: str = "Invalid input", details: dict = None):
        self.details = details or {}
        super().__init__(message)


class ProcessingError(Exception):
    """Raised when processing pipeline fails."""
    
    def __init__(self, stage: str, message: str = "Processing failed", details: dict = None):
        self.stage = stage
        self.details = details or {}
        super().__init__(f"{message} at stage: {stage}")


class SageMakerError(Exception):
    """Base exception for SageMaker-specific errors."""
    pass


class EndpointError(SageMakerError):
    """Raised when endpoint operations fail."""
    
    def __init__(self, endpoint_name: str, message: str = "Endpoint error"):
        self.endpoint_name = endpoint_name
        super().__init__(f"{message} for endpoint: {endpoint_name}")


class ModelArtifactError(SageMakerError):
    """Raised when model artifact operations fail."""
    
    def __init__(self, artifact_path: str, message: str = "Model artifact error"):
        self.artifact_path = artifact_path
        super().__init__(f"{message}: {artifact_path}")


class ContainerError(SageMakerError):
    """Raised when container operations fail."""
    pass


class HealthCheckError(SageMakerError):
    """Raised when health check fails."""
    
    def __init__(self, component: str, message: str = "Health check failed"):
        self.component = component
        super().__init__(f"{message} for component: {component}")


class ResourceExhaustionError(ProcessingError):
    """Raised when system resources are exhausted."""
    
    def __init__(self, resource_type: str, usage: float, limit: float):
        self.resource_type = resource_type
        self.usage = usage
        self.limit = limit
        super().__init__(
            "resource_exhaustion",
            f"{resource_type} usage {usage} exceeds limit {limit}"
        )