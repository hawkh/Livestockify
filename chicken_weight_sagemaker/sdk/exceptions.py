"""
Exception classes for the Chicken Weight Estimation SDK
"""


class ChickenWeightSDKError(Exception):
    """Base exception class for SDK errors."""
    
    def __init__(self, message: str, error_code: str = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
    
    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class EndpointError(ChickenWeightSDKError):
    """Exception raised when there are issues with the endpoint."""
    
    def __init__(self, message: str, status_code: int = None):
        super().__init__(message, "ENDPOINT_ERROR")
        self.status_code = status_code


class ProcessingError(ChickenWeightSDKError):
    """Exception raised when there are issues with image/video processing."""
    
    def __init__(self, message: str, frame_id: int = None):
        super().__init__(message, "PROCESSING_ERROR")
        self.frame_id = frame_id


class AuthenticationError(ChickenWeightSDKError):
    """Exception raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, "AUTH_ERROR")


class ValidationError(ChickenWeightSDKError):
    """Exception raised when input validation fails."""
    
    def __init__(self, message: str, field: str = None):
        super().__init__(message, "VALIDATION_ERROR")
        self.field = field


class TimeoutError(ChickenWeightSDKError):
    """Exception raised when requests timeout."""
    
    def __init__(self, message: str = "Request timed out", timeout_duration: float = None):
        super().__init__(message, "TIMEOUT_ERROR")
        self.timeout_duration = timeout_duration


class RateLimitError(ChickenWeightSDKError):
    """Exception raised when rate limits are exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = None):
        super().__init__(message, "RATE_LIMIT_ERROR")
        self.retry_after = retry_after


class ConfigurationError(ChickenWeightSDKError):
    """Exception raised when there are configuration issues."""
    
    def __init__(self, message: str, config_field: str = None):
        super().__init__(message, "CONFIG_ERROR")
        self.config_field = config_field