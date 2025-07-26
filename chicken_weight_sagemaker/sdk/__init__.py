"""
Chicken Weight Estimation SDK

A Python SDK for integrating with the chicken weight estimation system.
"""

from .client import ChickenWeightClient, AsyncChickenWeightClient
from .models import ChickenDetection, ProcessingResult, TrackingInfo
from .exceptions import ChickenWeightSDKError, EndpointError, ProcessingError
from .utils import ImageProcessor, VideoProcessor, BatchProcessor

__version__ = "1.0.0"
__author__ = "Chicken Weight Estimation Team"

__all__ = [
    'ChickenWeightClient',
    'AsyncChickenWeightClient',
    'ChickenDetection',
    'ProcessingResult',
    'TrackingInfo',
    'ChickenWeightSDKError',
    'EndpointError',
    'ProcessingError',
    'ImageProcessor',
    'VideoProcessor',
    'BatchProcessor'
]