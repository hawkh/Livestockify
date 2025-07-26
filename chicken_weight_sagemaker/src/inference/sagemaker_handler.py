"""
SageMaker inference handler for chicken weight estimation.
"""

import os
import json
import logging
import traceback
from typing import Dict, Any, List, Optional
import torch
import numpy as np

from .stream_handler import RealTimeStreamProcessor
from ..utils.config.config_manager import ConfigManager
from ..core.exceptions.inference_exceptions import ProcessingError, InvalidInputError, SageMakerError


class SageMakerInferenceHandler:
    """
    SageMaker inference handler that wraps the stream processor.
    This class follows SageMaker's inference handler pattern.
    """
    
    def __init__(self):
        self.processor = None
        self.config_manager = None
        self.logger = logging.getLogger(__name__)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def model_fn(self, model_dir: str):
        """
        Load the model for inference.
        This function is called by SageMaker when the container starts.
        
        Args:
            model_dir: Directory containing model artifacts
            
        Returns:
            Loaded model (in this case, our processor)
        """
        try:
            self.logger.info(f"Loading model from {model_dir}")
            
            # Initialize configuration manager
            self.config_manager = ConfigManager(config_dir=os.path.join(model_dir, "config"))
            
            # Initialize the stream processor
            self.processor = RealTimeStreamProcessor(self.config_manager)
            
            self.logger.info("Model loaded successfully")
            return self.processor
            
        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise SageMakerError(f"Model loading failed: {str(e)}")
    
    def input_fn(self, request_body: str, request_content_type: str) -> Dict[str, Any]:
        """
        Parse input data for inference.
        
        Args:
            request_body: Raw request body
            request_content_type: Content type of the request
            
        Returns:
            Parsed input data
        """
        try:
            if request_content_type == 'application/json':
                input_data = json.loads(request_body)
            else:
                raise InvalidInputError(f"Unsupported content type: {request_content_type}")
            
            # Validate input format
            if not self._validate_input(input_data):
                raise InvalidInputError("Invalid input format")
            
            return input_data
            
        except json.JSONDecodeError as e:
            raise InvalidInputError(f"JSON parsing failed: {str(e)}")
        except Exception as e:
            self.logger.error(f"Input parsing failed: {str(e)}")
            raise InvalidInputError(f"Input parsing failed: {str(e)}")
    
    def predict_fn(self, input_data: Dict[str, Any], model) -> Dict[str, Any]:
        """
        Run inference on the input data.
        
        Args:
            input_data: Parsed input data
            model: Loaded model (our processor)
            
        Returns:
            Inference results
        """
        try:
            if model is None:
                raise SageMakerError("Model not loaded")
            
            # Extract stream data
            stream_data = input_data.get('stream_data', {})
            
            # Process the frame
            result = model.process_frame(
                frame_data=stream_data.get('frame'),
                camera_id=stream_data.get('camera_id', 'default'),
                frame_sequence=stream_data.get('frame_sequence', 0)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            # Return error response in expected format
            return {
                "camera_id": stream_data.get('camera_id', 'unknown'),
                "timestamp": "",
                "frame_sequence": stream_data.get('frame_sequence', 0),
                "status": "error",
                "error_message": str(e),
                "detections": [],
                "total_chickens_detected": 0,
                "processing_time_ms": 0.0
            }
    
    def output_fn(self, prediction: Dict[str, Any], accept: str) -> str:
        """
        Format the prediction output.
        
        Args:
            prediction: Prediction results
            accept: Requested response content type
            
        Returns:
            Formatted response
        """
        try:
            if accept == 'application/json':
                return json.dumps(prediction, indent=2)
            else:
                raise InvalidInputError(f"Unsupported accept type: {accept}")
                
        except Exception as e:
            self.logger.error(f"Output formatting failed: {str(e)}")
            error_response = {
                "status": "error",
                "error_message": f"Output formatting failed: {str(e)}"
            }
            return json.dumps(error_response)
    
    def _validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data format."""
        try:
            # Check for required top-level structure
            if 'stream_data' not in input_data:
                return False
            
            stream_data = input_data['stream_data']
            
            # Check for required fields in stream_data
            required_fields = ['frame']
            for field in required_fields:
                if field not in stream_data:
                    return False
            
            # Validate frame data is not empty
            if not stream_data['frame']:
                return False
            
            return True
            
        except Exception:
            return False


# Global handler instance for SageMaker
handler = SageMakerInferenceHandler()


def model_fn(model_dir: str):
    """SageMaker model loading function."""
    return handler.model_fn(model_dir)


def input_fn(request_body: str, request_content_type: str):
    """SageMaker input parsing function."""
    return handler.input_fn(request_body, request_content_type)


def predict_fn(input_data: Dict[str, Any], model):
    """SageMaker prediction function."""
    return handler.predict_fn(input_data, model)


def output_fn(prediction: Dict[str, Any], accept: str):
    """SageMaker output formatting function."""
    return handler.output_fn(prediction, accept)


# Health check function for SageMaker
def health_check():
    """Health check for SageMaker container."""
    try:
        if handler.processor is None:
            return {"status": "unhealthy", "reason": "Model not loaded"}
        
        health_status = handler.processor.health_check()
        return health_status
        
    except Exception as e:
        return {
            "status": "unhealthy", 
            "reason": f"Health check failed: {str(e)}"
        }


class BatchInferenceHandler:
    """Handler for batch inference requests."""
    
    def __init__(self, processor: RealTimeStreamProcessor):
        self.processor = processor
        self.logger = logging.getLogger(__name__)
    
    def process_batch(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of frames.
        
        Args:
            batch_data: List of frame data dictionaries
            
        Returns:
            List of processing results
        """
        results = []
        
        for i, frame_data in enumerate(batch_data):
            try:
                # Extract frame information
                stream_data = frame_data.get('stream_data', {})
                
                # Process individual frame
                result = self.processor.process_frame(
                    frame_data=stream_data.get('frame'),
                    camera_id=stream_data.get('camera_id', f'batch_camera_{i}'),
                    frame_sequence=stream_data.get('frame_sequence', i)
                )
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Batch processing failed for frame {i}: {str(e)}")
                
                # Add error result
                error_result = {
                    "camera_id": stream_data.get('camera_id', f'batch_camera_{i}'),
                    "frame_sequence": i,
                    "status": "error",
                    "error_message": str(e),
                    "detections": [],
                    "total_chickens_detected": 0,
                    "processing_time_ms": 0.0
                }
                results.append(error_result)
        
        return results


class ModelMetadata:
    """Model metadata for SageMaker."""
    
    @staticmethod
    def get_model_metadata() -> Dict[str, Any]:
        """Get model metadata information."""
        return {
            "model_name": "chicken_weight_estimator",
            "model_version": "1.0.0",
            "framework": "pytorch",
            "framework_version": torch.__version__,
            "description": "Real-time chicken detection and weight estimation for poultry farms",
            "input_format": {
                "content_type": "application/json",
                "schema": {
                    "stream_data": {
                        "frame": "base64_encoded_image",
                        "camera_id": "string",
                        "frame_sequence": "integer",
                        "timestamp": "string (optional)",
                        "parameters": "object (optional)"
                    }
                }
            },
            "output_format": {
                "content_type": "application/json",
                "schema": {
                    "camera_id": "string",
                    "timestamp": "string",
                    "frame_sequence": "integer",
                    "detections": "array of detection objects",
                    "total_chickens_detected": "integer",
                    "average_weight": "float",
                    "processing_time_ms": "float",
                    "status": "string"
                }
            },
            "capabilities": [
                "real_time_processing",
                "occlusion_handling",
                "distance_compensation",
                "multi_object_tracking",
                "weight_estimation"
            ],
            "performance": {
                "target_latency_ms": 100,
                "max_batch_size": 10,
                "supported_image_formats": ["JPEG", "PNG"],
                "max_image_size": "5MB"
            }
        }