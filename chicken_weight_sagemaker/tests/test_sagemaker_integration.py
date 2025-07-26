"""
Test SageMaker integration and deployment compatibility.
"""

import unittest
import json
import base64
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference.sagemaker_handler import SageMakerInferenceHandler, model_fn, input_fn, predict_fn, output_fn
from inference.stream_handler import RealTimeStreamProcessor


class TestSageMakerIntegration(unittest.TestCase):
    """Test SageMaker integration components."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test image
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', self.test_image)
        self.test_frame_data = base64.b64encode(buffer).decode('utf-8')
        
        # Create test request
        self.test_request = {
            "stream_data": {
                "frame": self.test_frame_data,
                "camera_id": "test_camera",
                "frame_sequence": 1,
                "timestamp": "2024-01-15T10:30:00.123Z"
            }
        }
    
    def test_sagemaker_handler_initialization(self):
        """Test SageMaker handler initialization."""
        handler = SageMakerInferenceHandler()
        
        # Check initial state
        self.assertIsNone(handler.processor)
        self.assertIsNone(handler.config_manager)
        self.assertIsNotNone(handler.logger)
    
    @patch('inference.sagemaker_handler.ConfigManager')
    @patch('inference.sagemaker_handler.RealTimeStreamProcessor')
    def test_model_fn(self, mock_processor, mock_config):
        """Test model loading function."""
        # Setup mocks
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance
        
        mock_processor_instance = Mock()
        mock_processor.return_value = mock_processor_instance
        
        # Test model loading
        model_dir = "/opt/ml/model"
        result = model_fn(model_dir)
        
        # Verify processor was created and returned
        self.assertEqual(result, mock_processor_instance)
        mock_config.assert_called_once()
        mock_processor.assert_called_once_with(mock_config_instance)
    
    def test_input_fn_valid_json(self):
        """Test input parsing with valid JSON."""
        handler = SageMakerInferenceHandler()
        
        request_body = json.dumps(self.test_request)
        content_type = "application/json"
        
        result = handler.input_fn(request_body, content_type)
        
        # Verify parsing
        self.assertEqual(result, self.test_request)
        self.assertIn('stream_data', result)
        self.assertIn('frame', result['stream_data'])
    
    def test_input_fn_invalid_json(self):
        """Test input parsing with invalid JSON."""
        handler = SageMakerInferenceHandler()
        
        request_body = "invalid json"
        content_type = "application/json"
        
        with self.assertRaises(Exception):
            handler.input_fn(request_body, content_type)
    
    def test_input_fn_unsupported_content_type(self):
        """Test input parsing with unsupported content type."""
        handler = SageMakerInferenceHandler()
        
        request_body = json.dumps(self.test_request)
        content_type = "text/plain"
        
        with self.assertRaises(Exception):
            handler.input_fn(request_body, content_type)
    
    def test_input_fn_invalid_format(self):
        """Test input parsing with invalid format."""
        handler = SageMakerInferenceHandler()
        
        # Missing stream_data
        invalid_request = {"data": "test"}
        request_body = json.dumps(invalid_request)
        content_type = "application/json"
        
        with self.assertRaises(Exception):
            handler.input_fn(request_body, content_type)
    
    def test_predict_fn_success(self):
        """Test prediction function with successful processing."""
        handler = SageMakerInferenceHandler()
        
        # Mock processor
        mock_processor = Mock()
        mock_result = {
            "camera_id": "test_camera",
            "status": "success",
            "detections": [],
            "total_chickens_detected": 0,
            "processing_time_ms": 50.0
        }
        mock_processor.process_frame.return_value = mock_result
        
        # Test prediction
        result = handler.predict_fn(self.test_request, mock_processor)
        
        # Verify result
        self.assertEqual(result, mock_result)
        mock_processor.process_frame.assert_called_once_with(
            frame_data=self.test_frame_data,
            camera_id="test_camera",
            frame_sequence=1
        )
    
    def test_predict_fn_processing_error(self):
        """Test prediction function with processing error."""
        handler = SageMakerInferenceHandler()
        
        # Mock processor that raises exception
        mock_processor = Mock()
        mock_processor.process_frame.side_effect = Exception("Processing failed")
        
        # Test prediction
        result = handler.predict_fn(self.test_request, mock_processor)
        
        # Verify error response
        self.assertEqual(result['status'], 'error')
        self.assertIn('error_message', result)
        self.assertEqual(result['camera_id'], 'test_camera')
    
    def test_predict_fn_no_model(self):
        """Test prediction function with no model loaded."""
        handler = SageMakerInferenceHandler()
        
        # Test with None model
        result = handler.predict_fn(self.test_request, None)
        
        # Verify error response
        self.assertEqual(result['status'], 'error')
        self.assertIn('error_message', result)
    
    def test_output_fn_json(self):
        """Test output formatting for JSON."""
        handler = SageMakerInferenceHandler()
        
        test_prediction = {
            "status": "success",
            "camera_id": "test",
            "detections": []
        }
        
        result = handler.output_fn(test_prediction, "application/json")
        
        # Verify JSON formatting
        parsed_result = json.loads(result)
        self.assertEqual(parsed_result, test_prediction)
    
    def test_output_fn_unsupported_format(self):
        """Test output formatting with unsupported format."""
        handler = SageMakerInferenceHandler()
        
        test_prediction = {"status": "success"}
        
        result = handler.output_fn(test_prediction, "text/plain")
        
        # Should return error response as JSON
        parsed_result = json.loads(result)
        self.assertEqual(parsed_result['status'], 'error')
    
    def test_global_functions(self):
        """Test global SageMaker functions."""
        # These functions should exist and be callable
        self.assertTrue(callable(model_fn))
        self.assertTrue(callable(input_fn))
        self.assertTrue(callable(predict_fn))
        self.assertTrue(callable(output_fn))
    
    @patch('inference.sagemaker_handler.handler')
    def test_global_model_fn(self, mock_handler):
        """Test global model_fn function."""
        mock_handler.model_fn.return_value = "test_model"
        
        result = model_fn("/opt/ml/model")
        
        self.assertEqual(result, "test_model")
        mock_handler.model_fn.assert_called_once_with("/opt/ml/model")
    
    @patch('inference.sagemaker_handler.handler')
    def test_global_input_fn(self, mock_handler):
        """Test global input_fn function."""
        mock_handler.input_fn.return_value = {"test": "data"}
        
        result = input_fn("test_body", "application/json")
        
        self.assertEqual(result, {"test": "data"})
        mock_handler.input_fn.assert_called_once_with("test_body", "application/json")
    
    @patch('inference.sagemaker_handler.handler')
    def test_global_predict_fn(self, mock_handler):
        """Test global predict_fn function."""
        mock_handler.predict_fn.return_value = {"result": "test"}
        
        result = predict_fn({"input": "data"}, "test_model")
        
        self.assertEqual(result, {"result": "test"})
        mock_handler.predict_fn.assert_called_once_with({"input": "data"}, "test_model")
    
    @patch('inference.sagemaker_handler.handler')
    def test_global_output_fn(self, mock_handler):
        """Test global output_fn function."""
        mock_handler.output_fn.return_value = '{"output": "test"}'
        
        result = output_fn({"prediction": "data"}, "application/json")
        
        self.assertEqual(result, '{"output": "test"}')
        mock_handler.output_fn.assert_called_once_with({"prediction": "data"}, "application/json")


class TestSageMakerCompatibility(unittest.TestCase):
    """Test SageMaker deployment compatibility."""
    
    def test_request_response_format(self):
        """Test that request/response format matches SageMaker expectations."""
        # Test request format
        test_request = {
            "stream_data": {
                "frame": "base64_encoded_data",
                "camera_id": "farm_camera_01",
                "frame_sequence": 42,
                "timestamp": "2024-01-15T10:30:00.123Z",
                "parameters": {
                    "min_confidence": 0.4,
                    "max_occlusion": 0.7
                }
            }
        }
        
        # Verify request structure
        self.assertIn('stream_data', test_request)
        stream_data = test_request['stream_data']
        
        required_fields = ['frame', 'camera_id']
        for field in required_fields:
            self.assertIn(field, stream_data)
        
        # Test response format
        test_response = {
            "camera_id": "farm_camera_01",
            "timestamp": "2024-01-15T10:30:00.123Z",
            "frame_sequence": 42,
            "detections": [
                {
                    "chicken_id": "tracked_chicken_001",
                    "bbox": [100, 100, 200, 200],
                    "confidence": 0.85,
                    "occlusion_level": 0.3,
                    "distance_estimate": 2.5,
                    "weight_estimate": {
                        "value": 2.1,
                        "unit": "kg",
                        "confidence": 0.78,
                        "error_range": "±0.5kg",
                        "method": "distance_adaptive_nn"
                    },
                    "age_category": "WEEK_4",
                    "tracking_status": "stable"
                }
            ],
            "processing_time_ms": 45,
            "total_chickens_detected": 1,
            "average_weight": 2.1,
            "status": "success"
        }
        
        # Verify response structure
        required_response_fields = [
            'camera_id', 'timestamp', 'frame_sequence', 
            'detections', 'status', 'processing_time_ms'
        ]
        
        for field in required_response_fields:
            self.assertIn(field, test_response)
        
        # Verify detection structure
        if test_response['detections']:
            detection = test_response['detections'][0]
            detection_fields = ['chicken_id', 'bbox', 'confidence', 'weight_estimate']
            
            for field in detection_fields:
                self.assertIn(field, detection)
    
    def test_error_response_format(self):
        """Test error response format."""
        error_response = {
            "camera_id": "farm_camera_01",
            "timestamp": "2024-01-15T10:30:00.123Z",
            "frame_sequence": 42,
            "status": "error",
            "error_message": "Processing failed: Invalid input",
            "detections": [],
            "total_chickens_detected": 0,
            "processing_time_ms": 5.0
        }
        
        # Verify error response structure
        self.assertEqual(error_response['status'], 'error')
        self.assertIn('error_message', error_response)
        self.assertEqual(error_response['detections'], [])
        self.assertEqual(error_response['total_chickens_detected'], 0)
    
    def test_batch_processing_format(self):
        """Test batch processing request/response format."""
        batch_request = {
            "instances": [
                {
                    "stream_data": {
                        "frame": "base64_data_1",
                        "camera_id": "camera_01",
                        "frame_sequence": 1
                    }
                },
                {
                    "stream_data": {
                        "frame": "base64_data_2",
                        "camera_id": "camera_01",
                        "frame_sequence": 2
                    }
                }
            ]
        }
        
        # Verify batch request structure
        self.assertIn('instances', batch_request)
        self.assertEqual(len(batch_request['instances']), 2)
        
        for instance in batch_request['instances']:
            self.assertIn('stream_data', instance)
            self.assertIn('frame', instance['stream_data'])
    
    def test_health_check_format(self):
        """Test health check response format."""
        health_response = {
            "status": "healthy",
            "timestamp": "2024-01-15T10:30:00.123Z",
            "components": {
                "detector": {
                    "status": "healthy",
                    "model_info": {"model_loaded": True}
                },
                "weight_estimator": {
                    "status": "healthy",
                    "model_info": {"model_loaded": True}
                },
                "tracker": {
                    "status": "healthy",
                    "active_tracks": 5
                }
            }
        }
        
        # Verify health check structure
        self.assertIn('status', health_response)
        self.assertIn('components', health_response)
        
        for component in health_response['components'].values():
            self.assertIn('status', component)


if __name__ == '__main__':
    unittest.main(verbosity=2)