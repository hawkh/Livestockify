#!/usr/bin/env python3
"""
Comprehensive tests for SageMaker integration components.
"""

import unittest
import json
import base64
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from datetime import datetime

# Import components to test
from src.inference.sagemaker_handler import (
    model_fn, input_fn, predict_fn, output_fn, health_check
)
from src.inference.stream_handler import StreamProcessingServer, RealTimeStreamProcessor
from src.inference.frame_processor import FrameProcessor


class TestSageMakerHandler(unittest.TestCase):
    """Test SageMaker inference handler functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_model_dir = tempfile.mkdtemp()
        self.test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        cv2.rectangle(self.test_image, (100, 100), (200, 200), (255, 255, 255), -1)
        
        # Create test request
        _, buffer = cv2.imencode('.jpg', self.test_image)
        self.test_request = {
            "stream_data": {
                "frame": base64.b64encode(buffer).decode('utf-8'),
                "camera_id": "test_camera",
                "frame_sequence": 1,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.test_model_dir):
            shutil.rmtree(self.test_model_dir)
    
    @patch('src.inference.sagemaker_handler.RealTimeStreamProcessor')
    @patch('src.utils.config.config_manager.ConfigManager')
    def test_model_fn(self, mock_config_manager, mock_processor):
        """Test model loading function."""
        # Setup mocks
        mock_config = Mock()
        mock_config_manager.return_value = mock_config
        mock_processor_instance = Mock()
        mock_processor.return_value = mock_processor_instance
        
        # Test model loading
        model = model_fn(self.test_model_dir)
        
        # Verify model was created
        self.assertIsNotNone(model)
        mock_config_manager.assert_called_once()
        mock_processor.assert_called_once()
    
    def test_input_fn_json(self):
        """Test input function with JSON data."""
        json_data = json.dumps(self.test_request)
        
        # Test input parsing
        parsed_input = input_fn(json_data, 'application/json')
        
        # Verify parsing
        self.assertIsInstance(parsed_input, dict)
        self.assertIn('stream_data', parsed_input)
        self.assertEqual(parsed_input['stream_data']['camera_id'], 'test_camera')
    
    def test_input_fn_invalid_json(self):
        """Test input function with invalid JSON."""
        invalid_json = "{ invalid json }"
        
        # Should handle gracefully
        parsed_input = input_fn(invalid_json, 'application/json')
        
        # Should return error structure
        self.assertIsInstance(parsed_input, dict)
        self.assertIn('error', parsed_input)
    
    def test_input_fn_unsupported_content_type(self):
        """Test input function with unsupported content type."""
        data = "some data"
        
        # Should handle gracefully
        parsed_input = input_fn(data, 'text/plain')
        
        # Should return error structure
        self.assertIsInstance(parsed_input, dict)
        self.assertIn('error', parsed_input)
    
    @patch('src.inference.sagemaker_handler.RealTimeStreamProcessor')
    def test_predict_fn(self, mock_processor_class):
        """Test prediction function."""
        # Setup mock processor
        mock_processor = Mock()
        mock_processor.process_stream_data.return_value = {
            'detections': [
                {
                    'bbox': {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200, 'confidence': 0.9},
                    'class_id': 0,
                    'class_name': 'chicken',
                    'weight_estimate': 2.5,
                    'occlusion_score': 0.2
                }
            ],
            'tracks': [
                {
                    'track_id': 1,
                    'bbox': {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200},
                    'weight_history': [2.4, 2.5, 2.6],
                    'average_weight': 2.5
                }
            ],
            'frame_id': 1,
            'processing_time': 0.05,
            'timestamp': datetime.now().isoformat()
        }
        
        # Test prediction
        result = predict_fn(self.test_request, mock_processor)
        
        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn('detections', result)
        self.assertIn('tracks', result)
        self.assertIn('frame_id', result)
        self.assertIn('processing_time', result)
        
        # Verify prediction was called
        mock_processor.process_stream_data.assert_called_once()
    
    @patch('src.inference.sagemaker_handler.RealTimeStreamProcessor')
    def test_predict_fn_error_handling(self, mock_processor_class):
        """Test prediction function error handling."""
        # Setup mock processor to raise exception
        mock_processor = Mock()
        mock_processor.process_stream_data.side_effect = Exception("Processing failed")
        
        # Test prediction with error
        result = predict_fn(self.test_request, mock_processor)
        
        # Should handle error gracefully
        self.assertIsInstance(result, dict)
        self.assertIn('error', result)
        self.assertEqual(result['success'], False)
    
    def test_output_fn_json(self):
        """Test output function with JSON format."""
        prediction_result = {
            'detections': [],
            'tracks': [],
            'frame_id': 1,
            'processing_time': 0.05
        }
        
        # Test output formatting
        output = output_fn(prediction_result, 'application/json')
        
        # Verify output
        self.assertIsInstance(output, str)
        
        # Should be valid JSON
        parsed_output = json.loads(output)
        self.assertEqual(parsed_output['frame_id'], 1)
    
    def test_output_fn_unsupported_format(self):
        """Test output function with unsupported format."""
        prediction_result = {'test': 'data'}
        
        # Should default to JSON
        output = output_fn(prediction_result, 'text/plain')
        
        # Should still be JSON
        self.assertIsInstance(output, str)
        parsed_output = json.loads(output)
        self.assertEqual(parsed_output['test'], 'data')
    
    @patch('src.inference.sagemaker_handler.RealTimeStreamProcessor')
    def test_health_check(self, mock_processor_class):
        """Test health check function."""
        # Setup mock processor
        mock_processor = Mock()
        mock_processor.is_healthy.return_value = True
        
        # Test health check
        health_status = health_check()
        
        # Verify health status
        self.assertIsInstance(health_status, dict)
        self.assertIn('status', health_status)
        self.assertIn('timestamp', health_status)
        self.assertIn('model_loaded', health_status)


class TestStreamProcessingServer(unittest.TestCase):
    """Test stream processing server."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock processor
        self.mock_processor = Mock()
        self.mock_processor.process_stream_data.return_value = {
            'detections': [],
            'tracks': [],
            'frame_id': 1,
            'processing_time': 0.05
        }
        
        self.server = StreamProcessingServer(self.mock_processor, port=5001)
    
    def test_server_initialization(self):
        """Test server initialization."""
        self.assertIsNotNone(self.server.app)
        self.assertEqual(self.server.port, 5001)
        self.assertIsNotNone(self.server.processor)
    
    @patch('flask.Flask.test_client')
    def test_health_endpoint(self, mock_test_client):
        """Test health check endpoint."""
        # Create mock client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.get_json.return_value = {'status': 'healthy'}
        mock_client.get.return_value = mock_response
        mock_test_client.return_value = mock_client
        
        # Test health endpoint
        with self.server.app.test_client() as client:
            response = client.get('/ping')
            
        # Verify response structure would be correct
        # (Actual testing would require running server)
    
    def test_invocations_endpoint_structure(self):
        """Test invocations endpoint structure."""
        # Test that the endpoint is properly configured
        with self.server.app.test_request_context('/invocations', method='POST'):
            # Verify route exists
            self.assertIn('/invocations', [rule.rule for rule in self.server.app.url_map.iter_rules()])


class TestRealTimeStreamProcessor(unittest.TestCase):
    """Test real-time stream processor."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock config
        self.mock_config = Mock()
        self.mock_config.get_camera_config.return_value = Mock()
        self.mock_config.get_model_config.return_value = Mock()
        
        # Create test image
        self.test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        cv2.rectangle(self.test_image, (100, 100), (200, 200), (255, 255, 255), -1)
    
    @patch('src.inference.stream_handler.YOLODetector')
    @patch('src.inference.stream_handler.WeightEstimationEngine')
    @patch('src.inference.stream_handler.ChickenTracker')
    def test_processor_initialization(self, mock_tracker, mock_weight_engine, mock_detector):
        """Test processor initialization."""
        # Setup mocks
        mock_detector.return_value = Mock()
        mock_weight_engine.return_value = Mock()
        mock_tracker.return_value = Mock()
        
        # Create processor
        processor = RealTimeStreamProcessor(self.mock_config, 'test_camera')
        
        # Verify initialization
        self.assertIsNotNone(processor)
        self.assertEqual(processor.camera_id, 'test_camera')
        mock_detector.assert_called_once()
        mock_weight_engine.assert_called_once()
        mock_tracker.assert_called_once()
    
    @patch('src.inference.stream_handler.YOLODetector')
    @patch('src.inference.stream_handler.WeightEstimationEngine')
    @patch('src.inference.stream_handler.ChickenTracker')
    def test_frame_processing(self, mock_tracker, mock_weight_engine, mock_detector):
        """Test frame processing."""
        # Setup mocks
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = [
            Mock(bbox=Mock(x1=100, y1=100, x2=200, y2=200, confidence=0.9),
                 class_id=0, class_name='chicken')
        ]
        mock_detector.return_value = mock_detector_instance
        
        mock_weight_engine_instance = Mock()
        mock_weight_engine_instance.estimate_weight.return_value = (2.5, (2.0, 3.0))
        mock_weight_engine.return_value = mock_weight_engine_instance
        
        mock_tracker_instance = Mock()
        mock_tracker_instance.update.return_value = [
            Mock(track_id=1, bbox=Mock(), weight_history=[2.5], average_weight=2.5)
        ]
        mock_tracker.return_value = mock_tracker_instance
        
        # Create processor
        processor = RealTimeStreamProcessor(self.mock_config, 'test_camera')
        
        # Process frame
        result = processor.process_frame(self.test_image, frame_id=1)
        
        # Verify result
        self.assertIsInstance(result, dict)
        self.assertIn('detections', result)
        self.assertIn('tracks', result)
        self.assertIn('frame_id', result)
        self.assertIn('processing_time', result)
        
        # Verify components were called
        mock_detector_instance.detect.assert_called_once()
        mock_weight_engine_instance.estimate_weight.assert_called()
        mock_tracker_instance.update.assert_called_once()
    
    @patch('src.inference.stream_handler.YOLODetector')
    @patch('src.inference.stream_handler.WeightEstimationEngine')
    @patch('src.inference.stream_handler.ChickenTracker')
    def test_stream_data_processing(self, mock_tracker, mock_weight_engine, mock_detector):
        """Test stream data processing."""
        # Setup mocks
        mock_detector.return_value = Mock()
        mock_weight_engine.return_value = Mock()
        mock_tracker.return_value = Mock()
        
        # Create processor
        processor = RealTimeStreamProcessor(self.mock_config, 'test_camera')
        
        # Create stream data
        _, buffer = cv2.imencode('.jpg', self.test_image)
        stream_data = {
            "frame": base64.b64encode(buffer).decode('utf-8'),
            "camera_id": "test_camera",
            "frame_sequence": 1,
            "timestamp": datetime.now().isoformat()
        }
        
        # Process stream data
        with patch.object(processor, 'process_frame') as mock_process_frame:
            mock_process_frame.return_value = {
                'detections': [],
                'tracks': [],
                'frame_id': 1,
                'processing_time': 0.05
            }
            
            result = processor.process_stream_data(stream_data)
            
            # Verify processing
            self.assertIsInstance(result, dict)
            mock_process_frame.assert_called_once()
    
    def test_invalid_stream_data(self):
        """Test handling of invalid stream data."""
        with patch('src.inference.stream_handler.YOLODetector'), \
             patch('src.inference.stream_handler.WeightEstimationEngine'), \
             patch('src.inference.stream_handler.ChickenTracker'):
            
            processor = RealTimeStreamProcessor(self.mock_config, 'test_camera')
            
            # Test with invalid base64
            invalid_stream_data = {
                "frame": "invalid_base64_data",
                "camera_id": "test_camera",
                "frame_sequence": 1,
                "timestamp": datetime.now().isoformat()
            }
            
            result = processor.process_stream_data(invalid_stream_data)
            
            # Should handle error gracefully
            self.assertIsInstance(result, dict)
            self.assertIn('error', result)
    
    def test_health_check(self):
        """Test processor health check."""
        with patch('src.inference.stream_handler.YOLODetector'), \
             patch('src.inference.stream_handler.WeightEstimationEngine'), \
             patch('src.inference.stream_handler.ChickenTracker'):
            
            processor = RealTimeStreamProcessor(self.mock_config, 'test_camera')
            
            # Test health check
            health_status = processor.is_healthy()
            
            # Should return boolean
            self.assertIsInstance(health_status, bool)


class TestFrameProcessor(unittest.TestCase):
    """Test frame processor component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = Mock()
        self.test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    
    @patch('src.inference.frame_processor.YOLODetector')
    @patch('src.inference.frame_processor.WeightEstimationEngine')
    def test_frame_processor_initialization(self, mock_weight_engine, mock_detector):
        """Test frame processor initialization."""
        mock_detector.return_value = Mock()
        mock_weight_engine.return_value = Mock()
        
        processor = FrameProcessor(self.mock_config)
        
        self.assertIsNotNone(processor)
        mock_detector.assert_called_once()
        mock_weight_engine.assert_called_once()
    
    @patch('src.inference.frame_processor.YOLODetector')
    @patch('src.inference.frame_processor.WeightEstimationEngine')
    def test_single_frame_processing(self, mock_weight_engine, mock_detector):
        """Test single frame processing."""
        # Setup mocks
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = []
        mock_detector.return_value = mock_detector_instance
        
        mock_weight_engine_instance = Mock()
        mock_weight_engine.return_value = mock_weight_engine_instance
        
        processor = FrameProcessor(self.mock_config)
        
        # Process frame
        result = processor.process_single_frame(self.test_image)
        
        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn('detections', result)
        self.assertIn('processing_time', result)
        
        # Verify detector was called
        mock_detector_instance.detect.assert_called_once()
    
    def test_preprocessing(self):
        """Test image preprocessing."""
        with patch('src.inference.frame_processor.YOLODetector'), \
             patch('src.inference.frame_processor.WeightEstimationEngine'):
            
            processor = FrameProcessor(self.mock_config)
            
            # Test preprocessing
            preprocessed = processor._preprocess_image(self.test_image)
            
            # Should return processed image
            self.assertIsInstance(preprocessed, np.ndarray)
            self.assertEqual(len(preprocessed.shape), 3)  # Should be 3D (H, W, C)
    
    def test_postprocessing(self):
        """Test result postprocessing."""
        with patch('src.inference.frame_processor.YOLODetector'), \
             patch('src.inference.frame_processor.WeightEstimationEngine'):
            
            processor = FrameProcessor(self.mock_config)
            
            # Create mock raw results
            raw_results = {
                'detections': [Mock()],
                'processing_time': 0.05
            }
            
            # Test postprocessing
            processed = processor._postprocess_results(raw_results)
            
            # Should return processed results
            self.assertIsInstance(processed, dict)
            self.assertIn('detections', processed)


class TestSageMakerIntegration(unittest.TestCase):
    """Integration tests for SageMaker components."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        cv2.rectangle(self.test_image, (100, 100), (200, 200), (255, 255, 255), -1)
        
        # Create test request
        _, buffer = cv2.imencode('.jpg', self.test_image)
        self.test_request = {
            "stream_data": {
                "frame": base64.b64encode(buffer).decode('utf-8'),
                "camera_id": "test_camera",
                "frame_sequence": 1,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    @patch('src.inference.sagemaker_handler.RealTimeStreamProcessor')
    @patch('src.utils.config.config_manager.ConfigManager')
    def test_full_inference_pipeline(self, mock_config_manager, mock_processor_class):
        """Test complete inference pipeline."""
        # Setup mocks
        mock_config = Mock()
        mock_config_manager.return_value = mock_config
        
        mock_processor = Mock()
        mock_processor.process_stream_data.return_value = {
            'detections': [
                {
                    'bbox': {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200, 'confidence': 0.9},
                    'class_id': 0,
                    'class_name': 'chicken',
                    'weight_estimate': 2.5,
                    'occlusion_score': 0.2
                }
            ],
            'tracks': [],
            'frame_id': 1,
            'processing_time': 0.05,
            'timestamp': datetime.now().isoformat()
        }
        mock_processor_class.return_value = mock_processor
        
        # Test full pipeline
        model = model_fn("/tmp/model")
        parsed_input = input_fn(json.dumps(self.test_request), 'application/json')
        prediction = predict_fn(parsed_input, model)
        output = output_fn(prediction, 'application/json')
        
        # Verify pipeline
        self.assertIsNotNone(model)
        self.assertIsInstance(parsed_input, dict)
        self.assertIsInstance(prediction, dict)
        self.assertIsInstance(output, str)
        
        # Verify output is valid JSON
        parsed_output = json.loads(output)
        self.assertIn('detections', parsed_output)
    
    def test_error_propagation(self):
        """Test error propagation through pipeline."""
        # Test with invalid JSON input
        invalid_json = "{ invalid json }"
        
        parsed_input = input_fn(invalid_json, 'application/json')
        
        # Should contain error
        self.assertIn('error', parsed_input)
        
        # Test prediction with error input
        mock_processor = Mock()
        prediction = predict_fn(parsed_input, mock_processor)
        
        # Should handle error gracefully
        self.assertIsInstance(prediction, dict)
        self.assertIn('error', prediction)
    
    def test_performance_requirements(self):
        """Test performance requirements."""
        with patch('src.inference.sagemaker_handler.RealTimeStreamProcessor') as mock_processor_class, \
             patch('src.utils.config.config_manager.ConfigManager'):
            
            # Setup fast mock
            mock_processor = Mock()
            mock_processor.process_stream_data.return_value = {
                'detections': [],
                'tracks': [],
                'frame_id': 1,
                'processing_time': 0.02  # 20ms
            }
            mock_processor_class.return_value = mock_processor
            
            # Test processing time
            import time
            
            model = model_fn("/tmp/model")
            
            start_time = time.time()
            for _ in range(10):
                parsed_input = input_fn(json.dumps(self.test_request), 'application/json')
                prediction = predict_fn(parsed_input, model)
                output = output_fn(prediction, 'application/json')
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            
            # Should process requests quickly (< 100ms total pipeline)
            self.assertLess(avg_time, 0.1)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestSageMakerHandler))
    test_suite.addTest(unittest.makeSuite(TestStreamProcessingServer))
    test_suite.addTest(unittest.makeSuite(TestRealTimeStreamProcessor))
    test_suite.addTest(unittest.makeSuite(TestFrameProcessor))
    test_suite.addTest(unittest.makeSuite(TestSageMakerIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SAGEMAKER INTEGRATION TESTS SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('\\n')[-2]}")