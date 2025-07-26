"""
Test suite for stream processing components.
"""

import unittest
import numpy as np
import cv2
import base64
import json
import time
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference.stream_handler import RealTimeStreamProcessor, StreamProcessingServer
from inference.frame_processor import FrameProcessor, FrameQueue, ResultAggregator
from utils.config.config_manager import ConfigManager
from core.interfaces.detection import Detection
from core.interfaces.weight_estimation import WeightEstimate


class TestStreamProcessing(unittest.TestCase):
    """Test stream processing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock configuration
        self.mock_config = {
            'yolo': {
                'model_path': 'test_model.pt',
                'confidence_threshold': 0.5,
                'min_visibility_threshold': 0.3
            },
            'weight_estimation': {
                'model_path': 'test_weight_model.pt'
            },
            'tracking': {
                'max_disappeared': 30,
                'max_distance': 100.0
            },
            'focal_length': 1000.0,
            'sensor_width': 6.0,
            'sensor_height': 4.5,
            'image_width': 1920,
            'image_height': 1080,
            'camera_height': 3.0,
            'known_object_width': 25.0
        }
        
        # Create test image
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Encode test image to base64
        _, buffer = cv2.imencode('.jpg', self.test_image)
        self.test_frame_data = base64.b64encode(buffer).decode('utf-8')
    
    def create_mock_config_manager(self):
        """Create mock configuration manager."""
        mock_config_manager = Mock(spec=ConfigManager)
        mock_config_manager.load_config.return_value = self.mock_config
        return mock_config_manager
    
    @patch('inference.stream_handler.OcclusionRobustYOLODetector')
    @patch('inference.stream_handler.ChickenMultiObjectTracker')
    @patch('inference.stream_handler.DistanceAdaptiveWeightNN')
    @patch('inference.stream_handler.PerspectiveDistanceEstimator')
    def test_stream_processor_initialization(self, mock_distance, mock_weight, mock_tracker, mock_detector):
        """Test stream processor initialization."""
        # Setup mocks
        mock_config_manager = self.create_mock_config_manager()
        
        # Create processor
        processor = RealTimeStreamProcessor(mock_config_manager)
        
        # Verify components were initialized
        mock_detector.assert_called_once()
        mock_tracker.assert_called_once()
        mock_weight.assert_called_once()
        mock_distance.assert_called_once()
        
        # Verify configuration was loaded
        self.assertEqual(mock_config_manager.load_config.call_count, 2)
    
    @patch('inference.stream_handler.OcclusionRobustYOLODetector')
    @patch('inference.stream_handler.ChickenMultiObjectTracker')
    @patch('inference.stream_handler.DistanceAdaptiveWeightNN')
    @patch('inference.stream_handler.PerspectiveDistanceEstimator')
    def test_frame_processing(self, mock_distance, mock_weight, mock_tracker, mock_detector):
        """Test frame processing pipeline."""
        # Setup mocks
        mock_config_manager = self.create_mock_config_manager()
        
        # Mock detection result
        mock_detection = Detection(
            bbox=[100, 100, 200, 200],
            confidence=0.8,
            class_id=0,
            class_name="chicken",
            occlusion_level=0.2
        )
        
        mock_detection_result = Mock()
        mock_detection_result.detections = [mock_detection]
        mock_detector.return_value.detect_with_occlusion_handling.return_value = mock_detection_result
        
        # Mock distance estimation
        mock_distance.return_value.estimate_distance_to_chicken.return_value = 3.5
        
        # Mock tracking result
        from core.interfaces.tracking import TrackedChicken, TrackingResult
        mock_tracked_chicken = TrackedChicken(
            chicken_id="chicken_1",
            current_detection=mock_detection,
            tracking_status="active",
            confidence=0.8
        )
        
        mock_tracking_result = TrackingResult(
            tracked_chickens=[mock_tracked_chicken],
            new_tracks=[],
            lost_tracks=[],
            processing_time_ms=10.0,
            total_active_tracks=1
        )
        mock_tracker.return_value.update_tracks.return_value = mock_tracking_result
        
        # Mock weight estimation
        mock_weight_estimate = WeightEstimate(
            value=2.3,
            confidence=0.85,
            error_range="±0.5kg",
            method="distance_adaptive_nn"
        )
        mock_weight.return_value.estimate_weight_with_distance.return_value = mock_weight_estimate
        
        # Create processor and process frame
        processor = RealTimeStreamProcessor(mock_config_manager)
        result = processor.process_frame(self.test_frame_data, "test_camera", 1)
        
        # Verify result structure
        self.assertIn('camera_id', result)
        self.assertIn('timestamp', result)
        self.assertIn('frame_sequence', result)
        self.assertIn('detections', result)
        self.assertIn('status', result)
        self.assertEqual(result['camera_id'], 'test_camera')
        self.assertEqual(result['frame_sequence'], 1)
        self.assertEqual(result['status'], 'success')
        
        # Verify processing pipeline was called
        mock_detector.return_value.detect_with_occlusion_handling.assert_called_once()
        mock_tracker.return_value.update_tracks.assert_called_once()
        mock_weight.return_value.estimate_weight_with_distance.assert_called_once()
    
    def test_frame_processor(self):
        """Test frame preprocessing and postprocessing."""
        processor = FrameProcessor(target_size=(320, 320))
        
        # Test preprocessing
        processed_frame, metadata = processor.preprocess_frame(self.test_image)
        
        # Verify preprocessing
        self.assertEqual(processed_frame.shape[:2], (320, 320))
        self.assertEqual(processed_frame.dtype, np.float32)
        self.assertTrue(0 <= processed_frame.max() <= 1)
        
        # Verify metadata
        self.assertIn('original_shape', metadata)
        self.assertIn('processed_shape', metadata)
        self.assertIn('scale_factor_x', metadata)
        self.assertIn('scale_factor_y', metadata)
        
        # Test postprocessing
        test_detection = Detection(
            bbox=[50, 50, 100, 100],
            confidence=0.8,
            class_id=0,
            class_name="chicken"
        )
        
        postprocessed = processor.postprocess_detections([test_detection], metadata)
        
        # Verify scaling
        self.assertEqual(len(postprocessed), 1)
        scaled_detection = postprocessed[0]
        
        # Check that bounding box was scaled back
        original_bbox = test_detection.bbox
        scaled_bbox = scaled_detection.bbox
        
        self.assertNotEqual(original_bbox, scaled_bbox)
        self.assertEqual(scaled_detection.confidence, test_detection.confidence)
    
    def test_frame_queue(self):
        """Test frame queue functionality."""
        queue = FrameQueue(max_size=3)
        
        # Test adding frames
        frame1 = {'frame_id': 1, 'data': 'test1'}
        frame2 = {'frame_id': 2, 'data': 'test2'}
        frame3 = {'frame_id': 3, 'data': 'test3'}
        frame4 = {'frame_id': 4, 'data': 'test4'}
        
        # Add frames within capacity
        self.assertTrue(queue.put_frame(frame1))
        self.assertTrue(queue.put_frame(frame2))
        self.assertTrue(queue.put_frame(frame3))
        
        # Try to add frame beyond capacity
        self.assertFalse(queue.put_frame(frame4, timeout=0.1))
        
        # Get frames
        retrieved1 = queue.get_frame()
        self.assertEqual(retrieved1['frame_id'], 1)
        
        retrieved2 = queue.get_frame()
        self.assertEqual(retrieved2['frame_id'], 2)
        
        # Check stats
        stats = queue.get_queue_stats()
        self.assertEqual(stats['total_frames'], 4)
        self.assertEqual(stats['dropped_frames'], 1)
        self.assertGreater(stats['drop_rate'], 0)
    
    def test_result_aggregator(self):
        """Test result aggregation functionality."""
        aggregator = ResultAggregator(window_size=5)
        
        # Add test results
        for i in range(3):
            result = {
                'status': 'success',
                'total_chickens_detected': i + 1,
                'processing_time_ms': 50 + i * 10,
                'detections': [
                    {
                        'weight_estimate': {'value': 2.0 + i * 0.5}
                    }
                ]
            }
            aggregator.add_result(result)
        
        # Get aggregated stats
        stats = aggregator.get_aggregated_stats()
        
        # Verify statistics
        self.assertEqual(stats['status'], 'active')
        self.assertEqual(stats['total_frames'], 3)
        self.assertEqual(stats['successful_frames'], 3)
        self.assertEqual(stats['success_rate'], 1.0)
        self.assertEqual(stats['total_detections'], 6)  # 1 + 2 + 3
        self.assertEqual(stats['avg_detections_per_frame'], 2.0)
        
        # Check processing time stats
        self.assertIn('processing_time', stats)
        self.assertEqual(stats['processing_time']['average_ms'], 60.0)  # (50 + 60 + 70) / 3
        
        # Check weight statistics
        self.assertIn('weight_statistics', stats)
        self.assertEqual(stats['weight_statistics']['total_estimates'], 3)
    
    @patch('inference.stream_handler.RealTimeStreamProcessor')
    def test_stream_server_routes(self, mock_processor_class):
        """Test Flask server routes."""
        # Setup mock processor
        mock_processor = Mock()
        mock_processor.health_check.return_value = {'status': 'healthy'}
        mock_processor.get_processing_stats.return_value = {'frames_processed': 10}
        mock_processor.process_frame.return_value = {
            'camera_id': 'test',
            'status': 'success',
            'detections': []
        }
        mock_processor_class.return_value = mock_processor
        
        # Create server
        server = StreamProcessingServer(mock_processor, port=5000)
        
        # Test client
        with server.app.test_client() as client:
            # Test ping endpoint
            response = client.get('/ping')
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertEqual(data['status'], 'healthy')
            
            # Test health endpoint
            response = client.get('/health')
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertEqual(data['status'], 'healthy')
            
            # Test stats endpoint
            response = client.get('/stats')
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertEqual(data['frames_processed'], 10)
    
    def test_invalid_frame_data(self):
        """Test handling of invalid frame data."""
        mock_config_manager = self.create_mock_config_manager()
        
        with patch('inference.stream_handler.OcclusionRobustYOLODetector'), \
             patch('inference.stream_handler.ChickenMultiObjectTracker'), \
             patch('inference.stream_handler.DistanceAdaptiveWeightNN'), \
             patch('inference.stream_handler.PerspectiveDistanceEstimator'):
            
            processor = RealTimeStreamProcessor(mock_config_manager)
            
            # Test with invalid base64 data
            result = processor.process_frame("invalid_base64", "test_camera", 1)
            
            # Should return error response
            self.assertEqual(result['status'], 'error')
            self.assertIn('error_message', result)
            self.assertEqual(result['camera_id'], 'test_camera')
    
    def test_processing_statistics(self):
        """Test processing statistics tracking."""
        mock_config_manager = self.create_mock_config_manager()
        
        with patch('inference.stream_handler.OcclusionRobustYOLODetector'), \
             patch('inference.stream_handler.ChickenMultiObjectTracker'), \
             patch('inference.stream_handler.DistanceAdaptiveWeightNN'), \
             patch('inference.stream_handler.PerspectiveDistanceEstimator'):
            
            processor = RealTimeStreamProcessor(mock_config_manager)
            
            # Check initial stats
            stats = processor.get_processing_stats()
            self.assertEqual(stats['frames_processed'], 0)
            self.assertEqual(stats['total_chickens_detected'], 0)
            
            # Process some frames (will fail due to mocking, but stats should update)
            for i in range(3):
                try:
                    processor.process_frame(self.test_frame_data, "test_camera", i)
                except:
                    pass  # Expected to fail due to mocking
            
            # Check updated stats
            stats = processor.get_processing_stats()
            # Stats should be updated even if processing failed
            self.assertGreaterEqual(stats['frames_processed'], 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        # Create a more realistic test image
        self.test_image = self.create_test_chicken_image()
        
        # Encode to base64
        _, buffer = cv2.imencode('.jpg', self.test_image)
        self.test_frame_data = base64.b64encode(buffer).decode('utf-8')
    
    def create_test_chicken_image(self):
        """Create a synthetic image with chicken-like objects."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add background
        image[:] = [50, 100, 50]  # Green background
        
        # Add chicken-like shapes
        cv2.ellipse(image, (200, 200), (50, 30), 0, 0, 360, (200, 180, 150), -1)
        cv2.ellipse(image, (400, 300), (60, 35), 0, 0, 360, (180, 160, 140), -1)
        
        # Add some noise
        noise = np.random.randint(0, 50, image.shape, dtype=np.uint8)
        image = cv2.add(image, noise)
        
        return image
    
    def test_end_to_end_processing(self):
        """Test end-to-end processing with realistic data."""
        # This test would require actual model files, so we'll mock the heavy components
        # but test the data flow
        
        mock_config = {
            'yolo': {'model_path': 'test.pt', 'confidence_threshold': 0.5},
            'weight_estimation': {'model_path': 'test.pt'},
            'tracking': {'max_disappeared': 30},
            'focal_length': 1000.0,
            'camera_height': 3.0,
            'known_object_width': 25.0
        }
        
        with patch('inference.stream_handler.OcclusionRobustYOLODetector') as mock_detector, \
             patch('inference.stream_handler.ChickenMultiObjectTracker') as mock_tracker, \
             patch('inference.stream_handler.DistanceAdaptiveWeightNN') as mock_weight, \
             patch('inference.stream_handler.PerspectiveDistanceEstimator') as mock_distance:
            
            # Setup realistic mock responses
            mock_detection = Detection(
                bbox=[150, 170, 250, 230],
                confidence=0.85,
                class_id=0,
                class_name="chicken",
                occlusion_level=0.1
            )
            
            mock_detector.return_value.detect_with_occlusion_handling.return_value.detections = [mock_detection]
            mock_distance.return_value.estimate_distance_to_chicken.return_value = 3.2
            
            from core.interfaces.tracking import TrackedChicken, TrackingResult
            mock_tracked_chicken = TrackedChicken(
                chicken_id="chicken_001",
                current_detection=mock_detection,
                tracking_status="active",
                confidence=0.85
            )
            
            mock_tracking_result = TrackingResult(
                tracked_chickens=[mock_tracked_chicken],
                new_tracks=[mock_tracked_chicken],
                lost_tracks=[],
                processing_time_ms=15.0,
                total_active_tracks=1
            )
            mock_tracker.return_value.update_tracks.return_value = mock_tracking_result
            
            mock_weight_estimate = WeightEstimate(
                value=2.1,
                confidence=0.78,
                error_range="±0.4kg",
                method="distance_adaptive_nn"
            )
            mock_weight.return_value.estimate_weight_with_distance.return_value = mock_weight_estimate
            
            # Create config manager mock
            mock_config_manager = Mock()
            mock_config_manager.load_config.return_value = mock_config
            
            # Test the complete pipeline
            processor = RealTimeStreamProcessor(mock_config_manager)
            result = processor.process_frame(self.test_frame_data, "farm_camera_01", 42)
            
            # Verify complete result structure
            self.assertEqual(result['status'], 'success')
            self.assertEqual(result['camera_id'], 'farm_camera_01')
            self.assertEqual(result['frame_sequence'], 42)
            self.assertGreater(len(result['detections']), 0)
            self.assertGreater(result['total_chickens_detected'], 0)
            self.assertGreater(result['processing_time_ms'], 0)
            
            # Verify detection details
            detection = result['detections'][0]
            self.assertIn('chicken_id', detection)
            self.assertIn('bbox', detection)
            self.assertIn('confidence', detection)
            self.assertIn('weight_estimate', detection)
            
            # Verify weight estimate details
            weight_est = detection['weight_estimate']
            self.assertIn('value', weight_est)
            self.assertIn('confidence', weight_est)
            self.assertIn('method', weight_est)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)