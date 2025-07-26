"""
Comprehensive test suite for the chicken weight estimation system.
"""

import unittest
import numpy as np
import cv2
import time
import json
import base64
from unittest.mock import Mock, patch, MagicMock
import torch
from typing import List, Dict, Any

from ..src.models.detection.occlusion_robust_yolo import OcclusionRobustYOLODetector
from ..src.models.tracking.chicken_tracker import ChickenMultiObjectTracker
from ..src.models.weight_estimation.distance_adaptive_nn import DistanceAdaptiveWeightNN
from ..src.inference.stream_handler import RealTimeStreamProcessor
from ..src.utils.distance.perspective_distance import PerspectiveDistanceEstimator
from ..src.core.interfaces.detection import Detection
from ..src.core.interfaces.weight_estimation import WeightEstimate
from ..src.core.interfaces.camera import CameraParameters


class TestDetectionSystem(unittest.TestCase):
    """Test detection components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_image = np.ones((640, 640, 3), dtype=np.uint8) * 128
        # Add some test objects
        cv2.rectangle(self.test_image, (100, 100), (200, 180), (255, 255, 255), -1)
        cv2.rectangle(self.test_image, (300, 200), (400, 280), (200, 200, 200), -1)
    
    @patch('ultralytics.YOLO')
    def test_occlusion_detection(self, mock_yolo):
        """Test occlusion-robust detection."""
        # Mock YOLO model
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes = Mock()
        
        # Create mock detection boxes
        mock_boxes_data = torch.tensor([
            [100, 100, 200, 180, 0.9, 0],  # High confidence
            [150, 120, 250, 200, 0.6, 0],  # Overlapping, lower confidence
            [300, 200, 400, 280, 0.8, 0]   # Separate detection
        ])
        mock_result.boxes.data = Mock()
        mock_result.boxes.data.cpu.return_value.numpy.return_value = mock_boxes_data.numpy()
        
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model
        
        # Test detector
        detector = OcclusionRobustYOLODetector(
            model_path="mock_path",
            min_visibility_threshold=0.3
        )
        
        result = detector.detect_with_occlusion_handling(self.test_image)
        
        # Verify detections
        self.assertGreater(len(result.detections), 0)
        
        # Check occlusion levels are calculated
        for detection in result.detections:
            self.assertIsNotNone(detection.occlusion_level)
            self.assertGreaterEqual(detection.occlusion_level, 0.0)
            self.assertLessEqual(detection.occlusion_level, 1.0)
    
    def test_temporal_consistency(self):
        """Test temporal consistency across frames."""
        detector = OcclusionRobustYOLODetector("mock_path")
        
        # Create sequence of detections
        detections_sequence = []
        for i in range(5):
            # Simulate moving detection
            detection = Detection(
                bbox=[100 + i*5, 100, 200 + i*5, 180],
                confidence=0.8 + i*0.02,
                class_id=0,
                class_name="chicken"
            )
            detections_sequence.append([detection])
        
        # Test temporal smoothing
        smoothed_detections = []
        for i, detections in enumerate(detections_sequence):
            if i > 0:
                smoothed = detector.apply_temporal_smoothing(
                    detections, [detections_sequence[i-1]]
                )
                smoothed_detections.append(smoothed)
        
        # Verify smoothing effect
        self.assertGreater(len(smoothed_detections), 0)


class TestWeightEstimation(unittest.TestCase):
    """Test weight estimation components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.test_detection = Detection(
            bbox=[100, 100, 200, 180],
            confidence=0.8,
            class_id=0,
            class_name="chicken"
        )
    
    def test_distance_adaptive_estimation(self):
        """Test distance-adaptive weight estimation."""
        # Mock the neural network model
        with patch('torch.load'), patch('torch.device'):
            estimator = DistanceAdaptiveWeightNN()
            estimator.model = Mock()
            estimator.is_loaded = True
            
            # Mock model prediction
            mock_output = torch.tensor([[2.5]])  # 2.5 kg
            estimator.model.return_value = mock_output
            estimator.device = torch.device('cpu')
            
            # Test estimation
            weight_estimate = estimator.estimate_weight_with_distance(
                self.test_frame, self.test_detection, distance=3.0, occlusion_level=0.2
            )
            
            self.assertIsInstance(weight_estimate, WeightEstimate)
            self.assertGreater(weight_estimate.value, 0)
            self.assertLess(weight_estimate.value, 10)  # Reasonable range
            self.assertTrue(weight_estimate.distance_compensated)
    
    def test_feature_extraction(self):
        """Test feature extraction for weight estimation."""
        from ..src.models.weight_estimation.feature_extractor import ChickenFeatureExtractor
        
        extractor = ChickenFeatureExtractor()
        features = extractor.extract_features(
            self.test_frame, self.test_detection, distance=3.0
        )
        
        self.assertEqual(len(features), 25)  # Expected feature vector size
        self.assertTrue(np.all(np.isfinite(features)))  # No NaN or inf values
    
    def test_age_classification(self):
        """Test chicken age classification."""
        from ..src.models.weight_estimation.age_classifier import ChickenAgeClassifier
        
        classifier = ChickenAgeClassifier()
        
        # Test with different sized features (simulating different ages)
        small_features = np.array([50, 40, 2000] + [0.5] * 22)  # Small chicken
        large_features = np.array([150, 120, 18000] + [0.8] * 22)  # Large chicken
        
        small_age = classifier.classify_age_from_features(small_features)
        large_age = classifier.classify_age_from_features(large_features)
        
        # Verify age classification makes sense
        self.assertIn(small_age.name, ['DAY_OLD', 'WEEK_1', 'WEEK_2'])
        self.assertIn(large_age.name, ['WEEK_5', 'WEEK_6', 'ADULT'])


class TestTracking(unittest.TestCase):
    """Test tracking components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tracker = ChickenMultiObjectTracker()
        self.test_detections = [
            Detection(
                bbox=[100, 100, 200, 180],
                confidence=0.8,
                class_id=0,
                class_name="chicken"
            ),
            Detection(
                bbox=[300, 200, 400, 280],
                confidence=0.7,
                class_id=0,
                class_name="chicken"
            )
        ]
    
    def test_track_initialization(self):
        """Test track initialization."""
        result = self.tracker.update_tracks(self.test_detections, frame_id="1")
        
        self.assertEqual(len(result.new_tracks), 2)
        self.assertEqual(len(result.tracked_chickens), 2)
        
        # Verify track IDs are assigned
        for track in result.tracked_chickens:
            self.assertIsNotNone(track.chicken_id)
    
    def test_track_continuity(self):
        """Test track continuity across frames."""
        # First frame
        result1 = self.tracker.update_tracks(self.test_detections, frame_id="1")
        
        # Second frame with slightly moved detections
        moved_detections = [
            Detection(
                bbox=[105, 105, 205, 185],  # Moved slightly
                confidence=0.8,
                class_id=0,
                class_name="chicken"
            ),
            Detection(
                bbox=[305, 205, 405, 285],  # Moved slightly
                confidence=0.7,
                class_id=0,
                class_name="chicken"
            )
        ]
        
        result2 = self.tracker.update_tracks(moved_detections, frame_id="2")
        
        # Should maintain same tracks
        self.assertEqual(len(result2.new_tracks), 0)  # No new tracks
        self.assertEqual(len(result2.tracked_chickens), 2)  # Same number of tracks
    
    def test_occlusion_handling(self):
        """Test tracking through occlusions."""
        # Create detection with high occlusion
        occluded_detection = Detection(
            bbox=[100, 100, 200, 180],
            confidence=0.6,
            class_id=0,
            class_name="chicken",
            occlusion_level=0.8
        )
        
        result = self.tracker.update_tracks([occluded_detection], frame_id="1")
        tracks = self.tracker.handle_occlusion_tracking(result.tracked_chickens)
        
        # Verify occlusion handling
        for track in tracks:
            if track.current_detection.occlusion_level > 0.7:
                self.assertEqual(track.tracking_status, "occluded")


class TestDistanceEstimation(unittest.TestCase):
    """Test distance estimation components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.camera_params = CameraParameters(
            focal_length=1000.0,
            sensor_width=6.0,
            sensor_height=4.5,
            image_width=640,
            image_height=480,
            camera_height=3.0,
            known_object_width=25.0
        )
        self.estimator = PerspectiveDistanceEstimator(self.camera_params)
    
    def test_distance_estimation(self):
        """Test distance estimation accuracy."""
        # Test with different sized bounding boxes
        large_bbox = [100, 100, 300, 250]  # Close chicken (large bbox)
        small_bbox = [200, 200, 250, 230]  # Far chicken (small bbox)
        
        large_distance = self.estimator.estimate_distance_to_chicken(
            large_bbox, (480, 640)
        )
        small_distance = self.estimator.estimate_distance_to_chicken(
            small_bbox, (480, 640)
        )
        
        # Larger bbox should indicate closer distance
        self.assertLess(large_distance, small_distance)
        self.assertGreater(large_distance, 0.5)  # Reasonable minimum
        self.assertLess(small_distance, 15.0)    # Reasonable maximum
    
    def test_distance_validation(self):
        """Test distance estimate validation."""
        bbox = [100, 100, 200, 180]
        distance = 5.0
        
        is_valid = self.estimator.validate_distance_estimate(distance, 100.0)
        self.assertIsInstance(is_valid, bool)
    
    def test_confidence_calculation(self):
        """Test distance estimation confidence."""
        bbox = [100, 100, 200, 180]
        confidence = self.estimator.estimate_distance_confidence(bbox, (480, 640))
        
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)


class TestStreamProcessing(unittest.TestCase):
    """Test stream processing components."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock config manager
        self.mock_config = Mock()
        self.mock_config.load_config.return_value = {
            'yolo': {'model_path': 'mock.pt', 'confidence_threshold': 0.5},
            'weight_estimation': {'model_path': 'mock.pt'},
            'tracking': {'max_disappeared': 30}
        }
    
    @patch('chicken_weight_sagemaker.src.inference.stream_handler.OcclusionRobustYOLODetector')
    @patch('chicken_weight_sagemaker.src.inference.stream_handler.DistanceAdaptiveWeightNN')
    @patch('chicken_weight_sagemaker.src.inference.stream_handler.ChickenMultiObjectTracker')
    def test_frame_processing(self, mock_tracker, mock_weight, mock_detector):
        """Test complete frame processing pipeline."""
        # Setup mocks
        mock_detector.return_value.detect_with_occlusion_handling.return_value.detections = [
            Detection(bbox=[100, 100, 200, 180], confidence=0.8, class_id=0, class_name="chicken")
        ]
        
        mock_tracker.return_value.update_tracks.return_value.tracked_chickens = []
        mock_weight.return_value.estimate_weight_with_distance.return_value = WeightEstimate(
            value=2.5, confidence=0.8
        )
        
        # Create processor
        processor = RealTimeStreamProcessor(self.mock_config)
        
        # Create test frame
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', test_image)
        frame_data = base64.b64encode(buffer).decode('utf-8')
        
        # Process frame
        result = processor.process_frame(frame_data, "test_camera", 1)
        
        # Verify result structure
        self.assertIn('camera_id', result)
        self.assertIn('detections', result)
        self.assertIn('processing_time_ms', result)
        self.assertEqual(result['camera_id'], 'test_camera')
    
    def test_performance_monitoring(self):
        """Test performance monitoring capabilities."""
        from ..src.inference.frame_processor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring(interval=0.1)
        
        # Add some inference times
        for i in range(10):
            monitor.add_inference_time(0.05 + i * 0.01)
        
        time.sleep(0.2)  # Let monitor collect some data
        
        stats = monitor.get_performance_stats()
        monitor.stop_monitoring()
        
        self.assertIn('inference_times', stats)
        self.assertGreater(stats['inference_times']['average'], 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_end_to_end_processing(self):
        """Test complete end-to-end processing pipeline."""
        # This would test the entire pipeline from image input to final results
        # Using mock components to avoid dependency on actual models
        pass
    
    def test_sagemaker_compatibility(self):
        """Test SageMaker handler compatibility."""
        from ..src.inference.sagemaker_handler import SageMakerInferenceHandler
        
        handler = SageMakerInferenceHandler()
        
        # Test input validation
        valid_input = {
            "stream_data": {
                "frame": "base64_encoded_data",
                "camera_id": "test_camera"
            }
        }
        
        is_valid = handler._validate_input(valid_input)
        self.assertTrue(is_valid)
        
        # Test invalid input
        invalid_input = {"invalid": "data"}
        is_valid = handler._validate_input(invalid_input)
        self.assertFalse(is_valid)


def run_performance_tests():
    """Run performance-specific tests."""
    print("Running performance tests...")
    
    # Test processing speed
    start_time = time.time()
    test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    
    # Simulate processing
    for i in range(100):
        # Mock processing operations
        processed = cv2.resize(test_image, (320, 240))
        _ = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    
    end_time = time.time()
    processing_time = (end_time - start_time) / 100
    
    print(f"Average processing time per frame: {processing_time*1000:.2f}ms")
    print(f"Theoretical max FPS: {1/processing_time:.1f}")
    
    # Performance assertions
    assert processing_time < 0.1, "Processing too slow for real-time"
    assert 1/processing_time > 10, "FPS too low for practical use"


if __name__ == '__main__':
    # Run unit tests
    unittest.main(verbosity=2, exit=False)
    
    # Run performance tests
    run_performance_tests()
    
    print("\nAll tests completed!")

class TestPerformanceRequirements(unittest.TestCase):
    """Test performance requirements for real-time processing."""
    
    def test_processing_speed(self):
        """Test that processing meets real-time requirements."""
        # Create test processor with mocks
        processor = self._create_mock_processor()
        
        # Generate test frames
        test_frames = []
        for i in range(30):
            frame = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
            test_frames.append(frame)
        
        # Measure processing times
        processing_times = []
        for i, frame in enumerate(test_frames):
            start_time = time.time()
            
            # Encode frame as would be done in real scenario
            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = base64.b64encode(buffer).decode('utf-8')
            
            # Process frame
            result = processor.process_frame(frame_data, "test_camera", i)
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
        
        # Performance assertions
        avg_time = np.mean(processing_times)
        max_time = np.max(processing_times)
        
        print(f"Average processing time: {avg_time*1000:.2f}ms")
        print(f"Maximum processing time: {max_time*1000:.2f}ms")
        print(f"Theoretical FPS: {1/avg_time:.1f}")
        
        # Requirements: <100ms processing time for real-time
        self.assertLess(avg_time, 0.1, "Average processing time exceeds 100ms")
        self.assertLess(max_time, 0.2, "Maximum processing time exceeds 200ms")
        self.assertGreater(1/avg_time, 10, "FPS too low for practical use")
    
    def test_memory_usage(self):
        """Test memory usage during processing."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        processor = self._create_mock_processor()
        
        # Process multiple frames to test memory leaks
        for i in range(100):
            frame = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = base64.b64encode(buffer).decode('utf-8')
            
            result = processor.process_frame(frame_data, "test_camera", i)
            
            # Force garbage collection every 10 frames
            if i % 10 == 0:
                gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Initial memory: {initial_memory:.2f}MB")
        print(f"Final memory: {final_memory:.2f}MB")
        print(f"Memory increase: {memory_increase:.2f}MB")
        
        # Should not have significant memory leaks
        self.assertLess(memory_increase, 100, "Memory usage increased too much")
    
    def test_concurrent_processing(self):
        """Test concurrent frame processing."""
        import threading
        import queue
        
        processor = self._create_mock_processor()
        results_queue = queue.Queue()
        
        def process_frames(thread_id, num_frames):
            """Process frames in separate thread."""
            thread_results = []
            for i in range(num_frames):
                frame = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
                _, buffer = cv2.imencode('.jpg', frame)
                frame_data = base64.b64encode(buffer).decode('utf-8')
                
                start_time = time.time()
                result = processor.process_frame(frame_data, f"camera_{thread_id}", i)
                processing_time = time.time() - start_time
                
                thread_results.append({
                    'thread_id': thread_id,
                    'frame_id': i,
                    'processing_time': processing_time,
                    'success': result.get('status') != 'error'
                })
            
            results_queue.put(thread_results)
        
        # Start multiple threads
        threads = []
        num_threads = 3
        frames_per_thread = 20
        
        for thread_id in range(num_threads):
            thread = threading.Thread(
                target=process_frames,
                args=(thread_id, frames_per_thread)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Collect results
        all_results = []
        while not results_queue.empty():
            thread_results = results_queue.get()
            all_results.extend(thread_results)
        
        # Analyze results
        success_rate = sum(1 for r in all_results if r['success']) / len(all_results)
        avg_processing_time = np.mean([r['processing_time'] for r in all_results])
        
        print(f"Concurrent processing success rate: {success_rate*100:.1f}%")
        print(f"Average processing time under load: {avg_processing_time*1000:.2f}ms")
        
        # Performance requirements under concurrent load
        self.assertGreater(success_rate, 0.95, "Success rate too low under concurrent load")
        self.assertLess(avg_processing_time, 0.15, "Processing time too high under load")
    
    def _create_mock_processor(self):
        """Create a mock processor for testing."""
        from ..src.inference.stream_handler import RealTimeStreamProcessor
        
        # Mock config manager
        mock_config = Mock()
        mock_config.load_config.side_effect = lambda name: {
            'camera_config': {
                'focal_length': 1000.0,
                'sensor_width': 6.0,
                'sensor_height': 4.5,
                'image_width': 640,
                'image_height': 480,
                'camera_height': 3.0,
                'known_object_width': 25.0
            },
            'model_config': {
                'yolo': {
                    'model_path': 'mock.pt',
                    'confidence_threshold': 0.5,
                    'min_visibility_threshold': 0.3
                },
                'weight_estimation': {'model_path': 'mock.pt'},
                'tracking': {'max_disappeared': 30, 'max_distance': 100.0}
            }
        }.get(name, {})
        
        # Create processor
        processor = RealTimeStreamProcessor(mock_config)
        
        # Replace with mocks
        processor.detector = self._create_mock_detector()
        processor.weight_estimator = self._create_mock_weight_estimator()
        processor.tracker = self._create_mock_tracker()
        
        return processor
    
    def _create_mock_detector(self):
        """Create mock detector."""
        mock_detector = Mock()
        
        def mock_detect(frame):
            # Simulate processing time
            time.sleep(0.01)
            
            # Return mock detections
            detections = []
            for i in range(np.random.randint(1, 4)):
                detection = Detection(
                    bbox=[
                        np.random.randint(0, 500),
                        np.random.randint(0, 400),
                        np.random.randint(100, 600),
                        np.random.randint(100, 480)
                    ],
                    confidence=np.random.uniform(0.6, 0.95),
                    class_id=0,
                    class_name="chicken",
                    occlusion_level=np.random.uniform(0.0, 0.6)
                )
                detection.distance_estimate = np.random.uniform(2.0, 8.0)
                detections.append(detection)
            
            result = Mock()
            result.detections = detections
            return result
        
        mock_detector.detect_with_occlusion_handling = mock_detect
        return mock_detector
    
    def _create_mock_weight_estimator(self):
        """Create mock weight estimator."""
        mock_estimator = Mock()
        mock_estimator.is_loaded = True
        
        def mock_estimate(frame, detection, distance, occlusion_level):
            # Simulate processing time
            time.sleep(0.005)
            
            # Generate realistic weight
            bbox = detection.bbox
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            weight = 1.5 + (area / 10000.0) + np.random.normal(0, 0.2)
            weight = max(0.5, min(4.0, weight))
            
            return WeightEstimate(
                value=weight,
                confidence=np.random.uniform(0.7, 0.9),
                error_range=f"±{weight*0.2:.2f}kg",
                distance_compensated=True,
                method="mock_nn"
            )
        
        mock_estimator.estimate_weight_with_distance = mock_estimate
        mock_estimator.get_model_info.return_value = {"status": "mock"}
        return mock_estimator
    
    def _create_mock_tracker(self):
        """Create mock tracker."""
        mock_tracker = Mock()
        
        def mock_update(detections, frame_id=None):
            # Simulate tracking
            tracked_chickens = []
            for i, detection in enumerate(detections):
                from ..src.core.interfaces.tracking import TrackedChicken
                tracked_chicken = TrackedChicken(
                    chicken_id=f"chicken_{i}",
                    current_detection=detection,
                    tracking_status="active",
                    confidence=detection.confidence,
                    frames_tracked=1,
                    frames_lost=0,
                    last_seen_timestamp=time.time()
                )
                tracked_chickens.append(tracked_chicken)
            
            from ..src.core.interfaces.tracking import TrackingResult
            return TrackingResult(
                tracked_chickens=tracked_chickens,
                new_tracks=tracked_chickens,
                lost_tracks=[],
                processing_time_ms=5.0,
                total_active_tracks=len(tracked_chickens)
            )
        
        mock_tracker.update_tracks = mock_update
        mock_tracker.get_active_tracks.return_value = []
        mock_tracker.get_tracking_statistics.return_value = {
            'total_active_tracks': 0,
            'total_tracks_created': 0
        }
        return mock_tracker


def run_all_tests():
    """Run all test suites."""
    print("Running Chicken Weight Estimation System Test Suite")
    print("=" * 60)
    
    # Test suites to run
    test_suites = [
        TestDetectionSystem,
        TestWeightEstimation,
        TestTracking,
        TestDistanceEstimation,
        TestStreamProcessing,
        TestIntegration,
        TestPerformanceRequirements
    ]
    
    total_tests = 0
    total_failures = 0
    
    for test_suite in test_suites:
        print(f"\nRunning {test_suite.__name__}...")
        suite = unittest.TestLoader().loadTestsFromTestCase(test_suite)
        runner = unittest.TextTestRunner(verbosity=1)
        result = runner.run(suite)
        
        total_tests += result.testsRun
        total_failures += len(result.failures) + len(result.errors)
    
    print("\n" + "=" * 60)
    print(f"Total tests run: {total_tests}")
    print(f"Total failures: {total_failures}")
    print(f"Success rate: {((total_tests - total_failures) / total_tests * 100):.1f}%")
    
    if total_failures == 0:
        print("🎉 All tests passed!")
    else:
        print(f"❌ {total_failures} tests failed")
    
    return total_failures == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)