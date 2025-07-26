#!/usr/bin/env python3
"""
Comprehensive tests for detection components.
"""

import unittest
import numpy as np
import cv2
import torch
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path

# Import components to test
from src.models.detection.yolo_detector import YOLODetector
from src.models.detection.occlusion_robust_yolo import OcclusionRobustYOLO
from src.models.detection.temporal_consistency import TemporalConsistencyFilter
from src.core.interfaces.detection import Detection, BoundingBox


class TestYOLODetector(unittest.TestCase):
    """Test YOLO detection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_image = np.ones((640, 640, 3), dtype=np.uint8) * 128
        # Add some test objects
        cv2.rectangle(self.test_image, (100, 100), (200, 200), (255, 255, 255), -1)
        cv2.rectangle(self.test_image, (300, 300), (400, 400), (200, 200, 200), -1)
        
        # Create mock model path
        self.mock_model_path = "test_model.pt"
    
    @patch('ultralytics.YOLO')
    def test_detector_initialization(self, mock_yolo):
        """Test detector initialization."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        detector = YOLODetector(self.mock_model_path)
        
        self.assertIsNotNone(detector)
        mock_yolo.assert_called_once_with(self.mock_model_path)
    
    @patch('ultralytics.YOLO')
    def test_basic_detection(self, mock_yolo):
        """Test basic detection functionality."""
        # Setup mock
        mock_model = Mock()
        mock_result = Mock()
        
        # Create mock detection boxes
        mock_boxes = Mock()
        mock_boxes.xyxy = torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]])
        mock_boxes.conf = torch.tensor([0.9, 0.8])
        mock_boxes.cls = torch.tensor([0, 0])  # Both are chickens
        
        mock_result.boxes = mock_boxes
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model
        
        # Test detection
        detector = YOLODetector(self.mock_model_path)
        detections = detector.detect(self.test_image)
        
        # Verify results
        self.assertEqual(len(detections), 2)
        self.assertIsInstance(detections[0], Detection)
        self.assertEqual(detections[0].class_id, 0)
        self.assertGreaterEqual(detections[0].bbox.confidence, 0.8)
    
    @patch('ultralytics.YOLO')
    def test_confidence_filtering(self, mock_yolo):
        """Test confidence threshold filtering."""
        mock_model = Mock()
        mock_result = Mock()
        
        # Create detections with varying confidence
        mock_boxes = Mock()
        mock_boxes.xyxy = torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]])
        mock_boxes.conf = torch.tensor([0.9, 0.3])  # One high, one low confidence
        mock_boxes.cls = torch.tensor([0, 0])
        
        mock_result.boxes = mock_boxes
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model
        
        # Test with confidence threshold
        detector = YOLODetector(self.mock_model_path, confidence_threshold=0.5)
        detections = detector.detect(self.test_image)
        
        # Should only return high confidence detection
        self.assertEqual(len(detections), 1)
        self.assertGreaterEqual(detections[0].bbox.confidence, 0.5)
    
    def test_empty_detection(self):
        """Test handling of empty detection results."""
        with patch('ultralytics.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_result = Mock()
            mock_result.boxes = None  # No detections
            mock_model.return_value = [mock_result]
            mock_yolo.return_value = mock_model
            
            detector = YOLODetector(self.mock_model_path)
            detections = detector.detect(self.test_image)
            
            self.assertEqual(len(detections), 0)


class TestOcclusionRobustYOLO(unittest.TestCase):
    """Test occlusion-robust YOLO detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_image = np.ones((640, 640, 3), dtype=np.uint8) * 128
        self.mock_model_path = "test_model.pt"
    
    @patch('ultralytics.YOLO')
    def test_multi_scale_detection(self, mock_yolo):
        """Test multi-scale detection capability."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        # Mock different scale results
        def mock_predict(image, imgsz=640):
            mock_result = Mock()
            mock_boxes = Mock()
            
            # Different detections for different scales
            if imgsz == 640:
                mock_boxes.xyxy = torch.tensor([[100, 100, 200, 200]])
                mock_boxes.conf = torch.tensor([0.9])
                mock_boxes.cls = torch.tensor([0])
            elif imgsz == 1280:
                mock_boxes.xyxy = torch.tensor([[300, 300, 400, 400]])
                mock_boxes.conf = torch.tensor([0.8])
                mock_boxes.cls = torch.tensor([0])
            else:
                mock_boxes.xyxy = torch.tensor([])
                mock_boxes.conf = torch.tensor([])
                mock_boxes.cls = torch.tensor([])
            
            mock_result.boxes = mock_boxes
            return [mock_result]
        
        mock_model.side_effect = mock_predict
        
        detector = OcclusionRobustYOLO(self.mock_model_path)
        detections = detector.detect(self.test_image)
        
        # Should combine detections from multiple scales
        self.assertGreater(len(detections), 0)
        # Verify multiple scales were called
        self.assertGreater(mock_model.call_count, 1)
    
    @patch('ultralytics.YOLO')
    def test_occlusion_score_calculation(self, mock_yolo):
        """Test occlusion score calculation."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        # Create overlapping detections
        mock_result = Mock()
        mock_boxes = Mock()
        mock_boxes.xyxy = torch.tensor([
            [100, 100, 200, 200],  # Detection 1
            [150, 150, 250, 250]   # Detection 2 (overlapping)
        ])
        mock_boxes.conf = torch.tensor([0.9, 0.8])
        mock_boxes.cls = torch.tensor([0, 0])
        
        mock_result.boxes = mock_boxes
        mock_model.return_value = [mock_result]
        
        detector = OcclusionRobustYOLO(self.mock_model_path)
        detections = detector.detect(self.test_image)
        
        # Check that occlusion scores are calculated
        self.assertEqual(len(detections), 2)
        for detection in detections:
            self.assertIsNotNone(detection.occlusion_score)
            self.assertGreaterEqual(detection.occlusion_score, 0.0)
            self.assertLessEqual(detection.occlusion_score, 1.0)


class TestTemporalConsistencyFilter(unittest.TestCase):
    """Test temporal consistency filtering."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.filter = TemporalConsistencyFilter(buffer_size=5, consistency_threshold=0.7)
    
    def test_consistent_detection_tracking(self):
        """Test tracking of consistent detections."""
        # Create consistent detection across frames
        detection = Detection(
            bbox=BoundingBox(100, 100, 200, 200, 0.9),
            class_id=0,
            class_name='chicken'
        )
        
        # Add same detection to multiple frames
        for frame_id in range(5):
            detections = [detection]
            filtered = self.filter.filter_detections(detections, frame_id)
            
        # After several frames, detection should be highly consistent
        consistency = self.filter.get_detection_consistency(detection)
        self.assertGreater(consistency, 0.5)
    
    def test_inconsistent_detection_filtering(self):
        """Test filtering of inconsistent detections."""
        # Create different detections for each frame
        detections_per_frame = [
            [Detection(BoundingBox(100, 100, 200, 200, 0.9), 0, 'chicken')],
            [Detection(BoundingBox(300, 300, 400, 400, 0.8), 0, 'chicken')],
            [Detection(BoundingBox(500, 500, 600, 600, 0.7), 0, 'chicken')],
        ]
        
        filtered_results = []
        for frame_id, detections in enumerate(detections_per_frame):
            filtered = self.filter.filter_detections(detections, frame_id)
            filtered_results.append(filtered)
        
        # Early frames should have fewer detections due to low consistency
        self.assertLessEqual(len(filtered_results[0]), len(detections_per_frame[0]))
    
    def test_buffer_size_limit(self):
        """Test that buffer doesn't exceed maximum size."""
        # Add more detections than buffer size
        for frame_id in range(10):
            detection = Detection(
                BoundingBox(100 + frame_id, 100, 200 + frame_id, 200, 0.9),
                0, 'chicken'
            )
            self.filter.filter_detections([detection], frame_id)
        
        # Buffer should not exceed maximum size
        self.assertLessEqual(len(self.filter.detection_buffer), self.filter.buffer_size)


class TestDetectionIntegration(unittest.TestCase):
    """Integration tests for detection pipeline."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.test_video_frames = self._generate_test_video()
    
    def _generate_test_video(self):
        """Generate test video frames with moving objects."""
        frames = []
        for i in range(30):
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
            
            # Add moving chicken-like rectangles
            x1 = 100 + i * 5
            y1 = 100 + i * 2
            cv2.rectangle(frame, (x1, y1), (x1 + 80, y1 + 80), (255, 255, 255), -1)
            
            # Add second chicken
            x2 = 300 - i * 3
            y2 = 200 + i * 1
            cv2.rectangle(frame, (x2, y2), (x2 + 70, y2 + 70), (200, 200, 200), -1)
            
            frames.append(frame)
        
        return frames
    
    @patch('ultralytics.YOLO')
    def test_full_detection_pipeline(self, mock_yolo):
        """Test complete detection pipeline with video sequence."""
        # Setup mock YOLO
        mock_model = Mock()
        
        def mock_predict(image, imgsz=640):
            # Simulate detections based on frame content
            mock_result = Mock()
            mock_boxes = Mock()
            
            # Find white rectangles in image (our test chickens)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(
                (gray > 200).astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if contours:
                # Convert contours to bounding boxes
                boxes = []
                confidences = []
                classes = []
                
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if w > 50 and h > 50:  # Filter small objects
                        boxes.append([x, y, x + w, y + h])
                        confidences.append(0.9)
                        classes.append(0)
                
                if boxes:
                    mock_boxes.xyxy = torch.tensor(boxes, dtype=torch.float32)
                    mock_boxes.conf = torch.tensor(confidences)
                    mock_boxes.cls = torch.tensor(classes)
                else:
                    mock_boxes.xyxy = torch.tensor([])
                    mock_boxes.conf = torch.tensor([])
                    mock_boxes.cls = torch.tensor([])
            else:
                mock_boxes.xyxy = torch.tensor([])
                mock_boxes.conf = torch.tensor([])
                mock_boxes.cls = torch.tensor([])
            
            mock_result.boxes = mock_boxes
            return [mock_result]
        
        mock_model.side_effect = mock_predict
        mock_yolo.return_value = mock_model
        
        # Create detection pipeline
        detector = OcclusionRobustYOLO("test_model.pt")
        temporal_filter = TemporalConsistencyFilter()
        
        all_detections = []
        
        # Process video frames
        for frame_id, frame in enumerate(self.test_video_frames):
            # Detect objects
            detections = detector.detect(frame)
            
            # Apply temporal filtering
            filtered_detections = temporal_filter.filter_detections(detections, frame_id)
            
            all_detections.append({
                'frame_id': frame_id,
                'raw_detections': len(detections),
                'filtered_detections': len(filtered_detections),
                'detections': filtered_detections
            })
        
        # Verify results
        self.assertGreater(len(all_detections), 0)
        
        # Check that we detected objects in most frames
        frames_with_detections = sum(1 for frame in all_detections 
                                   if frame['raw_detections'] > 0)
        self.assertGreater(frames_with_detections, len(self.test_video_frames) * 0.5)
        
        # Check temporal consistency improved results
        total_raw = sum(frame['raw_detections'] for frame in all_detections)
        total_filtered = sum(frame['filtered_detections'] for frame in all_detections)
        
        # Temporal filtering should generally reduce noise
        self.assertLessEqual(total_filtered, total_raw * 1.2)  # Allow some variance
    
    def test_performance_requirements(self):
        """Test that detection meets performance requirements."""
        with patch('ultralytics.YOLO') as mock_yolo:
            # Setup fast mock
            mock_model = Mock()
            mock_result = Mock()
            mock_result.boxes = None
            mock_model.return_value = [mock_result]
            mock_yolo.return_value = mock_model
            
            detector = YOLODetector("test_model.pt")
            
            # Measure processing time
            import time
            start_time = time.time()
            
            for frame in self.test_video_frames[:10]:  # Test with 10 frames
                detections = detector.detect(frame)
            
            end_time = time.time()
            avg_time_per_frame = (end_time - start_time) / 10
            
            # Should process frames in reasonable time (< 100ms per frame)
            self.assertLess(avg_time_per_frame, 0.1)


class TestDetectionErrorHandling(unittest.TestCase):
    """Test error handling in detection components."""
    
    def test_invalid_image_handling(self):
        """Test handling of invalid images."""
        with patch('ultralytics.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model
            
            detector = YOLODetector("test_model.pt")
            
            # Test with None image
            detections = detector.detect(None)
            self.assertEqual(len(detections), 0)
            
            # Test with empty image
            empty_image = np.array([])
            detections = detector.detect(empty_image)
            self.assertEqual(len(detections), 0)
            
            # Test with wrong shape
            wrong_shape = np.ones((100,))  # 1D array
            detections = detector.detect(wrong_shape)
            self.assertEqual(len(detections), 0)
    
    def test_model_loading_error(self):
        """Test handling of model loading errors."""
        with patch('ultralytics.YOLO') as mock_yolo:
            mock_yolo.side_effect = Exception("Model loading failed")
            
            with self.assertRaises(Exception):
                detector = YOLODetector("invalid_model.pt")
    
    def test_inference_error_handling(self):
        """Test handling of inference errors."""
        with patch('ultralytics.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_model.side_effect = Exception("Inference failed")
            mock_yolo.return_value = mock_model
            
            detector = YOLODetector("test_model.pt")
            
            # Should handle inference errors gracefully
            test_image = np.ones((640, 640, 3), dtype=np.uint8)
            detections = detector.detect(test_image)
            self.assertEqual(len(detections), 0)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestYOLODetector))
    test_suite.addTest(unittest.makeSuite(TestOcclusionRobustYOLO))
    test_suite.addTest(unittest.makeSuite(TestTemporalConsistencyFilter))
    test_suite.addTest(unittest.makeSuite(TestDetectionIntegration))
    test_suite.addTest(unittest.makeSuite(TestDetectionErrorHandling))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("DETECTION TESTS SUMMARY")
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