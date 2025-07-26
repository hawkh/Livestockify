#!/usr/bin/env python3
"""
Comprehensive tests for tracking components.
"""

import unittest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
import time

# Import components to test
from src.models.tracking.deepsort_tracker import DeepSORTTracker
from src.models.tracking.kalman_filter import KalmanFilter
from src.models.tracking.reid_features import ReIDFeatureExtractor
from src.models.tracking.chicken_tracker import ChickenTracker
from src.core.interfaces.detection import Detection, BoundingBox
from src.core.interfaces.tracking import Track, TrackState


class TestKalmanFilter(unittest.TestCase):
    """Test Kalman filter for object tracking."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.kf = KalmanFilter()
        # Initial state: [x, y, vx, vy, w, h, vw, vh]
        self.initial_state = np.array([100, 100, 1, 1, 50, 50, 0, 0])
    
    def test_initialization(self):
        """Test Kalman filter initialization."""
        self.assertIsNotNone(self.kf.F)  # State transition matrix
        self.assertIsNotNone(self.kf.H)  # Observation matrix
        self.assertIsNotNone(self.kf.Q)  # Process noise
        self.assertIsNotNone(self.kf.R)  # Measurement noise
        self.assertIsNotNone(self.kf.P)  # Error covariance
    
    def test_prediction(self):
        """Test state prediction."""
        state = self.initial_state.copy()
        
        # Predict next state
        predicted_state, predicted_covariance = self.kf.predict(state)
        
        # Check dimensions
        self.assertEqual(len(predicted_state), 8)
        self.assertEqual(predicted_covariance.shape, (8, 8))
        
        # Position should move according to velocity
        self.assertAlmostEqual(predicted_state[0], 101, places=0)  # x + vx
        self.assertAlmostEqual(predicted_state[1], 101, places=0)  # y + vy
    
    def test_update(self):
        """Test measurement update."""
        state = self.initial_state.copy()
        
        # Predict
        predicted_state, predicted_cov = self.kf.predict(state)
        
        # Create measurement [x, y, w, h]
        measurement = np.array([102, 102, 52, 52])
        
        # Update with measurement
        updated_state, updated_cov = self.kf.update(predicted_state, predicted_cov, measurement)
        
        # Check dimensions
        self.assertEqual(len(updated_state), 8)
        self.assertEqual(updated_cov.shape, (8, 8))
        
        # State should be adjusted toward measurement
        self.assertGreater(updated_state[0], predicted_state[0])
        self.assertGreater(updated_state[1], predicted_state[1])
    
    def test_multiple_predictions(self):
        """Test multiple prediction steps."""
        state = self.initial_state.copy()
        
        # Run multiple predictions
        for i in range(5):
            state, _ = self.kf.predict(state)
        
        # Position should have moved according to velocity
        expected_x = 100 + 5 * 1  # initial_x + steps * velocity_x
        expected_y = 100 + 5 * 1  # initial_y + steps * velocity_y
        
        self.assertAlmostEqual(state[0], expected_x, places=0)
        self.assertAlmostEqual(state[1], expected_y, places=0)
    
    def test_noise_handling(self):
        """Test handling of process and measurement noise."""
        state = self.initial_state.copy()
        
        # Predict with noise
        predicted_state1, cov1 = self.kf.predict(state)
        predicted_state2, cov2 = self.kf.predict(state)
        
        # States should be identical (deterministic prediction)
        np.testing.assert_array_almost_equal(predicted_state1, predicted_state2)
        
        # But covariance should increase (uncertainty grows)
        self.assertGreater(np.trace(cov1), np.trace(self.kf.P))


class TestReIDFeatureExtractor(unittest.TestCase):
    """Test Re-identification feature extraction."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = ReIDFeatureExtractor()
        self.test_image = np.ones((64, 128, 3), dtype=np.uint8) * 128
        self.test_bbox = BoundingBox(10, 10, 50, 100, 0.9)
    
    def test_feature_extraction(self):
        """Test feature extraction from image crops."""
        features = self.extractor.extract_features(self.test_image, self.test_bbox)
        
        # Check feature dimensions
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features.shape), 1)  # 1D feature vector
        self.assertGreater(len(features), 0)
    
    def test_feature_consistency(self):
        """Test that same input produces consistent features."""
        features1 = self.extractor.extract_features(self.test_image, self.test_bbox)
        features2 = self.extractor.extract_features(self.test_image, self.test_bbox)
        
        # Should be identical for same input
        np.testing.assert_array_almost_equal(features1, features2)
    
    def test_feature_similarity(self):
        """Test feature similarity calculation."""
        # Create two similar images
        image1 = np.ones((64, 128, 3), dtype=np.uint8) * 128
        image2 = np.ones((64, 128, 3), dtype=np.uint8) * 130  # Slightly different
        
        features1 = self.extractor.extract_features(image1, self.test_bbox)
        features2 = self.extractor.extract_features(image2, self.test_bbox)
        
        # Calculate similarity
        similarity = self.extractor.calculate_similarity(features1, features2)
        
        # Should be high similarity for similar images
        self.assertGreater(similarity, 0.5)
        self.assertLessEqual(similarity, 1.0)
    
    def test_different_crops(self):
        """Test features from different image crops."""
        bbox1 = BoundingBox(10, 10, 50, 100, 0.9)
        bbox2 = BoundingBox(60, 60, 100, 150, 0.9)
        
        features1 = self.extractor.extract_features(self.test_image, bbox1)
        features2 = self.extractor.extract_features(self.test_image, bbox2)
        
        # Features should be different for different crops
        similarity = self.extractor.calculate_similarity(features1, features2)
        self.assertLess(similarity, 0.9)  # Should not be too similar


class TestDeepSORTTracker(unittest.TestCase):
    """Test DeepSORT tracking algorithm."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tracker = DeepSORTTracker(
            max_age=30,
            min_hits=3,
            iou_threshold=0.3
        )
        
        # Create test detections
        self.detections_frame1 = [
            Detection(BoundingBox(100, 100, 200, 200, 0.9), 0, 'chicken'),
            Detection(BoundingBox(300, 300, 400, 400, 0.8), 0, 'chicken')
        ]
        
        self.detections_frame2 = [
            Detection(BoundingBox(105, 105, 205, 205, 0.85), 0, 'chicken'),
            Detection(BoundingBox(295, 295, 395, 395, 0.82), 0, 'chicken')
        ]
    
    def test_tracker_initialization(self):
        """Test tracker initialization."""
        self.assertEqual(len(self.tracker.tracks), 0)
        self.assertEqual(self.tracker.next_id, 1)
    
    def test_first_frame_tracking(self):
        """Test tracking on first frame."""
        tracks = self.tracker.update(self.detections_frame1, frame_id=0)
        
        # Should create new tracks
        self.assertEqual(len(self.tracker.tracks), 2)
        self.assertEqual(len(tracks), 0)  # No confirmed tracks yet (need min_hits)
    
    def test_track_association(self):
        """Test detection-to-track association."""
        # Process first frame
        self.tracker.update(self.detections_frame1, frame_id=0)
        
        # Process second frame
        tracks = self.tracker.update(self.detections_frame2, frame_id=1)
        
        # Should maintain same tracks
        self.assertEqual(len(self.tracker.tracks), 2)
        
        # Check that tracks have been updated
        for track in self.tracker.tracks:
            self.assertEqual(track.hits, 2)
            self.assertEqual(track.time_since_update, 0)
    
    def test_track_confirmation(self):
        """Test track confirmation after min_hits."""
        detections_sequence = [
            self.detections_frame1,
            self.detections_frame2,
            [Detection(BoundingBox(110, 110, 210, 210, 0.8), 0, 'chicken'),
             Detection(BoundingBox(290, 290, 390, 390, 0.8), 0, 'chicken')]
        ]
        
        confirmed_tracks = []
        for frame_id, detections in enumerate(detections_sequence):
            tracks = self.tracker.update(detections, frame_id)
            confirmed_tracks.extend(tracks)
        
        # Should have confirmed tracks after min_hits
        self.assertGreater(len(confirmed_tracks), 0)
        
        # Check track properties
        for track in confirmed_tracks:
            self.assertGreaterEqual(track.hits, self.tracker.min_hits)
            self.assertEqual(track.state, TrackState.CONFIRMED)
    
    def test_track_deletion(self):
        """Test track deletion after max_age."""
        # Create tracks
        self.tracker.update(self.detections_frame1, frame_id=0)
        initial_track_count = len(self.tracker.tracks)
        
        # Update without detections for max_age frames
        for frame_id in range(1, self.tracker.max_age + 2):
            self.tracker.update([], frame_id)
        
        # Tracks should be deleted
        self.assertLess(len(self.tracker.tracks), initial_track_count)
    
    def test_iou_calculation(self):
        """Test IoU calculation for track association."""
        bbox1 = BoundingBox(100, 100, 200, 200, 0.9)
        bbox2 = BoundingBox(150, 150, 250, 250, 0.8)  # Overlapping
        bbox3 = BoundingBox(300, 300, 400, 400, 0.8)  # Non-overlapping
        
        iou1 = self.tracker._calculate_iou(bbox1, bbox2)
        iou2 = self.tracker._calculate_iou(bbox1, bbox3)
        
        # Overlapping boxes should have higher IoU
        self.assertGreater(iou1, iou2)
        self.assertGreater(iou1, 0)
        self.assertEqual(iou2, 0)  # No overlap


class TestChickenTracker(unittest.TestCase):
    """Test chicken-specific tracking functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tracker = ChickenTracker()
        
        # Create test detection with weight estimate
        self.detection_with_weight = Detection(
            bbox=BoundingBox(100, 100, 200, 200, 0.9),
            class_id=0,
            class_name='chicken'
        )
        self.detection_with_weight.weight_estimate = 2.5
        self.detection_with_weight.age_estimate = 28  # days
    
    def test_weight_tracking(self):
        """Test weight tracking over time."""
        # Add multiple weight observations
        weights = [2.3, 2.5, 2.4, 2.6, 2.5]
        
        for i, weight in enumerate(weights):
            detection = Detection(
                BoundingBox(100 + i, 100 + i, 200 + i, 200 + i, 0.9),
                0, 'chicken'
            )
            detection.weight_estimate = weight
            
            tracks = self.tracker.update([detection], frame_id=i)
        
        # Check weight history
        if self.tracker.tracks:
            track = self.tracker.tracks[0]
            self.assertGreater(len(track.weight_history), 0)
            
            # Check smoothed weight
            smoothed_weight = track.get_smoothed_weight()
            self.assertIsNotNone(smoothed_weight)
            self.assertGreater(smoothed_weight, 0)
    
    def test_growth_tracking(self):
        """Test growth rate calculation."""
        # Simulate growth over time
        initial_weight = 2.0
        growth_rate = 0.1  # kg per week
        
        for week in range(4):
            weight = initial_weight + week * growth_rate
            detection = Detection(
                BoundingBox(100, 100, 200, 200, 0.9),
                0, 'chicken'
            )
            detection.weight_estimate = weight
            detection.age_estimate = 21 + week * 7  # 3 weeks + growth
            
            tracks = self.tracker.update([detection], frame_id=week * 7)
        
        # Check growth calculation
        if self.tracker.tracks:
            track = self.tracker.tracks[0]
            growth = track.calculate_growth_rate()
            
            if growth is not None:
                # Should detect positive growth
                self.assertGreater(growth, 0)
    
    def test_health_monitoring(self):
        """Test health status monitoring."""
        # Create detection with health indicators
        detection = Detection(
            BoundingBox(100, 100, 200, 200, 0.9),
            0, 'chicken'
        )
        detection.weight_estimate = 1.5  # Low weight
        detection.age_estimate = 35      # 5 weeks old
        detection.activity_level = 0.3   # Low activity
        
        tracks = self.tracker.update([detection], frame_id=0)
        
        # Check health assessment
        if self.tracker.tracks:
            track = self.tracker.tracks[0]
            health_score = track.assess_health()
            
            # Should detect potential health issues
            self.assertLess(health_score, 0.8)  # Below normal health threshold
    
    def test_behavior_analysis(self):
        """Test behavior pattern analysis."""
        # Simulate movement pattern
        positions = [
            (100, 100), (105, 102), (110, 105), (115, 108), (120, 110)
        ]
        
        for i, (x, y) in enumerate(positions):
            detection = Detection(
                BoundingBox(x, y, x + 100, y + 100, 0.9),
                0, 'chicken'
            )
            
            tracks = self.tracker.update([detection], frame_id=i)
        
        # Check movement analysis
        if self.tracker.tracks:
            track = self.tracker.tracks[0]
            movement_pattern = track.analyze_movement()
            
            self.assertIsNotNone(movement_pattern)
            self.assertIn('velocity', movement_pattern)
            self.assertIn('direction', movement_pattern)


class TestTrackingIntegration(unittest.TestCase):
    """Integration tests for tracking pipeline."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.tracker = ChickenTracker()
        self.test_sequence = self._generate_test_sequence()
    
    def _generate_test_sequence(self):
        """Generate test detection sequence."""
        sequence = []
        
        # Simulate two chickens moving
        for frame_id in range(20):
            detections = []
            
            # Chicken 1: moving right
            x1 = 100 + frame_id * 5
            y1 = 100 + frame_id * 2
            detection1 = Detection(
                BoundingBox(x1, y1, x1 + 80, y1 + 80, 0.9),
                0, 'chicken'
            )
            detection1.weight_estimate = 2.5 + np.random.normal(0, 0.1)
            detections.append(detection1)
            
            # Chicken 2: moving left (sometimes occluded)
            if frame_id < 15:  # Disappears later
                x2 = 400 - frame_id * 3
                y2 = 200 + frame_id * 1
                detection2 = Detection(
                    BoundingBox(x2, y2, x2 + 75, y2 + 75, 0.8),
                    0, 'chicken'
                )
                detection2.weight_estimate = 3.0 + np.random.normal(0, 0.15)
                detections.append(detection2)
            
            sequence.append((frame_id, detections))
        
        return sequence
    
    def test_full_tracking_pipeline(self):
        """Test complete tracking pipeline."""
        all_tracks = []
        
        # Process sequence
        for frame_id, detections in self.test_sequence:
            tracks = self.tracker.update(detections, frame_id)
            all_tracks.extend(tracks)
        
        # Verify tracking results
        self.assertGreater(len(all_tracks), 0)
        
        # Check track continuity
        track_ids = set()
        for track in all_tracks:
            track_ids.add(track.track_id)
        
        # Should have consistent track IDs
        self.assertLessEqual(len(track_ids), 3)  # Allow for some tracking errors
    
    def test_occlusion_handling(self):
        """Test handling of temporary occlusions."""
        # Create sequence with temporary occlusion
        detections_sequence = [
            [Detection(BoundingBox(100, 100, 200, 200, 0.9), 0, 'chicken')],  # Frame 0
            [Detection(BoundingBox(105, 105, 205, 205, 0.8), 0, 'chicken')],  # Frame 1
            [],  # Frame 2: Occluded
            [],  # Frame 3: Still occluded
            [Detection(BoundingBox(115, 115, 215, 215, 0.85), 0, 'chicken')], # Frame 4: Reappears
        ]
        
        tracks_per_frame = []
        for frame_id, detections in enumerate(detections_sequence):
            tracks = self.tracker.update(detections, frame_id)
            tracks_per_frame.append(tracks)
        
        # Should maintain track through occlusion
        final_tracks = self.tracker.get_active_tracks()
        self.assertGreater(len(final_tracks), 0)
        
        # Track should have survived the occlusion
        surviving_track = final_tracks[0]
        self.assertGreater(surviving_track.hits, 2)
    
    def test_performance_requirements(self):
        """Test tracking performance requirements."""
        import time
        
        # Create large detection set
        large_detections = []
        for i in range(50):  # 50 detections
            detection = Detection(
                BoundingBox(i * 20, i * 15, i * 20 + 80, i * 15 + 80, 0.8),
                0, 'chicken'
            )
            large_detections.append(detection)
        
        # Measure processing time
        start_time = time.time()
        
        for frame_id in range(10):
            tracks = self.tracker.update(large_detections, frame_id)
        
        end_time = time.time()
        avg_time_per_frame = (end_time - start_time) / 10
        
        # Should process frames quickly (< 50ms per frame)
        self.assertLess(avg_time_per_frame, 0.05)
    
    def test_memory_management(self):
        """Test memory management for long sequences."""
        initial_track_count = len(self.tracker.tracks)
        
        # Process very long sequence
        for frame_id in range(1000):
            # Create random detections
            detections = []
            if np.random.random() > 0.3:  # 70% chance of detection
                detection = Detection(
                    BoundingBox(
                        np.random.randint(0, 500),
                        np.random.randint(0, 400),
                        np.random.randint(50, 150),
                        np.random.randint(50, 150),
                        np.random.random() * 0.5 + 0.5
                    ),
                    0, 'chicken'
                )
                detections.append(detection)
            
            tracks = self.tracker.update(detections, frame_id)
        
        # Should not accumulate unlimited tracks
        final_track_count = len(self.tracker.tracks)
        self.assertLess(final_track_count, 100)  # Reasonable upper bound


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestKalmanFilter))
    test_suite.addTest(unittest.makeSuite(TestReIDFeatureExtractor))
    test_suite.addTest(unittest.makeSuite(TestDeepSORTTracker))
    test_suite.addTest(unittest.makeSuite(TestChickenTracker))
    test_suite.addTest(unittest.makeSuite(TestTrackingIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("TRACKING TESTS SUMMARY")
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