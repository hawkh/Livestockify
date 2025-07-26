"""
Comprehensive test suite for stream processing system.
"""

import unittest
import time
import json
import base64
import threading
import asyncio
import requests
import numpy as np
import cv2
from collections import defaultdict, deque
from unittest.mock import Mock, patch, MagicMock
import matplotlib.pyplot as plt
import pandas as pd

from ..src.inference.stream_handler import RealTimeStreamProcessor, StreamProcessingServer
from ..src.utils.config.config_manager import ConfigManager
from ..src.core.interfaces.detection import Detection
from ..src.core.interfaces.weight_estimation import WeightEstimate
from ..src.core.interfaces.tracking import TrackedChicken


class MockComponents:
    """Mock components for testing."""
    
    class MockYOLODetector:
        def __init__(self):
            self.model = True
            
        def detect_with_occlusion_handling(self, frame, previous_detections=None):
            # Generate mock detections
            detections = []
            num_chickens = np.random.randint(1, 6)  # 1-5 chickens
            
            for i in range(num_chickens):
                x1 = np.random.randint(0, frame.shape[1] - 100)
                y1 = np.random.randint(0, frame.shape[0] - 100)
                x2 = x1 + np.random.randint(50, 150)
                y2 = y1 + np.random.randint(50, 150)
                
                detection = Detection(
                    bbox=[x1, y1, x2, y2],
                    confidence=np.random.uniform(0.5, 0.95),
                    class_id=0,
                    class_name="chicken",
                    occlusion_level=np.random.uniform(0.0, 0.7),
                    visibility_score=np.random.uniform(0.3, 1.0)
                )
                detection.distance_estimate = np.random.uniform(2.0, 8.0)
                detections.append(detection)
            
            return Mock(detections=detections, processing_time_ms=np.random.uniform(20, 80))
        
        def get_model_info(self):
            return {"status": "loaded", "model_path": "mock"}
    
    class MockWeightEstimator:
        def __init__(self):
            self.is_loaded = True
            
        def estimate_weight_with_distance(self, frame, detection, distance, occlusion_level):
            # Generate realistic weight based on detection size
            bbox_area = (detection.bbox[2] - detection.bbox[0]) * (detection.bbox[3] - detection.bbox[1])
            base_weight = 1.5 + (bbox_area / 10000) * 2.0  # Scale with size
            
            # Add some randomness
            weight = base_weight + np.random.normal(0, 0.3)
            weight = max(0.5, min(5.0, weight))  # Clamp to reasonable range
            
            return WeightEstimate(
                value=weight,
                confidence=np.random.uniform(0.6, 0.9),
                error_range=f"±{weight * 0.2:.2f}kg",
                distance_compensated=True,
                occlusion_adjusted=occlusion_level > 0.1,
                age_category=np.random.choice(["WEEK_4", "WEEK_5", "WEEK_6", "ADULT"]),
                method="distance_adaptive_nn"
            )
        
        def get_model_info(self):
            return {"model_loaded": True, "device": "cpu"}
    
    class MockTracker:
        def __init__(self):
            self.active_tracks = {}
            self.track_id_counter = 0
            
        def update_tracks(self, detections, frame_id=None):
            tracked_chickens = []
            
            for detection in detections:
                # Simple tracking - create or update tracks
                track_id = f"chicken_{self.track_id_counter}"
                self.track_id_counter += 1
                
                tracked_chicken = TrackedChicken(
                    chicken_id=track_id,
                    current_detection=detection,
                    tracking_status="active",
                    confidence=detection.confidence,
                    frames_tracked=np.random.randint(1, 50),
                    frames_lost=0,
                    last_seen_timestamp=time.time()
                )
                
                tracked_chickens.append(tracked_chicken)
            
            return Mock(
                tracked_chickens=tracked_chickens,
                new_tracks=[],
                lost_tracks=[],
                processing_time_ms=np.random.uniform(5, 15),
                total_active_tracks=len(tracked_chickens)
            )
        
        def get_active_tracks(self):
            return list(self.active_tracks.values())
        
        def get_tracking_statistics(self):
            return {
                'total_active_tracks': len(self.active_tracks),
                'frames_processed': 100
            }
        
        def reset_tracker(self):
            self.active_tracks.clear()
    
    class MockDistanceEstimator:
        def estimate_distance_to_chicken(self, bbox, frame_shape):
            # Simple distance estimation based on bbox size
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            frame_area = frame_shape[0] * frame_shape[1]
            relative_size = bbox_area / frame_area
            
            # Inverse relationship: larger objects are closer
            distance = 10.0 / (relative_size * 100 + 1.0)
            return max(1.0, min(10.0, distance))


class TestVideoGenerator:
    """Generate test video frames with synthetic chickens."""
    
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.frame_count = 0
        
    def generate_frame(self):
        """Generate a synthetic frame with chicken-like objects."""
        # Create background
        frame = np.random.randint(50, 150, (self.height, self.width, 3), dtype=np.uint8)
        
        # Add some texture
        noise = np.random.randint(-20, 20, (self.height, self.width, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add synthetic "chickens" (elliptical shapes)
        num_chickens = np.random.randint(1, 5)
        
        for _ in range(num_chickens):
            center_x = np.random.randint(50, self.width - 50)
            center_y = np.random.randint(50, self.height - 50)
            
            # Chicken size varies
            width_radius = np.random.randint(20, 60)
            height_radius = np.random.randint(15, 45)
            
            # Chicken color (brownish)
            color = (
                np.random.randint(80, 150),   # Blue
                np.random.randint(100, 180),  # Green
                np.random.randint(120, 200)   # Red
            )
            
            # Draw ellipse
            cv2.ellipse(frame, (center_x, center_y), (width_radius, height_radius), 
                       0, 0, 360, color, -1)
            
            # Add some details (head)
            head_x = center_x + np.random.randint(-10, 10)
            head_y = center_y - height_radius - np.random.randint(5, 15)
            cv2.circle(frame, (head_x, head_y), np.random.randint(8, 15), color, -1)
        
        self.frame_count += 1
        return frame


class StreamTestAnalyzer:
    """Analyze stream processing performance and accuracy."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.tracking_data = defaultdict(list)
        
    def add_result(self, result):
        """Add a processing result for analysis."""
        self.metrics['processing_time'].append(result.get('processing_time_ms', 0) / 1000.0)
        self.metrics['num_detections'].append(result.get('total_chickens_detected', 0))
        self.metrics['timestamp'].append(time.time())
        
        # Track individual chickens
        for detection in result.get('detections', []):
            if 'chicken_id' in detection:
                self.tracking_data[detection['chicken_id']].append({
                    'timestamp': time.time(),
                    'weight': detection.get('weight_estimate', {}).get('value', 0),
                    'confidence': detection.get('confidence', 0)
                })
    
    def generate_report(self):
        """Generate performance report."""
        if not self.metrics['processing_time']:
            return {'status': 'no_data'}
        
        processing_times = self.metrics['processing_time']
        detection_counts = self.metrics['num_detections']
        
        report = {
            'performance': {
                'avg_processing_time_ms': np.mean(processing_times) * 1000,
                'max_processing_time_ms': np.max(processing_times) * 1000,
                'min_processing_time_ms': np.min(processing_times) * 1000,
                'avg_fps': 1.0 / np.mean(processing_times) if np.mean(processing_times) > 0 else 0,
                'total_frames': len(processing_times)
            },
            'detection': {
                'avg_detections_per_frame': np.mean(detection_counts),
                'max_detections': np.max(detection_counts) if detection_counts else 0,
                'min_detections': np.min(detection_counts) if detection_counts else 0
            },
            'tracking': {
                'total_tracks': len(self.tracking_data),
                'track_continuity': self._calculate_track_continuity()
            }
        }
        
        return report
    
    def _calculate_track_continuity(self):
        """Calculate how continuous tracks are."""
        if not self.tracking_data:
            return 0.0
        
        continuities = []
        for track_id, data in self.tracking_data.items():
            if len(data) > 1:
                timestamps = [d['timestamp'] for d in data]
                gaps = np.diff(timestamps)
                # Consider gaps < 0.1 seconds as continuous
                continuity = np.sum(gaps < 0.1) / len(gaps) if gaps.size > 0 else 0
                continuities.append(continuity)
        
        return np.mean(continuities) if continuities else 0.0
    
    def plot_performance(self, save_path='performance_analysis.png'):
        """Plot performance metrics."""
        if not self.metrics['processing_time']:
            print("No data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Processing time
        axes[0, 0].plot(self.metrics['processing_time'])
        axes[0, 0].set_title('Processing Time per Frame')
        axes[0, 0].set_xlabel('Frame')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].axhline(y=0.033, color='r', linestyle='--', label='30 FPS target')
        axes[0, 0].legend()
        
        # Detection count
        axes[0, 1].plot(self.metrics['num_detections'])
        axes[0, 1].set_title('Detections per Frame')
        axes[0, 1].set_xlabel('Frame')
        axes[0, 1].set_ylabel('Number of Detections')
        
        # FPS over time
        fps_values = [1.0/t if t > 0 else 0 for t in self.metrics['processing_time']]
        axes[1, 0].plot(fps_values)
        axes[1, 0].set_title('FPS Over Time')
        axes[1, 0].set_xlabel('Frame')
        axes[1, 0].set_ylabel('FPS')
        axes[1, 0].axhline(y=30, color='r', linestyle='--', label='30 FPS target')
        axes[1, 0].legend()
        
        # Weight estimates (if available)
        axes[1, 1].set_title('Weight Estimates by Track')
        track_count = 0
        for track_id, data in list(self.tracking_data.items())[:5]:  # Show first 5 tracks
            if data:
                weights = [d['weight'] for d in data if d['weight'] > 0]
                if weights:
                    axes[1, 1].plot(weights, label=f'Track {track_id}')
                    track_count += 1
        
        if track_count > 0:
            axes[1, 1].set_xlabel('Measurement')
            axes[1, 1].set_ylabel('Weight (kg)')
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'No weight data available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Performance plot saved to {save_path}")
        return fig


class TestStreamProcessing(unittest.TestCase):
    """Comprehensive stream processing tests."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_video_gen = TestVideoGenerator()
        self.mock_components = MockComponents()
        
    def create_mock_processor(self):
        """Create processor with mock components."""
        # Mock config manager
        config_manager = Mock()
        config_manager.load_config.return_value = {
            'yolo': {'model_path': 'mock', 'confidence_threshold': 0.5},
            'weight_estimation': {'model_path': 'mock'},
            'tracking': {'max_disappeared': 30}
        }
        
        # Create processor
        processor = RealTimeStreamProcessor(config_manager)
        
        # Replace with mock components
        processor.detector = self.mock_components.MockYOLODetector()
        processor.weight_estimator = self.mock_components.MockWeightEstimator()
        processor.tracker = self.mock_components.MockTracker()
        processor.distance_estimator = self.mock_components.MockDistanceEstimator()
        
        return processor
    
    def test_basic_frame_processing(self):
        """Test basic frame processing functionality."""
        print("Testing basic frame processing...")
        
        processor = self.create_mock_processor()
        
        # Generate test frame
        frame = self.test_video_gen.generate_frame()
        frame_data = base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode('utf-8')
        
        # Process frame
        result = processor.process_frame(frame_data, "test_camera", 0)
        
        # Verify result structure
        self.assertIn('camera_id', result)
        self.assertIn('detections', result)
        self.assertIn('processing_time_ms', result)
        self.assertIn('status', result)
        
        # Verify processing was successful
        self.assertEqual(result['status'], 'success')
        self.assertGreater(result['processing_time_ms'], 0)
        
        print(f"✓ Processed frame with {result['total_chickens_detected']} detections")
        print(f"✓ Processing time: {result['processing_time_ms']:.2f}ms")
    
    def test_performance_requirements(self):
        """Test system meets performance requirements."""
        print("Testing performance requirements...")
        
        processor = self.create_mock_processor()
        analyzer = StreamTestAnalyzer()
        
        # Process multiple frames
        num_frames = 50
        for i in range(num_frames):
            frame = self.test_video_gen.generate_frame()
            frame_data = base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode('utf-8')
            
            result = processor.process_frame(frame_data, "test_camera", i)
            analyzer.add_result(result)
        
        # Generate report
        report = analyzer.generate_report()
        
        # Check performance requirements
        avg_processing_time = report['performance']['avg_processing_time_ms']
        avg_fps = report['performance']['avg_fps']
        
        print(f"✓ Average processing time: {avg_processing_time:.2f}ms")
        print(f"✓ Average FPS: {avg_fps:.2f}")
        
        # Performance assertions
        self.assertLess(avg_processing_time, 100, "Processing time should be under 100ms")
        self.assertGreater(avg_fps, 10, "Should achieve at least 10 FPS")
        
        # Generate performance plot
        analyzer.plot_performance('test_performance.png')
    
    def test_error_handling(self):
        """Test error handling for various failure scenarios."""
        print("Testing error handling...")
        
        processor = self.create_mock_processor()
        
        # Test invalid frame data
        invalid_inputs = [
            "",  # Empty string
            "invalid_base64",  # Invalid base64
            base64.b64encode(b"not_an_image").decode('utf-8'),  # Invalid image
        ]
        
        for invalid_input in invalid_inputs:
            result = processor.process_frame(invalid_input, "test_camera", 0)
            self.assertEqual(result['status'], 'error')
            self.assertIn('error_message', result)
        
        print("✓ Error handling works correctly")
    
    def test_concurrent_processing(self):
        """Test concurrent frame processing."""
        print("Testing concurrent processing...")
        
        processor = self.create_mock_processor()
        results = []
        
        def process_frames(thread_id, num_frames=10):
            thread_results = []
            for i in range(num_frames):
                frame = self.test_video_gen.generate_frame()
                frame_data = base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode('utf-8')
                
                result = processor.process_frame(frame_data, f"camera_{thread_id}", i)
                thread_results.append(result)
            
            results.extend(thread_results)
        
        # Run multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=process_frames, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify all frames were processed
        self.assertEqual(len(results), 30)  # 3 threads * 10 frames
        
        # Check for successful processing
        successful_results = [r for r in results if r['status'] == 'success']
        success_rate = len(successful_results) / len(results)
        
        print(f"✓ Processed {len(results)} frames concurrently")
        print(f"✓ Success rate: {success_rate * 100:.1f}%")
        
        self.assertGreater(success_rate, 0.9, "Success rate should be > 90%")
    
    def test_memory_usage(self):
        """Test memory usage during processing."""
        print("Testing memory usage...")
        
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            processor = self.create_mock_processor()
            
            # Measure initial memory
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Process many frames
            for i in range(100):
                frame = self.test_video_gen.generate_frame()
                frame_data = base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode('utf-8')
                
                processor.process_frame(frame_data, "test_camera", i)
            
            # Measure final memory
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            print(f"✓ Initial memory: {initial_memory:.1f}MB")
            print(f"✓ Final memory: {final_memory:.1f}MB")
            print(f"✓ Memory increase: {memory_increase:.1f}MB")
            
            # Memory should not increase dramatically
            self.assertLess(memory_increase, 100, "Memory increase should be < 100MB")
            
        except ImportError:
            print("⚠ psutil not available, skipping memory test")
    
    def test_health_check(self):
        """Test health check functionality."""
        print("Testing health check...")
        
        processor = self.create_mock_processor()
        health_status = processor.health_check()
        
        # Verify health check structure
        self.assertIn('status', health_status)
        self.assertIn('components', health_status)
        
        # Should be healthy with mock components
        self.assertEqual(health_status['status'], 'healthy')
        
        print("✓ Health check passed")
    
    def run_all_tests(self):
        """Run all tests in sequence."""
        print("=" * 60)
        print("CHICKEN WEIGHT ESTIMATION - STREAM PROCESSING TESTS")
        print("=" * 60)
        
        test_methods = [
            self.test_basic_frame_processing,
            self.test_performance_requirements,
            self.test_error_handling,
            self.test_concurrent_processing,
            self.test_memory_usage,
            self.test_health_check
        ]
        
        passed = 0
        failed = 0
        
        for test_method in test_methods:
            try:
                print(f"\n{test_method.__name__.replace('_', ' ').title()}")
                print("-" * 40)
                test_method()
                passed += 1
                print("✅ PASSED")
            except Exception as e:
                failed += 1
                print(f"❌ FAILED: {str(e)}")
        
        print("\n" + "=" * 60)
        print(f"TEST SUMMARY: {passed} passed, {failed} failed")
        print("=" * 60)
        
        return failed == 0


def test_with_video_file(video_path=None):
    """Test with actual video file or webcam."""
    print("Testing with video input...")
    
    # Create processor
    test_case = TestStreamProcessing()
    processor = test_case.create_mock_processor()
    
    # Open video
    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        # Use webcam
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("⚠ Could not open video source, using synthetic frames")
        cap = None
    
    analyzer = StreamTestAnalyzer()
    frame_id = 0
    
    print("Processing video... Press 'q' to quit")
    
    try:
        while frame_id < 100:  # Process up to 100 frames
            if cap:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (640, 480))
            else:
                # Use synthetic frame
                video_gen = TestVideoGenerator()
                frame = video_gen.generate_frame()
            
            # Process frame
            frame_data = base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode('utf-8')
            result = processor.process_frame(frame_data, "video_camera", frame_id)
            analyzer.add_result(result)
            
            # Visualize results (optional)
            vis_frame = draw_results_on_frame(frame, result)
            
            # Display (if running interactively)
            try:
                cv2.imshow('Chicken Tracking', vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except:
                pass  # Skip display if not available
            
            frame_id += 1
            
            # Print progress
            if frame_id % 30 == 0:
                stats = processor.get_processing_stats()
                print(f"Processed {frame_id} frames, FPS: {stats.get('current_fps', 0):.2f}")
    
    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()
    
    # Generate report
    report = analyzer.generate_report()
    print("\nPerformance Report:")
    print(json.dumps(report, indent=2))
    
    # Plot analysis
    analyzer.plot_performance('video_test_performance.png')


def draw_results_on_frame(frame, result):
    """Draw detection and tracking results on frame."""
    vis_frame = frame.copy()
    
    # Draw detections
    for detection in result.get('detections', []):
        bbox = detection.get('bbox', [0, 0, 100, 100])
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Color based on confidence
        confidence = detection.get('confidence', 0.5)
        color = (0, int(255 * confidence), int(255 * (1 - confidence)))
        
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        
        # Detection info
        weight_info = detection.get('weight_estimate', {})
        weight = weight_info.get('value', 0)
        text = f"C:{confidence:.2f} W:{weight:.2f}kg"
        
        cv2.putText(vis_frame, text, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # FPS counter
    processing_time = result.get('processing_time_ms', 0)
    fps = 1000.0 / processing_time if processing_time > 0 else 0
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(vis_frame, fps_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Detection count
    detection_count = result.get('total_chickens_detected', 0)
    count_text = f"Chickens: {detection_count}"
    cv2.putText(vis_frame, count_text, (10, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return vis_frame


if __name__ == "__main__":
    # Run comprehensive tests
    test_case = TestStreamProcessing()
    success = test_case.run_all_tests()
    
    # Optionally test with video
    print("\n" + "=" * 60)
    print("VIDEO PROCESSING TEST")
    print("=" * 60)
    test_with_video_file()
    
    print("\nAll tests completed!")
    if success:
        print("🎉 All tests passed!")
    else:
        print("⚠ Some tests failed. Check the output above.")