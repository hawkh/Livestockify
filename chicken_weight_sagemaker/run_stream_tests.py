#!/usr/bin/env python3
"""
Quick test runner for stream processing system.
"""

import os
import sys
import time
import cv2
import numpy as np
import base64
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_functionality():
    """Test basic functionality with minimal setup."""
    print("🐔 Testing Basic Stream Processing Functionality")
    print("=" * 50)
    
    try:
        # Import components
        from src.inference.stream_handler import RealTimeStreamProcessor
        from src.utils.config.config_manager import ConfigManager
        
        # Create mock config manager
        class MockConfigManager:
            def load_config(self, config_name):
                if config_name == "model_config":
                    return {
                        'yolo': {
                            'model_path': 'mock_yolo.pt',
                            'confidence_threshold': 0.4,
                            'min_visibility_threshold': 0.3
                        },
                        'weight_estimation': {
                            'model_path': 'mock_weight.pt'
                        },
                        'tracking': {
                            'max_disappeared': 30,
                            'max_distance': 100.0
                        }
                    }
                elif config_name == "camera_config":
                    return {
                        'focal_length': 1000.0,
                        'sensor_width': 6.0,
                        'sensor_height': 4.5,
                        'image_width': 640,
                        'image_height': 480,
                        'camera_height': 3.0,
                        'known_object_width': 25.0
                    }
                return {}
        
        print("✅ Imports successful")
        
        # Create processor
        config_manager = MockConfigManager()
        processor = RealTimeStreamProcessor(config_manager)
        
        print("✅ Processor created")
        
        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add some colored rectangles to simulate chickens
        cv2.rectangle(test_frame, (100, 100), (200, 180), (150, 100, 80), -1)
        cv2.rectangle(test_frame, (300, 200), (400, 280), (140, 110, 90), -1)
        cv2.rectangle(test_frame, (450, 150), (550, 230), (160, 90, 70), -1)
        
        # Encode to base64
        _, buffer = cv2.imencode('.jpg', test_frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        print("✅ Test frame created and encoded")
        
        # Mock the components that would normally load models
        from tests.test_stream_processing_comprehensive import MockYOLODetector, MockWeightEstimator
        
        processor.detector = MockYOLODetector()
        processor.weight_estimator = MockWeightEstimator()
        
        print("✅ Mock components attached")
        
        # Process frame
        start_time = time.time()
        result = processor.process_frame(frame_b64, "test_camera", 1)
        processing_time = time.time() - start_time
        
        print("✅ Frame processed successfully")
        print(f"   Processing time: {processing_time*1000:.2f}ms")
        print(f"   Detections: {result.get('total_chickens_detected', 0)}")
        print(f"   Status: {result.get('status', 'unknown')}")
        
        # Test health check
        health = processor.health_check()
        print(f"✅ Health check: {health.get('status', 'unknown')}")
        
        # Test multiple frames
        print("\n📊 Processing multiple frames for performance test...")
        processing_times = []
        
        for i in range(10):
            # Create slightly different frame
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.rectangle(test_frame, (100+i*5, 100), (200+i*5, 180), (150, 100, 80), -1)
            
            _, buffer = cv2.imencode('.jpg', test_frame)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            start_time = time.time()
            result = processor.process_frame(frame_b64, "test_camera", i)
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            if i % 3 == 0:
                print(f"   Frame {i+1}: {processing_time*1000:.2f}ms, "
                      f"Detections: {result.get('total_chickens_detected', 0)}")
        
        # Performance summary
        avg_time = np.mean(processing_times) * 1000
        max_time = np.max(processing_times) * 1000
        min_time = np.min(processing_times) * 1000
        avg_fps = 1000 / avg_time
        
        print(f"\n📈 Performance Summary:")
        print(f"   Average processing time: {avg_time:.2f}ms")
        print(f"   Min/Max processing time: {min_time:.2f}ms / {max_time:.2f}ms")
        print(f"   Average FPS: {avg_fps:.2f}")
        print(f"   Real-time capable (>30 FPS): {'✅ Yes' if avg_fps > 30 else '❌ No'}")
        
        # Test statistics
        stats = processor.get_processing_stats()
        print(f"\n📊 Processor Statistics:")
        print(f"   Frames processed: {stats.get('frames_processed', 0)}")
        print(f"   Current FPS: {stats.get('current_fps', 0):.2f}")
        print(f"   Active tracks: {stats.get('active_tracks', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_sagemaker_handler():
    """Test SageMaker handler functionality."""
    print("\n🚀 Testing SageMaker Handler")
    print("=" * 30)
    
    try:
        from src.inference.sagemaker_handler import SageMakerInferenceHandler
        
        # Create handler
        handler = SageMakerInferenceHandler()
        
        print("✅ SageMaker handler created")
        
        # Test input validation
        test_input = {
            "stream_data": {
                "frame": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
                "camera_id": "test_camera",
                "frame_sequence": 1
            }
        }
        
        # Test input parsing
        parsed_input = handler.input_fn(json.dumps(test_input), "application/json")
        print("✅ Input parsing successful")
        
        # Test input validation
        is_valid = handler._validate_input(parsed_input)
        print(f"✅ Input validation: {'Valid' if is_valid else 'Invalid'}")
        
        # Test output formatting
        mock_prediction = {
            "camera_id": "test_camera",
            "timestamp": "2024-01-01T00:00:00Z",
            "status": "success",
            "detections": [],
            "total_chickens_detected": 0
        }
        
        output = handler.output_fn(mock_prediction, "application/json")
        print("✅ Output formatting successful")
        
        # Test health check
        health = handler.health_check()
        print(f"✅ Health check: {health.get('status', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ SageMaker handler test failed: {str(e)}")
        return False


def test_frame_processor():
    """Test frame processor utilities."""
    print("\n🖼️  Testing Frame Processor")
    print("=" * 25)
    
    try:
        from src.inference.frame_processor import FrameProcessor, FrameQueue
        
        # Test frame processor
        processor = FrameProcessor(target_size=(320, 320))
        
        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test preprocessing
        processed_frame, metadata = processor.preprocess_frame(test_frame)
        
        print(f"✅ Frame preprocessing successful")
        print(f"   Original shape: {metadata['original_shape']}")
        print(f"   Processed shape: {metadata['processed_shape']}")
        print(f"   Scale factors: {metadata['scale_factor_x']:.2f}, {metadata['scale_factor_y']:.2f}")
        
        # Test frame queue
        frame_queue = FrameQueue(max_size=5)
        
        # Add frames to queue
        for i in range(7):  # More than max_size to test overflow
            frame_data = {"frame_id": i, "data": f"frame_{i}"}
            success = frame_queue.put_frame(frame_data)
            if not success:
                print(f"   Frame {i} dropped (queue full)")
        
        # Get queue stats
        stats = frame_queue.get_queue_stats()
        print(f"✅ Frame queue test completed")
        print(f"   Queue utilization: {stats['utilization']*100:.1f}%")
        print(f"   Dropped frames: {stats['dropped_frames']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Frame processor test failed: {str(e)}")
        return False


def create_sample_video():
    """Create a sample video for testing."""
    print("\n🎥 Creating Sample Test Video")
    print("=" * 30)
    
    try:
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('sample_test_video.mp4', fourcc, 20.0, (640, 480))
        
        print("Creating 100 frames with moving 'chickens'...")
        
        for frame_num in range(100):
            # Create background
            frame = np.random.randint(80, 120, (480, 640, 3), dtype=np.uint8)
            
            # Add moving "chickens" (colored rectangles)
            for i in range(3):
                # Calculate position based on frame number for movement
                x = int(100 + 200 * np.sin(frame_num * 0.1 + i * 2))
                y = int(150 + 100 * np.cos(frame_num * 0.05 + i))
                
                # Ensure within bounds
                x = max(50, min(550, x))
                y = max(50, min(400, y))
                
                # Draw "chicken"
                color = (120 + i * 30, 100, 80 + i * 20)
                cv2.rectangle(frame, (x, y), (x + 80, y + 60), color, -1)
                
                # Add some details
                cv2.circle(frame, (x + 40, y + 30), 8, (200, 200, 200), -1)  # Head
                cv2.rectangle(frame, (x + 10, y + 45), (x + 70, y + 55), (100, 100, 100), -1)  # Body detail
            
            # Add frame number
            cv2.putText(frame, f"Frame {frame_num}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
            
            if frame_num % 20 == 0:
                print(f"   Generated frame {frame_num}/100")
        
        out.release()
        print("✅ Sample video created: sample_test_video.mp4")
        
        return True
        
    except Exception as e:
        print(f"❌ Video creation failed: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("🐔 Chicken Weight Estimation - Stream Processing Test Suite")
    print("=" * 65)
    print("This test suite validates the stream processing components")
    print("without requiring actual model files or SageMaker deployment.")
    print()
    
    test_results = {}
    
    # Test 1: Basic functionality
    test_results['basic'] = test_basic_functionality()
    
    # Test 2: SageMaker handler
    test_results['sagemaker'] = test_sagemaker_handler()
    
    # Test 3: Frame processor
    test_results['frame_processor'] = test_frame_processor()
    
    # Test 4: Create sample video
    test_results['video_creation'] = create_sample_video()
    
    # Summary
    print("\n" + "=" * 65)
    print("🏁 TEST SUMMARY")
    print("=" * 65)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! The stream processing system is ready.")
        print("\nNext steps:")
        print("   1. Add your actual YOLO model weights to model_artifacts/")
        print("   2. Train the weight estimation neural network")
        print("   3. Test with real poultry farm footage")
        print("   4. Deploy to SageMaker using the Docker container")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)