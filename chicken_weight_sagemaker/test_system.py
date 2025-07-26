"""
Quick system test for the chicken weight estimation stream processing.
"""

import sys
import os
import json
import base64
import numpy as np
import cv2
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_test_image():
    """Create a test image with chicken-like objects."""
    print("Creating test image...")
    
    # Create a realistic farm scene
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add background (barn floor)
    image[:] = [101, 67, 33]  # Brown background
    
    # Add some chicken-like ellipses
    chickens = [
        ((200, 200), (50, 30), (200, 180, 150)),  # Chicken 1
        ((400, 300), (60, 35), (180, 160, 140)),  # Chicken 2
        ((150, 350), (45, 28), (190, 170, 145)),  # Chicken 3
    ]
    
    for (center, axes, color) in chickens:
        cv2.ellipse(image, center, axes, 0, 0, 360, color, -1)
        # Add some texture
        cv2.ellipse(image, center, (axes[0]-10, axes[1]-5), 0, 0, 360, 
                   (color[0]-20, color[1]-20, color[2]-20), 2)
    
    # Add some noise for realism
    noise = np.random.randint(0, 30, image.shape, dtype=np.uint8)
    image = cv2.add(image, noise)
    
    print(f"✓ Created test image: {image.shape}")
    return image

def test_frame_encoding():
    """Test frame encoding/decoding."""
    print("\nTesting frame encoding...")
    
    # Create test image
    test_image = create_test_image()
    
    # Encode to base64
    _, buffer = cv2.imencode('.jpg', test_image)
    encoded_frame = base64.b64encode(buffer).decode('utf-8')
    
    print(f"✓ Encoded frame size: {len(encoded_frame)} characters")
    
    # Test decoding
    decoded_bytes = base64.b64decode(encoded_frame)
    decoded_array = np.frombuffer(decoded_bytes, np.uint8)
    decoded_image = cv2.imdecode(decoded_array, cv2.IMREAD_COLOR)
    
    print(f"✓ Decoded image shape: {decoded_image.shape}")
    
    return encoded_frame

def test_stream_processor():
    """Test the stream processor with mocked components."""
    print("\nTesting stream processor...")
    
    try:
        # Mock all the heavy dependencies
        with patch('inference.stream_handler.OcclusionRobustYOLODetector') as mock_detector, \
             patch('inference.stream_handler.ChickenMultiObjectTracker') as mock_tracker, \
             patch('inference.stream_handler.DistanceAdaptiveWeightNN') as mock_weight, \
             patch('inference.stream_handler.PerspectiveDistanceEstimator') as mock_distance:
            
            # Setup mock responses
            from core.interfaces.detection import Detection
            from core.interfaces.weight_estimation import WeightEstimate
            from core.interfaces.tracking import TrackedChicken, TrackingResult
            
            # Mock detection
            mock_detection = Detection(
                bbox=[150, 170, 250, 230],
                confidence=0.85,
                class_id=0,
                class_name="chicken",
                occlusion_level=0.1
            )
            
            mock_detection_result = Mock()
            mock_detection_result.detections = [mock_detection]
            mock_detector.return_value.detect_with_occlusion_handling.return_value = mock_detection_result
            
            # Mock distance estimation
            mock_distance.return_value.estimate_distance_to_chicken.return_value = 3.2
            
            # Mock tracking
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
            
            # Mock weight estimation
            mock_weight_estimate = WeightEstimate(
                value=2.1,
                confidence=0.78,
                error_range="±0.4kg",
                method="distance_adaptive_nn"
            )
            mock_weight.return_value.estimate_weight_with_distance.return_value = mock_weight_estimate
            
            # Create mock config manager
            from utils.config.config_manager import ConfigManager
            mock_config_manager = Mock(spec=ConfigManager)
            mock_config_manager.load_config.return_value = {
                'yolo': {'model_path': 'test.pt', 'confidence_threshold': 0.5, 'min_visibility_threshold': 0.3},
                'weight_estimation': {'model_path': 'test.pt'},
                'tracking': {'max_disappeared': 30, 'max_distance': 100.0},
                'focal_length': 1000.0,
                'sensor_width': 6.0,
                'sensor_height': 4.5,
                'image_width': 1920,
                'image_height': 1080,
                'camera_height': 3.0,
                'known_object_width': 25.0
            }
            
            # Test processor initialization
            from inference.stream_handler import RealTimeStreamProcessor
            processor = RealTimeStreamProcessor(mock_config_manager)
            print("✓ Stream processor initialized")
            
            # Test frame processing
            encoded_frame = test_frame_encoding()
            result = processor.process_frame(encoded_frame, "test_camera", 1)
            
            print("✓ Frame processed successfully")
            print(f"  - Status: {result.get('status')}")
            print(f"  - Camera ID: {result.get('camera_id')}")
            print(f"  - Detections: {len(result.get('detections', []))}")
            print(f"  - Processing time: {result.get('processing_time_ms', 0):.1f}ms")
            
            # Verify result structure
            expected_fields = ['camera_id', 'timestamp', 'frame_sequence', 'status', 'detections']
            for field in expected_fields:
                if field not in result:
                    print(f"✗ Missing field: {field}")
                    return False
            
            print("✓ Result structure validated")
            return True
            
    except Exception as e:
        print(f"✗ Stream processor test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_sagemaker_handler():
    """Test SageMaker handler functionality."""
    print("\nTesting SageMaker handler...")
    
    try:
        from inference.sagemaker_handler import SageMakerInferenceHandler
        
        # Create handler
        handler = SageMakerInferenceHandler()
        print("✓ SageMaker handler created")
        
        # Test input parsing
        test_request = {
            "stream_data": {
                "frame": test_frame_encoding(),
                "camera_id": "test_camera",
                "frame_sequence": 1
            }
        }
        
        request_body = json.dumps(test_request)
        parsed_input = handler.input_fn(request_body, "application/json")
        
        print("✓ Input parsing successful")
        print(f"  - Parsed camera_id: {parsed_input['stream_data']['camera_id']}")
        
        # Test output formatting
        test_prediction = {
            "status": "success",
            "camera_id": "test_camera",
            "detections": [],
            "processing_time_ms": 50.0
        }
        
        formatted_output = handler.output_fn(test_prediction, "application/json")
        parsed_output = json.loads(formatted_output)
        
        print("✓ Output formatting successful")
        print(f"  - Output status: {parsed_output['status']}")
        
        return True
        
    except Exception as e:
        print(f"✗ SageMaker handler test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_frame_processor():
    """Test frame processor utilities."""
    print("\nTesting frame processor...")
    
    try:
        from inference.frame_processor import FrameProcessor, FrameQueue, ResultAggregator
        
        # Test frame processor
        processor = FrameProcessor(target_size=(320, 320))
        test_image = create_test_image()
        
        processed_frame, metadata = processor.preprocess_frame(test_image)
        print("✓ Frame preprocessing successful")
        print(f"  - Original shape: {metadata['original_shape']}")
        print(f"  - Processed shape: {metadata['processed_shape']}")
        print(f"  - Scale factors: {metadata['scale_factor_x']:.2f}, {metadata['scale_factor_y']:.2f}")
        
        # Test frame queue
        queue = FrameQueue(max_size=3)
        
        for i in range(5):  # Try to add more than capacity
            frame_data = {'frame_id': i, 'data': f'test_{i}'}
            success = queue.put_frame(frame_data, timeout=0.1)
            if not success:
                print(f"  - Frame {i} dropped (expected)")
        
        stats = queue.get_queue_stats()
        print("✓ Frame queue tested")
        print(f"  - Queue utilization: {stats['utilization']:.1%}")
        print(f"  - Drop rate: {stats['drop_rate']:.1%}")
        
        # Test result aggregator
        aggregator = ResultAggregator(window_size=5)
        
        for i in range(3):
            result = {
                'status': 'success',
                'total_chickens_detected': i + 1,
                'processing_time_ms': 50 + i * 10
            }
            aggregator.add_result(result)
        
        agg_stats = aggregator.get_aggregated_stats()
        print("✓ Result aggregator tested")
        print(f"  - Success rate: {agg_stats['success_rate']:.1%}")
        print(f"  - Avg processing time: {agg_stats['processing_time']['average_ms']:.1f}ms")
        
        return True
        
    except Exception as e:
        print(f"✗ Frame processor test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_system_test():
    """Run complete system test."""
    print("=" * 60)
    print("CHICKEN WEIGHT ESTIMATION SYSTEM - QUICK TEST")
    print("=" * 60)
    
    tests = [
        ("Frame Encoding/Decoding", test_frame_encoding),
        ("Stream Processor", test_stream_processor),
        ("SageMaker Handler", test_sagemaker_handler),
        ("Frame Processor", test_frame_processor),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_name == "Frame Encoding/Decoding":
                # This test returns the encoded frame, not a boolean
                test_func()
                results.append((test_name, True))
            else:
                success = test_func()
                results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {str(e)}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:<8} {test_name}")
        if success:
            passed += 1
    
    print("-" * 60)
    print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready for deployment.")
        return True
    else:
        print(f"\n❌ {total-passed} tests failed. Please check the issues above.")
        return False

if __name__ == '__main__':
    success = run_system_test()
    sys.exit(0 if success else 1)