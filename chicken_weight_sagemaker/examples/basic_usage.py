#!/usr/bin/env python3
"""
Basic usage examples for the Chicken Weight Estimation SDK
"""

import sys
from pathlib import Path
import logging

# Add SDK to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sdk import ChickenWeightClient, setup_logging
from sdk.utils import ImageProcessor, VideoProcessor, BatchProcessor


def example_single_image():
    """Example: Process a single image"""
    print("🖼️  Example: Processing a single image")
    print("=" * 50)
    
    # Initialize client
    client = ChickenWeightClient(
        endpoint_url="https://your-endpoint.amazonaws.com/prod",
        api_key="your-api-key-here",
        timeout=30.0
    )
    
    # Process image
    try:
        result = client.process_image(
            image="path/to/your/chicken_image.jpg",
            camera_id="farm_camera_01"
        )
        
        print(f"✅ Processing successful!")
        print(f"   Frame ID: {result.frame_id}")
        print(f"   Detections: {result.detection_count}")
        print(f"   Processing time: {result.processing_time:.3f}s")
        print(f"   FPS: {result.fps:.1f}")
        
        # Print detection details
        for i, detection in enumerate(result.detections):
            print(f"   Detection {i+1}:")
            print(f"     Confidence: {detection.confidence:.2f}")
            print(f"     Weight: {detection.weight_estimate:.2f}kg" if detection.weight_estimate else "     Weight: Not estimated")
            print(f"     Occlusion: {detection.occlusion_score:.2f}")
        
        # Print tracking info
        for track in result.tracks:
            print(f"   Track {track.track_id}:")
            print(f"     Average weight: {track.average_weight:.2f}kg" if track.average_weight else "     Average weight: Not available")
            print(f"     Weight trend: {track.weight_trend}")
        
    except Exception as e:
        print(f"❌ Error processing image: {str(e)}")


def example_video_processing():
    """Example: Process a video file"""
    print("\n🎥 Example: Processing a video file")
    print("=" * 50)
    
    # Initialize client
    client = ChickenWeightClient(
        endpoint_url="https://your-endpoint.amazonaws.com/prod",
        api_key="your-api-key-here"
    )
    
    def progress_callback(current, total):
        """Progress callback for video processing"""
        percentage = (current / total) * 100
        print(f"   Progress: {current}/{total} frames ({percentage:.1f}%)")
    
    def frame_callback(result):
        """Callback for each processed frame"""
        if result.detection_count > 0:
            avg_weight = result.average_weight
            print(f"   Frame {result.frame_id}: {result.detection_count} chickens" + 
                  (f", avg weight: {avg_weight:.2f}kg" if avg_weight else ""))
    
    try:
        results = client.process_video(
            video_path="path/to/your/chicken_video.mp4",
            camera_id="farm_camera_01",
            callback=frame_callback,
            skip_frames=5,  # Process every 6th frame
            max_frames=100,  # Limit to first 100 frames
            progress_callback=progress_callback
        )
        
        print(f"\n✅ Video processing complete!")
        print(f"   Total frames processed: {len(results)}")
        
        # Calculate summary statistics
        total_detections = sum(r.detection_count for r in results)
        avg_fps = sum(r.fps for r in results) / len(results) if results else 0
        
        print(f"   Total detections: {total_detections}")
        print(f"   Average processing FPS: {avg_fps:.1f}")
        
        # Weight statistics
        all_weights = []
        for result in results:
            for detection in result.detections:
                if detection.weight_estimate:
                    all_weights.append(detection.weight_estimate)
        
        if all_weights:
            print(f"   Weight statistics:")
            print(f"     Count: {len(all_weights)}")
            print(f"     Average: {sum(all_weights)/len(all_weights):.2f}kg")
            print(f"     Min: {min(all_weights):.2f}kg")
            print(f"     Max: {max(all_weights):.2f}kg")
        
    except Exception as e:
        print(f"❌ Error processing video: {str(e)}")


def example_live_stream():
    """Example: Process a live stream"""
    print("\n📡 Example: Processing a live stream")
    print("=" * 50)
    
    # Initialize client
    client = ChickenWeightClient(
        endpoint_url="https://your-endpoint.amazonaws.com/prod",
        api_key="your-api-key-here"
    )
    
    def stream_callback(result):
        """Callback for each processed frame"""
        timestamp = result.timestamp
        detections = result.detection_count
        avg_weight = result.average_weight
        
        print(f"   [{timestamp}] {detections} chickens detected" + 
              (f", avg weight: {avg_weight:.2f}kg" if avg_weight else ""))
    
    try:
        print("   Starting live stream processing (30 seconds)...")
        
        results = client.process_live_stream(
            stream_url="rtsp://your-camera-ip:554/stream",
            camera_id="live_camera_01",
            duration_seconds=30,
            callback=stream_callback
        )
        
        print(f"\n✅ Live stream processing complete!")
        print(f"   Frames processed: {len(results)}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Stream processing stopped by user")
    except Exception as e:
        print(f"❌ Error processing stream: {str(e)}")


def example_batch_processing():
    """Example: Batch process multiple images"""
    print("\n📁 Example: Batch processing images")
    print("=" * 50)
    
    # Initialize client and batch processor
    client = ChickenWeightClient(
        endpoint_url="https://your-endpoint.amazonaws.com/prod",
        api_key="your-api-key-here"
    )
    
    batch_processor = BatchProcessor(client)
    
    def progress_callback(current, total):
        """Progress callback"""
        percentage = (current / total) * 100
        print(f"   Progress: {current}/{total} images ({percentage:.1f}%)")
    
    try:
        # Process all images in directory
        batch_result = batch_processor.process_image_directory(
            directory_path="path/to/your/image_directory",
            camera_id="batch_processing",
            file_pattern="*.jpg",
            progress_callback=progress_callback
        )
        
        print(f"\n✅ Batch processing complete!")
        print(f"   Total images: {batch_result.total_frames}")
        print(f"   Successful: {batch_result.successful_frames}")
        print(f"   Failed: {batch_result.failed_frames}")
        print(f"   Success rate: {batch_result.success_rate:.1f}%")
        print(f"   Average FPS: {batch_result.average_fps:.1f}")
        print(f"   Total detections: {batch_result.total_detections}")
        
        # Weight statistics
        weight_stats = batch_result.get_weight_statistics()
        if weight_stats['count'] > 0:
            print(f"   Weight statistics:")
            print(f"     Count: {weight_stats['count']}")
            print(f"     Mean: {weight_stats['mean']:.2f}kg")
            print(f"     Min: {weight_stats['min']:.2f}kg")
            print(f"     Max: {weight_stats['max']:.2f}kg")
            print(f"     Std Dev: {weight_stats['std']:.2f}kg")
        
        # Generate report
        report_path = batch_processor.generate_batch_report(
            batch_result, 
            "batch_processing_report.json"
        )
        print(f"   Report saved to: {report_path}")
        
    except Exception as e:
        print(f"❌ Error in batch processing: {str(e)}")


def example_image_utilities():
    """Example: Using image processing utilities"""
    print("\n🛠️  Example: Image processing utilities")
    print("=" * 50)
    
    try:
        import cv2
        import numpy as np
        
        # Load test image
        image = cv2.imread("path/to/your/test_image.jpg")
        if image is None:
            print("   ⚠️  Could not load test image, creating synthetic image")
            image = np.ones((480, 640, 3), dtype=np.uint8) * 128
            cv2.rectangle(image, (100, 100), (200, 200), (255, 255, 255), -1)
        
        print(f"   Original image shape: {image.shape}")
        
        # Resize image
        resized = ImageProcessor.resize_image(image, target_size=(640, 640))
        print(f"   Resized image shape: {resized.shape}")
        
        # Enhance image
        enhanced = ImageProcessor.enhance_image(image, brightness=1.2, contrast=1.1)
        print(f"   Enhanced image created")
        
        # Create mock detection for cropping
        mock_detection_bbox = {
            'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200, 'confidence': 0.9
        }
        
        cropped = ImageProcessor.crop_detection(image, mock_detection_bbox, padding=20)
        print(f"   Cropped detection shape: {cropped.shape}")
        
        # Save processed images
        cv2.imwrite("resized_image.jpg", resized)
        cv2.imwrite("enhanced_image.jpg", enhanced)
        cv2.imwrite("cropped_detection.jpg", cropped)
        
        print("   ✅ Processed images saved")
        
    except Exception as e:
        print(f"   ❌ Error in image processing: {str(e)}")


def example_health_check():
    """Example: Check endpoint health"""
    print("\n🏥 Example: Endpoint health check")
    print("=" * 50)
    
    # Initialize client
    client = ChickenWeightClient(
        endpoint_url="https://your-endpoint.amazonaws.com/prod",
        api_key="your-api-key-here"
    )
    
    try:
        health_status = client.get_health_status()
        
        print("   ✅ Health check successful!")
        print(f"   Status: {health_status.get('status', 'Unknown')}")
        print(f"   Timestamp: {health_status.get('timestamp', 'Unknown')}")
        print(f"   Model loaded: {health_status.get('model_loaded', 'Unknown')}")
        
        if 'version' in health_status:
            print(f"   Version: {health_status['version']}")
        
        if 'uptime' in health_status:
            print(f"   Uptime: {health_status['uptime']}")
        
    except Exception as e:
        print(f"   ❌ Health check failed: {str(e)}")


def main():
    """Run all examples"""
    print("🐔 Chicken Weight Estimation SDK - Usage Examples")
    print("=" * 60)
    
    # Setup logging
    setup_logging(level='INFO')
    
    # Run examples
    try:
        example_health_check()
        example_single_image()
        example_video_processing()
        example_live_stream()
        example_batch_processing()
        example_image_utilities()
        
    except KeyboardInterrupt:
        print("\n⏹️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
    
    print("\n🎉 Examples completed!")
    print("\nNext steps:")
    print("1. Replace endpoint URL and API key with your actual values")
    print("2. Update image/video paths to point to your test files")
    print("3. Customize processing parameters for your use case")
    print("4. Integrate the SDK into your farm management system")


if __name__ == "__main__":
    main()