#!/usr/bin/env python3
"""
Run comprehensive tests for the chicken weight estimation system.
"""

import sys
import os
import time
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def run_tests():
    """Run all comprehensive tests."""
    print("🐔 CHICKEN WEIGHT ESTIMATION SYSTEM - COMPREHENSIVE TESTS")
    print("=" * 70)
    
    # Import and run stream processing tests
    try:
        from tests.test_stream_processing_comprehensive import TestStreamProcessing, test_with_video_file
        
        print("\n📊 RUNNING STREAM PROCESSING TESTS")
        print("-" * 50)
        
        # Run main test suite
        test_case = TestStreamProcessing()
        success = test_case.run_all_tests()
        
        if success:
            print("\n✅ All stream processing tests passed!")
        else:
            print("\n❌ Some stream processing tests failed!")
            return False
        
        # Ask user if they want to run video tests
        print("\n🎥 VIDEO PROCESSING TEST")
        print("-" * 30)
        print("This will test with synthetic video frames and optionally webcam.")
        
        try:
            response = input("Run video test? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                test_with_video_file()
            else:
                print("Skipping video test.")
        except KeyboardInterrupt:
            print("\nSkipping video test.")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import test modules: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install -r docker/requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False


def check_dependencies():
    """Check if required dependencies are available."""
    print("🔍 CHECKING DEPENDENCIES")
    print("-" * 30)
    
    required_packages = [
        'numpy', 'opencv-python', 'torch', 'matplotlib', 
        'pandas', 'requests', 'flask'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("\n✅ All dependencies available!")
    return True


def run_quick_test():
    """Run a quick smoke test."""
    print("\n🚀 QUICK SMOKE TEST")
    print("-" * 25)
    
    try:
        # Test basic imports
        from src.inference.stream_handler import RealTimeStreamProcessor
        from src.utils.config.config_manager import ConfigManager
        print("✅ Core modules import successfully")
        
        # Test mock processor creation
        from tests.test_stream_processing_comprehensive import TestStreamProcessing
        test_case = TestStreamProcessing()
        processor = test_case.create_mock_processor()
        print("✅ Mock processor created successfully")
        
        # Test basic frame processing
        from tests.test_stream_processing_comprehensive import TestVideoGenerator
        import base64
        import cv2
        
        video_gen = TestVideoGenerator()
        frame = video_gen.generate_frame()
        frame_data = base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode('utf-8')
        
        result = processor.process_frame(frame_data, "test_camera", 0)
        
        if result['status'] == 'success':
            print("✅ Basic frame processing works")
            print(f"   - Detected {result['total_chickens_detected']} chickens")
            print(f"   - Processing time: {result['processing_time_ms']:.2f}ms")
        else:
            print("❌ Basic frame processing failed")
            return False
        
        print("\n🎉 Quick test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False


def main():
    """Main test runner."""
    start_time = time.time()
    
    print("Starting comprehensive test suite...")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install missing packages.")
        return 1
    
    # Run quick test first
    if not run_quick_test():
        print("\n❌ Quick test failed. Check your setup.")
        return 1
    
    # Ask user what tests to run
    print("\n" + "=" * 70)
    print("TEST OPTIONS")
    print("=" * 70)
    print("1. Quick test only (already completed)")
    print("2. Full comprehensive tests")
    print("3. Performance benchmarks")
    print("4. All tests")
    
    try:
        choice = input("\nSelect option (1-4) or press Enter for option 2: ").strip()
        if not choice:
            choice = "2"
    except KeyboardInterrupt:
        print("\nTest cancelled by user.")
        return 0
    
    success = True
    
    if choice in ["2", "4"]:
        # Run comprehensive tests
        success = run_tests()
    
    if choice in ["3", "4"]:
        # Run performance benchmarks
        print("\n📈 PERFORMANCE BENCHMARKS")
        print("-" * 35)
        print("Running extended performance tests...")
        
        try:
            from tests.test_stream_processing_comprehensive import TestStreamProcessing
            test_case = TestStreamProcessing()
            
            # Extended performance test
            processor = test_case.create_mock_processor()
            
            print("Processing 200 frames for performance analysis...")
            processing_times = []
            
            for i in range(200):
                from tests.test_stream_processing_comprehensive import TestVideoGenerator
                import base64
                import cv2
                
                video_gen = TestVideoGenerator()
                frame = video_gen.generate_frame()
                frame_data = base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode('utf-8')
                
                start = time.time()
                result = processor.process_frame(frame_data, "benchmark_camera", i)
                end = time.time()
                
                processing_times.append((end - start) * 1000)  # Convert to ms
                
                if i % 50 == 0:
                    print(f"  Progress: {i}/200 frames")
            
            # Calculate statistics
            import numpy as np
            avg_time = np.mean(processing_times)
            max_time = np.max(processing_times)
            min_time = np.min(processing_times)
            std_time = np.std(processing_times)
            
            print(f"\n📊 PERFORMANCE RESULTS:")
            print(f"   Average processing time: {avg_time:.2f}ms")
            print(f"   Maximum processing time: {max_time:.2f}ms")
            print(f"   Minimum processing time: {min_time:.2f}ms")
            print(f"   Standard deviation: {std_time:.2f}ms")
            print(f"   Average FPS: {1000/avg_time:.2f}")
            
            # Performance requirements check
            if avg_time < 100:
                print("✅ Meets latency requirement (<100ms)")
            else:
                print("❌ Exceeds latency requirement (>100ms)")
                success = False
            
            if 1000/avg_time > 10:
                print("✅ Meets throughput requirement (>10 FPS)")
            else:
                print("❌ Below throughput requirement (<10 FPS)")
                success = False
                
        except Exception as e:
            print(f"❌ Performance benchmark failed: {e}")
            success = False
    
    # Final summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Total test time: {total_time:.2f} seconds")
    
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("\nYour chicken weight estimation system is ready for deployment!")
        print("\nNext steps:")
        print("1. Build Docker container: docker build -t chicken-weight-estimator .")
        print("2. Deploy to SageMaker using deployment scripts")
        print("3. Test with real poultry farm footage")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        print("\nPlease review the test output and fix any issues before deployment.")
        return 1


if __name__ == "__main__":
    sys.exit(main())