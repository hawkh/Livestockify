"""
Test runner for the chicken weight estimation system.
"""

import unittest
import sys
import os
import time
from io import StringIO

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def run_all_tests():
    """Run all test suites."""
    print("=" * 70)
    print("CHICKEN WEIGHT ESTIMATION SYSTEM - TEST SUITE")
    print("=" * 70)
    print()
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Custom test result class for better output
    class DetailedTestResult(unittest.TextTestResult):
        def __init__(self, stream, descriptions, verbosity):
            super().__init__(stream, descriptions, verbosity)
            self.test_results = []
        
        def startTest(self, test):
            super().startTest(test)
            self.start_time = time.time()
        
        def stopTest(self, test):
            super().stopTest(test)
            duration = time.time() - self.start_time
            self.test_results.append({
                'test': str(test),
                'duration': duration,
                'status': 'PASS' if not self.failures and not self.errors else 'FAIL'
            })
    
    # Run tests with custom result class
    stream = StringIO()
    runner = unittest.TextTestRunner(
        stream=stream,
        verbosity=2,
        resultclass=DetailedTestResult
    )
    
    print("Running tests...")
    print("-" * 50)
    
    start_time = time.time()
    result = runner.run(suite)
    total_time = time.time() - start_time
    
    # Print results
    print("\nTEST RESULTS:")
    print("-" * 50)
    
    # Print individual test results
    if hasattr(result, 'test_results'):
        for test_result in result.test_results:
            status_symbol = "✓" if test_result['status'] == 'PASS' else "✗"
            print(f"{status_symbol} {test_result['test']} ({test_result['duration']:.3f}s)")
    
    print("-" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"Total time: {total_time:.3f}s")
    
    # Print failure details
    if result.failures:
        print("\nFAILURES:")
        print("-" * 50)
        for test, traceback in result.failures:
            print(f"FAIL: {test}")
            print(traceback)
            print()
    
    # Print error details
    if result.errors:
        print("\nERRORS:")
        print("-" * 50)
        for test, traceback in result.errors:
            print(f"ERROR: {test}")
            print(traceback)
            print()
    
    # Overall result
    if result.wasSuccessful():
        print("🎉 ALL TESTS PASSED!")
        return True
    else:
        print("❌ SOME TESTS FAILED!")
        return False


def run_specific_test(test_name):
    """Run a specific test module."""
    print(f"Running specific test: {test_name}")
    print("-" * 50)
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(test_name)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_integration_tests():
    """Run only integration tests."""
    print("Running integration tests...")
    print("-" * 50)
    
    # Import test modules
    from test_stream_processing import TestIntegration
    from test_sagemaker_integration import TestSageMakerCompatibility
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add integration tests
    suite.addTest(unittest.makeSuite(TestIntegration))
    suite.addTest(unittest.makeSuite(TestSageMakerCompatibility))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_unit_tests():
    """Run only unit tests."""
    print("Running unit tests...")
    print("-" * 50)
    
    # Import test modules
    from test_stream_processing import TestStreamProcessing
    from test_sagemaker_integration import TestSageMakerIntegration
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add unit tests
    suite.addTest(unittest.makeSuite(TestStreamProcessing))
    suite.addTest(unittest.makeSuite(TestSageMakerIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def check_dependencies():
    """Check if all required dependencies are available."""
    print("Checking dependencies...")
    print("-" * 30)
    
    required_packages = [
        'numpy', 'cv2', 'torch', 'flask', 'unittest'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'torch':
                import torch
            elif package == 'flask':
                import flask
            elif package == 'numpy':
                import numpy
            elif package == 'unittest':
                import unittest
            
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install missing packages before running tests.")
        return False
    
    print("\nAll dependencies available!")
    return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run tests for chicken weight estimation system')
    parser.add_argument('--test-type', choices=['all', 'unit', 'integration'], 
                       default='all', help='Type of tests to run')
    parser.add_argument('--specific', help='Run a specific test module')
    parser.add_argument('--check-deps', action='store_true', 
                       help='Check dependencies before running tests')
    
    args = parser.parse_args()
    
    # Check dependencies if requested
    if args.check_deps:
        if not check_dependencies():
            sys.exit(1)
        print()
    
    # Run specific test if requested
    if args.specific:
        success = run_specific_test(args.specific)
        sys.exit(0 if success else 1)
    
    # Run tests based on type
    if args.test_type == 'all':
        success = run_all_tests()
    elif args.test_type == 'unit':
        success = run_unit_tests()
    elif args.test_type == 'integration':
        success = run_integration_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)