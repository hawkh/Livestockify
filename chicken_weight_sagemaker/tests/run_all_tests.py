#!/usr/bin/env python3
"""
Comprehensive test runner for all chicken weight estimation components.
"""

import unittest
import sys
import os
import time
from pathlib import Path
import argparse
import json
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import all test modules
from tests.test_detection_comprehensive import *
from tests.test_weight_estimation_comprehensive import *
from tests.test_tracking_comprehensive import *
from tests.test_sagemaker_integration_comprehensive import *
from tests.test_stream_processing_comprehensive import *


class TestResult:
    """Custom test result class for detailed reporting."""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.end_time = None
    
    def add_result(self, test_name, result):
        """Add test result."""
        self.test_results[test_name] = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success_rate': ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0,
            'failure_details': [f"{test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}" 
                              for test, traceback in result.failures],
            'error_details': [f"{test}: {traceback.split('\\n')[-2]}" 
                            for test, traceback in result.errors]
        }
    
    def get_overall_stats(self):
        """Get overall test statistics."""
        total_tests = sum(r['tests_run'] for r in self.test_results.values())
        total_failures = sum(r['failures'] for r in self.test_results.values())
        total_errors = sum(r['errors'] for r in self.test_results.values())
        
        return {
            'total_tests': total_tests,
            'total_failures': total_failures,
            'total_errors': total_errors,
            'overall_success_rate': ((total_tests - total_failures - total_errors) / total_tests * 100) if total_tests > 0 else 0,
            'duration': (self.end_time - self.start_time) if self.start_time and self.end_time else 0
        }


def run_test_suite(test_suite_name, test_classes, verbosity=1):
    """Run a specific test suite."""
    print(f"\n{'='*60}")
    print(f"RUNNING {test_suite_name.upper()} TESTS")
    print(f"{'='*60}")
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test classes
    for test_class in test_classes:
        suite.addTest(unittest.makeSuite(test_class))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout)
    result = runner.run(suite)
    
    return result


def generate_html_report(test_result, output_file):
    """Generate HTML test report."""
    overall_stats = test_result.get_overall_stats()
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chicken Weight Estimation - Test Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .summary {{ background-color: #e8f5e8; padding: 15px; margin: 20px 0; border-radius: 5px; }}
            .failure {{ background-color: #ffe8e8; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            .success {{ background-color: #e8f5e8; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            .test-suite {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .pass {{ color: green; font-weight: bold; }}
            .fail {{ color: red; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🐔 Chicken Weight Estimation System - Test Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="summary">
            <h2>📊 Overall Summary</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Tests</td><td>{overall_stats['total_tests']}</td></tr>
                <tr><td>Passed</td><td class="pass">{overall_stats['total_tests'] - overall_stats['total_failures'] - overall_stats['total_errors']}</td></tr>
                <tr><td>Failed</td><td class="fail">{overall_stats['total_failures']}</td></tr>
                <tr><td>Errors</td><td class="fail">{overall_stats['total_errors']}</td></tr>
                <tr><td>Success Rate</td><td>{overall_stats['overall_success_rate']:.1f}%</td></tr>
                <tr><td>Duration</td><td>{overall_stats['duration']:.2f} seconds</td></tr>
            </table>
        </div>
    """
    
    # Add detailed results for each test suite
    for test_name, result in test_result.test_results.items():
        status_class = "success" if result['success_rate'] == 100 else "failure"
        
        html_content += f"""
        <div class="test-suite {status_class}">
            <h3>🧪 {test_name}</h3>
            <table>
                <tr><th>Tests Run</th><td>{result['tests_run']}</td></tr>
                <tr><th>Failures</th><td>{result['failures']}</td></tr>
                <tr><th>Errors</th><td>{result['errors']}</td></tr>
                <tr><th>Success Rate</th><td>{result['success_rate']:.1f}%</td></tr>
            </table>
        """
        
        if result['failure_details']:
            html_content += "<h4>❌ Failures:</h4><ul>"
            for failure in result['failure_details']:
                html_content += f"<li>{failure}</li>"
            html_content += "</ul>"
        
        if result['error_details']:
            html_content += "<h4>💥 Errors:</h4><ul>"
            for error in result['error_details']:
                html_content += f"<li>{error}</li>"
            html_content += "</ul>"
        
        html_content += "</div>"
    
    html_content += """
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"📄 HTML report generated: {output_file}")


def generate_json_report(test_result, output_file):
    """Generate JSON test report."""
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'overall_stats': test_result.get_overall_stats(),
        'test_suites': test_result.test_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"📄 JSON report generated: {output_file}")


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description='Run comprehensive tests for chicken weight estimation system')
    parser.add_argument('--suite', choices=['detection', 'weight', 'tracking', 'sagemaker', 'stream', 'all'], 
                       default='all', help='Test suite to run')
    parser.add_argument('--verbosity', type=int, choices=[0, 1, 2], default=1, 
                       help='Test output verbosity')
    parser.add_argument('--output-dir', default='test_reports', 
                       help='Output directory for test reports')
    parser.add_argument('--html-report', action='store_true', 
                       help='Generate HTML report')
    parser.add_argument('--json-report', action='store_true', 
                       help='Generate JSON report')
    parser.add_argument('--fail-fast', action='store_true', 
                       help='Stop on first failure')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize test result tracker
    test_result = TestResult()
    test_result.start_time = time.time()
    
    # Define test suites
    test_suites = {
        'detection': [
            TestYOLODetector,
            TestOcclusionRobustYOLO,
            TestTemporalConsistencyFilter,
            TestDetectionIntegration,
            TestDetectionErrorHandling
        ],
        'weight': [
            TestDistanceAdaptiveWeightNet,
            TestFeatureExtractor,
            TestAgeClassifier,
            TestWeightValidator,
            TestDistanceCompensationEngine,
            TestWeightEstimationIntegration
        ],
        'tracking': [
            TestKalmanFilter,
            TestReIDFeatureExtractor,
            TestDeepSORTTracker,
            TestChickenTracker,
            TestTrackingIntegration
        ],
        'sagemaker': [
            TestSageMakerHandler,
            TestStreamProcessingServer,
            TestRealTimeStreamProcessor,
            TestFrameProcessor,
            TestSageMakerIntegration
        ],
        'stream': [
            # Add stream processing tests if they exist
        ]
    }
    
    # Run selected test suites
    suites_to_run = [args.suite] if args.suite != 'all' else list(test_suites.keys())
    
    overall_success = True
    
    for suite_name in suites_to_run:
        if suite_name in test_suites and test_suites[suite_name]:
            try:
                result = run_test_suite(suite_name, test_suites[suite_name], args.verbosity)
                test_result.add_result(suite_name, result)
                
                # Check if we should fail fast
                if args.fail_fast and (result.failures or result.errors):
                    print(f"\n❌ Failing fast due to failures in {suite_name} tests")
                    overall_success = False
                    break
                    
                if result.failures or result.errors:
                    overall_success = False
                    
            except Exception as e:
                print(f"\n💥 Error running {suite_name} tests: {str(e)}")
                overall_success = False
                if args.fail_fast:
                    break
    
    test_result.end_time = time.time()
    
    # Print final summary
    overall_stats = test_result.get_overall_stats()
    
    print(f"\n{'='*80}")
    print("🎯 FINAL TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total Tests: {overall_stats['total_tests']}")
    print(f"Passed: {overall_stats['total_tests'] - overall_stats['total_failures'] - overall_stats['total_errors']}")
    print(f"Failed: {overall_stats['total_failures']}")
    print(f"Errors: {overall_stats['total_errors']}")
    print(f"Success Rate: {overall_stats['overall_success_rate']:.1f}%")
    print(f"Duration: {overall_stats['duration']:.2f} seconds")
    
    # Generate reports
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if args.html_report:
        html_file = output_dir / f"test_report_{timestamp}.html"
        generate_html_report(test_result, html_file)
    
    if args.json_report:
        json_file = output_dir / f"test_report_{timestamp}.json"
        generate_json_report(test_result, json_file)
    
    # Print component-specific summaries
    print(f"\n📋 Component Test Results:")
    for suite_name, result in test_result.test_results.items():
        status = "✅ PASS" if result['success_rate'] == 100 else "❌ FAIL"
        print(f"  {suite_name.capitalize()}: {status} ({result['success_rate']:.1f}%)")
    
    # Exit with appropriate code
    if overall_success:
        print(f"\n🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print(f"\n💥 Some tests failed. Check the output above for details.")
        sys.exit(1)


if __name__ == '__main__':
    main()