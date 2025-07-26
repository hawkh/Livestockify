# Comprehensive Testing Suite

This directory contains a comprehensive testing suite for the chicken weight estimation system, covering all major components and integration scenarios.

## Test Structure

```
tests/
├── test_detection_comprehensive.py           # Detection component tests
├── test_weight_estimation_comprehensive.py   # Weight estimation tests
├── test_tracking_comprehensive.py            # Tracking component tests
├── test_sagemaker_integration_comprehensive.py # SageMaker integration tests
├── test_stream_processing_comprehensive.py   # Stream processing tests
├── run_all_tests.py                          # Main test runner
└── README.md                                 # This file
```

## Test Categories

### 1. Detection Tests (`test_detection_comprehensive.py`)
- **YOLODetector**: Basic detection functionality, confidence filtering, error handling
- **OcclusionRobustYOLO**: Multi-scale detection, occlusion score calculation
- **TemporalConsistencyFilter**: Temporal filtering, consistency tracking
- **DetectionIntegration**: Full detection pipeline, performance requirements
- **DetectionErrorHandling**: Invalid input handling, model loading errors

### 2. Weight Estimation Tests (`test_weight_estimation_comprehensive.py`)
- **DistanceAdaptiveWeightNet**: Neural network functionality, distance adaptation
- **FeatureExtractor**: Feature extraction from images, backbone architectures
- **AgeClassifier**: Age classification, confidence scores
- **WeightValidator**: Age-based validation, distance/occlusion confidence
- **DistanceCompensationEngine**: Distance estimation, feature compensation
- **WeightEstimationIntegration**: Full pipeline, batch processing, performance

### 3. Tracking Tests (`test_tracking_comprehensive.py`)
- **KalmanFilter**: State prediction, measurement updates, noise handling
- **ReIDFeatureExtractor**: Feature extraction, similarity calculation
- **DeepSORTTracker**: Track association, confirmation, deletion
- **ChickenTracker**: Weight tracking, growth monitoring, health assessment
- **TrackingIntegration**: Full pipeline, occlusion handling, performance

### 4. SageMaker Integration Tests (`test_sagemaker_integration_comprehensive.py`)
- **SageMakerHandler**: Model loading, input/output functions, health checks
- **StreamProcessingServer**: Server initialization, endpoint structure
- **RealTimeStreamProcessor**: Frame processing, stream data handling
- **FrameProcessor**: Single frame processing, preprocessing/postprocessing
- **SageMakerIntegration**: Full inference pipeline, error propagation

### 5. Stream Processing Tests (`test_stream_processing_comprehensive.py`)
- Real-time processing capabilities
- Concurrent stream handling
- Performance under load
- Error recovery mechanisms

## Running Tests

### Quick Start

```bash
# Run all tests
python tests/run_all_tests.py

# Run specific test suite
python tests/run_all_tests.py --suite detection

# Run with verbose output
python tests/run_all_tests.py --verbosity 2

# Generate HTML report
python tests/run_all_tests.py --html-report --output-dir reports
```

### Individual Test Files

```bash
# Run detection tests only
python -m pytest tests/test_detection_comprehensive.py -v

# Run weight estimation tests
python -m pytest tests/test_weight_estimation_comprehensive.py -v

# Run tracking tests
python -m pytest tests/test_tracking_comprehensive.py -v

# Run SageMaker integration tests
python -m pytest tests/test_sagemaker_integration_comprehensive.py -v
```

### Test Runner Options

```bash
python tests/run_all_tests.py --help
```

**Available options:**
- `--suite`: Choose specific test suite (detection, weight, tracking, sagemaker, stream, all)
- `--verbosity`: Set output verbosity (0=quiet, 1=normal, 2=verbose)
- `--output-dir`: Directory for test reports
- `--html-report`: Generate HTML test report
- `--json-report`: Generate JSON test report
- `--fail-fast`: Stop on first failure

## Test Reports

### HTML Report
Generates a comprehensive HTML report with:
- Overall test statistics
- Component-wise results
- Detailed failure and error information
- Visual indicators for pass/fail status

### JSON Report
Generates machine-readable JSON report with:
- Timestamp and metadata
- Overall statistics
- Detailed test suite results
- Failure and error details

## Test Coverage

### Component Coverage
- ✅ **Detection**: YOLO detector, occlusion handling, temporal consistency
- ✅ **Weight Estimation**: Neural networks, feature extraction, validation
- ✅ **Tracking**: Kalman filtering, DeepSORT, chicken-specific tracking
- ✅ **SageMaker Integration**: Handlers, processors, inference pipeline
- ✅ **Stream Processing**: Real-time processing, concurrent handling

### Test Types
- **Unit Tests**: Individual component functionality
- **Integration Tests**: Component interaction and data flow
- **Performance Tests**: Speed and resource usage requirements
- **Error Handling Tests**: Graceful failure and recovery
- **End-to-End Tests**: Complete pipeline functionality

## Performance Requirements

The test suite validates these performance requirements:

### Detection
- Frame processing: < 100ms per frame
- Multi-scale detection: < 200ms per frame
- Temporal consistency: < 10ms overhead

### Weight Estimation
- Single inference: < 10ms per sample
- Batch processing: < 5ms per sample (batch of 10)
- Feature extraction: < 20ms per image

### Tracking
- Track update: < 50ms per frame (50 detections)
- Association: < 20ms per frame
- Memory management: < 100 active tracks

### SageMaker Integration
- Full pipeline: < 100ms per request
- Input parsing: < 5ms
- Output formatting: < 5ms

## Mock Strategy

Tests use comprehensive mocking to:
- **Isolate components**: Test individual functionality without dependencies
- **Control inputs**: Provide predictable test data
- **Simulate errors**: Test error handling paths
- **Performance testing**: Remove I/O bottlenecks

### Mock Patterns
```python
# Mock external dependencies
@patch('ultralytics.YOLO')
def test_detector(self, mock_yolo):
    mock_model = Mock()
    mock_yolo.return_value = mock_model
    # Test implementation

# Mock with side effects
mock_model.side_effect = Exception("Model loading failed")

# Mock return values
mock_model.return_value = expected_result
```

## Test Data

### Synthetic Test Data
- **Images**: Generated test frames with known objects
- **Detections**: Mock detection results with controlled properties
- **Sequences**: Simulated video sequences for temporal testing

### Test Scenarios
- **Normal operation**: Typical use cases with expected inputs
- **Edge cases**: Boundary conditions and unusual inputs
- **Error conditions**: Invalid inputs and system failures
- **Performance stress**: High load and concurrent processing

## Continuous Integration

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run tests before commit
pre-commit run --all-files
```

### CI/CD Pipeline
The test suite integrates with CI/CD systems:

```yaml
# Example GitHub Actions workflow
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.10
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: python tests/run_all_tests.py --json-report
      - name: Upload test results
        uses: actions/upload-artifact@v2
        with:
          name: test-results
          path: test_reports/
```

## Debugging Tests

### Running Individual Tests
```bash
# Run specific test class
python -m pytest tests/test_detection_comprehensive.py::TestYOLODetector -v

# Run specific test method
python -m pytest tests/test_detection_comprehensive.py::TestYOLODetector::test_basic_detection -v

# Run with debugging
python -m pytest tests/test_detection_comprehensive.py --pdb
```

### Test Output
```bash
# Capture print statements
python -m pytest tests/test_detection_comprehensive.py -s

# Show local variables on failure
python -m pytest tests/test_detection_comprehensive.py -l

# Show full diff on assertion failures
python -m pytest tests/test_detection_comprehensive.py --tb=long
```

## Adding New Tests

### Test Structure Template
```python
import unittest
from unittest.mock import Mock, patch

class TestNewComponent(unittest.TestCase):
    """Test new component functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.component = NewComponent()
        self.test_data = create_test_data()
    
    def test_basic_functionality(self):
        """Test basic component functionality."""
        result = self.component.process(self.test_data)
        self.assertIsNotNone(result)
        self.assertEqual(result.status, 'success')
    
    def test_error_handling(self):
        """Test error handling."""
        with self.assertRaises(ValueError):
            self.component.process(invalid_data)
    
    def tearDown(self):
        """Clean up test fixtures."""
        cleanup_test_data()
```

### Integration with Test Runner
1. Add test class to appropriate test file
2. Import test class in `run_all_tests.py`
3. Add to relevant test suite in `test_suites` dictionary
4. Update documentation

## Best Practices

### Test Design
- **Isolation**: Each test should be independent
- **Repeatability**: Tests should produce consistent results
- **Clarity**: Test names should describe what is being tested
- **Coverage**: Test both success and failure paths

### Mock Usage
- **Minimal mocking**: Only mock external dependencies
- **Realistic mocks**: Mock behavior should match real components
- **Verification**: Assert that mocks were called correctly

### Performance Testing
- **Realistic data**: Use representative test data sizes
- **Multiple runs**: Average results over multiple iterations
- **Resource monitoring**: Check memory usage and cleanup

### Error Testing
- **Expected errors**: Test known error conditions
- **Graceful degradation**: Verify system continues operating
- **Error messages**: Check error messages are informative

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Add src to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or use relative imports
python -m tests.run_all_tests
```

**Mock Issues**
```python
# Patch at the right location
@patch('module.where.used.Class')  # Not where defined
def test_function(self, mock_class):
    pass
```

**Async Testing**
```python
import asyncio

class TestAsync(unittest.TestCase):
    def test_async_function(self):
        async def run_test():
            result = await async_function()
            self.assertIsNotNone(result)
        
        asyncio.run(run_test())
```

### Getting Help

1. **Check test output**: Look for specific error messages
2. **Run individual tests**: Isolate failing components
3. **Check mocks**: Verify mock setup and expectations
4. **Review logs**: Check application logs for additional context
5. **Debug mode**: Use `--pdb` flag for interactive debugging

## Contributing

When adding new features:

1. **Write tests first**: Follow TDD principles
2. **Test all paths**: Cover success, failure, and edge cases
3. **Update documentation**: Keep README and comments current
4. **Run full suite**: Ensure no regressions
5. **Check coverage**: Aim for >90% test coverage

### Test Coverage Report
```bash
# Install coverage tool
pip install coverage

# Run tests with coverage
coverage run -m pytest tests/

# Generate report
coverage report -m

# Generate HTML report
coverage html
```

This comprehensive testing suite ensures the reliability, performance, and maintainability of the chicken weight estimation system across all components and deployment scenarios.