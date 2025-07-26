# Chicken Weight Estimation SDK - Examples

This directory contains comprehensive examples demonstrating how to use the Chicken Weight Estimation SDK in various scenarios.

## Examples Overview

### 1. Basic Usage (`basic_usage.py`)
**Purpose**: Introduction to core SDK functionality
**Features**:
- Single image processing
- Video file processing
- Live stream processing
- Batch processing
- Image utilities
- Health checks

**Usage**:
```bash
python examples/basic_usage.py
```

### 2. Farm Integration (`farm_integration.py`)
**Purpose**: Complete farm management system integration
**Features**:
- Multi-camera monitoring
- Database storage
- Alert system
- Daily reports
- Data export
- Scheduled tasks

**Usage**:
```bash
python examples/farm_integration.py
```

### 3. Async Batch Processing (`async_batch_processing.py`)
**Purpose**: High-performance batch processing
**Features**:
- Asynchronous processing
- Multiple directory handling
- Performance benchmarking
- Error handling
- Throughput optimization

**Usage**:
```bash
python examples/async_batch_processing.py
```

## Prerequisites

### 1. SDK Installation
```bash
# Install the SDK (from project root)
pip install -e .

# Or install dependencies manually
pip install -r requirements.txt
```

### 2. SageMaker Endpoint
- Deploy the chicken weight estimation model to SageMaker
- Note the endpoint URL
- Obtain API key if authentication is enabled

### 3. Test Data
- Prepare test images of chickens
- Set up video files for testing
- Configure camera streams (for live examples)

## Configuration

### Environment Variables
```bash
export CHICKEN_WEIGHT_ENDPOINT_URL="https://your-endpoint.amazonaws.com/prod"
export CHICKEN_WEIGHT_API_KEY="your-api-key-here"
```

### Configuration Files
Most examples support configuration files:

**config.json**:
```json
{
  "client": {
    "endpoint_url": "https://your-endpoint.amazonaws.com/prod",
    "api_key": "your-api-key-here",
    "timeout": 30.0,
    "retry_attempts": 3
  }
}
```

## Example Scenarios

### Scenario 1: Research and Development
**Use Case**: Testing the system with sample images
**Recommended Example**: `basic_usage.py`
**Setup**:
1. Collect sample chicken images
2. Update endpoint URL in the script
3. Run single image processing examples

### Scenario 2: Small Farm Deployment
**Use Case**: Monitor 1-3 cameras continuously
**Recommended Example**: `farm_integration.py`
**Setup**:
1. Configure camera streams in `farm_config.json`
2. Set up database for data storage
3. Configure alert thresholds
4. Run continuous monitoring

### Scenario 3: Large-Scale Processing
**Use Case**: Process thousands of historical images
**Recommended Example**: `async_batch_processing.py`
**Setup**:
1. Organize images in directories
2. Configure optimal batch sizes
3. Run performance benchmarks
4. Process multiple directories concurrently

### Scenario 4: Production Farm Network
**Use Case**: Monitor multiple farms with many cameras
**Recommended Example**: Combination of all examples
**Setup**:
1. Deploy multiple SageMaker endpoints
2. Configure load balancing
3. Set up centralized monitoring
4. Implement data aggregation

## Performance Optimization

### Batch Processing Tips
1. **Optimal Batch Size**: Test different batch sizes (10-30 images)
2. **Concurrency**: Start with 20-50 concurrent requests
3. **Network**: Ensure sufficient bandwidth for image uploads
4. **Endpoint**: Use appropriate SageMaker instance types

### Memory Management
```python
# Process large datasets in chunks
async def process_large_dataset(image_paths, chunk_size=1000):
    for i in range(0, len(image_paths), chunk_size):
        chunk = image_paths[i:i + chunk_size]
        results = await process_batch(chunk)
        # Process results immediately to free memory
        save_results(results)
        del results  # Explicit cleanup
```

### Error Handling Best Practices
```python
# Implement retry logic with exponential backoff
async def robust_process_image(client, image_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await client.process_image_async(session, image_path)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

## Monitoring and Logging

### Enable Detailed Logging
```python
from sdk import setup_logging

# Enable debug logging
setup_logging(level='DEBUG', log_file='chicken_weight.log')
```

### Performance Metrics
```python
# Track processing metrics
metrics = {
    'total_images': 0,
    'successful_images': 0,
    'total_processing_time': 0,
    'average_fps': 0
}

# Update metrics after each batch
def update_metrics(batch_result):
    metrics['total_images'] += batch_result.total_frames
    metrics['successful_images'] += batch_result.successful_frames
    metrics['total_processing_time'] += batch_result.total_processing_time
    metrics['average_fps'] = metrics['successful_images'] / metrics['total_processing_time']
```

## Troubleshooting

### Common Issues

#### 1. Connection Errors
**Problem**: Cannot connect to SageMaker endpoint
**Solutions**:
- Verify endpoint URL is correct
- Check API key authentication
- Ensure endpoint is in "InService" status
- Test network connectivity

#### 2. Timeout Errors
**Problem**: Requests timing out
**Solutions**:
- Increase timeout value in client configuration
- Reduce batch size
- Check endpoint instance capacity
- Monitor CloudWatch metrics

#### 3. Memory Issues
**Problem**: Out of memory during batch processing
**Solutions**:
- Reduce batch size
- Process images in smaller chunks
- Implement explicit memory cleanup
- Use streaming processing for large datasets

#### 4. Low Throughput
**Problem**: Processing is slower than expected
**Solutions**:
- Increase concurrency level
- Optimize batch size
- Use async processing
- Consider multiple endpoints

### Debug Mode
```python
# Enable debug mode for detailed error information
import logging
logging.basicConfig(level=logging.DEBUG)

# Test endpoint connectivity
try:
    health = client.get_health_status()
    print(f"Endpoint health: {health}")
except Exception as e:
    print(f"Health check failed: {e}")
```

## Integration Patterns

### Pattern 1: Real-time Processing
```python
# Continuous processing with immediate results
def real_time_processor(camera_stream):
    for frame in camera_stream:
        result = client.process_image(frame)
        handle_result_immediately(result)
```

### Pattern 2: Batch Processing
```python
# Collect images and process in batches
async def batch_processor(image_queue):
    while True:
        batch = await collect_batch(image_queue, size=20)
        results = await process_batch_async(batch)
        store_results(results)
```

### Pattern 3: Hybrid Processing
```python
# Combine real-time and batch processing
class HybridProcessor:
    def __init__(self):
        self.priority_queue = asyncio.Queue()
        self.batch_queue = asyncio.Queue()
    
    async def process_priority(self):
        # Process high-priority images immediately
        while True:
            image = await self.priority_queue.get()
            result = await self.client.process_image_async(image)
            handle_priority_result(result)
    
    async def process_batch(self):
        # Process normal images in batches
        while True:
            batch = await collect_batch_from_queue(self.batch_queue)
            results = await self.client.process_batch_async(batch)
            handle_batch_results(results)
```

## Data Management

### Result Storage
```python
# Store results in different formats
class ResultStorage:
    def store_json(self, results, filename):
        with open(filename, 'w') as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
    
    def store_csv(self, results, filename):
        import pandas as pd
        data = []
        for result in results:
            for detection in result.detections:
                data.append({
                    'timestamp': result.timestamp,
                    'camera_id': result.camera_id,
                    'confidence': detection.confidence,
                    'weight': detection.weight_estimate,
                    'occlusion': detection.occlusion_score
                })
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
```

### Data Analysis
```python
# Analyze processing results
def analyze_results(results):
    total_detections = sum(r.detection_count for r in results)
    weights = [d.weight_estimate for r in results 
               for d in r.detections if d.weight_estimate]
    
    analysis = {
        'total_frames': len(results),
        'total_detections': total_detections,
        'average_detections_per_frame': total_detections / len(results),
        'weight_statistics': {
            'count': len(weights),
            'mean': np.mean(weights) if weights else 0,
            'std': np.std(weights) if weights else 0,
            'min': min(weights) if weights else 0,
            'max': max(weights) if weights else 0
        }
    }
    
    return analysis
```

## Next Steps

After running the examples:

1. **Customize for Your Use Case**
   - Modify processing parameters
   - Adapt alert thresholds
   - Customize data storage format

2. **Scale Your Deployment**
   - Deploy multiple endpoints
   - Implement load balancing
   - Set up monitoring dashboards

3. **Integrate with Existing Systems**
   - Connect to farm management software
   - Integrate with databases
   - Set up automated reporting

4. **Optimize Performance**
   - Benchmark different configurations
   - Monitor resource usage
   - Implement caching strategies

## Support

For additional help:
- Check the main SDK documentation
- Review the SageMaker deployment guide
- Monitor CloudWatch logs for errors
- Test with the health check endpoints

## Contributing

To contribute new examples:
1. Follow the existing code structure
2. Include comprehensive error handling
3. Add detailed documentation
4. Test with various scenarios
5. Submit a pull request with examples and documentation