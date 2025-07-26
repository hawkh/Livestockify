#!/usr/bin/env python3
"""
Performance testing suite for chicken weight estimation system.
"""

import asyncio
import aiohttp
import time
import json
import base64
import argparse
import logging
import statistics
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import cv2
import boto3
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class PerformanceResult:
    """Container for performance test results."""
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    average_response_time: float
    min_response_time: float
    max_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    errors: List[str]
    timestamp: str


@dataclass
class LoadTestConfig:
    """Configuration for load testing."""
    endpoint_url: str
    concurrent_users: int
    duration_seconds: int
    ramp_up_time: int
    test_data_size: int
    request_timeout: int
    think_time: float


class TestDataGenerator:
    """Generates test data for performance testing."""
    
    def __init__(self):
        self.test_images = []
        self._generate_test_images()
    
    def _generate_test_images(self, count: int = 10):
        """Generate test images for performance testing."""
        logger.info(f"Generating {count} test images...")
        
        for i in range(count):
            # Create varied test images
            image = self._create_test_image(
                width=640 + (i * 10),
                height=480 + (i * 10),
                chicken_count=2 + (i % 4)
            )
            
            # Encode as base64
            _, buffer = cv2.imencode('.jpg', image)
            base64_data = base64.b64encode(buffer).decode('utf-8')
            
            self.test_images.append({
                'frame': base64_data,
                'camera_id': f'test_camera_{i:02d}',
                'frame_sequence': i,
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            })
        
        logger.info(f"Generated {len(self.test_images)} test images")
    
    def _create_test_image(self, width: int, height: int, chicken_count: int) -> np.ndarray:
        """Create a test image with specified parameters."""
        # Create base image
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:] = [101, 67, 33]  # Brown background
        
        # Add chickens
        for i in range(chicken_count):
            center_x = np.random.randint(50, width - 50)
            center_y = np.random.randint(50, height - 50)
            size_x = np.random.randint(30, 70)
            size_y = np.random.randint(20, 50)
            color = (
                np.random.randint(150, 220),
                np.random.randint(130, 200),
                np.random.randint(120, 180)
            )
            
            cv2.ellipse(image, (center_x, center_y), (size_x, size_y), 0, 0, 360, color, -1)
        
        # Add noise
        noise = np.random.randint(0, 30, image.shape, dtype=np.uint8)
        image = cv2.add(image, noise)
        
        return image
    
    def get_random_test_data(self) -> Dict[str, Any]:
        """Get random test data."""
        test_image = np.random.choice(self.test_images)
        return {
            'stream_data': {
                **test_image,
                'frame_sequence': int(time.time() * 1000) % 1000000,
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'parameters': {
                    'min_confidence': 0.4,
                    'max_occlusion': 0.7
                }
            }
        }


class PerformanceTester:
    """Main performance testing class."""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.data_generator = TestDataGenerator()
        self.results = []
        self.session = None
    
    async def run_load_test(self) -> PerformanceResult:
        """Run a comprehensive load test."""
        logger.info(f"Starting load test with {self.config.concurrent_users} concurrent users")
        logger.info(f"Duration: {self.config.duration_seconds}s, Ramp-up: {self.config.ramp_up_time}s")
        
        start_time = time.time()
        
        # Create aiohttp session
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            self.session = session
            
            # Run the load test
            tasks = []
            
            # Gradual ramp-up
            ramp_up_delay = self.config.ramp_up_time / self.config.concurrent_users
            
            for i in range(self.config.concurrent_users):
                task = asyncio.create_task(
                    self._user_simulation(i, ramp_up_delay * i)
                )
                tasks.append(task)
            
            # Wait for all tasks to complete or timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self.config.duration_seconds + self.config.ramp_up_time + 60
                )
            except asyncio.TimeoutError:
                logger.warning("Load test timed out, cancelling remaining tasks")
                for task in tasks:
                    task.cancel()
        
        total_time = time.time() - start_time
        
        # Analyze results
        return self._analyze_results("Load Test", total_time)
    
    async def _user_simulation(self, user_id: int, initial_delay: float):
        """Simulate a single user's behavior."""
        # Initial delay for ramp-up
        await asyncio.sleep(initial_delay)
        
        user_start_time = time.time()
        user_results = []
        
        while (time.time() - user_start_time) < self.config.duration_seconds:
            try:
                # Make request
                start_time = time.time()
                test_data = self.data_generator.get_random_test_data()
                
                async with self.session.post(
                    self.config.endpoint_url,
                    json=test_data,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        result_data = await response.json()
                        user_results.append({
                            'success': True,
                            'response_time': response_time,
                            'status_code': response.status,
                            'user_id': user_id
                        })
                    else:
                        error_text = await response.text()
                        user_results.append({
                            'success': False,
                            'response_time': response_time,
                            'status_code': response.status,
                            'error': f"HTTP {response.status}: {error_text}",
                            'user_id': user_id
                        })
                
                # Think time between requests
                if self.config.think_time > 0:
                    await asyncio.sleep(self.config.think_time)
                
            except Exception as e:
                user_results.append({
                    'success': False,
                    'response_time': 0,
                    'status_code': 0,
                    'error': str(e),
                    'user_id': user_id
                })
        
        # Store user results
        self.results.extend(user_results)
        logger.debug(f"User {user_id} completed {len(user_results)} requests")
    
    def run_stress_test(self) -> PerformanceResult:
        """Run a stress test to find breaking point."""
        logger.info("Starting stress test to find system limits")
        
        start_time = time.time()
        max_concurrent = 1
        step_size = 5
        max_response_time_threshold = 5.0  # 5 seconds
        error_rate_threshold = 0.1  # 10% error rate
        
        while max_concurrent <= 100:  # Safety limit
            logger.info(f"Testing with {max_concurrent} concurrent users")
            
            # Run test with current concurrency level
            test_config = LoadTestConfig(
                endpoint_url=self.config.endpoint_url,
                concurrent_users=max_concurrent,
                duration_seconds=60,  # 1 minute per test
                ramp_up_time=10,
                test_data_size=self.config.test_data_size,
                request_timeout=self.config.request_timeout,
                think_time=0.1
            )
            
            # Run synchronous test for stress testing
            result = self._run_sync_test(test_config)
            
            # Check if we've hit the breaking point
            error_rate = result.failed_requests / result.total_requests if result.total_requests > 0 else 1
            
            if (result.average_response_time > max_response_time_threshold or 
                error_rate > error_rate_threshold):
                logger.info(f"Breaking point found at {max_concurrent} concurrent users")
                break
            
            max_concurrent += step_size
        
        total_time = time.time() - start_time
        return self._analyze_results("Stress Test", total_time)
    
    def _run_sync_test(self, config: LoadTestConfig) -> PerformanceResult:
        """Run a synchronous test for stress testing."""
        results = []
        
        def make_request():
            try:
                import requests
                test_data = self.data_generator.get_random_test_data()
                start_time = time.time()
                
                response = requests.post(
                    config.endpoint_url,
                    json=test_data,
                    timeout=config.request_timeout
                )
                
                response_time = time.time() - start_time
                
                return {
                    'success': response.status_code == 200,
                    'response_time': response_time,
                    'status_code': response.status_code,
                    'error': None if response.status_code == 200 else f"HTTP {response.status_code}"
                }
            except Exception as e:
                return {
                    'success': False,
                    'response_time': 0,
                    'status_code': 0,
                    'error': str(e)
                }
        
        # Run concurrent requests
        with ThreadPoolExecutor(max_workers=config.concurrent_users) as executor:
            futures = []
            
            # Submit requests for the duration
            start_time = time.time()
            while (time.time() - start_time) < config.duration_seconds:
                for _ in range(config.concurrent_users):
                    future = executor.submit(make_request)
                    futures.append(future)
                
                time.sleep(1)  # Submit batch every second
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=config.request_timeout + 5)
                    results.append(result)
                except Exception as e:
                    results.append({
                        'success': False,
                        'response_time': 0,
                        'status_code': 0,
                        'error': str(e)
                    })
        
        self.results = results
        return self._analyze_results("Sync Test", time.time() - start_time)
    
    def run_spike_test(self) -> PerformanceResult:
        """Run a spike test with sudden load increases."""
        logger.info("Starting spike test")
        
        start_time = time.time()
        
        # Normal load phase
        logger.info("Phase 1: Normal load (10 users)")
        normal_config = LoadTestConfig(
            endpoint_url=self.config.endpoint_url,
            concurrent_users=10,
            duration_seconds=60,
            ramp_up_time=5,
            test_data_size=self.config.test_data_size,
            request_timeout=self.config.request_timeout,
            think_time=0.5
        )
        self._run_sync_test(normal_config)
        
        # Spike phase
        logger.info("Phase 2: Spike load (100 users)")
        spike_config = LoadTestConfig(
            endpoint_url=self.config.endpoint_url,
            concurrent_users=100,
            duration_seconds=30,
            ramp_up_time=2,
            test_data_size=self.config.test_data_size,
            request_timeout=self.config.request_timeout,
            think_time=0.1
        )
        spike_results = self._run_sync_test(spike_config)
        self.results.extend(spike_results.errors)
        
        # Recovery phase
        logger.info("Phase 3: Recovery (10 users)")
        self._run_sync_test(normal_config)
        
        total_time = time.time() - start_time
        return self._analyze_results("Spike Test", total_time)
    
    def run_endurance_test(self) -> PerformanceResult:
        """Run an endurance test for extended periods."""
        logger.info("Starting endurance test (30 minutes)")
        
        endurance_config = LoadTestConfig(
            endpoint_url=self.config.endpoint_url,
            concurrent_users=20,
            duration_seconds=1800,  # 30 minutes
            ramp_up_time=60,
            test_data_size=self.config.test_data_size,
            request_timeout=self.config.request_timeout,
            think_time=1.0
        )
        
        return asyncio.run(self._run_endurance_test(endurance_config))
    
    async def _run_endurance_test(self, config: LoadTestConfig) -> PerformanceResult:
        """Run the actual endurance test."""
        start_time = time.time()
        
        timeout = aiohttp.ClientTimeout(total=config.request_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            self.session = session
            
            tasks = []
            for i in range(config.concurrent_users):
                task = asyncio.create_task(
                    self._endurance_user_simulation(i, config.duration_seconds)
                )
                tasks.append(task)
            
            await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        return self._analyze_results("Endurance Test", total_time)
    
    async def _endurance_user_simulation(self, user_id: int, duration: int):
        """Simulate user behavior for endurance testing."""
        start_time = time.time()
        request_count = 0
        
        while (time.time() - start_time) < duration:
            try:
                test_data = self.data_generator.get_random_test_data()
                request_start = time.time()
                
                async with self.session.post(
                    self.config.endpoint_url,
                    json=test_data,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    response_time = time.time() - request_start
                    
                    self.results.append({
                        'success': response.status == 200,
                        'response_time': response_time,
                        'status_code': response.status,
                        'user_id': user_id,
                        'request_count': request_count
                    })
                
                request_count += 1
                
                # Longer think time for endurance
                await asyncio.sleep(2.0)
                
            except Exception as e:
                self.results.append({
                    'success': False,
                    'response_time': 0,
                    'status_code': 0,
                    'error': str(e),
                    'user_id': user_id
                })
        
        logger.info(f"Endurance user {user_id} completed {request_count} requests")
    
    def _analyze_results(self, test_name: str, total_time: float) -> PerformanceResult:
        """Analyze test results and create performance result."""
        if not self.results:
            return PerformanceResult(
                test_name=test_name,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                total_time=total_time,
                average_response_time=0,
                min_response_time=0,
                max_response_time=0,
                p50_response_time=0,
                p95_response_time=0,
                p99_response_time=0,
                requests_per_second=0,
                errors=[],
                timestamp=datetime.utcnow().isoformat()
            )
        
        # Calculate metrics
        successful_results = [r for r in self.results if r.get('success', False)]
        failed_results = [r for r in self.results if not r.get('success', True)]
        
        response_times = [r['response_time'] for r in successful_results if r['response_time'] > 0]
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            p50_response_time = statistics.median(response_times)
            p95_response_time = np.percentile(response_times, 95)
            p99_response_time = np.percentile(response_times, 99)
        else:
            avg_response_time = min_response_time = max_response_time = 0
            p50_response_time = p95_response_time = p99_response_time = 0
        
        total_requests = len(self.results)
        successful_requests = len(successful_results)
        failed_requests = len(failed_results)
        requests_per_second = total_requests / total_time if total_time > 0 else 0
        
        errors = [r.get('error', 'Unknown error') for r in failed_results if r.get('error')]
        
        return PerformanceResult(
            test_name=test_name,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_time=total_time,
            average_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p50_response_time=p50_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            requests_per_second=requests_per_second,
            errors=errors[:10],  # Keep only first 10 errors
            timestamp=datetime.utcnow().isoformat()
        )
    
    def save_results(self, result: PerformanceResult, filename: str):
        """Save results to file."""
        with open(filename, 'w') as f:
            json.dump(asdict(result), f, indent=2)
        logger.info(f"Results saved to {filename}")


class SageMakerPerformanceTester(PerformanceTester):
    """Performance tester specifically for SageMaker endpoints."""
    
    def __init__(self, endpoint_name: str, region: str = "us-east-1"):
        self.endpoint_name = endpoint_name
        self.region = region
        self.sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=region)
        
        # Create config for SageMaker testing
        config = LoadTestConfig(
            endpoint_url=f"sagemaker://{endpoint_name}",
            concurrent_users=10,
            duration_seconds=300,
            ramp_up_time=30,
            test_data_size=100,
            request_timeout=30,
            think_time=0.5
        )
        
        super().__init__(config)
    
    async def run_sagemaker_load_test(self) -> PerformanceResult:
        """Run load test specifically for SageMaker endpoint."""
        logger.info(f"Starting SageMaker load test for endpoint: {self.endpoint_name}")
        
        start_time = time.time()
        
        # Use ThreadPoolExecutor for SageMaker calls (not async)
        with ThreadPoolExecutor(max_workers=self.config.concurrent_users) as executor:
            futures = []
            
            # Submit requests
            for i in range(self.config.concurrent_users):
                future = executor.submit(self._sagemaker_user_simulation, i)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    user_results = future.result()
                    self.results.extend(user_results)
                except Exception as e:
                    logger.error(f"User simulation failed: {e}")
        
        total_time = time.time() - start_time
        return self._analyze_results("SageMaker Load Test", total_time)
    
    def _sagemaker_user_simulation(self, user_id: int) -> List[Dict]:
        """Simulate user behavior for SageMaker endpoint."""
        user_results = []
        start_time = time.time()
        
        while (time.time() - start_time) < self.config.duration_seconds:
            try:
                test_data = self.data_generator.get_random_test_data()
                request_start = time.time()
                
                response = self.sagemaker_runtime.invoke_endpoint(
                    EndpointName=self.endpoint_name,
                    ContentType='application/json',
                    Body=json.dumps(test_data)
                )
                
                response_time = time.time() - request_start
                
                # Read response
                result = json.loads(response['Body'].read().decode())
                
                user_results.append({
                    'success': True,
                    'response_time': response_time,
                    'status_code': 200,
                    'user_id': user_id
                })
                
                time.sleep(self.config.think_time)
                
            except Exception as e:
                user_results.append({
                    'success': False,
                    'response_time': 0,
                    'status_code': 0,
                    'error': str(e),
                    'user_id': user_id
                })
        
        return user_results


def main():
    """Main function for running performance tests."""
    parser = argparse.ArgumentParser(description="Performance testing for chicken weight estimation")
    parser.add_argument("--endpoint-url", help="HTTP endpoint URL")
    parser.add_argument("--endpoint-name", help="SageMaker endpoint name")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--test-type", choices=["load", "stress", "spike", "endurance", "all"],
                       default="load", help="Type of test to run")
    parser.add_argument("--concurrent-users", type=int, default=10,
                       help="Number of concurrent users")
    parser.add_argument("--duration", type=int, default=300,
                       help="Test duration in seconds")
    parser.add_argument("--ramp-up-time", type=int, default=30,
                       help="Ramp-up time in seconds")
    parser.add_argument("--output-file", default="performance_results.json",
                       help="Output file for results")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        if args.endpoint_name:
            # SageMaker endpoint testing
            tester = SageMakerPerformanceTester(args.endpoint_name, args.region)
            
            if args.test_type == "load" or args.test_type == "all":
                result = asyncio.run(tester.run_sagemaker_load_test())
                tester.save_results(result, f"sagemaker_load_{args.output_file}")
                print(f"SageMaker Load Test Results:")
                print(f"  Total Requests: {result.total_requests}")
                print(f"  Success Rate: {result.successful_requests/result.total_requests*100:.2f}%")
                print(f"  Average Response Time: {result.average_response_time:.3f}s")
                print(f"  Requests/Second: {result.requests_per_second:.2f}")
        
        elif args.endpoint_url:
            # HTTP endpoint testing
            config = LoadTestConfig(
                endpoint_url=args.endpoint_url,
                concurrent_users=args.concurrent_users,
                duration_seconds=args.duration,
                ramp_up_time=args.ramp_up_time,
                test_data_size=100,
                request_timeout=30,
                think_time=0.5
            )
            
            tester = PerformanceTester(config)
            
            if args.test_type == "load" or args.test_type == "all":
                result = asyncio.run(tester.run_load_test())
                tester.save_results(result, f"load_{args.output_file}")
                print(f"Load Test Results:")
                print(f"  Total Requests: {result.total_requests}")
                print(f"  Success Rate: {result.successful_requests/result.total_requests*100:.2f}%")
                print(f"  Average Response Time: {result.average_response_time:.3f}s")
                print(f"  P95 Response Time: {result.p95_response_time:.3f}s")
                print(f"  Requests/Second: {result.requests_per_second:.2f}")
            
            if args.test_type == "stress" or args.test_type == "all":
                result = tester.run_stress_test()
                tester.save_results(result, f"stress_{args.output_file}")
                print(f"Stress Test Results:")
                print(f"  Breaking Point: {result.total_requests} total requests")
                print(f"  Max Response Time: {result.max_response_time:.3f}s")
            
            if args.test_type == "spike" or args.test_type == "all":
                result = tester.run_spike_test()
                tester.save_results(result, f"spike_{args.output_file}")
                print(f"Spike Test Results:")
                print(f"  Recovery Success Rate: {result.successful_requests/result.total_requests*100:.2f}%")
            
            if args.test_type == "endurance" or args.test_type == "all":
                result = tester.run_endurance_test()
                tester.save_results(result, f"endurance_{args.output_file}")
                print(f"Endurance Test Results:")
                print(f"  Total Duration: {result.total_time:.0f}s")
                print(f"  Stability: {result.successful_requests/result.total_requests*100:.2f}%")
        
        else:
            print("Error: Must specify either --endpoint-url or --endpoint-name")
            return 1
    
    except Exception as e:
        logger.error(f"Performance test failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())