#!/usr/bin/env python3
"""
Asynchronous Batch Processing Example

This example demonstrates high-performance batch processing of images
using the async client for maximum throughput.
"""

import sys
from pathlib import Path
import asyncio
import aiohttp
import time
import logging
from typing import List, Dict, Any
import json

# Add SDK to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sdk import AsyncChickenWeightClient, setup_logging
from sdk.models import BatchProcessingResult, ProcessingResult
from sdk.utils import BatchProcessor


class HighThroughputProcessor:
    """High-throughput batch processor for large-scale operations."""
    
    def __init__(self, endpoint_url: str, api_key: str = None, max_concurrent: int = 50):
        """
        Initialize the high-throughput processor.
        
        Args:
            endpoint_url: SageMaker endpoint URL
            api_key: Optional API key
            max_concurrent: Maximum concurrent requests
        """
        self.client = AsyncChickenWeightClient(
            endpoint_url=endpoint_url,
            api_key=api_key,
            max_concurrent=max_concurrent
        )
        self.logger = logging.getLogger(__name__)
        self.stats = {
            'total_processed': 0,
            'total_failed': 0,
            'total_time': 0,
            'batch_times': []
        }
    
    async def process_image_directory_fast(self, 
                                         directory_path: Path,
                                         batch_size: int = 20,
                                         file_pattern: str = "*.jpg") -> BatchProcessingResult:
        """
        Process all images in a directory with maximum speed.
        
        Args:
            directory_path: Path to image directory
            batch_size: Number of images to process concurrently
            file_pattern: File pattern to match
            
        Returns:
            BatchProcessingResult
        """
        # Find all image files
        image_files = list(directory_path.glob(file_pattern))
        if not image_files:
            raise ValueError(f"No images found matching {file_pattern}")
        
        image_files.sort()
        self.logger.info(f"Found {len(image_files)} images to process")
        
        start_time = time.time()
        all_results = []
        successful_count = 0
        failed_count = 0
        total_processing_time = 0.0
        
        # Process in batches
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
            connector=aiohttp.TCPConnector(limit=100)
        ) as session:
            
            for i in range(0, len(image_files), batch_size):
                batch_files = image_files[i:i + batch_size]
                batch_start = time.time()
                
                # Create tasks for this batch
                tasks = [
                    self.client.process_image_async(
                        session, 
                        str(image_path), 
                        camera_id="batch_processing",
                        frame_id=j
                    )
                    for j, image_path in enumerate(batch_files, start=i)
                ]
                
                # Execute batch with error handling
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for result in batch_results:
                    if isinstance(result, ProcessingResult):
                        all_results.append(result)
                        successful_count += 1
                        total_processing_time += result.processing_time
                    else:
                        self.logger.error(f"Batch processing error: {result}")
                        failed_count += 1
                
                batch_time = time.time() - batch_start
                self.stats['batch_times'].append(batch_time)
                
                # Progress update
                processed = i + len(batch_files)
                percentage = (processed / len(image_files)) * 100
                batch_fps = len(batch_files) / batch_time
                
                self.logger.info(
                    f"Batch {i//batch_size + 1}: {len(batch_files)} images in {batch_time:.2f}s "
                    f"({batch_fps:.1f} FPS) - Progress: {percentage:.1f}%"
                )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Update stats
        self.stats['total_processed'] += successful_count
        self.stats['total_failed'] += failed_count
        self.stats['total_time'] += total_time
        
        return BatchProcessingResult(
            total_frames=len(image_files),
            successful_frames=successful_count,
            failed_frames=failed_count,
            total_processing_time=total_processing_time,
            results=all_results,
            start_time=start_time,
            end_time=end_time
        )
    
    async def process_multiple_directories(self, 
                                         directories: List[Path],
                                         batch_size: int = 20) -> List[BatchProcessingResult]:
        """
        Process multiple directories concurrently.
        
        Args:
            directories: List of directories to process
            batch_size: Batch size for each directory
            
        Returns:
            List of BatchProcessingResult objects
        """
        self.logger.info(f"Processing {len(directories)} directories concurrently")
        
        # Create tasks for each directory
        tasks = [
            self.process_image_directory_fast(directory, batch_size)
            for directory in directories
        ]
        
        # Execute all directory processing concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, BatchProcessingResult):
                valid_results.append(result)
            else:
                self.logger.error(f"Directory {directories[i]} processing failed: {result}")
        
        return valid_results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.stats['batch_times']:
            return {'message': 'No processing completed yet'}
        
        avg_batch_time = sum(self.stats['batch_times']) / len(self.stats['batch_times'])
        total_images = self.stats['total_processed'] + self.stats['total_failed']
        overall_fps = total_images / self.stats['total_time'] if self.stats['total_time'] > 0 else 0
        
        return {
            'total_images_processed': self.stats['total_processed'],
            'total_images_failed': self.stats['total_failed'],
            'success_rate': (self.stats['total_processed'] / total_images * 100) if total_images > 0 else 0,
            'total_processing_time': self.stats['total_time'],
            'overall_fps': overall_fps,
            'average_batch_time': avg_batch_time,
            'batches_completed': len(self.stats['batch_times']),
            'min_batch_time': min(self.stats['batch_times']),
            'max_batch_time': max(self.stats['batch_times'])
        }


async def example_single_directory():
    """Example: Process a single directory with high throughput."""
    print("📁 Example: High-throughput single directory processing")
    print("=" * 60)
    
    processor = HighThroughputProcessor(
        endpoint_url="https://your-endpoint.amazonaws.com/prod",
        api_key="your-api-key-here",
        max_concurrent=30
    )
    
    try:
        # Process directory
        directory = Path("path/to/your/image_directory")
        if not directory.exists():
            print(f"⚠️  Directory not found: {directory}")
            print("   Creating example directory structure...")
            directory.mkdir(parents=True, exist_ok=True)
            print(f"   Please add images to {directory} and run again")
            return
        
        print(f"🚀 Processing directory: {directory}")
        
        result = await processor.process_image_directory_fast(
            directory_path=directory,
            batch_size=25,
            file_pattern="*.jpg"
        )
        
        print(f"\n✅ Processing complete!")
        print(f"   Total images: {result.total_frames}")
        print(f"   Successful: {result.successful_frames}")
        print(f"   Failed: {result.failed_frames}")
        print(f"   Success rate: {result.success_rate:.1f}%")
        print(f"   Average FPS: {result.average_fps:.1f}")
        print(f"   Total detections: {result.total_detections}")
        
        # Performance stats
        stats = processor.get_performance_stats()
        print(f"\n📊 Performance Statistics:")
        print(f"   Overall FPS: {stats['overall_fps']:.1f}")
        print(f"   Average batch time: {stats['average_batch_time']:.2f}s")
        print(f"   Min batch time: {stats['min_batch_time']:.2f}s")
        print(f"   Max batch time: {stats['max_batch_time']:.2f}s")
        
        # Weight statistics
        weight_stats = result.get_weight_statistics()
        if weight_stats['count'] > 0:
            print(f"\n⚖️  Weight Statistics:")
            print(f"   Chickens with weight estimates: {weight_stats['count']}")
            print(f"   Average weight: {weight_stats['mean']:.2f}kg")
            print(f"   Weight range: {weight_stats['min']:.2f} - {weight_stats['max']:.2f}kg")
            print(f"   Standard deviation: {weight_stats['std']:.2f}kg")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


async def example_multiple_directories():
    """Example: Process multiple directories concurrently."""
    print("\n📁📁 Example: Multiple directory concurrent processing")
    print("=" * 60)
    
    processor = HighThroughputProcessor(
        endpoint_url="https://your-endpoint.amazonaws.com/prod",
        api_key="your-api-key-here",
        max_concurrent=50
    )
    
    # Define directories to process
    directories = [
        Path("path/to/directory1"),
        Path("path/to/directory2"),
        Path("path/to/directory3"),
        Path("path/to/directory4")
    ]
    
    # Filter existing directories
    existing_dirs = [d for d in directories if d.exists()]
    if not existing_dirs:
        print("⚠️  No directories found. Creating example structure...")
        for i, directory in enumerate(directories[:2]):  # Create first 2
            directory.mkdir(parents=True, exist_ok=True)
            print(f"   Created: {directory}")
        print("   Please add images to these directories and run again")
        return
    
    try:
        print(f"🚀 Processing {len(existing_dirs)} directories concurrently...")
        
        start_time = time.time()
        results = await processor.process_multiple_directories(
            directories=existing_dirs,
            batch_size=20
        )
        end_time = time.time()
        
        print(f"\n✅ All directories processed in {end_time - start_time:.2f}s!")
        
        # Aggregate results
        total_images = sum(r.total_frames for r in results)
        total_successful = sum(r.successful_frames for r in results)
        total_failed = sum(r.failed_frames for r in results)
        total_detections = sum(r.total_detections for r in results)
        
        print(f"\n📊 Aggregate Results:")
        print(f"   Directories processed: {len(results)}")
        print(f"   Total images: {total_images}")
        print(f"   Successful: {total_successful}")
        print(f"   Failed: {total_failed}")
        print(f"   Success rate: {(total_successful/total_images*100):.1f}%")
        print(f"   Total detections: {total_detections}")
        
        # Per-directory breakdown
        print(f"\n📋 Per-Directory Breakdown:")
        for i, result in enumerate(results):
            directory_name = existing_dirs[i].name
            print(f"   {directory_name}:")
            print(f"     Images: {result.total_frames}")
            print(f"     Success rate: {result.success_rate:.1f}%")
            print(f"     FPS: {result.average_fps:.1f}")
            print(f"     Detections: {result.total_detections}")
        
        # Performance stats
        stats = processor.get_performance_stats()
        print(f"\n⚡ Performance Statistics:")
        print(f"   Overall throughput: {stats['overall_fps']:.1f} images/second")
        print(f"   Total processing time: {stats['total_processing_time']:.2f}s")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


async def example_performance_benchmark():
    """Example: Performance benchmarking with different configurations."""
    print("\n⚡ Example: Performance benchmarking")
    print("=" * 60)
    
    # Test different configurations
    configs = [
        {"max_concurrent": 10, "batch_size": 10},
        {"max_concurrent": 20, "batch_size": 15},
        {"max_concurrent": 30, "batch_size": 20},
        {"max_concurrent": 50, "batch_size": 25}
    ]
    
    test_directory = Path("path/to/benchmark_images")
    if not test_directory.exists():
        print(f"⚠️  Benchmark directory not found: {test_directory}")
        print("   Please create this directory with test images")
        return
    
    benchmark_results = []
    
    for i, config in enumerate(configs):
        print(f"\n🧪 Test {i+1}: max_concurrent={config['max_concurrent']}, batch_size={config['batch_size']}")
        
        processor = HighThroughputProcessor(
            endpoint_url="https://your-endpoint.amazonaws.com/prod",
            api_key="your-api-key-here",
            max_concurrent=config['max_concurrent']
        )
        
        try:
            start_time = time.time()
            result = await processor.process_image_directory_fast(
                directory_path=test_directory,
                batch_size=config['batch_size']
            )
            end_time = time.time()
            
            stats = processor.get_performance_stats()
            
            benchmark_result = {
                'config': config,
                'total_time': end_time - start_time,
                'fps': stats['overall_fps'],
                'success_rate': result.success_rate,
                'total_images': result.total_frames
            }
            
            benchmark_results.append(benchmark_result)
            
            print(f"   ✅ Completed in {benchmark_result['total_time']:.2f}s")
            print(f"   📊 FPS: {benchmark_result['fps']:.1f}")
            print(f"   ✔️  Success rate: {benchmark_result['success_rate']:.1f}%")
            
        except Exception as e:
            print(f"   ❌ Test failed: {str(e)}")
    
    # Find best configuration
    if benchmark_results:
        best_config = max(benchmark_results, key=lambda x: x['fps'])
        
        print(f"\n🏆 Best Configuration:")
        print(f"   Max concurrent: {best_config['config']['max_concurrent']}")
        print(f"   Batch size: {best_config['config']['batch_size']}")
        print(f"   FPS: {best_config['fps']:.1f}")
        print(f"   Total time: {best_config['total_time']:.2f}s")
        
        # Save benchmark results
        with open("benchmark_results.json", "w") as f:
            json.dump(benchmark_results, f, indent=2)
        print(f"   📄 Results saved to: benchmark_results.json")


async def example_error_handling():
    """Example: Robust error handling in batch processing."""
    print("\n🛡️  Example: Error handling and resilience")
    print("=" * 60)
    
    processor = HighThroughputProcessor(
        endpoint_url="https://invalid-endpoint-url.com/test",  # Intentionally invalid
        api_key="invalid-key",
        max_concurrent=10
    )
    
    # Create test directory with mixed content
    test_dir = Path("error_test_directory")
    test_dir.mkdir(exist_ok=True)
    
    # Create some test files (empty files to simulate errors)
    for i in range(5):
        (test_dir / f"test_image_{i}.jpg").touch()
    
    try:
        print("🧪 Testing error handling with invalid endpoint...")
        
        result = await processor.process_image_directory_fast(
            directory_path=test_dir,
            batch_size=3
        )
        
        print(f"📊 Results with errors:")
        print(f"   Total files: {result.total_frames}")
        print(f"   Successful: {result.successful_frames}")
        print(f"   Failed: {result.failed_frames}")
        print(f"   Success rate: {result.success_rate:.1f}%")
        
        if result.failed_frames > 0:
            print("   ✅ Error handling working correctly - failures detected and handled")
        
    except Exception as e:
        print(f"   ✅ Exception caught and handled: {type(e).__name__}")
    
    finally:
        # Cleanup
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)


async def main():
    """Run all async batch processing examples."""
    print("🚀 Asynchronous Batch Processing Examples")
    print("=" * 70)
    
    # Setup logging
    setup_logging(level='INFO')
    
    try:
        await example_single_directory()
        await example_multiple_directories()
        await example_performance_benchmark()
        await example_error_handling()
        
    except KeyboardInterrupt:
        print("\n⏹️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        logging.exception("Async batch processing error")
    
    print("\n🎉 Async batch processing examples completed!")
    print("\nKey takeaways:")
    print("- Async processing can achieve much higher throughput")
    print("- Optimal batch size and concurrency depend on your endpoint")
    print("- Error handling is crucial for production deployments")
    print("- Monitor performance metrics to optimize configuration")


if __name__ == "__main__":
    asyncio.run(main())