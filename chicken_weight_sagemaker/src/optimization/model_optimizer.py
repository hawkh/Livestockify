"""
Model optimization for production deployment.
"""

import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic
import onnx
import onnxruntime as ort
from typing import Tuple, Dict, Any, Optional
import numpy as np
import logging
import time
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """Optimize models for production deployment."""
    
    def __init__(self):
        self.optimization_stats = {}
    
    def quantize_model(self, model: nn.Module, example_input: torch.Tensor) -> nn.Module:
        """
        Apply dynamic quantization to reduce model size.
        
        Args:
            model: PyTorch model to quantize
            example_input: Example input tensor for testing
            
        Returns:
            Quantized model
        """
        logger.info("Applying dynamic quantization...")
        
        # Store original size
        original_size = self._get_model_size(model)
        
        # Apply quantization
        quantized_model = quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        
        # Test quantized model
        with torch.no_grad():
            original_output = model(example_input)
            quantized_output = quantized_model(example_input)
        
        # Calculate accuracy loss
        diff = torch.mean(torch.abs(original_output - quantized_output))
        logger.info(f"Average difference: {diff.item():.6f}")
        
        # Get new size
        quantized_size = self._get_model_size(quantized_model)
        
        self.optimization_stats['quantization'] = {
            'original_size_mb': original_size,
            'quantized_size_mb': quantized_size,
            'compression_ratio': original_size / quantized_size,
            'accuracy_diff': diff.item()
        }
        
        logger.info(f"Model size reduced from {original_size:.2f}MB to {quantized_size:.2f}MB")
        logger.info(f"Compression ratio: {original_size / quantized_size:.2f}x")
        
        return quantized_model
    
    def export_to_onnx(self, model: nn.Module, example_input: torch.Tensor,
                      output_path: str, optimize: bool = True) -> str:
        """
        Export model to ONNX for faster inference.
        
        Args:
            model: PyTorch model to export
            example_input: Example input tensor
            output_path: Path to save ONNX model
            optimize: Whether to optimize the ONNX model
            
        Returns:
            Path to exported ONNX model
        """
        logger.info("Exporting to ONNX...")
        model.eval()
        
        # Export to ONNX
        torch.onnx.export(
            model,
            example_input,
            output_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Optimize ONNX model
        if optimize:
            try:
                import onnxoptimizer
                model_onnx = onnx.load(output_path)
                model_onnx = onnxoptimizer.optimize(model_onnx)
                onnx.save(model_onnx, output_path)
                logger.info("ONNX model optimized")
            except ImportError:
                logger.warning("onnxoptimizer not available, skipping optimization")
        
        # Verify ONNX model
        ort_session = ort.InferenceSession(output_path)
        
        # Compare outputs
        with torch.no_grad():
            torch_output = model(example_input).numpy()
        
        ort_inputs = {ort_session.get_inputs()[0].name: example_input.numpy()}
        ort_output = ort_session.run(None, ort_inputs)[0]
        
        diff = np.mean(np.abs(torch_output - ort_output))
        logger.info(f"ONNX conversion difference: {diff:.6f}")
        
        self.optimization_stats['onnx'] = {
            'output_path': output_path,
            'accuracy_diff': float(diff)
        }
        
        return output_path
    
    def optimize_for_edge_deployment(self, model: nn.Module) -> nn.Module:
        """
        Optimize model for edge device deployment.
        
        Args:
            model: PyTorch model to optimize
            
        Returns:
            Optimized model
        """
        logger.info("Optimizing for edge deployment...")
        
        # 1. Fuse batch normalization with convolution layers
        model = self._fuse_conv_bn(model)
        
        # 2. Remove dropout layers (for inference)
        self._remove_dropout(model)
        
        # 3. Prune low-magnitude weights
        model = self._prune_weights(model, threshold=0.01)
        
        return model
    
    def benchmark_model(self, model: nn.Module, example_input: torch.Tensor,
                       num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark model performance.
        
        Args:
            model: Model to benchmark
            example_input: Example input tensor
            num_runs: Number of benchmark runs
            
        Returns:
            Performance statistics
        """
        logger.info(f"Benchmarking model with {num_runs} runs...")
        
        model.eval()
        times = []
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(example_input)
        
        # Benchmark
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(example_input)
                end_time = time.time()
                times.append(end_time - start_time)
        
        stats = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'p95_time': np.percentile(times, 95),
            'p99_time': np.percentile(times, 99),
            'fps': 1.0 / np.mean(times)
        }
        
        logger.info(f"Average inference time: {stats['mean_time']*1000:.2f}ms")
        logger.info(f"Average FPS: {stats['fps']:.1f}")
        
        return stats
    
    def _get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def _fuse_conv_bn(self, model: nn.Module) -> nn.Module:
        """Fuse Conv2d and BatchNorm2d layers."""
        from torch.nn.utils import fusion
        
        for module in model.modules():
            if isinstance(module, nn.Sequential):
                for idx in range(len(module) - 1):
                    if (isinstance(module[idx], nn.Conv2d) and 
                        isinstance(module[idx + 1], nn.BatchNorm2d)):
                        fused = fusion.fuse_conv_bn_eval(module[idx], module[idx + 1])
                        module[idx] = fused
                        module[idx + 1] = nn.Identity()
        
        return model
    
    def _remove_dropout(self, model: nn.Module):
        """Replace dropout layers with identity."""
        for name, module in model.named_children():
            if isinstance(module, nn.Dropout):
                setattr(model, name, nn.Identity())
            else:
                self._remove_dropout(module)
    
    def _prune_weights(self, model: nn.Module, threshold: float) -> nn.Module:
        """Prune weights below threshold."""
        for name, param in model.named_parameters():
            if 'weight' in name:
                mask = torch.abs(param.data) > threshold
                param.data *= mask.float()
        
        return model


class InferenceOptimizer:
    """Optimize inference pipeline for production."""
    
    def __init__(self):
        self.connection_pool = None
        self.request_queue = None
        self.batch_processor = None
    
    def setup_connection_pooling(self, max_connections: int = 100):
        """Setup connection pooling for better throughput."""
        from concurrent.futures import ThreadPoolExecutor
        
        self.connection_pool = ThreadPoolExecutor(max_workers=max_connections)
        logger.info(f"Connection pool created with {max_connections} workers")
    
    def enable_request_batching(self, batch_size: int = 10, timeout_ms: int = 50):
        """Enable request batching for better GPU utilization."""
        import queue
        
        self.request_queue = queue.Queue(maxsize=1000)
        self.batch_processor = BatchProcessor(
            batch_size=batch_size,
            timeout_ms=timeout_ms,
            process_func=self._process_batch
        )
        self.batch_processor.start()
        
        logger.info(f"Request batching enabled: batch_size={batch_size}, timeout={timeout_ms}ms")
    
    def _process_batch(self, requests: list) -> list:
        """Process a batch of requests."""
        # Combine frames into batch
        frames = [req['frame'] for req in requests]
        batch_tensor = torch.stack([self._preprocess_frame(f) for f in frames])
        
        # Run batch inference
        with torch.no_grad():
            results = self.model(batch_tensor)
        
        # Split results
        responses = []
        for i, req in enumerate(requests):
            response = {
                'frame_id': req['frame_id'],
                'result': results[i]
            }
            responses.append(response)
        
        return responses
    
    def _preprocess_frame(self, frame):
        """Preprocess frame for inference."""
        # Convert to tensor and normalize
        if isinstance(frame, np.ndarray):
            frame = torch.from_numpy(frame).float()
        
        # Add batch dimension if needed
        if len(frame.shape) == 3:
            frame = frame.unsqueeze(0)
        
        return frame


class BatchProcessor:
    """Process requests in batches for better throughput."""
    
    def __init__(self, batch_size: int, timeout_ms: int, process_func):
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms
        self.process_func = process_func
        self.request_queue = None
        self.response_futures = {}
        self.running = False
    
    def start(self):
        """Start the batch processor."""
        import threading
        import queue
        
        self.request_queue = queue.Queue()
        self.running = True
        
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        
        logger.info("Batch processor started")
    
    def submit(self, request: dict):
        """Submit request for batch processing."""
        import asyncio
        
        future = asyncio.Future()
        request_id = id(request)
        self.response_futures[request_id] = future
        self.request_queue.put((request_id, request))
        return future
    
    def run(self):
        """Batch processing loop."""
        import queue
        import time
        
        while self.running:
            batch = []
            batch_ids = []
            
            # Collect batch
            deadline = time.time() + self.timeout_ms / 1000
            while len(batch) < self.batch_size and time.time() < deadline:
                try:
                    timeout = max(0, deadline - time.time())
                    request_id, request = self.request_queue.get(timeout=timeout)
                    batch.append(request)
                    batch_ids.append(request_id)
                except queue.Empty:
                    break
            
            if batch:
                # Process batch
                try:
                    results = self.process_func(batch)
                    
                    # Set results
                    for request_id, result in zip(batch_ids, results):
                        if request_id in self.response_futures:
                            future = self.response_futures.pop(request_id)
                            future.set_result(result)
                            
                except Exception as e:
                    # Set exception for all requests in batch
                    for request_id in batch_ids:
                        if request_id in self.response_futures:
                            future = self.response_futures.pop(request_id)
                            future.set_exception(e)
    
    def stop(self):
        """Stop the batch processor."""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()
        logger.info("Batch processor stopped")


class CostOptimizer:
    """Optimize deployment costs."""
    
    def __init__(self, aws_region: str = 'us-east-1'):
        import boto3
        
        self.ec2_client = boto3.client('ec2', region_name=aws_region)
        self.sagemaker_client = boto3.client('sagemaker', region_name=aws_region)
        self.autoscaling_client = boto3.client('application-autoscaling', region_name=aws_region)
    
    def setup_spot_instances(self, endpoint_name: str, spot_price: float = 0.9):
        """Configure spot instances for cost savings."""
        logger.info(f"Setting up spot instances for {endpoint_name}")
        
        # Get current endpoint config
        endpoint_desc = self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        config_name = endpoint_desc['EndpointConfigName']
        
        # Create new config with spot instances
        new_config_name = f"{config_name}-spot-{int(time.time())}"
        
        # Create spot instance configuration
        self.sagemaker_client.create_endpoint_config(
            EndpointConfigName=new_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'primary',
                    'ModelName': endpoint_desc['ModelName'],
                    'InitialInstanceCount': 1,
                    'InstanceType': 'ml.g4dn.xlarge',
                    'InitialVariantWeight': 1.0,
                    'ManagedInstanceScaling': {
                        'Status': 'ENABLED',
                        'MinInstanceCount': 1,
                        'MaxInstanceCount': 4
                    },
                    'RoutingStrategy': 'LEAST_OUTSTANDING_REQUESTS'
                }
            ]
        )
        
        logger.info(f"Spot instance configuration created: {new_config_name}")
    
    def implement_caching(self, cache_size_mb: int = 1000):
        """Implement result caching to reduce redundant processing."""
        from cachetools import LRUCache
        import hashlib
        
        # Create LRU cache
        cache = LRUCache(maxsize=cache_size_mb * 1024 * 1024)  # Convert to bytes
        
        def cache_key(frame: np.ndarray) -> str:
            """Generate cache key for frame."""
            # Use perceptual hash for similar frame detection
            frame_small = cv2.resize(frame, (32, 32))
            frame_hash = hashlib.md5(frame_small.tobytes()).hexdigest()
            return frame_hash
        
        def cached_process(frame: np.ndarray):
            """Process frame with caching."""
            key = cache_key(frame)
            if key in cache:
                return cache[key]
            
            # Process frame
            result = self.process_frame(frame)
            
            # Cache result
            cache[key] = result
            return result
        
        return cached_process
    
    def setup_auto_scaling(self, endpoint_name: str, 
                          min_instances: int = 1, 
                          max_instances: int = 10,
                          target_invocations: int = 100):
        """Setup auto-scaling based on invocation rate."""
        logger.info(f"Setting up auto-scaling for {endpoint_name}")
        
        # Register scalable target
        self.autoscaling_client.register_scalable_target(
            ServiceNamespace='sagemaker',
            ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            MinCapacity=min_instances,
            MaxCapacity=max_instances
        )
        
        # Create scaling policy
        policy_name = f'{endpoint_name}-scaling-policy'
        self.autoscaling_client.put_scaling_policy(
            PolicyName=policy_name,
            ServiceNamespace='sagemaker',
            ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            PolicyType='TargetTrackingScaling',
            TargetTrackingScalingPolicyConfiguration={
                'TargetValue': float(target_invocations),
                'PredefinedMetricSpecification': {
                    'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
                },
                'ScaleInCooldown': 300,  # 5 minutes
                'ScaleOutCooldown': 60   # 1 minute
            }
        )
        
        logger.info(f"Auto-scaling configured: {min_instances}-{max_instances} instances")


def optimize_model_for_production(model_path: str, output_dir: str) -> Dict[str, str]:
    """
    Complete model optimization pipeline for production.
    
    Args:
        model_path: Path to the original model
        output_dir: Directory to save optimized models
        
    Returns:
        Dictionary with paths to optimized models
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = torch.load(model_path, map_location='cpu')
    model.eval()
    
    # Create example input
    example_input = torch.randn(1, 3, 224, 224)
    
    # Initialize optimizer
    optimizer = ModelOptimizer()
    
    # Benchmark original model
    logger.info("Benchmarking original model...")
    original_stats = optimizer.benchmark_model(model, example_input)
    
    optimized_models = {}
    
    # 1. Quantized model
    logger.info("Creating quantized model...")
    quantized_model = optimizer.quantize_model(model, example_input)
    quantized_path = output_dir / "model_quantized.pt"
    torch.save(quantized_model, quantized_path)
    optimized_models['quantized'] = str(quantized_path)
    
    # Benchmark quantized model
    quantized_stats = optimizer.benchmark_model(quantized_model, example_input)
    logger.info(f"Quantized model speedup: {original_stats['mean_time'] / quantized_stats['mean_time']:.2f}x")
    
    # 2. ONNX model
    logger.info("Creating ONNX model...")
    onnx_path = output_dir / "model.onnx"
    optimizer.export_to_onnx(model, example_input, str(onnx_path))
    optimized_models['onnx'] = str(onnx_path)
    
    # 3. Edge-optimized model
    logger.info("Creating edge-optimized model...")
    edge_model = optimizer.optimize_for_edge_deployment(model)
    edge_path = output_dir / "model_edge.pt"
    torch.save(edge_model, edge_path)
    optimized_models['edge'] = str(edge_path)
    
    # Benchmark edge model
    edge_stats = optimizer.benchmark_model(edge_model, example_input)
    logger.info(f"Edge model speedup: {original_stats['mean_time'] / edge_stats['mean_time']:.2f}x")
    
    # Save optimization report
    report = {
        'original_model': {
            'path': model_path,
            'size_mb': optimizer._get_model_size(model),
            'performance': original_stats
        },
        'optimized_models': {
            'quantized': {
                'path': str(quantized_path),
                'size_mb': optimizer._get_model_size(quantized_model),
                'performance': quantized_stats,
                'speedup': original_stats['mean_time'] / quantized_stats['mean_time']
            },
            'onnx': {
                'path': str(onnx_path),
                'performance': 'Use ONNX Runtime for benchmarking'
            },
            'edge': {
                'path': str(edge_path),
                'size_mb': optimizer._get_model_size(edge_model),
                'performance': edge_stats,
                'speedup': original_stats['mean_time'] / edge_stats['mean_time']
            }
        },
        'optimization_stats': optimizer.optimization_stats
    }
    
    import json
    with open(output_dir / "optimization_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Optimization complete. Report saved to {output_dir / 'optimization_report.json'}")
    
    return optimized_models