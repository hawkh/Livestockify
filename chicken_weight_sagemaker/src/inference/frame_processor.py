"""
Frame processing utilities for real-time inference.
"""

import time
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import threading
import queue

from ..core.interfaces.detection import Detection
from ..core.interfaces.weight_estimation import WeightEstimate
from ..core.interfaces.tracking import TrackedChicken
from ..core.exceptions.inference_exceptions import ProcessingError


class FrameProcessor:
    """Handles frame preprocessing and postprocessing for inference."""
    
    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        self.target_size = target_size
        self.preprocessing_stats = {
            'frames_processed': 0,
            'average_preprocessing_time': 0.0,
            'resize_operations': 0,
            'normalization_operations': 0
        }
        
    def preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Preprocess frame for inference.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (processed_frame, metadata)
        """
        start_time = time.time()
        
        try:
            original_shape = frame.shape
            
            # Resize frame if needed
            if frame.shape[:2] != self.target_size:
                processed_frame = cv2.resize(frame, self.target_size)
                self.preprocessing_stats['resize_operations'] += 1
            else:
                processed_frame = frame.copy()
            
            # Normalize pixel values to [0, 1]
            if processed_frame.dtype == np.uint8:
                processed_frame = processed_frame.astype(np.float32) / 255.0
                self.preprocessing_stats['normalization_operations'] += 1
            
            # Create metadata
            metadata = {
                'original_shape': original_shape,
                'processed_shape': processed_frame.shape,
                'scale_factor_x': original_shape[1] / self.target_size[0],
                'scale_factor_y': original_shape[0] / self.target_size[1],
                'preprocessing_time': (time.time() - start_time) * 1000
            }
            
            # Update statistics
            self.preprocessing_stats['frames_processed'] += 1
            self._update_preprocessing_stats(metadata['preprocessing_time'])
            
            return processed_frame, metadata
            
        except Exception as e:
            raise ProcessingError("preprocessing", f"Frame preprocessing failed: {str(e)}")
    
    def postprocess_detections(
        self, 
        detections: List[Detection], 
        metadata: Dict[str, Any]
    ) -> List[Detection]:
        """
        Postprocess detections to original frame coordinates.
        
        Args:
            detections: List of detections in processed frame coordinates
            metadata: Preprocessing metadata
            
        Returns:
            Detections in original frame coordinates
        """
        try:
            scale_x = metadata['scale_factor_x']
            scale_y = metadata['scale_factor_y']
            
            postprocessed_detections = []
            
            for detection in detections:
                # Scale bounding box back to original coordinates
                scaled_bbox = [
                    detection.bbox[0] * scale_x,
                    detection.bbox[1] * scale_y,
                    detection.bbox[2] * scale_x,
                    detection.bbox[3] * scale_y
                ]
                
                # Create new detection with scaled coordinates
                scaled_detection = Detection(
                    bbox=scaled_bbox,
                    confidence=detection.confidence,
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                    occlusion_level=detection.occlusion_level,
                    visibility_score=detection.visibility_score
                )
                
                # Copy additional attributes if present
                if hasattr(detection, 'distance_estimate'):
                    scaled_detection.distance_estimate = detection.distance_estimate
                if hasattr(detection, 'reid_features'):
                    scaled_detection.reid_features = detection.reid_features
                
                postprocessed_detections.append(scaled_detection)
            
            return postprocessed_detections
            
        except Exception as e:
            raise ProcessingError("postprocessing", f"Detection postprocessing failed: {str(e)}")
    
    def _update_preprocessing_stats(self, processing_time: float):
        """Update preprocessing statistics."""
        current_avg = self.preprocessing_stats['average_preprocessing_time']
        frame_count = self.preprocessing_stats['frames_processed']
        
        # Calculate running average
        self.preprocessing_stats['average_preprocessing_time'] = (
            (current_avg * (frame_count - 1) + processing_time) / frame_count
        )
    
    def get_preprocessing_stats(self) -> Dict[str, Any]:
        """Get preprocessing statistics."""
        return self.preprocessing_stats.copy()
    
    def reset_stats(self):
        """Reset preprocessing statistics."""
        for key in self.preprocessing_stats:
            self.preprocessing_stats[key] = 0.0


class FrameQueue:
    """Thread-safe frame queue for continuous processing."""
    
    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self.queue = queue.Queue(maxsize=max_size)
        self.dropped_frames = 0
        self.total_frames = 0
        self.lock = threading.Lock()
    
    def put_frame(self, frame_data: Dict[str, Any], timeout: float = 0.1) -> bool:
        """
        Add frame to queue.
        
        Args:
            frame_data: Frame data dictionary
            timeout: Timeout for queue operation
            
        Returns:
            True if frame was added successfully
        """
        with self.lock:
            self.total_frames += 1
        
        try:
            self.queue.put(frame_data, timeout=timeout)
            return True
        except queue.Full:
            with self.lock:
                self.dropped_frames += 1
            return False
    
    def get_frame(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """
        Get frame from queue.
        
        Args:
            timeout: Timeout for queue operation
            
        Returns:
            Frame data or None if timeout
        """
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self.lock:
            return {
                'queue_size': self.queue.qsize(),
                'max_size': self.max_size,
                'total_frames': self.total_frames,
                'dropped_frames': self.dropped_frames,
                'drop_rate': self.dropped_frames / max(1, self.total_frames),
                'utilization': self.queue.qsize() / self.max_size
            }
    
    def clear_queue(self):
        """Clear all frames from queue."""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break


class ResultAggregator:
    """Aggregates processing results for analysis and reporting."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.results_history = deque(maxlen=window_size)
        self.lock = threading.Lock()
    
    def add_result(self, result: Dict[str, Any]):
        """Add processing result to history."""
        with self.lock:
            self.results_history.append({
                **result,
                'timestamp': time.time()
            })
    
    def get_aggregated_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics from recent results."""
        with self.lock:
            if not self.results_history:
                return {'status': 'no_data'}
            
            results = list(self.results_history)
        
        # Calculate statistics
        total_frames = len(results)
        successful_frames = sum(1 for r in results if r.get('status') == 'success')
        
        # Detection statistics
        total_detections = sum(r.get('total_chickens_detected', 0) for r in results)
        avg_detections_per_frame = total_detections / total_frames if total_frames > 0 else 0
        
        # Processing time statistics
        processing_times = [r.get('processing_time_ms', 0) for r in results]
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        max_processing_time = max(processing_times) if processing_times else 0
        min_processing_time = min(processing_times) if processing_times else 0
        
        # Weight statistics (if available)
        weights = []
        for result in results:
            if 'detections' in result:
                for detection in result['detections']:
                    if 'weight_estimate' in detection:
                        weights.append(detection['weight_estimate'].get('value', 0))
        
        avg_weight = np.mean(weights) if weights else 0
        weight_std = np.std(weights) if weights else 0
        
        # Time range
        timestamps = [r['timestamp'] for r in results]
        time_range = max(timestamps) - min(timestamps) if timestamps else 0
        
        return {
            'status': 'active',
            'time_range_seconds': time_range,
            'total_frames': total_frames,
            'successful_frames': successful_frames,
            'success_rate': successful_frames / total_frames if total_frames > 0 else 0,
            'total_detections': total_detections,
            'avg_detections_per_frame': avg_detections_per_frame,
            'processing_time': {
                'average_ms': avg_processing_time,
                'max_ms': max_processing_time,
                'min_ms': min_processing_time,
                'std_ms': np.std(processing_times) if processing_times else 0
            },
            'weight_statistics': {
                'total_estimates': len(weights),
                'average_weight_kg': avg_weight,
                'weight_std_kg': weight_std,
                'min_weight_kg': min(weights) if weights else 0,
                'max_weight_kg': max(weights) if weights else 0
            },
            'fps': total_frames / time_range if time_range > 0 else 0
        }
    
    def get_recent_results(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get most recent results."""
        with self.lock:
            return list(self.results_history)[-count:]
    
    def clear_history(self):
        """Clear results history."""
        with self.lock:
            self.results_history.clear()


class PerformanceMonitor:
    """Monitors system performance during inference."""
    
    def __init__(self):
        self.metrics = {
            'cpu_usage': deque(maxlen=60),  # Last 60 measurements
            'memory_usage': deque(maxlen=60),
            'gpu_usage': deque(maxlen=60),
            'inference_times': deque(maxlen=100)
        }
        self.monitoring_active = False
        self.monitor_thread = None
    
    def start_monitoring(self, interval: float = 1.0):
        """Start performance monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop."""
        try:
            import psutil
            
            while self.monitoring_active:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.metrics['cpu_usage'].append(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.metrics['memory_usage'].append(memory.percent)
                
                # GPU usage (if available)
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_usage = gpus[0].load * 100
                        self.metrics['gpu_usage'].append(gpu_usage)
                except ImportError:
                    pass
                
                time.sleep(interval)
                
        except ImportError:
            # psutil not available, skip monitoring
            pass
    
    def add_inference_time(self, inference_time: float):
        """Add inference time measurement."""
        self.metrics['inference_times'].append(inference_time)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        stats = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                stats[metric_name] = {
                    'current': values[-1],
                    'average': np.mean(values),
                    'max': max(values),
                    'min': min(values),
                    'std': np.std(values)
                }
            else:
                stats[metric_name] = {
                    'current': 0,
                    'average': 0,
                    'max': 0,
                    'min': 0,
                    'std': 0
                }
        
        return stats