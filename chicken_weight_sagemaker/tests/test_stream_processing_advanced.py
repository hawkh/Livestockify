"""
Advanced stream processing test suite with performance analysis.
"""

import asyncio
import time
import json
import threading
import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict, deque
from typing import Dict, Any, List, Optional
import base64

from ..src.inference.stream_handler import RealTimeStreamProcessor, StreamProcessingServer
from ..src.utils.config.config_manager import ConfigManager
from ..src.core.interfaces.detection import Detection
from ..src.core.interfaces.weight_estimation import WeightEstimate


class StreamTestAnalyzer:
    """Analyze stream processing performance and accuracy."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.tracking_data = defaultdict(list)
        self.start_time = time.time()
    
    def add_result(self, result: Dict[str, Any]):
        """Add a processing result for analysis."""
        self.metrics['processing_time'].append(result.get('processing_time_ms', 0) / 1000.0)
        self.metrics['num_detections'].append(result.get('total_chickens_detected', 0))
        self.metrics['timestamp'].append(time.time() - self.start_time)
        
        # Track individual chickens
        detections = result.get('detections', [])
        for i, detection in enumerate(detections):
            chicken_id = detection.get('chicken_id', f'chicken_{i}')
            weight_data = detection.get('weight_estimate', {})
            
            self.tracking_data[chicken_id].append({
                'timestamp': time.time() - self.start_time,
                'weight': weight_data.get('value', 0),
                'confidence': weight_data.get('confidence', 0),
                'bbox': detection.get('bbox', {}),
                'occlusion': detection.get('occlusion_level', 0)
            })
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.metrics['processing_time']:
            return {'status': 'no_data'}
        
        processing_times = self.metrics['processing_time']
        detections = self.metrics['num_detections']
        
        report = {
            'performance': {
                'avg_processing_time_ms': np.mean(processing_times) * 1000,
                'max_processing_time_ms': np.max(processing_times) * 1000,
                'min_processing_time_ms': np.min(processing_times) * 1000,
                'std_processing_time_ms': np.std(processing_times) * 1000,
                'avg_fps': 1.0 / np.mean(processing_times) if np.mean(processing_times) > 0 else 0,
                'total_frames': len(processing_times),
                'frames_under_100ms': sum(1 for t in processing_times if t < 0.1),
                'real_time_percentage': sum(1 for t in processing_times if t < 0.033) / len(processing_times) * 100
            },
            'detection': {
                'avg_detections_per_frame': np.mean(detections),
                'max_detections': np.max(detections),
                'min_detections': np.min(detections),
                'std_detections': np.std(detections),
                'total_detections': sum(detections)
            },
            'tracking': {
                'total_unique_chickens': len(self.tracking_data),
                'avg_track_length': np.mean([len(data) for data in self.tracking_data.values()]),
                'track_continuity': self._calculate_track_continuity()
            }
        }
        
        return report  
  
    def _calculate_track_continuity(self) -> float:
        """Calculate how continuous tracks are."""
        if not self.tracking_data:
            return 0.0
        
        continuities = []
        for track_id, data in self.tracking_data.items():
            if len(data) > 1:
                timestamps = [d['timestamp'] for d in data]
                gaps = np.diff(timestamps)
                # Consider continuous if gaps are less than 2 frame intervals (assuming 30fps)
                continuous_gaps = sum(1 for gap in gaps if gap < 0.067)
                continuity = continuous_gaps / len(gaps) if gaps else 0
                continuities.append(continuity)
        
        return np.mean(continuities) if continuities else 0.0
    
    def plot_performance(self, save_path: str = 'performance_analysis.png'):
        """Plot comprehensive performance metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Processing time
        axes[0, 0].plot(self.metrics['timestamp'], 
                       [t * 1000 for t in self.metrics['processing_time']])
        axes[0, 0].set_title('Processing Time per Frame')
        axes[0, 0].set_xlabel('Time (seconds)')
        axes[0, 0].set_ylabel('Processing Time (ms)')
        axes[0, 0].axhline(y=33.3, color='r', linestyle='--', label='30 FPS target')
        axes[0, 0].axhline(y=100, color='orange', linestyle='--', label='Real-time target')
        axes[0, 0].legend()
        
        # Detection count
        axes[0, 1].plot(self.metrics['timestamp'], self.metrics['num_detections'])
        axes[0, 1].set_title('Detections per Frame')
        axes[0, 1].set_xlabel('Time (seconds)')
        axes[0, 1].set_ylabel('Number of Detections')
        
        # Processing time histogram
        axes[0, 2].hist([t * 1000 for t in self.metrics['processing_time']], bins=20)
        axes[0, 2].set_title('Processing Time Distribution')
        axes[0, 2].set_xlabel('Processing Time (ms)')
        axes[0, 2].set_ylabel('Frequency')
        
        # Weight estimates by track (top 5 tracks)
        axes[1, 0].set_title('Weight Estimates by Track')
        track_items = list(self.tracking_data.items())[:5]
        for track_id, data in track_items:
            timestamps = [d['timestamp'] for d in data]
            weights = [d['weight'] for d in data]
            axes[1, 0].plot(timestamps, weights, label=f'Track {track_id}', marker='o', markersize=3)
        axes[1, 0].set_xlabel('Time (seconds)')
        axes[1, 0].set_ylabel('Weight (kg)')
        axes[1, 0].legend()
        
        # Confidence scores
        axes[1, 1].set_title('Weight Confidence Scores')
        all_confidences = []
        all_timestamps = []
        for data_list in self.tracking_data.values():
            for data in data_list:
                all_confidences.append(data['confidence'])
                all_timestamps.append(data['timestamp'])
        
        if all_confidences:
            axes[1, 1].scatter(all_timestamps, all_confidences, alpha=0.6, s=10)
            axes[1, 1].set_xlabel('Time (seconds)')
            axes[1, 1].set_ylabel('Confidence Score')
        
        # Occlusion levels
        axes[1, 2].set_title('Occlusion Levels')
        all_occlusions = []
        all_timestamps_occ = []
        for data_list in self.tracking_data.values():
            for data in data_list:
                all_occlusions.append(data['occlusion'])
                all_timestamps_occ.append(data['timestamp'])
        
        if all_occlusions:
            axes[1, 2].scatter(all_timestamps_occ, all_occlusions, alpha=0.6, s=10)
            axes[1, 2].set_xlabel('Time (seconds)')
            axes[1, 2].set_ylabel('Occlusion Level')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class TestVideoGenerator:
    """Generate synthetic test video frames with moving objects."""
    
    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.frame_count = 0
        self.objects = self._initialize_objects()
    
    def _initialize_objects(self) -> List[Dict[str, Any]]:
        """Initialize moving objects (simulated chickens)."""
        objects = []
        for i in range(np.random.randint(3, 8)):  # 3-7 chickens
            obj = {
                'id': i,
                'x': np.random.randint(50, self.width - 50),
                'y': np.random.randint(50, self.height - 50),
                'vx': np.random.uniform(-2, 2),
                'vy': np.random.uniform(-2, 2),
                'size': np.random.randint(30, 60),
                'color': (np.random.randint(100, 200), np.random.randint(100, 200), np.random.randint(100, 200))
            }
            objects.append(obj)
        return objects 
   
    def generate_frame(self) -> np.ndarray:
        """Generate a synthetic frame with moving objects."""
        # Create background
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 50
        
        # Add some texture to background
        noise = np.random.randint(0, 30, (self.height, self.width, 3))
        frame = cv2.add(frame, noise.astype(np.uint8))
        
        # Update and draw objects
        for obj in self.objects:
            # Update position
            obj['x'] += obj['vx']
            obj['y'] += obj['vy']
            
            # Bounce off walls
            if obj['x'] <= obj['size']//2 or obj['x'] >= self.width - obj['size']//2:
                obj['vx'] *= -1
            if obj['y'] <= obj['size']//2 or obj['y'] >= self.height - obj['size']//2:
                obj['vy'] *= -1
            
            # Keep in bounds
            obj['x'] = max(obj['size']//2, min(self.width - obj['size']//2, obj['x']))
            obj['y'] = max(obj['size']//2, min(self.height - obj['size']//2, obj['y']))
            
            # Draw object (ellipse to simulate chicken shape)
            center = (int(obj['x']), int(obj['y']))
            axes = (obj['size']//2, int(obj['size']//2 * 0.7))
            cv2.ellipse(frame, center, axes, 0, 0, 360, obj['color'], -1)
            
            # Add some random occlusion
            if np.random.random() < 0.1:  # 10% chance of occlusion
                # Draw occluding rectangle
                occlusion_size = np.random.randint(20, 40)
                occlusion_x = center[0] + np.random.randint(-20, 20)
                occlusion_y = center[1] + np.random.randint(-20, 20)
                cv2.rectangle(frame, 
                            (occlusion_x - occlusion_size//2, occlusion_y - occlusion_size//2),
                            (occlusion_x + occlusion_size//2, occlusion_y + occlusion_size//2),
                            (80, 80, 80), -1)
        
        self.frame_count += 1
        return frame


def create_mock_processor() -> RealTimeStreamProcessor:
    """Create processor with mock components for testing."""
    
    class MockConfigManager(ConfigManager):
        def load_config(self, config_name: str):
            if config_name == "camera_config":
                return {
                    'focal_length': 1000.0,
                    'sensor_width': 6.0,
                    'sensor_height': 4.5,
                    'image_width': 640,
                    'image_height': 480,
                    'camera_height': 3.0,
                    'known_object_width': 25.0
                }
            elif config_name == "model_config":
                return {
                    'yolo': {
                        'model_path': 'mock_yolo.pt',
                        'confidence_threshold': 0.4,
                        'min_visibility_threshold': 0.3
                    },
                    'weight_estimation': {
                        'model_path': 'mock_weight.pt'
                    },
                    'tracking': {
                        'max_disappeared': 30,
                        'max_distance': 100.0
                    }
                }
            return {}
    
    # Create processor with mock config
    processor = RealTimeStreamProcessor(MockConfigManager())
    
    # Replace components with mocks
    processor.detector = MockYOLODetector()
    processor.weight_estimator = MockWeightEstimator()
    
    return processor


class MockYOLODetector:
    """Mock YOLO detector for testing."""
    
    def __init__(self):
        self.model = "mock_model"
    
    def detect_with_occlusion_handling(self, frame: np.ndarray):
        """Mock detection with synthetic results."""
        # Simulate detection processing time
        time.sleep(np.random.uniform(0.01, 0.05))
        
        # Generate mock detections
        detections = []
        num_detections = np.random.randint(2, 6)
        
        for i in range(num_detections):
            # Random bounding box
            x1 = np.random.randint(0, frame.shape[1] - 100)
            y1 = np.random.randint(0, frame.shape[0] - 100)
            x2 = x1 + np.random.randint(50, 100)
            y2 = y1 + np.random.randint(40, 80)
            
            detection = Detection(
                bbox=[x1, y1, x2, y2],
                confidence=np.random.uniform(0.5, 0.95),
                class_id=0,
                class_name="chicken",
                occlusion_level=np.random.uniform(0.0, 0.7),
                visibility_score=np.random.uniform(0.3, 1.0)
            )
            detection.distance_estimate = np.random.uniform(2.0, 8.0)
            detections.append(detection)
        
        # Mock result object
        class MockResult:
            def __init__(self, detections):
                self.detections = detections
        
        return MockResult(detections)
    
    def get_model_info(self):
        return {"status": "mock", "model_path": "mock_yolo.pt"}


class MockWeightEstimator:
    """Mock weight estimator for testing."""
    
    def __init__(self):
        self.is_loaded = True
    
    def estimate_weight_with_distance(self, frame, detection, distance, occlusion_level):
        """Mock weight estimation."""
        # Simulate processing time
        time.sleep(np.random.uniform(0.005, 0.02))
        
        # Generate realistic weight based on detection size
        bbox = detection.bbox
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        base_weight = 1.5 + (area / 5000.0)  # Scale with detection size
        
        # Add some randomness
        weight = base_weight + np.random.normal(0, 0.3)
        weight = max(0.5, min(4.0, weight))  # Clamp to reasonable range
        
        return WeightEstimate(
            value=weight,
            unit="kg",
            confidence=np.random.uniform(0.6, 0.9),
            error_range=f"±{weight * 0.2:.2f}kg",
            distance_compensated=True,
            occlusion_adjusted=occlusion_level > 0.3,
            method="mock_nn"
        )
    
    def get_model_info(self):
        return {"status": "mock", "model_path": "mock_weight.pt"}