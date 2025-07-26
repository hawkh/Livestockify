"""
Real-time stream processing handler for live poultry farm footage.
"""

import time
import json
import base64
from typing import Dict, Any, Optional, List
from collections import deque
import numpy as np
import cv2
from flask import Flask, request, jsonify
import threading
import queue
import logging

from ..models.detection.occlusion_robust_yolo import OcclusionRobustYOLODetector
from ..models.tracking.chicken_tracker import ChickenMultiObjectTracker
from ..models.weight_estimation.distance_adaptive_nn import DistanceAdaptiveWeightNN
from ..utils.distance.perspective_distance import PerspectiveDistanceEstimator
from ..utils.config.config_manager import ConfigManager
from ..core.interfaces.inference import InferenceRequest, InferenceResponse
from ..core.exceptions.inference_exceptions import ProcessingError, InvalidInputError


class RealTimeStreamProcessor:
    """Real-time stream processor optimized for live poultry farm footage."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Load configurations
        self.model_config = config_manager.load_config("model_config")
        self.camera_config = config_manager.load_config("camera_config")
        
        # Initialize processing components
        self._initialize_components()
        
        # Performance monitoring
        self.frame_times = deque(maxlen=100)
        self.processing_stats = {
            'frames_processed': 0,
            'total_chickens_detected': 0,
            'average_processing_time': 0.0,
            'current_fps': 0.0
        }
        
        # Frame buffering for continuous streams
        self.frame_buffer = queue.Queue(maxsize=10)
        self.result_buffer = queue.Queue(maxsize=50)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _initialize_components(self):
        """Initialize all processing components."""
        try:
            # YOLO detector with occlusion handling
            self.detector = OcclusionRobustYOLODetector(
                model_path=self.model_config.get('yolo', {}).get('model_path', 'model_artifacts/yolo_best.pt'),
                confidence_threshold=self.model_config.get('yolo', {}).get('confidence_threshold', 0.4),
                min_visibility_threshold=self.model_config.get('yolo', {}).get('min_visibility_threshold', 0.3)
            )
            
            # Multi-object tracker
            self.tracker = ChickenMultiObjectTracker(
                max_disappeared=self.model_config.get('tracking', {}).get('max_disappeared', 30),
                max_distance=self.model_config.get('tracking', {}).get('max_distance', 100.0)
            )
            
            # Weight estimation neural network
            self.weight_estimator = DistanceAdaptiveWeightNN(
                model_path=self.model_config.get('weight_estimation', {}).get('model_path', 'model_artifacts/weight_nn.pt')
            )
            
            # Distance estimator
            from ..core.interfaces.camera import CameraParameters
            camera_params = CameraParameters(
                focal_length=self.camera_config.get('focal_length', 1000.0),
                sensor_width=self.camera_config.get('sensor_width', 6.0),
                sensor_height=self.camera_config.get('sensor_height', 4.5),
                image_width=self.camera_config.get('image_width', 1920),
                image_height=self.camera_config.get('image_height', 1080),
                camera_height=self.camera_config.get('camera_height', 3.0),
                known_object_width=self.camera_config.get('known_object_width', 25.0)
            )
            
            self.distance_estimator = PerspectiveDistanceEstimator(camera_params)
            
            self.logger.info("All processing components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {str(e)}")
            raise ProcessingError("initialization", f"Component initialization failed: {str(e)}")
    
    def process_frame(self, frame_data: str, camera_id: str, frame_sequence: int = 0) -> Dict[str, Any]:
        """
        Process a single frame from live footage.
        
        Args:
            frame_data: Base64 encoded frame data
            camera_id: Camera identifier
            frame_sequence: Frame sequence number
            
        Returns:
            Processing results with detections and weight estimates
        """
        start_time = time.time()
        
        try:
            # Decode frame
            frame = self._decode_frame(frame_data)
            
            # Detect chickens with occlusion handling
            detection_result = self.detector.detect_with_occlusion_handling(frame)
            detections = detection_result.detections
            
            # Estimate distances for each detection
            for detection in detections:
                distance = self.distance_estimator.estimate_distance_to_chicken(
                    detection.bbox, frame.shape[:2]
                )
                detection.distance_estimate = distance
            
            # Update tracking
            tracking_result = self.tracker.update_tracks(
                detections, frame_id=str(frame_sequence)
            )
            
            # Estimate weights for tracked chickens
            weight_estimates = []
            for tracked_chicken in tracking_result.tracked_chickens:
                if tracked_chicken.current_detection:
                    try:
                        weight_estimate = self.weight_estimator.estimate_weight_with_distance(
                            frame,
                            tracked_chicken.current_detection,
                            tracked_chicken.current_detection.distance_estimate or 3.0,
                            tracked_chicken.current_detection.occlusion_level or 0.0
                        )
                        weight_estimates.append(weight_estimate)
                        
                        # Update tracked chicken with weight
                        tracked_chicken.current_weight = weight_estimate
                        
                    except Exception as e:
                        self.logger.warning(f"Weight estimation failed for chicken {tracked_chicken.chicken_id}: {str(e)}")
                        # Create default weight estimate
                        from ..core.interfaces.weight_estimation import WeightEstimate
                        default_weight = WeightEstimate(
                            value=2.0,  # Default weight
                            confidence=0.1,
                            error_range="±1.0kg",
                            method="fallback"
                        )
                        weight_estimates.append(default_weight)
            
            # Calculate processing statistics
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            self.frame_times.append(processing_time)
            self._update_stats(len(detections), processing_time)
            
            # Create response
            response = InferenceResponse.from_processing_results(
                camera_id=camera_id,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                frame_sequence=frame_sequence,
                detections=detections,
                weight_estimates=weight_estimates,
                tracked_chickens=tracking_result.tracked_chickens,
                processing_time_ms=processing_time,
                status="success"
            )
            
            return response.frame_results
            
        except Exception as e:
            self.logger.error(f"Frame processing failed: {str(e)}")
            error_response = {
                "camera_id": camera_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "frame_sequence": frame_sequence,
                "status": "error",
                "error_message": str(e),
                "detections": [],
                "processing_time_ms": (time.time() - start_time) * 1000
            }
            return error_response
    
    def _decode_frame(self, frame_data: str) -> np.ndarray:
        """Decode base64 frame data to numpy array."""
        try:
            # Remove data URL prefix if present
            if frame_data.startswith('data:image'):
                frame_data = frame_data.split(',')[1]
            
            # Decode base64
            img_bytes = base64.b64decode(frame_data)
            
            # Convert to numpy array
            nparr = np.frombuffer(img_bytes, np.uint8)
            
            # Decode image
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise ValueError("Failed to decode image data")
            
            return frame
            
        except Exception as e:
            raise InvalidInputError(f"Frame decoding failed: {str(e)}")
    
    def _update_stats(self, num_detections: int, processing_time: float):
        """Update processing statistics."""
        self.processing_stats['frames_processed'] += 1
        self.processing_stats['total_chickens_detected'] += num_detections
        
        # Calculate average processing time
        if len(self.frame_times) > 0:
            self.processing_stats['average_processing_time'] = np.mean(list(self.frame_times))
            self.processing_stats['current_fps'] = 1000.0 / self.processing_stats['average_processing_time']
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return {
            **self.processing_stats,
            'buffer_size': self.frame_buffer.qsize(),
            'result_buffer_size': self.result_buffer.qsize(),
            'active_tracks': len(self.tracker.get_active_tracks()),
            'tracking_stats': self.tracker.get_tracking_statistics()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components."""
        health_status = {
            'status': 'healthy',
            'timestamp': time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            'components': {}
        }
        
        try:
            # Check detector
            health_status['components']['detector'] = {
                'status': 'healthy' if self.detector.model is not None else 'unhealthy',
                'model_info': self.detector.get_model_info()
            }
            
            # Check weight estimator
            health_status['components']['weight_estimator'] = {
                'status': 'healthy' if self.weight_estimator.is_loaded else 'unhealthy',
                'model_info': self.weight_estimator.get_model_info()
            }
            
            # Check tracker
            health_status['components']['tracker'] = {
                'status': 'healthy',
                'active_tracks': len(self.tracker.get_active_tracks())
            }
            
            # Overall health
            component_statuses = [comp['status'] for comp in health_status['components'].values()]
            if all(status == 'healthy' for status in component_statuses):
                health_status['status'] = 'healthy'
            else:
                health_status['status'] = 'degraded'
            
        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['error'] = str(e)
        
        return health_status


class StreamProcessingServer:
    """Flask server for real-time stream processing."""
    
    def __init__(self, processor: RealTimeStreamProcessor, port: int = 8080):
        self.processor = processor
        self.app = Flask(__name__)
        self.port = port
        
        # Setup routes
        self._setup_routes()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _setup_routes(self):
        """Setup Flask routes for the server."""
        
        @self.app.route('/ping', methods=['GET'])
        def ping():
            """Health check endpoint for SageMaker."""
            return jsonify({'status': 'healthy'}), 200
        
        @self.app.route('/invocations', methods=['POST'])
        def invocations():
            """Main inference endpoint for SageMaker."""
            try:
                # Parse request
                request_data = request.get_json()
                
                if not request_data or 'stream_data' not in request_data:
                    return jsonify({'error': 'Invalid request format'}), 400
                
                stream_data = request_data['stream_data']
                
                # Extract required fields
                frame_data = stream_data.get('frame')
                camera_id = stream_data.get('camera_id', 'default')
                frame_sequence = stream_data.get('frame_sequence', 0)
                
                if not frame_data:
                    return jsonify({'error': 'No frame data provided'}), 400
                
                # Process frame
                result = self.processor.process_frame(frame_data, camera_id, frame_sequence)
                
                return jsonify(result), 200
                
            except Exception as e:
                self.logger.error(f"Inference failed: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/health', methods=['GET'])
        def health():
            """Detailed health check endpoint."""
            health_status = self.processor.health_check()
            status_code = 200 if health_status['status'] == 'healthy' else 503
            return jsonify(health_status), status_code
        
        @self.app.route('/stats', methods=['GET'])
        def stats():
            """Get processing statistics."""
            return jsonify(self.processor.get_processing_stats()), 200
        
        @self.app.route('/reset', methods=['POST'])
        def reset():
            """Reset tracker state."""
            try:
                self.processor.tracker.reset_tracker()
                return jsonify({'status': 'reset_complete'}), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def run(self, host: str = '0.0.0.0', debug: bool = False):
        """Start the Flask server."""
        self.logger.info(f"Starting stream processing server on {host}:{self.port}")
        self.app.run(host=host, port=self.port, debug=debug, threaded=True)


class FrameBuffer:
    """Thread-safe frame buffer for continuous stream processing."""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.buffer = queue.Queue(maxsize=max_size)
        self.dropped_frames = 0
        self.lock = threading.Lock()
    
    def add_frame(self, frame_data: Dict[str, Any]) -> bool:
        """
        Add frame to buffer.
        
        Returns:
            True if frame was added, False if buffer is full
        """
        try:
            self.buffer.put_nowait(frame_data)
            return True
        except queue.Full:
            with self.lock:
                self.dropped_frames += 1
            return False
    
    def get_frame(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Get frame from buffer with timeout."""
        try:
            return self.buffer.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self.lock:
            return {
                'buffer_size': self.buffer.qsize(),
                'max_size': self.max_size,
                'dropped_frames': self.dropped_frames,
                'utilization': self.buffer.qsize() / self.max_size
            }
    
    def clear(self):
        """Clear the buffer."""
        while not self.buffer.empty():
            try:
                self.buffer.get_nowait()
            except queue.Empty:
                break
        
        with self.lock:
            self.dropped_frames = 0


# Utility functions for frame processing
def validate_frame_request(request_data: Dict[str, Any]) -> bool:
    """Validate frame processing request."""
    required_fields = ['stream_data']
    
    if not all(field in request_data for field in required_fields):
        return False
    
    stream_data = request_data['stream_data']
    if 'frame' not in stream_data:
        return False
    
    return True


def create_error_response(error_message: str, camera_id: str = "unknown") -> Dict[str, Any]:
    """Create standardized error response."""
    return {
        "frame_results": {
            "camera_id": camera_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "status": "error",
            "error_message": error_message,
            "detections": [],
            "total_chickens_detected": 0,
            "processing_time_ms": 0.0
        }
    }