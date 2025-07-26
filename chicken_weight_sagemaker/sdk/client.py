"""
Client SDK for Chicken Weight Estimation System
"""

import requests
import asyncio
import aiohttp
import base64
import json
import numpy as np
import cv2
from typing import Dict, List, Optional, Callable, Union, Any
from datetime import datetime
import logging
import time
from pathlib import Path

from .models import ChickenDetection, ProcessingResult, TrackingInfo
from .exceptions import ChickenWeightSDKError, EndpointError, ProcessingError

logger = logging.getLogger(__name__)


class ChickenWeightClient:
    """Synchronous client for chicken weight estimation system."""
    
    def __init__(self, 
                 endpoint_url: str,
                 api_key: Optional[str] = None,
                 timeout: float = 30.0,
                 retry_attempts: int = 3,
                 retry_delay: float = 1.0):
        """
        Initialize the client.
        
        Args:
            endpoint_url: SageMaker endpoint URL
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts for failed requests
            retry_delay: Delay between retry attempts
        """
        self.endpoint_url = endpoint_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        # Setup session
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({'X-API-Key': api_key})
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def process_image(self, 
                     image: Union[np.ndarray, str, Path],
                     camera_id: str = 'default',
                     frame_id: Optional[int] = None) -> ProcessingResult:
        """
        Process a single image for chicken detection and weight estimation.
        
        Args:
            image: Image as numpy array, file path, or base64 string
            camera_id: Camera identifier
            frame_id: Optional frame identifier
            
        Returns:
            ProcessingResult containing detections and tracking info
        """
        # Convert image to base64
        image_b64 = self._prepare_image(image)
        
        # Prepare request
        request_data = {
            "stream_data": {
                "frame": image_b64,
                "camera_id": camera_id,
                "frame_sequence": frame_id or int(time.time()),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Make request with retries
        for attempt in range(self.retry_attempts):
            try:
                response = self.session.post(
                    f"{self.endpoint_url}/invocations",
                    json=request_data,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                # Parse response
                result_data = response.json()
                return self._parse_response(result_data)
                
            except requests.exceptions.RequestException as e:
                if attempt == self.retry_attempts - 1:
                    raise EndpointError(f"Failed to process image after {self.retry_attempts} attempts: {str(e)}")
                
                self.logger.warning(f"Request attempt {attempt + 1} failed: {str(e)}")
                time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
        
        raise EndpointError("Unexpected error in request processing")
    
    def process_video(self, 
                     video_path: Union[str, Path],
                     camera_id: str = 'default',
                     callback: Optional[Callable[[ProcessingResult], None]] = None,
                     skip_frames: int = 0,
                     max_frames: Optional[int] = None,
                     progress_callback: Optional[Callable[[int, int], None]] = None) -> List[ProcessingResult]:
        """
        Process a video file frame by frame.
        
        Args:
            video_path: Path to video file
            camera_id: Camera identifier
            callback: Optional callback function called for each frame result
            skip_frames: Number of frames to skip between processing
            max_frames: Maximum number of frames to process
            progress_callback: Optional progress callback (current_frame, total_frames)
            
        Returns:
            List of ProcessingResult objects
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ProcessingError(f"Could not open video file: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        results = []
        frame_id = 0
        processed_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if requested
                if frame_id % (skip_frames + 1) != 0:
                    frame_id += 1
                    continue
                
                # Process frame
                try:
                    result = self.process_image(frame, camera_id, frame_id)
                    results.append(result)
                    
                    if callback:
                        callback(result)
                    
                    processed_count += 1
                    
                    # Progress callback
                    if progress_callback:
                        progress_callback(processed_count, total_frames // (skip_frames + 1))
                    
                    # Check max frames limit
                    if max_frames and processed_count >= max_frames:
                        break
                        
                except Exception as e:
                    self.logger.error(f"Error processing frame {frame_id}: {str(e)}")
                    continue
                
                frame_id += 1
                
        finally:
            cap.release()
        
        return results
    
    def process_live_stream(self,
                           stream_url: str,
                           camera_id: str,
                           duration_seconds: Optional[int] = None,
                           callback: Optional[Callable[[ProcessingResult], None]] = None) -> List[ProcessingResult]:
        """
        Process a live video stream.
        
        Args:
            stream_url: RTSP or HTTP stream URL
            camera_id: Camera identifier
            duration_seconds: Optional duration limit
            callback: Optional callback for each frame result
            
        Returns:
            List of ProcessingResult objects
        """
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            raise ProcessingError(f"Could not open stream: {stream_url}")
        
        results = []
        frame_id = 0
        start_time = time.time()
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame from stream")
                    continue
                
                # Check duration limit
                if duration_seconds and (time.time() - start_time) > duration_seconds:
                    break
                
                try:
                    result = self.process_image(frame, camera_id, frame_id)
                    results.append(result)
                    
                    if callback:
                        callback(result)
                        
                except Exception as e:
                    self.logger.error(f"Error processing stream frame {frame_id}: {str(e)}")
                
                frame_id += 1
                
        except KeyboardInterrupt:
            self.logger.info("Stream processing interrupted by user")
        finally:
            cap.release()
        
        return results
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Check the health status of the endpoint.
        
        Returns:
            Health status information
        """
        try:
            response = self.session.get(f"{self.endpoint_url}/ping", timeout=10.0)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise EndpointError(f"Health check failed: {str(e)}")
    
    def _prepare_image(self, image: Union[np.ndarray, str, Path]) -> str:
        """Convert image to base64 string."""
        if isinstance(image, (str, Path)):
            # Load from file
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            with open(image_path, 'rb') as f:
                image_data = f.read()
            return base64.b64encode(image_data).decode('utf-8')
        
        elif isinstance(image, np.ndarray):
            # Encode numpy array
            success, buffer = cv2.imencode('.jpg', image)
            if not success:
                raise ProcessingError("Failed to encode image")
            return base64.b64encode(buffer).decode('utf-8')
        
        elif isinstance(image, str):
            # Assume it's already base64
            return image
        
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def _parse_response(self, response_data: Dict[str, Any]) -> ProcessingResult:
        """Parse API response into ProcessingResult."""
        if 'error' in response_data:
            raise ProcessingError(f"Server error: {response_data['error']}")
        
        # Parse detections
        detections = []
        for det_data in response_data.get('detections', []):
            detection = ChickenDetection(
                bbox=det_data['bbox'],
                confidence=det_data['bbox']['confidence'],
                class_name=det_data.get('class_name', 'chicken'),
                weight_estimate=det_data.get('weight_estimate'),
                occlusion_score=det_data.get('occlusion_score', 0.0),
                distance_estimate=det_data.get('distance_estimate')
            )
            detections.append(detection)
        
        # Parse tracking info
        tracks = []
        for track_data in response_data.get('tracks', []):
            track = TrackingInfo(
                track_id=track_data['track_id'],
                bbox=track_data['bbox'],
                weight_history=track_data.get('weight_history', []),
                average_weight=track_data.get('average_weight'),
                confidence=track_data.get('confidence', 1.0)
            )
            tracks.append(track)
        
        return ProcessingResult(
            frame_id=response_data.get('frame_id', 0),
            timestamp=response_data.get('timestamp', datetime.now().isoformat()),
            detections=detections,
            tracks=tracks,
            processing_time=response_data.get('processing_time', 0.0),
            camera_id=response_data.get('camera_id', 'unknown')
        )


class AsyncChickenWeightClient:
    """Asynchronous client for high-throughput processing."""
    
    def __init__(self,
                 endpoint_url: str,
                 api_key: Optional[str] = None,
                 timeout: float = 30.0,
                 max_concurrent: int = 10,
                 retry_attempts: int = 3):
        """
        Initialize the async client.
        
        Args:
            endpoint_url: SageMaker endpoint URL
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            max_concurrent: Maximum concurrent requests
            retry_attempts: Number of retry attempts
        """
        self.endpoint_url = endpoint_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.retry_attempts = retry_attempts
        
        # Semaphore for controlling concurrency
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def process_image_async(self,
                                 session: aiohttp.ClientSession,
                                 image: Union[np.ndarray, str, Path],
                                 camera_id: str = 'default',
                                 frame_id: Optional[int] = None) -> ProcessingResult:
        """
        Process a single image asynchronously.
        
        Args:
            session: aiohttp session
            image: Image to process
            camera_id: Camera identifier
            frame_id: Optional frame identifier
            
        Returns:
            ProcessingResult
        """
        async with self.semaphore:
            # Prepare image
            image_b64 = self._prepare_image(image)
            
            # Prepare request
            request_data = {
                "stream_data": {
                    "frame": image_b64,
                    "camera_id": camera_id,
                    "frame_sequence": frame_id or int(time.time()),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            headers = {}
            if self.api_key:
                headers['X-API-Key'] = self.api_key
            
            # Make request with retries
            for attempt in range(self.retry_attempts):
                try:
                    async with session.post(
                        f"{self.endpoint_url}/invocations",
                        json=request_data,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        response.raise_for_status()
                        result_data = await response.json()
                        return self._parse_response(result_data)
                        
                except aiohttp.ClientError as e:
                    if attempt == self.retry_attempts - 1:
                        raise EndpointError(f"Failed to process image: {str(e)}")
                    
                    await asyncio.sleep(1.0 * (2 ** attempt))  # Exponential backoff
            
            raise EndpointError("Unexpected error in async request processing")
    
    async def process_video_batch(self,
                                 video_path: Union[str, Path],
                                 camera_id: str = 'default',
                                 batch_size: int = 10,
                                 skip_frames: int = 0,
                                 max_frames: Optional[int] = None) -> List[ProcessingResult]:
        """
        Process video in batches asynchronously.
        
        Args:
            video_path: Path to video file
            camera_id: Camera identifier
            batch_size: Number of frames to process concurrently
            skip_frames: Frames to skip between processing
            max_frames: Maximum frames to process
            
        Returns:
            List of ProcessingResult objects
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Extract frames
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        frame_ids = []
        frame_id = 0
        processed_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_id % (skip_frames + 1) == 0:
                    frames.append(frame)
                    frame_ids.append(frame_id)
                    processed_count += 1
                    
                    if max_frames and processed_count >= max_frames:
                        break
                
                frame_id += 1
        finally:
            cap.release()
        
        # Process in batches
        results = []
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(frames), batch_size):
                batch_frames = frames[i:i + batch_size]
                batch_ids = frame_ids[i:i + batch_size]
                
                # Create tasks for batch
                tasks = [
                    self.process_image_async(session, frame, camera_id, fid)
                    for frame, fid in zip(batch_frames, batch_ids)
                ]
                
                # Execute batch
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Filter out exceptions and add successful results
                for result in batch_results:
                    if isinstance(result, ProcessingResult):
                        results.append(result)
                    else:
                        self.logger.error(f"Batch processing error: {result}")
        
        return results
    
    def _prepare_image(self, image: Union[np.ndarray, str, Path]) -> str:
        """Convert image to base64 string."""
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            with open(image_path, 'rb') as f:
                image_data = f.read()
            return base64.b64encode(image_data).decode('utf-8')
        
        elif isinstance(image, np.ndarray):
            success, buffer = cv2.imencode('.jpg', image)
            if not success:
                raise ProcessingError("Failed to encode image")
            return base64.b64encode(buffer).decode('utf-8')
        
        elif isinstance(image, str):
            return image
        
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def _parse_response(self, response_data: Dict[str, Any]) -> ProcessingResult:
        """Parse API response into ProcessingResult."""
        if 'error' in response_data:
            raise ProcessingError(f"Server error: {response_data['error']}")
        
        # Parse detections
        detections = []
        for det_data in response_data.get('detections', []):
            detection = ChickenDetection(
                bbox=det_data['bbox'],
                confidence=det_data['bbox']['confidence'],
                class_name=det_data.get('class_name', 'chicken'),
                weight_estimate=det_data.get('weight_estimate'),
                occlusion_score=det_data.get('occlusion_score', 0.0),
                distance_estimate=det_data.get('distance_estimate')
            )
            detections.append(detection)
        
        # Parse tracking info
        tracks = []
        for track_data in response_data.get('tracks', []):
            track = TrackingInfo(
                track_id=track_data['track_id'],
                bbox=track_data['bbox'],
                weight_history=track_data.get('weight_history', []),
                average_weight=track_data.get('average_weight'),
                confidence=track_data.get('confidence', 1.0)
            )
            tracks.append(track)
        
        return ProcessingResult(
            frame_id=response_data.get('frame_id', 0),
            timestamp=response_data.get('timestamp', datetime.now().isoformat()),
            detections=detections,
            tracks=tracks,
            processing_time=response_data.get('processing_time', 0.0),
            camera_id=response_data.get('camera_id', 'unknown')
        )