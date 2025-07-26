"""
Utility classes and functions for the Chicken Weight Estimation SDK
"""

import cv2
import numpy as np
import asyncio
from typing import List, Dict, Any, Optional, Callable, Union
from pathlib import Path
import logging
import time
from datetime import datetime
import json

from .models import ProcessingResult, BatchProcessingResult, ChickenDetection
from .exceptions import ProcessingError, ValidationError

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Utility class for image processing operations."""
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: tuple = (640, 640)) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio.
        
        Args:
            image: Input image
            target_size: Target size (width, height)
            
        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create padded image
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Calculate padding
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        
        # Place resized image in center
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        return padded
    
    @staticmethod
    def enhance_image(image: np.ndarray, brightness: float = 1.0, contrast: float = 1.0) -> np.ndarray:
        """
        Enhance image brightness and contrast.
        
        Args:
            image: Input image
            brightness: Brightness factor (1.0 = no change)
            contrast: Contrast factor (1.0 = no change)
            
        Returns:
            Enhanced image
        """
        # Apply brightness and contrast
        enhanced = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness * 50)
        return enhanced
    
    @staticmethod
    def crop_detection(image: np.ndarray, bbox: Dict[str, float], padding: int = 10) -> np.ndarray:
        """
        Crop image around detection bounding box.
        
        Args:
            image: Input image
            bbox: Bounding box coordinates
            padding: Padding around bounding box
            
        Returns:
            Cropped image
        """
        h, w = image.shape[:2]
        
        # Extract coordinates
        x1 = max(0, int(bbox['x1']) - padding)
        y1 = max(0, int(bbox['y1']) - padding)
        x2 = min(w, int(bbox['x2']) + padding)
        y2 = min(h, int(bbox['y2']) + padding)
        
        return image[y1:y2, x1:x2]
    
    @staticmethod
    def draw_detections(image: np.ndarray, detections: List[ChickenDetection]) -> np.ndarray:
        """
        Draw detection bounding boxes on image.
        
        Args:
            image: Input image
            detections: List of detections
            
        Returns:
            Image with drawn detections
        """
        result_image = image.copy()
        
        for detection in detections:
            bbox = detection.bbox
            
            # Draw bounding box
            cv2.rectangle(
                result_image,
                (int(bbox['x1']), int(bbox['y1'])),
                (int(bbox['x2']), int(bbox['y2'])),
                (0, 255, 0), 2
            )
            
            # Draw confidence and weight
            label = f"Conf: {detection.confidence:.2f}"
            if detection.weight_estimate:
                label += f", Weight: {detection.weight_estimate:.2f}kg"
            
            cv2.putText(
                result_image, label,
                (int(bbox['x1']), int(bbox['y1']) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )
        
        return result_image


class VideoProcessor:
    """Utility class for video processing operations."""
    
    def __init__(self, client):
        """
        Initialize video processor.
        
        Args:
            client: ChickenWeightClient instance
        """
        self.client = client
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def extract_frames(self, video_path: Union[str, Path], 
                      output_dir: Union[str, Path],
                      interval: int = 30) -> List[str]:
        """
        Extract frames from video at specified intervals.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            interval: Frame interval (extract every N frames)
            
        Returns:
            List of extracted frame paths
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ProcessingError(f"Could not open video: {video_path}")
        
        frame_paths = []
        frame_id = 0
        saved_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_id % interval == 0:
                    frame_filename = f"frame_{saved_count:06d}.jpg"
                    frame_path = output_dir / frame_filename
                    
                    cv2.imwrite(str(frame_path), frame)
                    frame_paths.append(str(frame_path))
                    saved_count += 1
                
                frame_id += 1
                
        finally:
            cap.release()
        
        self.logger.info(f"Extracted {len(frame_paths)} frames from {video_path}")
        return frame_paths
    
    def create_summary_video(self, results: List[ProcessingResult],
                           original_video_path: Union[str, Path],
                           output_path: Union[str, Path],
                           show_detections: bool = True) -> str:
        """
        Create summary video with detection overlays.
        
        Args:
            results: Processing results for each frame
            original_video_path: Path to original video
            output_path: Path for output video
            show_detections: Whether to show detection overlays
            
        Returns:
            Path to created summary video
        """
        original_video_path = Path(original_video_path)
        output_path = Path(output_path)
        
        # Open original video
        cap = cv2.VideoCapture(str(original_video_path))
        if not cap.isOpened():
            raise ProcessingError(f"Could not open video: {original_video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Create results lookup
        results_dict = {r.frame_id: r for r in results}
        
        frame_id = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Add detection overlays if available
                if show_detections and frame_id in results_dict:
                    result = results_dict[frame_id]
                    frame = ImageProcessor.draw_detections(frame, result.detections)
                    
                    # Add summary text
                    summary_text = f"Frame {frame_id}: {len(result.detections)} chickens"
                    if result.average_weight:
                        summary_text += f", Avg weight: {result.average_weight:.2f}kg"
                    
                    cv2.putText(frame, summary_text, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                out.write(frame)
                frame_id += 1
                
        finally:
            cap.release()
            out.release()
        
        self.logger.info(f"Created summary video: {output_path}")
        return str(output_path)


class BatchProcessor:
    """Utility class for batch processing operations."""
    
    def __init__(self, client):
        """
        Initialize batch processor.
        
        Args:
            client: ChickenWeightClient or AsyncChickenWeightClient
        """
        self.client = client
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def process_image_directory(self, 
                               directory_path: Union[str, Path],
                               camera_id: str = 'batch',
                               file_pattern: str = "*.jpg",
                               progress_callback: Optional[Callable[[int, int], None]] = None) -> BatchProcessingResult:
        """
        Process all images in a directory.
        
        Args:
            directory_path: Path to directory containing images
            camera_id: Camera identifier for processing
            file_pattern: File pattern to match (e.g., "*.jpg", "*.png")
            progress_callback: Optional progress callback
            
        Returns:
            BatchProcessingResult
        """
        directory_path = Path(directory_path)
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find image files
        image_files = list(directory_path.glob(file_pattern))
        if not image_files:
            raise ValidationError(f"No images found matching pattern: {file_pattern}")
        
        # Sort files for consistent processing order
        image_files.sort()
        
        start_time = datetime.now()
        results = []
        successful_count = 0
        failed_count = 0
        total_processing_time = 0.0
        
        for i, image_path in enumerate(image_files):
            try:
                result = self.client.process_image(image_path, camera_id, frame_id=i)
                results.append(result)
                successful_count += 1
                total_processing_time += result.processing_time
                
                if progress_callback:
                    progress_callback(i + 1, len(image_files))
                    
            except Exception as e:
                self.logger.error(f"Failed to process {image_path}: {str(e)}")
                failed_count += 1
        
        end_time = datetime.now()
        
        return BatchProcessingResult(
            total_frames=len(image_files),
            successful_frames=successful_count,
            failed_frames=failed_count,
            total_processing_time=total_processing_time,
            results=results,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat()
        )
    
    async def process_image_directory_async(self,
                                          directory_path: Union[str, Path],
                                          camera_id: str = 'batch',
                                          file_pattern: str = "*.jpg",
                                          batch_size: int = 10) -> BatchProcessingResult:
        """
        Process all images in a directory asynchronously.
        
        Args:
            directory_path: Path to directory containing images
            camera_id: Camera identifier for processing
            file_pattern: File pattern to match
            batch_size: Number of images to process concurrently
            
        Returns:
            BatchProcessingResult
        """
        directory_path = Path(directory_path)
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find image files
        image_files = list(directory_path.glob(file_pattern))
        if not image_files:
            raise ValidationError(f"No images found matching pattern: {file_pattern}")
        
        image_files.sort()
        
        start_time = datetime.now()
        all_results = []
        successful_count = 0
        failed_count = 0
        total_processing_time = 0.0
        
        # Process in batches
        import aiohttp
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(image_files), batch_size):
                batch_files = image_files[i:i + batch_size]
                
                # Create tasks for batch
                tasks = [
                    self.client.process_image_async(session, str(image_path), camera_id, j)
                    for j, image_path in enumerate(batch_files, start=i)
                ]
                
                # Execute batch
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
        
        end_time = datetime.now()
        
        return BatchProcessingResult(
            total_frames=len(image_files),
            successful_frames=successful_count,
            failed_frames=failed_count,
            total_processing_time=total_processing_time,
            results=all_results,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat()
        )
    
    def generate_batch_report(self, batch_result: BatchProcessingResult,
                            output_path: Union[str, Path]) -> str:
        """
        Generate a detailed batch processing report.
        
        Args:
            batch_result: Batch processing result
            output_path: Path for output report
            
        Returns:
            Path to generated report
        """
        output_path = Path(output_path)
        
        # Generate report data
        report_data = {
            'summary': batch_result.to_dict(),
            'detailed_results': [],
            'statistics': {
                'weight_stats': batch_result.get_weight_statistics(),
                'confidence_stats': self._calculate_confidence_stats(batch_result.results),
                'processing_time_stats': self._calculate_processing_time_stats(batch_result.results)
            }
        }
        
        # Add detailed results
        for result in batch_result.results:
            report_data['detailed_results'].append(result.to_dict())
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        self.logger.info(f"Generated batch report: {output_path}")
        return str(output_path)
    
    def _calculate_confidence_stats(self, results: List[ProcessingResult]) -> Dict[str, float]:
        """Calculate confidence statistics."""
        all_confidences = []
        for result in results:
            for detection in result.detections:
                all_confidences.append(detection.confidence)
        
        if not all_confidences:
            return {'count': 0, 'mean': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0}
        
        import statistics
        return {
            'count': len(all_confidences),
            'mean': statistics.mean(all_confidences),
            'min': min(all_confidences),
            'max': max(all_confidences),
            'std': statistics.stdev(all_confidences) if len(all_confidences) > 1 else 0.0
        }
    
    def _calculate_processing_time_stats(self, results: List[ProcessingResult]) -> Dict[str, float]:
        """Calculate processing time statistics."""
        processing_times = [r.processing_time for r in results if r.processing_time > 0]
        
        if not processing_times:
            return {'count': 0, 'mean': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0}
        
        import statistics
        return {
            'count': len(processing_times),
            'mean': statistics.mean(processing_times),
            'min': min(processing_times),
            'max': max(processing_times),
            'std': statistics.stdev(processing_times) if len(processing_times) > 1 else 0.0
        }


class ConfigurationManager:
    """Utility class for managing SDK configuration."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = Path(config_path) if config_path else None
        self.config = {}
        
        if self.config_path and self.config_path.exists():
            self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if not self.config_path or not self.config_path.exists():
            return {}
        
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        return self.config
    
    def save_config(self) -> None:
        """Save configuration to file."""
        if not self.config_path:
            raise ValidationError("No config path specified")
        
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_client_config(self) -> Dict[str, Any]:
        """Get client configuration."""
        return {
            'endpoint_url': self.get('client.endpoint_url'),
            'api_key': self.get('client.api_key'),
            'timeout': self.get('client.timeout', 30.0),
            'retry_attempts': self.get('client.retry_attempts', 3),
            'retry_delay': self.get('client.retry_delay', 1.0)
        }


def setup_logging(level: str = 'INFO', log_file: Optional[str] = None) -> None:
    """
    Setup logging for the SDK.
    
    Args:
        level: Logging level
        log_file: Optional log file path
    """
    log_level = getattr(logging, level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # Setup file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def validate_image(image_path: Union[str, Path]) -> bool:
    """
    Validate that an image file can be processed.
    
    Args:
        image_path: Path to image file
        
    Returns:
        True if image is valid
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        return False
    
    # Check file extension
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    if image_path.suffix.lower() not in valid_extensions:
        return False
    
    # Try to load image
    try:
        image = cv2.imread(str(image_path))
        return image is not None and image.size > 0
    except Exception:
        return False


def validate_video(video_path: Union[str, Path]) -> bool:
    """
    Validate that a video file can be processed.
    
    Args:
        video_path: Path to video file
        
    Returns:
        True if video is valid
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        return False
    
    # Check file extension
    valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
    if video_path.suffix.lower() not in valid_extensions:
        return False
    
    # Try to open video
    try:
        cap = cv2.VideoCapture(str(video_path))
        is_valid = cap.isOpened()
        cap.release()
        return is_valid
    except Exception:
        return False