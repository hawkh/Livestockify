#!/usr/bin/env python3
"""
CloudWatch metrics and monitoring for chicken weight estimation system.
"""

import boto3
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class MetricData:
    """Container for metric data."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    dimensions: Dict[str, str]


@dataclass
class AlarmConfig:
    """Configuration for CloudWatch alarms."""
    name: str
    metric_name: str
    threshold: float
    comparison_operator: str
    evaluation_periods: int
    period: int
    statistic: str
    alarm_actions: List[str]
    ok_actions: List[str]
    treat_missing_data: str = "notBreaching"


class CloudWatchMetrics:
    """CloudWatch metrics publisher for chicken weight estimation system."""
    
    def __init__(self, namespace: str = "ChickenWeightEstimation", region: str = "us-east-1"):
        self.namespace = namespace
        self.region = region
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        
        # Metric buffer for batch publishing
        self.metric_buffer = deque(maxlen=1000)
        self.buffer_lock = threading.Lock()
        
        # Background publishing thread
        self.publishing_thread = None
        self.stop_publishing = threading.Event()
        
        # Metric aggregation
        self.metric_aggregator = defaultdict(list)
        self.last_publish_time = time.time()
        
        logger.info(f"CloudWatch metrics initialized for namespace: {namespace}")
    
    def start_background_publishing(self, interval: int = 60):
        """Start background thread for publishing metrics."""
        if self.publishing_thread and self.publishing_thread.is_alive():
            logger.warning("Background publishing already running")
            return
        
        self.stop_publishing.clear()
        self.publishing_thread = threading.Thread(
            target=self._background_publisher,
            args=(interval,),
            daemon=True
        )
        self.publishing_thread.start()
        logger.info(f"Started background metric publishing (interval: {interval}s)")
    
    def stop_background_publishing(self):
        """Stop background publishing thread."""
        if self.publishing_thread:
            self.stop_publishing.set()
            self.publishing_thread.join(timeout=10)
            logger.info("Stopped background metric publishing")
    
    def _background_publisher(self, interval: int):
        """Background thread for publishing metrics."""
        while not self.stop_publishing.wait(interval):
            try:
                self.publish_buffered_metrics()
            except Exception as e:
                logger.error(f"Error in background metric publishing: {e}")
    
    def put_metric(self, name: str, value: float, unit: str = "Count", 
                   dimensions: Optional[Dict[str, str]] = None):
        """Add a metric to the buffer for publishing."""
        metric = MetricData(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.utcnow(),
            dimensions=dimensions or {}
        )
        
        with self.buffer_lock:
            self.metric_buffer.append(metric)
        
        logger.debug(f"Added metric: {name}={value} {unit}")
    
    def put_inference_metrics(self, processing_time: float, detections: int, 
                            errors: int = 0, camera_id: str = "default"):
        """Put inference-specific metrics."""
        dimensions = {"CameraId": camera_id}
        
        self.put_metric("ProcessingTime", processing_time, "Milliseconds", dimensions)
        self.put_metric("DetectionCount", detections, "Count", dimensions)
        self.put_metric("InferenceRequests", 1, "Count", dimensions)
        
        if errors > 0:
            self.put_metric("InferenceErrors", errors, "Count", dimensions)
    
    def put_weight_estimation_metrics(self, weights: List[float], confidences: List[float],
                                    camera_id: str = "default"):
        """Put weight estimation specific metrics."""
        if not weights:
            return
        
        dimensions = {"CameraId": camera_id}
        
        # Average weight and confidence
        avg_weight = sum(weights) / len(weights)
        avg_confidence = sum(confidences) / len(confidences)
        
        self.put_metric("AverageWeight", avg_weight, "None", dimensions)
        self.put_metric("AverageConfidence", avg_confidence, "Percent", dimensions)
        self.put_metric("WeightEstimations", len(weights), "Count", dimensions)
        
        # Weight distribution metrics
        if len(weights) > 1:
            weight_std = (sum((w - avg_weight) ** 2 for w in weights) / len(weights)) ** 0.5
            self.put_metric("WeightStandardDeviation", weight_std, "None", dimensions)
    
    def put_tracking_metrics(self, active_tracks: int, new_tracks: int, 
                           lost_tracks: int, camera_id: str = "default"):
        """Put tracking-specific metrics."""
        dimensions = {"CameraId": camera_id}
        
        self.put_metric("ActiveTracks", active_tracks, "Count", dimensions)
        self.put_metric("NewTracks", new_tracks, "Count", dimensions)
        self.put_metric("LostTracks", lost_tracks, "Count", dimensions)
    
    def put_system_metrics(self, cpu_usage: float, memory_usage: float, 
                          gpu_usage: float = 0.0):
        """Put system resource metrics."""
        self.put_metric("CPUUsage", cpu_usage, "Percent")
        self.put_metric("MemoryUsage", memory_usage, "Percent")
        
        if gpu_usage > 0:
            self.put_metric("GPUUsage", gpu_usage, "Percent")
    
    def put_api_metrics(self, endpoint: str, response_time: float, status_code: int):
        """Put API endpoint metrics."""
        dimensions = {"Endpoint": endpoint, "StatusCode": str(status_code)}
        
        self.put_metric("APIResponseTime", response_time, "Milliseconds", dimensions)
        self.put_metric("APIRequests", 1, "Count", dimensions)
        
        if status_code >= 400:
            self.put_metric("APIErrors", 1, "Count", dimensions)
    
    def publish_buffered_metrics(self):
        """Publish all buffered metrics to CloudWatch."""
        if not self.metric_buffer:
            return
        
        with self.buffer_lock:
            metrics_to_publish = list(self.metric_buffer)
            self.metric_buffer.clear()
        
        if not metrics_to_publish:
            return
        
        # Group metrics by 20 (CloudWatch limit)
        batch_size = 20
        for i in range(0, len(metrics_to_publish), batch_size):
            batch = metrics_to_publish[i:i + batch_size]
            self._publish_metric_batch(batch)
        
        logger.info(f"Published {len(metrics_to_publish)} metrics to CloudWatch")
    
    def _publish_metric_batch(self, metrics: List[MetricData]):
        """Publish a batch of metrics to CloudWatch."""
        try:
            metric_data = []
            
            for metric in metrics:
                data_point = {
                    'MetricName': metric.name,
                    'Value': metric.value,
                    'Unit': metric.unit,
                    'Timestamp': metric.timestamp
                }
                
                if metric.dimensions:
                    data_point['Dimensions'] = [
                        {'Name': k, 'Value': v} for k, v in metric.dimensions.items()
                    ]
                
                metric_data.append(data_point)
            
            self.cloudwatch.put_metric_data(
                Namespace=self.namespace,
                MetricData=metric_data
            )
            
        except Exception as e:
            logger.error(f"Failed to publish metric batch: {e}")
            # Re-add metrics to buffer for retry
            with self.buffer_lock:
                self.metric_buffer.extend(metrics)
    
    def create_alarm(self, config: AlarmConfig):
        """Create a CloudWatch alarm."""
        try:
            alarm_kwargs = {
                'AlarmName': config.name,
                'ComparisonOperator': config.comparison_operator,
                'EvaluationPeriods': config.evaluation_periods,
                'MetricName': config.metric_name,
                'Namespace': self.namespace,
                'Period': config.period,
                'Statistic': config.statistic,
                'Threshold': config.threshold,
                'ActionsEnabled': True,
                'TreatMissingData': config.treat_missing_data
            }
            
            if config.alarm_actions:
                alarm_kwargs['AlarmActions'] = config.alarm_actions
            
            if config.ok_actions:
                alarm_kwargs['OKActions'] = config.ok_actions
            
            self.cloudwatch.put_metric_alarm(**alarm_kwargs)
            logger.info(f"Created CloudWatch alarm: {config.name}")
            
        except Exception as e:
            logger.error(f"Failed to create alarm {config.name}: {e}")
    
    def create_default_alarms(self, sns_topic_arn: Optional[str] = None):
        """Create default alarms for the chicken weight estimation system."""
        alarm_actions = [sns_topic_arn] if sns_topic_arn else []
        
        alarms = [
            AlarmConfig(
                name="ChickenWeightEstimation-HighProcessingTime",
                metric_name="ProcessingTime",
                threshold=1000.0,  # 1 second
                comparison_operator="GreaterThanThreshold",
                evaluation_periods=2,
                period=300,  # 5 minutes
                statistic="Average",
                alarm_actions=alarm_actions,
                ok_actions=alarm_actions
            ),
            AlarmConfig(
                name="ChickenWeightEstimation-HighErrorRate",
                metric_name="InferenceErrors",
                threshold=10.0,
                comparison_operator="GreaterThanThreshold",
                evaluation_periods=1,
                period=300,
                statistic="Sum",
                alarm_actions=alarm_actions,
                ok_actions=alarm_actions
            ),
            AlarmConfig(
                name="ChickenWeightEstimation-LowDetectionRate",
                metric_name="DetectionCount",
                threshold=1.0,
                comparison_operator="LessThanThreshold",
                evaluation_periods=3,
                period=300,
                statistic="Average",
                alarm_actions=alarm_actions,
                ok_actions=alarm_actions
            ),
            AlarmConfig(
                name="ChickenWeightEstimation-HighCPUUsage",
                metric_name="CPUUsage",
                threshold=80.0,
                comparison_operator="GreaterThanThreshold",
                evaluation_periods=2,
                period=300,
                statistic="Average",
                alarm_actions=alarm_actions,
                ok_actions=alarm_actions
            ),
            AlarmConfig(
                name="ChickenWeightEstimation-HighMemoryUsage",
                metric_name="MemoryUsage",
                threshold=85.0,
                comparison_operator="GreaterThanThreshold",
                evaluation_periods=2,
                period=300,
                statistic="Average",
                alarm_actions=alarm_actions,
                ok_actions=alarm_actions
            )
        ]
        
        for alarm in alarms:
            self.create_alarm(alarm)
    
    def get_metric_statistics(self, metric_name: str, start_time: datetime, 
                            end_time: datetime, period: int = 300,
                            statistic: str = "Average",
                            dimensions: Optional[Dict[str, str]] = None) -> List[Dict]:
        """Get metric statistics from CloudWatch."""
        try:
            kwargs = {
                'Namespace': self.namespace,
                'MetricName': metric_name,
                'StartTime': start_time,
                'EndTime': end_time,
                'Period': period,
                'Statistics': [statistic]
            }
            
            if dimensions:
                kwargs['Dimensions'] = [
                    {'Name': k, 'Value': v} for k, v in dimensions.items()
                ]
            
            response = self.cloudwatch.get_metric_statistics(**kwargs)
            return sorted(response['Datapoints'], key=lambda x: x['Timestamp'])
            
        except Exception as e:
            logger.error(f"Failed to get metric statistics for {metric_name}: {e}")
            return []
    
    def get_dashboard_data(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get dashboard data for the last N hours."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours_back)
        
        dashboard_data = {
            'time_range': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            },
            'metrics': {}
        }
        
        # Key metrics to retrieve
        metrics = [
            'ProcessingTime',
            'DetectionCount',
            'InferenceRequests',
            'InferenceErrors',
            'AverageWeight',
            'ActiveTracks',
            'CPUUsage',
            'MemoryUsage'
        ]
        
        for metric in metrics:
            data = self.get_metric_statistics(
                metric_name=metric,
                start_time=start_time,
                end_time=end_time,
                period=300  # 5 minutes
            )
            dashboard_data['metrics'][metric] = data
        
        return dashboard_data


class MetricsCollector:
    """Collects system and application metrics for publishing."""
    
    def __init__(self, cloudwatch_metrics: CloudWatchMetrics):
        self.cloudwatch = cloudwatch_metrics
        self.collection_thread = None
        self.stop_collection = threading.Event()
        
    def start_collection(self, interval: int = 60):
        """Start metric collection thread."""
        if self.collection_thread and self.collection_thread.is_alive():
            logger.warning("Metric collection already running")
            return
        
        self.stop_collection.clear()
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            args=(interval,),
            daemon=True
        )
        self.collection_thread.start()
        logger.info(f"Started metric collection (interval: {interval}s)")
    
    def stop_collection(self):
        """Stop metric collection thread."""
        if self.collection_thread:
            self.stop_collection.set()
            self.collection_thread.join(timeout=10)
            logger.info("Stopped metric collection")
    
    def _collection_loop(self, interval: int):
        """Main collection loop."""
        while not self.stop_collection.wait(interval):
            try:
                self._collect_system_metrics()
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
    
    def _collect_system_metrics(self):
        """Collect system resource metrics."""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cloudwatch.put_metric("CPUUsage", cpu_percent, "Percent")
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.cloudwatch.put_metric("MemoryUsage", memory_percent, "Percent")
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.cloudwatch.put_metric("DiskUsage", disk_percent, "Percent")
            
            # Network I/O
            network = psutil.net_io_counters()
            self.cloudwatch.put_metric("NetworkBytesReceived", network.bytes_recv, "Bytes")
            self.cloudwatch.put_metric("NetworkBytesSent", network.bytes_sent, "Bytes")
            
        except ImportError:
            logger.warning("psutil not available for system metrics collection")
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")


def create_monitoring_setup(namespace: str = "ChickenWeightEstimation",
                          region: str = "us-east-1",
                          sns_topic_arn: Optional[str] = None) -> CloudWatchMetrics:
    """Create and configure monitoring setup."""
    
    # Create CloudWatch metrics publisher
    metrics = CloudWatchMetrics(namespace=namespace, region=region)
    
    # Start background publishing
    metrics.start_background_publishing(interval=60)
    
    # Create default alarms
    if sns_topic_arn:
        metrics.create_default_alarms(sns_topic_arn=sns_topic_arn)
    
    # Create metrics collector
    collector = MetricsCollector(metrics)
    collector.start_collection(interval=60)
    
    logger.info("Monitoring setup complete")
    return metrics


# Context manager for metric timing
class MetricTimer:
    """Context manager for timing operations and publishing metrics."""
    
    def __init__(self, cloudwatch_metrics: CloudWatchMetrics, metric_name: str,
                 dimensions: Optional[Dict[str, str]] = None):
        self.cloudwatch = cloudwatch_metrics
        self.metric_name = metric_name
        self.dimensions = dimensions
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (time.time() - self.start_time) * 1000  # Convert to milliseconds
            self.cloudwatch.put_metric(
                self.metric_name,
                duration,
                "Milliseconds",
                self.dimensions
            )


# Decorator for automatic metric collection
def monitor_function(cloudwatch_metrics: CloudWatchMetrics, metric_name: str = None,
                    dimensions: Optional[Dict[str, str]] = None):
    """Decorator to automatically monitor function execution time."""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            name = metric_name or f"{func.__name__}ExecutionTime"
            
            with MetricTimer(cloudwatch_metrics, name, dimensions):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create monitoring setup
    metrics = create_monitoring_setup()
    
    # Example metric publishing
    metrics.put_inference_metrics(
        processing_time=45.2,
        detections=3,
        camera_id="farm_camera_01"
    )
    
    metrics.put_weight_estimation_metrics(
        weights=[2.3, 2.7, 1.9],
        confidences=[0.85, 0.92, 0.78],
        camera_id="farm_camera_01"
    )
    
    # Publish metrics
    metrics.publish_buffered_metrics()
    
    # Example of using timer context manager
    with MetricTimer(metrics, "DatabaseQuery", {"Table": "chickens"}):
        time.sleep(0.1)  # Simulate database query
    
    logger.info("Example metrics published")