#!/usr/bin/env python3
"""
Monitoring script for SageMaker chicken weight estimation endpoint.
"""

import boto3
import json
import time
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SageMakerMonitor:
    """Monitor SageMaker endpoint performance and health."""
    
    def __init__(self, region: str = 'us-east-1'):
        self.region = region
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        self.sagemaker = boto3.client('sagemaker', region_name=region)
        self.sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=region)
        
        logger.info(f"Initialized monitor for region {region}")
    
    def get_endpoint_metrics(self, endpoint_name: str, hours: int = 24) -> Dict[str, Any]:
        """Get CloudWatch metrics for an endpoint."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        metrics = {}
        
        # Define metrics to collect
        metric_queries = [
            {
                'name': 'Invocations',
                'metric_name': 'Invocations',
                'statistic': 'Sum'
            },
            {
                'name': 'ModelLatency',
                'metric_name': 'ModelLatency',
                'statistic': 'Average'
            },
            {
                'name': 'OverheadLatency',
                'metric_name': 'OverheadLatency',
                'statistic': 'Average'
            },
            {
                'name': 'Invocation4XXErrors',
                'metric_name': 'Invocation4XXErrors',
                'statistic': 'Sum'
            },
            {
                'name': 'Invocation5XXErrors',
                'metric_name': 'Invocation5XXErrors',
                'statistic': 'Sum'
            },
            {
                'name': 'InvocationsPerInstance',
                'metric_name': 'InvocationsPerInstance',
                'statistic': 'Average'
            }
        ]
        
        for query in metric_queries:
            try:
                response = self.cloudwatch.get_metric_statistics(
                    Namespace='AWS/SageMaker',
                    MetricName=query['metric_name'],
                    Dimensions=[
                        {'Name': 'EndpointName', 'Value': endpoint_name},
                        {'Name': 'VariantName', 'Value': 'AllTraffic'}
                    ],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=300,  # 5 minutes
                    Statistics=[query['statistic']]
                )
                
                datapoints = sorted(response['Datapoints'], key=lambda x: x['Timestamp'])
                metrics[query['name']] = {
                    'datapoints': datapoints,
                    'latest_value': datapoints[-1][query['statistic']] if datapoints else 0,
                    'unit': response.get('Label', '')
                }
                
            except Exception as e:
                logger.error(f"Failed to get metric {query['name']}: {str(e)}")
                metrics[query['name']] = {'datapoints': [], 'latest_value': 0, 'unit': ''}
        
        return metrics
    
    def get_endpoint_status(self, endpoint_name: str) -> Dict[str, Any]:
        """Get current endpoint status and configuration."""
        try:
            response = self.sagemaker.describe_endpoint(EndpointName=endpoint_name)
            
            status_info = {
                'endpoint_name': endpoint_name,
                'endpoint_status': response['EndpointStatus'],
                'creation_time': response['CreationTime'],
                'last_modified_time': response['LastModifiedTime'],
                'endpoint_config_name': response['EndpointConfigName']
            }
            
            # Get endpoint configuration details
            config_response = self.sagemaker.describe_endpoint_config(
                EndpointConfigName=response['EndpointConfigName']
            )
            
            status_info['production_variants'] = []
            for variant in config_response['ProductionVariants']:
                variant_info = {
                    'variant_name': variant['VariantName'],
                    'model_name': variant['ModelName'],
                    'instance_type': variant['InstanceType'],
                    'initial_instance_count': variant['InitialInstanceCount'],
                    'initial_variant_weight': variant['InitialVariantWeight']
                }
                status_info['production_variants'].append(variant_info)
            
            return status_info
            
        except Exception as e:
            logger.error(f"Failed to get endpoint status: {str(e)}")
            return {'error': str(e)}
    
    def test_endpoint_health(self, endpoint_name: str) -> Dict[str, Any]:
        """Test endpoint health with a sample request."""
        try:
            # Create test payload
            test_payload = {
                "stream_data": {
                    "frame": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
                    "camera_id": "health_check",
                    "frame_sequence": 1,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Make request
            start_time = time.time()
            response = self.sagemaker_runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=json.dumps(test_payload)
            )
            end_time = time.time()
            
            # Parse response
            result = json.loads(response['Body'].read().decode())
            
            health_info = {
                'status': 'healthy',
                'response_time_ms': (end_time - start_time) * 1000,
                'response_keys': list(result.keys()),
                'timestamp': datetime.now().isoformat()
            }
            
            # Check response structure
            expected_keys = ['detections', 'tracks', 'frame_id', 'processing_time']
            missing_keys = [key for key in expected_keys if key not in result]
            if missing_keys:
                health_info['warnings'] = f"Missing response keys: {missing_keys}"
            
            return health_info
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def generate_performance_report(self, endpoint_name: str, hours: int = 24, 
                                  output_dir: str = 'reports') -> str:
        """Generate a comprehensive performance report."""
        logger.info(f"Generating performance report for {endpoint_name}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get data
        metrics = self.get_endpoint_metrics(endpoint_name, hours)
        status = self.get_endpoint_status(endpoint_name)
        health = self.test_endpoint_health(endpoint_name)
        
        # Generate report
        report = {
            'report_generated': datetime.now().isoformat(),
            'endpoint_name': endpoint_name,
            'time_period_hours': hours,
            'endpoint_status': status,
            'health_check': health,
            'performance_metrics': {},
            'summary': {},
            'recommendations': []
        }
        
        # Process metrics
        for metric_name, metric_data in metrics.items():
            if metric_data['datapoints']:
                values = [dp[list(dp.keys())[1]] for dp in metric_data['datapoints'] 
                         if list(dp.keys())[1] in dp]  # Get the statistic value
                
                report['performance_metrics'][metric_name] = {
                    'latest_value': metric_data['latest_value'],
                    'average': sum(values) / len(values) if values else 0,
                    'max': max(values) if values else 0,
                    'min': min(values) if values else 0,
                    'data_points': len(values)
                }
        
        # Generate summary
        invocations = report['performance_metrics'].get('Invocations', {}).get('latest_value', 0)
        avg_latency = report['performance_metrics'].get('ModelLatency', {}).get('average', 0)
        error_4xx = report['performance_metrics'].get('Invocation4XXErrors', {}).get('latest_value', 0)
        error_5xx = report['performance_metrics'].get('Invocation5XXErrors', {}).get('latest_value', 0)
        
        report['summary'] = {
            'total_invocations': invocations,
            'average_latency_ms': avg_latency,
            'error_rate_4xx': error_4xx,
            'error_rate_5xx': error_5xx,
            'health_status': health.get('status', 'unknown')
        }
        
        # Generate recommendations
        if avg_latency > 2000:  # > 2 seconds
            report['recommendations'].append("High latency detected. Consider using faster instance type or optimizing model.")
        
        if error_4xx > 10:
            report['recommendations'].append("High 4XX error rate. Check input validation and client requests.")
        
        if error_5xx > 5:
            report['recommendations'].append("5XX errors detected. Check endpoint health and model loading.")
        
        if invocations < 10:
            report['recommendations'].append("Low invocation rate. Consider cost optimization or endpoint scaling.")
        
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = output_path / f"performance_report_{endpoint_name}_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to: {report_file}")
        
        # Generate visualizations
        self._create_performance_charts(metrics, endpoint_name, output_path, timestamp)
        
        return str(report_file)
    
    def _create_performance_charts(self, metrics: Dict[str, Any], endpoint_name: str, 
                                 output_path: Path, timestamp: str):
        """Create performance visualization charts."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Performance Metrics - {endpoint_name}', fontsize=16)
            
            # Invocations over time
            if metrics['Invocations']['datapoints']:
                invocation_data = metrics['Invocations']['datapoints']
                times = [dp['Timestamp'] for dp in invocation_data]
                values = [dp['Sum'] for dp in invocation_data]
                
                axes[0, 0].plot(times, values, 'b-', marker='o')
                axes[0, 0].set_title('Invocations Over Time')
                axes[0, 0].set_ylabel('Invocations')
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Latency over time
            if metrics['ModelLatency']['datapoints']:
                latency_data = metrics['ModelLatency']['datapoints']
                times = [dp['Timestamp'] for dp in latency_data]
                values = [dp['Average'] for dp in latency_data]
                
                axes[0, 1].plot(times, values, 'r-', marker='o')
                axes[0, 1].set_title('Model Latency Over Time')
                axes[0, 1].set_ylabel('Latency (ms)')
                axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Error rates
            error_4xx = metrics['Invocation4XXErrors']['datapoints']
            error_5xx = metrics['Invocation5XXErrors']['datapoints']
            
            if error_4xx or error_5xx:
                times_4xx = [dp['Timestamp'] for dp in error_4xx]
                values_4xx = [dp['Sum'] for dp in error_4xx]
                times_5xx = [dp['Timestamp'] for dp in error_5xx]
                values_5xx = [dp['Sum'] for dp in error_5xx]
                
                axes[1, 0].plot(times_4xx, values_4xx, 'orange', label='4XX Errors', marker='o')
                axes[1, 0].plot(times_5xx, values_5xx, 'red', label='5XX Errors', marker='s')
                axes[1, 0].set_title('Error Rates Over Time')
                axes[1, 0].set_ylabel('Error Count')
                axes[1, 0].legend()
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Invocations per instance
            if metrics['InvocationsPerInstance']['datapoints']:
                ipi_data = metrics['InvocationsPerInstance']['datapoints']
                times = [dp['Timestamp'] for dp in ipi_data]
                values = [dp['Average'] for dp in ipi_data]
                
                axes[1, 1].plot(times, values, 'g-', marker='o')
                axes[1, 1].set_title('Invocations Per Instance')
                axes[1, 1].set_ylabel('Invocations/Instance')
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save chart
            chart_file = output_path / f"performance_charts_{endpoint_name}_{timestamp}.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Performance charts saved to: {chart_file}")
            
        except Exception as e:
            logger.error(f"Failed to create performance charts: {str(e)}")
    
    def continuous_monitoring(self, endpoint_name: str, interval_minutes: int = 5, 
                            duration_hours: int = 1):
        """Run continuous monitoring for specified duration."""
        logger.info(f"Starting continuous monitoring for {endpoint_name}")
        logger.info(f"Interval: {interval_minutes} minutes, Duration: {duration_hours} hours")
        
        end_time = datetime.now() + timedelta(hours=duration_hours)
        
        while datetime.now() < end_time:
            try:
                # Get current status
                status = self.get_endpoint_status(endpoint_name)
                health = self.test_endpoint_health(endpoint_name)
                
                # Log status
                logger.info(f"Status: {status.get('endpoint_status', 'Unknown')}")
                logger.info(f"Health: {health.get('status', 'Unknown')}")
                
                if health.get('status') == 'healthy':
                    logger.info(f"Response time: {health.get('response_time_ms', 0):.2f}ms")
                else:
                    logger.warning(f"Health check failed: {health.get('error', 'Unknown error')}")
                
                # Wait for next check
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Monitoring error: {str(e)}")
                time.sleep(60)  # Wait 1 minute before retrying


def main():
    """Main monitoring function."""
    parser = argparse.ArgumentParser(description='Monitor SageMaker endpoint')
    parser.add_argument('--endpoint-name', required=True, help='SageMaker endpoint name')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--action', choices=['status', 'health', 'metrics', 'report', 'monitor'], 
                       default='status', help='Monitoring action')
    parser.add_argument('--hours', type=int, default=24, help='Hours of metrics to retrieve')
    parser.add_argument('--output-dir', default='reports', help='Output directory for reports')
    parser.add_argument('--interval', type=int, default=5, help='Monitoring interval in minutes')
    parser.add_argument('--duration', type=int, default=1, help='Monitoring duration in hours')
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = SageMakerMonitor(region=args.region)
    
    try:
        if args.action == 'status':
            # Get endpoint status
            status = monitor.get_endpoint_status(args.endpoint_name)
            print(json.dumps(status, indent=2, default=str))
            
        elif args.action == 'health':
            # Run health check
            health = monitor.test_endpoint_health(args.endpoint_name)
            print(json.dumps(health, indent=2, default=str))
            
        elif args.action == 'metrics':
            # Get metrics
            metrics = monitor.get_endpoint_metrics(args.endpoint_name, args.hours)
            
            # Print summary
            print(f"\n📊 METRICS SUMMARY ({args.hours} hours)")
            print("=" * 50)
            for metric_name, metric_data in metrics.items():
                print(f"{metric_name}: {metric_data['latest_value']}")
            
        elif args.action == 'report':
            # Generate comprehensive report
            report_file = monitor.generate_performance_report(
                args.endpoint_name, args.hours, args.output_dir
            )
            print(f"📋 Performance report generated: {report_file}")
            
        elif args.action == 'monitor':
            # Run continuous monitoring
            monitor.continuous_monitoring(
                args.endpoint_name, args.interval, args.duration
            )
        
        return 0
        
    except Exception as e:
        logger.error(f"Monitoring failed: {str(e)}")
        return 1


if __name__ == '__main__':
    exit(main())