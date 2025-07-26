#!/usr/bin/env python3
"""
Setup CloudWatch dashboards for chicken weight estimation system.
"""

import boto3
import json
import argparse
import logging
from typing import Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class DashboardManager:
    """Manages CloudWatch dashboards for the chicken weight estimation system."""
    
    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        
    def create_main_dashboard(self, environment: str, endpoint_name: str) -> str:
        """Create the main monitoring dashboard."""
        dashboard_name = f"ChickenWeightEstimator-{environment}"
        
        dashboard_body = {
            "widgets": [
                self._create_sagemaker_metrics_widget(endpoint_name),
                self._create_application_metrics_widget(),
                self._create_system_metrics_widget(),
                self._create_error_metrics_widget(endpoint_name),
                self._create_performance_metrics_widget(),
                self._create_business_metrics_widget(),
                self._create_cost_metrics_widget(endpoint_name),
                self._create_alerts_widget(environment)
            ]
        }
        
        try:
            self.cloudwatch.put_dashboard(
                DashboardName=dashboard_name,
                DashboardBody=json.dumps(dashboard_body)
            )
            logger.info(f"Created main dashboard: {dashboard_name}")
            return dashboard_name
        except Exception as e:
            logger.error(f"Failed to create main dashboard: {e}")
            raise
    
    def create_performance_dashboard(self, environment: str, endpoint_name: str) -> str:
        """Create a detailed performance monitoring dashboard."""
        dashboard_name = f"ChickenWeightEstimator-Performance-{environment}"
        
        dashboard_body = {
            "widgets": [
                self._create_latency_distribution_widget(endpoint_name),
                self._create_throughput_widget(endpoint_name),
                self._create_resource_utilization_widget(),
                self._create_queue_metrics_widget(),
                self._create_processing_pipeline_widget(),
                self._create_model_performance_widget(),
                self._create_scaling_metrics_widget(endpoint_name),
                self._create_network_metrics_widget()
            ]
        }
        
        try:
            self.cloudwatch.put_dashboard(
                DashboardName=dashboard_name,
                DashboardBody=json.dumps(dashboard_body)
            )
            logger.info(f"Created performance dashboard: {dashboard_name}")
            return dashboard_name
        except Exception as e:
            logger.error(f"Failed to create performance dashboard: {e}")
            raise
    
    def create_business_dashboard(self, environment: str) -> str:
        """Create a business metrics dashboard."""
        dashboard_name = f"ChickenWeightEstimator-Business-{environment}"
        
        dashboard_body = {
            "widgets": [
                self._create_detection_accuracy_widget(),
                self._create_weight_distribution_widget(),
                self._create_farm_analytics_widget(),
                self._create_usage_analytics_widget(),
                self._create_roi_metrics_widget(),
                self._create_data_quality_widget(),
                self._create_trend_analysis_widget(),
                self._create_comparative_analysis_widget()
            ]
        }
        
        try:
            self.cloudwatch.put_dashboard(
                DashboardName=dashboard_name,
                DashboardBody=json.dumps(dashboard_body)
            )
            logger.info(f"Created business dashboard: {dashboard_name}")
            return dashboard_name
        except Exception as e:
            logger.error(f"Failed to create business dashboard: {e}")
            raise
    
    def _create_sagemaker_metrics_widget(self, endpoint_name: str) -> Dict[str, Any]:
        """Create SageMaker endpoint metrics widget."""
        return {
            "type": "metric",
            "x": 0,
            "y": 0,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    ["AWS/SageMaker", "Invocations", "EndpointName", endpoint_name],
                    [".", "InvocationErrors", ".", "."],
                    [".", "ModelLatency", ".", "."],
                    [".", "OverheadLatency", ".", "."],
                    [".", "Invocation4XXErrors", ".", "."],
                    [".", "Invocation5XXErrors", ".", "."]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": self.region,
                "title": "SageMaker Endpoint Metrics",
                "period": 300,
                "stat": "Average",
                "yAxis": {
                    "left": {
                        "min": 0
                    }
                }
            }
        }
    
    def _create_application_metrics_widget(self) -> Dict[str, Any]:
        """Create application-specific metrics widget."""
        return {
            "type": "metric",
            "x": 12,
            "y": 0,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    ["ChickenWeightEstimation", "ProcessingTime"],
                    [".", "DetectionCount"],
                    [".", "AverageWeight"],
                    [".", "AverageConfidence"],
                    [".", "ActiveTracks"],
                    [".", "WeightEstimations"]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": self.region,
                "title": "Application Metrics",
                "period": 300,
                "stat": "Average"
            }
        }
    
    def _create_system_metrics_widget(self) -> Dict[str, Any]:
        """Create system resource metrics widget."""
        return {
            "type": "metric",
            "x": 0,
            "y": 6,
            "width": 8,
            "height": 6,
            "properties": {
                "metrics": [
                    ["ChickenWeightEstimation", "CPUUsage"],
                    [".", "MemoryUsage"],
                    [".", "GPUUsage"],
                    [".", "DiskUsage"]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": self.region,
                "title": "System Resources",
                "period": 300,
                "stat": "Average",
                "yAxis": {
                    "left": {
                        "min": 0,
                        "max": 100
                    }
                }
            }
        }
    
    def _create_error_metrics_widget(self, endpoint_name: str) -> Dict[str, Any]:
        """Create error metrics widget."""
        return {
            "type": "metric",
            "x": 8,
            "y": 6,
            "width": 8,
            "height": 6,
            "properties": {
                "metrics": [
                    ["ChickenWeightEstimation", "InferenceErrors"],
                    ["AWS/SageMaker", "InvocationErrors", "EndpointName", endpoint_name],
                    ["ChickenWeightEstimation", "APIErrors"],
                    [".", "ValidationErrors"]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": self.region,
                "title": "Error Metrics",
                "period": 300,
                "stat": "Sum"
            }
        }
    
    def _create_performance_metrics_widget(self) -> Dict[str, Any]:
        """Create performance metrics widget."""
        return {
            "type": "metric",
            "x": 16,
            "y": 6,
            "width": 8,
            "height": 6,
            "properties": {
                "metrics": [
                    ["ChickenWeightEstimation", "APIResponseTime"],
                    [".", "ProcessingTime"],
                    [".", "DatabaseQueryTime"],
                    [".", "ModelInferenceTime"]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": self.region,
                "title": "Performance Metrics",
                "period": 300,
                "stat": "Average"
            }
        }
    
    def _create_business_metrics_widget(self) -> Dict[str, Any]:
        """Create business metrics widget."""
        return {
            "type": "metric",
            "x": 0,
            "y": 12,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    ["ChickenWeightEstimation", "TotalChickensProcessed"],
                    [".", "AverageWeightPerFarm"],
                    [".", "WeightAccuracy"],
                    [".", "FarmProductivity"],
                    [".", "CostPerInference"]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": self.region,
                "title": "Business Metrics",
                "period": 3600,
                "stat": "Average"
            }
        }
    
    def _create_cost_metrics_widget(self, endpoint_name: str) -> Dict[str, Any]:
        """Create cost metrics widget."""
        return {
            "type": "metric",
            "x": 12,
            "y": 12,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    ["AWS/SageMaker", "CPUUtilization", "EndpointName", endpoint_name],
                    [".", "MemoryUtilization", ".", "."],
                    ["ChickenWeightEstimation", "CostPerHour"],
                    [".", "CostPerInference"]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": self.region,
                "title": "Cost & Utilization",
                "period": 3600,
                "stat": "Average"
            }
        }
    
    def _create_alerts_widget(self, environment: str) -> Dict[str, Any]:
        """Create alerts status widget."""
        return {
            "type": "log",
            "x": 0,
            "y": 18,
            "width": 24,
            "height": 6,
            "properties": {
                "query": f"SOURCE '/aws/sagemaker/chicken-weight-estimator-{environment}'\n| fields @timestamp, @message\n| filter @message like /ERROR/\n| sort @timestamp desc\n| limit 100",
                "region": self.region,
                "title": "Recent Errors",
                "view": "table"
            }
        }
    
    def _create_latency_distribution_widget(self, endpoint_name: str) -> Dict[str, Any]:
        """Create latency distribution widget."""
        return {
            "type": "metric",
            "x": 0,
            "y": 0,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    ["AWS/SageMaker", "ModelLatency", "EndpointName", endpoint_name, {"stat": "p50"}],
                    [".", ".", ".", ".", {"stat": "p90"}],
                    [".", ".", ".", ".", {"stat": "p95"}],
                    [".", ".", ".", ".", {"stat": "p99"}]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": self.region,
                "title": "Latency Distribution",
                "period": 300
            }
        }
    
    def _create_throughput_widget(self, endpoint_name: str) -> Dict[str, Any]:
        """Create throughput widget."""
        return {
            "type": "metric",
            "x": 12,
            "y": 0,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    ["AWS/SageMaker", "Invocations", "EndpointName", endpoint_name, {"stat": "Sum"}],
                    ["ChickenWeightEstimation", "RequestsPerSecond"],
                    [".", "SuccessfulInferences"],
                    [".", "FailedInferences"]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": self.region,
                "title": "Throughput Metrics",
                "period": 300
            }
        }
    
    def _create_resource_utilization_widget(self) -> Dict[str, Any]:
        """Create resource utilization widget."""
        return {
            "type": "metric",
            "x": 0,
            "y": 6,
            "width": 8,
            "height": 6,
            "properties": {
                "metrics": [
                    ["ChickenWeightEstimation", "CPUUsage"],
                    [".", "MemoryUsage"],
                    [".", "GPUUsage"],
                    [".", "NetworkUtilization"]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": self.region,
                "title": "Resource Utilization",
                "period": 300,
                "stat": "Average"
            }
        }
    
    def _create_queue_metrics_widget(self) -> Dict[str, Any]:
        """Create queue metrics widget."""
        return {
            "type": "metric",
            "x": 8,
            "y": 6,
            "width": 8,
            "height": 6,
            "properties": {
                "metrics": [
                    ["ChickenWeightEstimation", "QueueSize"],
                    [".", "QueueWaitTime"],
                    [".", "ProcessingBacklog"],
                    [".", "ConcurrentRequests"]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": self.region,
                "title": "Queue Metrics",
                "period": 300,
                "stat": "Average"
            }
        }
    
    def _create_processing_pipeline_widget(self) -> Dict[str, Any]:
        """Create processing pipeline widget."""
        return {
            "type": "metric",
            "x": 16,
            "y": 6,
            "width": 8,
            "height": 6,
            "properties": {
                "metrics": [
                    ["ChickenWeightEstimation", "ImagePreprocessingTime"],
                    [".", "DetectionTime"],
                    [".", "WeightEstimationTime"],
                    [".", "TrackingTime"],
                    [".", "PostprocessingTime"]
                ],
                "view": "timeSeries",
                "stacked": True,
                "region": self.region,
                "title": "Processing Pipeline",
                "period": 300,
                "stat": "Average"
            }
        }
    
    def _create_model_performance_widget(self) -> Dict[str, Any]:
        """Create model performance widget."""
        return {
            "type": "metric",
            "x": 0,
            "y": 12,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    ["ChickenWeightEstimation", "DetectionAccuracy"],
                    [".", "WeightEstimationAccuracy"],
                    [".", "TrackingAccuracy"],
                    [".", "FalsePositiveRate"],
                    [".", "FalseNegativeRate"]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": self.region,
                "title": "Model Performance",
                "period": 3600,
                "stat": "Average"
            }
        }
    
    def _create_scaling_metrics_widget(self, endpoint_name: str) -> Dict[str, Any]:
        """Create scaling metrics widget."""
        return {
            "type": "metric",
            "x": 12,
            "y": 12,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    ["AWS/ApplicationAutoScaling", "TargetValue", "ResourceId", f"endpoint/{endpoint_name}/variant/AllTraffic"],
                    [".", "ActualValue", ".", "."],
                    ["AWS/SageMaker", "CPUUtilization", "EndpointName", endpoint_name],
                    [".", "MemoryUtilization", ".", "."]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": self.region,
                "title": "Auto Scaling Metrics",
                "period": 300,
                "stat": "Average"
            }
        }
    
    def _create_network_metrics_widget(self) -> Dict[str, Any]:
        """Create network metrics widget."""
        return {
            "type": "metric",
            "x": 0,
            "y": 18,
            "width": 24,
            "height": 6,
            "properties": {
                "metrics": [
                    ["ChickenWeightEstimation", "NetworkBytesReceived"],
                    [".", "NetworkBytesSent"],
                    [".", "NetworkLatency"],
                    [".", "ConnectionCount"]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": self.region,
                "title": "Network Metrics",
                "period": 300,
                "stat": "Average"
            }
        }
    
    def _create_detection_accuracy_widget(self) -> Dict[str, Any]:
        """Create detection accuracy widget."""
        return {
            "type": "metric",
            "x": 0,
            "y": 0,
            "width": 8,
            "height": 6,
            "properties": {
                "metrics": [
                    ["ChickenWeightEstimation", "DetectionAccuracy"],
                    [".", "Precision"],
                    [".", "Recall"],
                    [".", "F1Score"]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": self.region,
                "title": "Detection Accuracy",
                "period": 3600,
                "stat": "Average"
            }
        }
    
    def _create_weight_distribution_widget(self) -> Dict[str, Any]:
        """Create weight distribution widget."""
        return {
            "type": "metric",
            "x": 8,
            "y": 0,
            "width": 8,
            "height": 6,
            "properties": {
                "metrics": [
                    ["ChickenWeightEstimation", "AverageWeight"],
                    [".", "WeightStandardDeviation"],
                    [".", "MinWeight"],
                    [".", "MaxWeight"]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": self.region,
                "title": "Weight Distribution",
                "period": 3600,
                "stat": "Average"
            }
        }
    
    def _create_farm_analytics_widget(self) -> Dict[str, Any]:
        """Create farm analytics widget."""
        return {
            "type": "metric",
            "x": 16,
            "y": 0,
            "width": 8,
            "height": 6,
            "properties": {
                "metrics": [
                    ["ChickenWeightEstimation", "ChickensPerFarm"],
                    [".", "AverageWeightPerFarm"],
                    [".", "FarmProductivity"],
                    [".", "GrowthRate"]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": self.region,
                "title": "Farm Analytics",
                "period": 3600,
                "stat": "Average"
            }
        }
    
    def _create_usage_analytics_widget(self) -> Dict[str, Any]:
        """Create usage analytics widget."""
        return {
            "type": "metric",
            "x": 0,
            "y": 6,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    ["ChickenWeightEstimation", "DailyActiveUsers"],
                    [".", "TotalInferences"],
                    [".", "UniqueCustomers"],
                    [".", "SessionDuration"]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": self.region,
                "title": "Usage Analytics",
                "period": 3600,
                "stat": "Sum"
            }
        }
    
    def _create_roi_metrics_widget(self) -> Dict[str, Any]:
        """Create ROI metrics widget."""
        return {
            "type": "metric",
            "x": 12,
            "y": 6,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    ["ChickenWeightEstimation", "CostSavings"],
                    [".", "RevenueGenerated"],
                    [".", "ROI"],
                    [".", "CostPerChicken"]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": self.region,
                "title": "ROI Metrics",
                "period": 3600,
                "stat": "Average"
            }
        }
    
    def _create_data_quality_widget(self) -> Dict[str, Any]:
        """Create data quality widget."""
        return {
            "type": "metric",
            "x": 0,
            "y": 12,
            "width": 8,
            "height": 6,
            "properties": {
                "metrics": [
                    ["ChickenWeightEstimation", "DataQualityScore"],
                    [".", "ImageQuality"],
                    [".", "AnnotationAccuracy"],
                    [".", "DataCompleteness"]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": self.region,
                "title": "Data Quality",
                "period": 3600,
                "stat": "Average"
            }
        }
    
    def _create_trend_analysis_widget(self) -> Dict[str, Any]:
        """Create trend analysis widget."""
        return {
            "type": "metric",
            "x": 8,
            "y": 12,
            "width": 8,
            "height": 6,
            "properties": {
                "metrics": [
                    ["ChickenWeightEstimation", "WeightTrend"],
                    [".", "GrowthTrend"],
                    [".", "SeasonalVariation"],
                    [".", "PredictedWeight"]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": self.region,
                "title": "Trend Analysis",
                "period": 3600,
                "stat": "Average"
            }
        }
    
    def _create_comparative_analysis_widget(self) -> Dict[str, Any]:
        """Create comparative analysis widget."""
        return {
            "type": "metric",
            "x": 16,
            "y": 12,
            "width": 8,
            "height": 6,
            "properties": {
                "metrics": [
                    ["ChickenWeightEstimation", "AccuracyVsManual"],
                    [".", "SpeedImprovement"],
                    [".", "CostReduction"],
                    [".", "EfficiencyGain"]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": self.region,
                "title": "Comparative Analysis",
                "period": 3600,
                "stat": "Average"
            }
        }
    
    def create_all_dashboards(self, environment: str, endpoint_name: str) -> List[str]:
        """Create all dashboards for the environment."""
        dashboards = []
        
        try:
            # Main dashboard
            main_dashboard = self.create_main_dashboard(environment, endpoint_name)
            dashboards.append(main_dashboard)
            
            # Performance dashboard
            perf_dashboard = self.create_performance_dashboard(environment, endpoint_name)
            dashboards.append(perf_dashboard)
            
            # Business dashboard
            business_dashboard = self.create_business_dashboard(environment)
            dashboards.append(business_dashboard)
            
            logger.info(f"Created {len(dashboards)} dashboards for {environment}")
            return dashboards
            
        except Exception as e:
            logger.error(f"Failed to create dashboards: {e}")
            raise
    
    def delete_dashboard(self, dashboard_name: str):
        """Delete a dashboard."""
        try:
            self.cloudwatch.delete_dashboards(DashboardNames=[dashboard_name])
            logger.info(f"Deleted dashboard: {dashboard_name}")
        except Exception as e:
            logger.error(f"Failed to delete dashboard {dashboard_name}: {e}")
            raise
    
    def list_dashboards(self) -> List[str]:
        """List all dashboards."""
        try:
            response = self.cloudwatch.list_dashboards()
            return [dashboard['DashboardName'] for dashboard in response['DashboardEntries']]
        except Exception as e:
            logger.error(f"Failed to list dashboards: {e}")
            raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Setup CloudWatch dashboards")
    parser.add_argument("--environment", required=True, choices=["dev", "staging", "prod"],
                       help="Environment name")
    parser.add_argument("--endpoint-name", required=True,
                       help="SageMaker endpoint name")
    parser.add_argument("--region", default="us-east-1",
                       help="AWS region")
    parser.add_argument("--action", choices=["create", "delete", "list"], default="create",
                       help="Action to perform")
    parser.add_argument("--dashboard-type", choices=["main", "performance", "business", "all"],
                       default="all", help="Type of dashboard to create")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    dashboard_manager = DashboardManager(region=args.region)
    
    try:
        if args.action == "create":
            if args.dashboard_type == "all":
                dashboards = dashboard_manager.create_all_dashboards(
                    args.environment, args.endpoint_name
                )
                print(f"Created dashboards: {', '.join(dashboards)}")
            elif args.dashboard_type == "main":
                dashboard = dashboard_manager.create_main_dashboard(
                    args.environment, args.endpoint_name
                )
                print(f"Created main dashboard: {dashboard}")
            elif args.dashboard_type == "performance":
                dashboard = dashboard_manager.create_performance_dashboard(
                    args.environment, args.endpoint_name
                )
                print(f"Created performance dashboard: {dashboard}")
            elif args.dashboard_type == "business":
                dashboard = dashboard_manager.create_business_dashboard(args.environment)
                print(f"Created business dashboard: {dashboard}")
        
        elif args.action == "list":
            dashboards = dashboard_manager.list_dashboards()
            print("Existing dashboards:")
            for dashboard in dashboards:
                print(f"  - {dashboard}")
        
        elif args.action == "delete":
            dashboard_name = f"ChickenWeightEstimator-{args.environment}"
            dashboard_manager.delete_dashboard(dashboard_name)
            print(f"Deleted dashboard: {dashboard_name}")
    
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())