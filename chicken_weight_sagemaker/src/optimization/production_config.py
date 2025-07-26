"""
Production configuration and auto-scaling setup.
"""

import boto3
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class ProductionConfigManager:
    """Manage production deployment configurations."""
    
    def __init__(self, region: str = 'us-east-1'):
        self.region = region
        self.sagemaker = boto3.client('sagemaker', region_name=region)
        self.autoscaling = boto3.client('application-autoscaling', region_name=region)
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        self.ec2 = boto3.client('ec2', region_name=region)
    
    def create_production_endpoint_config(self, 
                                        model_name: str,
                                        config_name: str,
                                        instance_type: str = 'ml.g4dn.xlarge',
                                        initial_instance_count: int = 2,
                                        enable_data_capture: bool = True,
                                        enable_multi_model: bool = False) -> str:
        """
        Create production-ready endpoint configuration.
        
        Args:
            model_name: Name of the SageMaker model
            config_name: Name for the endpoint configuration
            instance_type: EC2 instance type
            initial_instance_count: Initial number of instances
            enable_data_capture: Whether to enable data capture
            enable_multi_model: Whether to enable multi-model endpoint
            
        Returns:
            Endpoint configuration name
        """
        logger.info(f"Creating production endpoint config: {config_name}")
        
        # Base configuration
        config = {
            'EndpointConfigName': config_name,
            'ProductionVariants': [
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InitialInstanceCount': initial_instance_count,
                    'InstanceType': instance_type,
                    'InitialVariantWeight': 1.0,
                    'AcceleratorType': None  # Use GPU instances instead
                }
            ]
        }
        
        # Add data capture configuration
        if enable_data_capture:
            config['DataCaptureConfig'] = {
                'EnableCapture': True,
                'InitialSamplingPercentage': 5,  # Capture 5% of requests
                'DestinationS3Uri': f's3://sagemaker-{self.region}-{self._get_account_id()}/chicken-weight-data-capture',
                'CaptureOptions': [
                    {'CaptureMode': 'Input'},
                    {'CaptureMode': 'Output'}
                ],
                'CaptureContentTypeHeader': {
                    'CsvContentTypes': ['text/csv'],
                    'JsonContentTypes': ['application/json']
                }
            }
        
        # Add multi-model configuration
        if enable_multi_model:
            config['ProductionVariants'][0]['ModelDataDownloadTimeoutInSeconds'] = 600
            config['ProductionVariants'][0]['ContainerStartupHealthCheckTimeoutInSeconds'] = 600
        
        # Create endpoint configuration
        response = self.sagemaker.create_endpoint_config(**config)
        
        logger.info(f"Endpoint configuration created: {config_name}")
        return config_name
    
    def setup_auto_scaling(self, 
                          endpoint_name: str,
                          variant_name: str = 'AllTraffic',
                          min_capacity: int = 1,
                          max_capacity: int = 10,
                          target_invocations: int = 70,
                          scale_in_cooldown: int = 300,
                          scale_out_cooldown: int = 60) -> Dict[str, str]:
        """
        Setup auto-scaling for production endpoint.
        
        Args:
            endpoint_name: Name of the endpoint
            variant_name: Name of the variant
            min_capacity: Minimum number of instances
            max_capacity: Maximum number of instances
            target_invocations: Target invocations per instance
            scale_in_cooldown: Scale-in cooldown period (seconds)
            scale_out_cooldown: Scale-out cooldown period (seconds)
            
        Returns:
            Dictionary with scaling configuration details
        """
        logger.info(f"Setting up auto-scaling for {endpoint_name}")
        
        resource_id = f'endpoint/{endpoint_name}/variant/{variant_name}'
        
        # Register scalable target
        self.autoscaling.register_scalable_target(
            ServiceNamespace='sagemaker',
            ResourceId=resource_id,
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            MinCapacity=min_capacity,
            MaxCapacity=max_capacity
        )
        
        # Create scaling policy
        policy_name = f'{endpoint_name}-target-tracking-policy'
        policy_response = self.autoscaling.put_scaling_policy(
            PolicyName=policy_name,
            ServiceNamespace='sagemaker',
            ResourceId=resource_id,
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            PolicyType='TargetTrackingScaling',
            TargetTrackingScalingPolicyConfiguration={
                'TargetValue': float(target_invocations),
                'PredefinedMetricSpecification': {
                    'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
                },
                'ScaleInCooldown': scale_in_cooldown,
                'ScaleOutCooldown': scale_out_cooldown
            }
        )
        
        # Create scheduled scaling for predictable patterns
        self._setup_scheduled_scaling(endpoint_name, variant_name)
        
        scaling_config = {
            'resource_id': resource_id,
            'policy_name': policy_name,
            'policy_arn': policy_response['PolicyARN'],
            'min_capacity': min_capacity,
            'max_capacity': max_capacity,
            'target_invocations': target_invocations
        }
        
        logger.info(f"Auto-scaling configured: {min_capacity}-{max_capacity} instances")
        return scaling_config
    
    def _setup_scheduled_scaling(self, endpoint_name: str, variant_name: str):
        """Setup scheduled scaling for predictable load patterns."""
        resource_id = f'endpoint/{endpoint_name}/variant/{variant_name}'
        
        # Business hours scale-up (8 AM)
        self.autoscaling.put_scheduled_action(
            ServiceNamespace='sagemaker',
            ScheduledActionName=f'{endpoint_name}-scale-up-business-hours',
            ResourceId=resource_id,
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            Schedule='cron(0 8 * * MON-FRI)',  # 8 AM weekdays
            ScalableTargetAction={
                'MinCapacity': 2,
                'MaxCapacity': 10
            }
        )
        
        # Off-hours scale-down (6 PM)
        self.autoscaling.put_scheduled_action(
            ServiceNamespace='sagemaker',
            ScheduledActionName=f'{endpoint_name}-scale-down-off-hours',
            ResourceId=resource_id,
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            Schedule='cron(0 18 * * MON-FRI)',  # 6 PM weekdays
            ScalableTargetAction={
                'MinCapacity': 1,
                'MaxCapacity': 4
            }
        )
        
        logger.info("Scheduled scaling configured for business hours")
    
    def create_cloudwatch_alarms(self, endpoint_name: str) -> List[str]:
        """
        Create comprehensive CloudWatch alarms for monitoring.
        
        Args:
            endpoint_name: Name of the endpoint
            
        Returns:
            List of created alarm names
        """
        logger.info(f"Creating CloudWatch alarms for {endpoint_name}")
        
        alarms = []
        
        # High latency alarm
        latency_alarm = f'{endpoint_name}-high-latency'
        self.cloudwatch.put_metric_alarm(
            AlarmName=latency_alarm,
            ComparisonOperator='GreaterThanThreshold',
            EvaluationPeriods=2,
            MetricName='ModelLatency',
            Namespace='AWS/SageMaker',
            Period=300,
            Statistic='Average',
            Threshold=2000.0,  # 2 seconds
            ActionsEnabled=True,
            AlarmDescription='Triggers when model latency exceeds 2 seconds',
            Dimensions=[
                {'Name': 'EndpointName', 'Value': endpoint_name},
                {'Name': 'VariantName', 'Value': 'AllTraffic'}
            ],
            Unit='Milliseconds'
        )
        alarms.append(latency_alarm)
        
        # High error rate alarm
        error_alarm = f'{endpoint_name}-high-error-rate'
        self.cloudwatch.put_metric_alarm(
            AlarmName=error_alarm,
            ComparisonOperator='GreaterThanThreshold',
            EvaluationPeriods=1,
            MetricName='Invocation4XXErrors',
            Namespace='AWS/SageMaker',
            Period=300,
            Statistic='Sum',
            Threshold=10.0,
            ActionsEnabled=True,
            AlarmDescription='Triggers when 4XX error rate is high',
            Dimensions=[
                {'Name': 'EndpointName', 'Value': endpoint_name},
                {'Name': 'VariantName', 'Value': 'AllTraffic'}
            ]
        )
        alarms.append(error_alarm)
        
        # Low invocation rate alarm (potential issues)
        invocation_alarm = f'{endpoint_name}-low-invocations'
        self.cloudwatch.put_metric_alarm(
            AlarmName=invocation_alarm,
            ComparisonOperator='LessThanThreshold',
            EvaluationPeriods=3,
            MetricName='Invocations',
            Namespace='AWS/SageMaker',
            Period=900,  # 15 minutes
            Statistic='Sum',
            Threshold=1.0,
            ActionsEnabled=True,
            AlarmDescription='Triggers when endpoint receives no invocations',
            Dimensions=[
                {'Name': 'EndpointName', 'Value': endpoint_name},
                {'Name': 'VariantName', 'Value': 'AllTraffic'}
            ]
        )
        alarms.append(invocation_alarm)
        
        # CPU utilization alarm
        cpu_alarm = f'{endpoint_name}-high-cpu'
        self.cloudwatch.put_metric_alarm(
            AlarmName=cpu_alarm,
            ComparisonOperator='GreaterThanThreshold',
            EvaluationPeriods=2,
            MetricName='CPUUtilization',
            Namespace='AWS/SageMaker',
            Period=300,
            Statistic='Average',
            Threshold=80.0,
            ActionsEnabled=True,
            AlarmDescription='Triggers when CPU utilization exceeds 80%',
            Dimensions=[
                {'Name': 'EndpointName', 'Value': endpoint_name},
                {'Name': 'VariantName', 'Value': 'AllTraffic'}
            ],
            Unit='Percent'
        )
        alarms.append(cpu_alarm)
        
        # Memory utilization alarm
        memory_alarm = f'{endpoint_name}-high-memory'
        self.cloudwatch.put_metric_alarm(
            AlarmName=memory_alarm,
            ComparisonOperator='GreaterThanThreshold',
            EvaluationPeriods=2,
            MetricName='MemoryUtilization',
            Namespace='AWS/SageMaker',
            Period=300,
            Statistic='Average',
            Threshold=85.0,
            ActionsEnabled=True,
            AlarmDescription='Triggers when memory utilization exceeds 85%',
            Dimensions=[
                {'Name': 'EndpointName', 'Value': endpoint_name},
                {'Name': 'VariantName', 'Value': 'AllTraffic'}
            ],
            Unit='Percent'
        )
        alarms.append(memory_alarm)
        
        logger.info(f"Created {len(alarms)} CloudWatch alarms")
        return alarms
    
    def setup_spot_instance_config(self, 
                                  model_name: str,
                                  config_name: str,
                                  instance_type: str = 'ml.g4dn.xlarge',
                                  max_spot_price: float = 0.5) -> str:
        """
        Setup endpoint configuration with spot instances for cost optimization.
        
        Args:
            model_name: Name of the SageMaker model
            config_name: Name for the endpoint configuration
            instance_type: EC2 instance type
            max_spot_price: Maximum spot price per hour
            
        Returns:
            Endpoint configuration name
        """
        logger.info(f"Creating spot instance endpoint config: {config_name}")
        
        # Note: SageMaker doesn't directly support spot instances for endpoints
        # This creates a configuration optimized for cost with managed scaling
        config = {
            'EndpointConfigName': config_name,
            'ProductionVariants': [
                {
                    'VariantName': 'SpotOptimized',
                    'ModelName': model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': instance_type,
                    'InitialVariantWeight': 1.0,
                    'ManagedInstanceScaling': {
                        'Status': 'ENABLED',
                        'MinInstanceCount': 0,  # Scale to zero when not needed
                        'MaxInstanceCount': 5
                    },
                    'RoutingStrategy': 'LEAST_OUTSTANDING_REQUESTS'
                }
            ],
            'AsyncInferenceConfig': {
                'OutputConfig': {
                    'S3OutputPath': f's3://sagemaker-{self.region}-{self._get_account_id()}/async-inference-output'
                },
                'ClientConfig': {
                    'MaxConcurrentInvocationsPerInstance': 4
                }
            }
        }
        
        # Create endpoint configuration
        response = self.sagemaker.create_endpoint_config(**config)
        
        logger.info(f"Spot-optimized endpoint configuration created: {config_name}")
        return config_name
    
    def create_multi_az_config(self, 
                              model_name: str,
                              config_name: str,
                              instance_type: str = 'ml.g4dn.xlarge') -> str:
        """
        Create multi-AZ endpoint configuration for high availability.
        
        Args:
            model_name: Name of the SageMaker model
            config_name: Name for the endpoint configuration
            instance_type: EC2 instance type
            
        Returns:
            Endpoint configuration name
        """
        logger.info(f"Creating multi-AZ endpoint config: {config_name}")
        
        # Get available AZs
        azs = self._get_available_azs()
        
        # Create variants for multiple AZs
        variants = []
        for i, az in enumerate(azs[:3]):  # Use up to 3 AZs
            variants.append({
                'VariantName': f'AZ{i+1}',
                'ModelName': model_name,
                'InitialInstanceCount': 1,
                'InstanceType': instance_type,
                'InitialVariantWeight': 1.0 / len(azs[:3])  # Equal weight distribution
            })
        
        config = {
            'EndpointConfigName': config_name,
            'ProductionVariants': variants
        }
        
        # Create endpoint configuration
        response = self.sagemaker.create_endpoint_config(**config)
        
        logger.info(f"Multi-AZ endpoint configuration created with {len(variants)} variants")
        return config_name
    
    def generate_production_deployment_config(self, 
                                            endpoint_name: str,
                                            model_name: str,
                                            deployment_type: str = 'standard') -> Dict[str, Any]:
        """
        Generate complete production deployment configuration.
        
        Args:
            endpoint_name: Name for the endpoint
            model_name: Name of the SageMaker model
            deployment_type: Type of deployment ('standard', 'cost-optimized', 'high-availability')
            
        Returns:
            Complete deployment configuration
        """
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        config_name = f'{endpoint_name}-config-{timestamp}'
        
        base_config = {
            'endpoint_name': endpoint_name,
            'model_name': model_name,
            'config_name': config_name,
            'deployment_type': deployment_type,
            'timestamp': timestamp
        }
        
        if deployment_type == 'standard':
            base_config.update({
                'instance_type': 'ml.g4dn.xlarge',
                'initial_instance_count': 2,
                'auto_scaling': {
                    'min_capacity': 1,
                    'max_capacity': 10,
                    'target_invocations': 70
                },
                'data_capture': True,
                'monitoring': True
            })
        
        elif deployment_type == 'cost-optimized':
            base_config.update({
                'instance_type': 'ml.g4dn.large',  # Smaller instance
                'initial_instance_count': 1,
                'auto_scaling': {
                    'min_capacity': 0,  # Scale to zero
                    'max_capacity': 5,
                    'target_invocations': 100
                },
                'data_capture': False,  # Reduce costs
                'monitoring': True,
                'spot_instances': True
            })
        
        elif deployment_type == 'high-availability':
            base_config.update({
                'instance_type': 'ml.g4dn.xlarge',
                'initial_instance_count': 3,  # Higher initial count
                'auto_scaling': {
                    'min_capacity': 2,  # Always have 2 instances
                    'max_capacity': 15,
                    'target_invocations': 50  # Lower target for better performance
                },
                'data_capture': True,
                'monitoring': True,
                'multi_az': True
            })
        
        return base_config
    
    def _get_account_id(self) -> str:
        """Get AWS account ID."""
        sts = boto3.client('sts')
        return sts.get_caller_identity()['Account']
    
    def _get_available_azs(self) -> List[str]:
        """Get available availability zones."""
        response = self.ec2.describe_availability_zones(
            Filters=[{'Name': 'state', 'Values': ['available']}]
        )
        return [az['ZoneName'] for az in response['AvailabilityZones']]


class LoadBalancingManager:
    """Manage load balancing across multiple endpoints."""
    
    def __init__(self, region: str = 'us-east-1'):
        self.region = region
        self.sagemaker = boto3.client('sagemaker', region_name=region)
        self.route53 = boto3.client('route53', region_name=region)
    
    def create_multi_endpoint_config(self, 
                                   endpoints: List[str],
                                   weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Create configuration for load balancing across multiple endpoints.
        
        Args:
            endpoints: List of endpoint names
            weights: Optional weights for each endpoint (must sum to 1.0)
            
        Returns:
            Load balancing configuration
        """
        if weights is None:
            weights = [1.0 / len(endpoints)] * len(endpoints)
        
        if len(weights) != len(endpoints):
            raise ValueError("Number of weights must match number of endpoints")
        
        if abs(sum(weights) - 1.0) > 0.001:
            raise ValueError("Weights must sum to 1.0")
        
        config = {
            'endpoints': [
                {'name': endpoint, 'weight': weight}
                for endpoint, weight in zip(endpoints, weights)
            ],
            'routing_strategy': 'weighted_round_robin',
            'health_check_interval': 30,
            'failover_enabled': True
        }
        
        logger.info(f"Multi-endpoint configuration created for {len(endpoints)} endpoints")
        return config
    
    def setup_dns_failover(self, 
                          primary_endpoint: str,
                          secondary_endpoint: str,
                          hosted_zone_id: str,
                          domain_name: str) -> str:
        """
        Setup DNS-based failover between endpoints.
        
        Args:
            primary_endpoint: Primary endpoint name
            secondary_endpoint: Secondary endpoint name
            hosted_zone_id: Route53 hosted zone ID
            domain_name: Domain name for the service
            
        Returns:
            Record set ID
        """
        logger.info(f"Setting up DNS failover: {primary_endpoint} -> {secondary_endpoint}")
        
        # Create primary record
        primary_response = self.route53.change_resource_record_sets(
            HostedZoneId=hosted_zone_id,
            ChangeBatch={
                'Changes': [{
                    'Action': 'CREATE',
                    'ResourceRecordSet': {
                        'Name': domain_name,
                        'Type': 'CNAME',
                        'SetIdentifier': 'primary',
                        'Failover': 'PRIMARY',
                        'TTL': 60,
                        'ResourceRecords': [{'Value': f'{primary_endpoint}.sagemaker.{self.region}.amazonaws.com'}],
                        'HealthCheckId': self._create_health_check(primary_endpoint)
                    }
                }]
            }
        )
        
        # Create secondary record
        secondary_response = self.route53.change_resource_record_sets(
            HostedZoneId=hosted_zone_id,
            ChangeBatch={
                'Changes': [{
                    'Action': 'CREATE',
                    'ResourceRecordSet': {
                        'Name': domain_name,
                        'Type': 'CNAME',
                        'SetIdentifier': 'secondary',
                        'Failover': 'SECONDARY',
                        'TTL': 60,
                        'ResourceRecords': [{'Value': f'{secondary_endpoint}.sagemaker.{self.region}.amazonaws.com'}]
                    }
                }]
            }
        )
        
        logger.info("DNS failover configured successfully")
        return primary_response['ChangeInfo']['Id']
    
    def _create_health_check(self, endpoint_name: str) -> str:
        """Create Route53 health check for endpoint."""
        response = self.route53.create_health_check(
            Type='HTTPS',
            ResourcePath='/ping',
            FullyQualifiedDomainName=f'{endpoint_name}.sagemaker.{self.region}.amazonaws.com',
            Port=443,
            RequestInterval=30,
            FailureThreshold=3
        )
        
        return response['HealthCheck']['Id']


def create_production_deployment_plan(config: Dict[str, Any]) -> str:
    """
    Create a complete production deployment plan.
    
    Args:
        config: Deployment configuration
        
    Returns:
        Path to the deployment plan file
    """
    plan = {
        'deployment_info': {
            'endpoint_name': config['endpoint_name'],
            'model_name': config['model_name'],
            'deployment_type': config['deployment_type'],
            'created_at': datetime.now().isoformat()
        },
        'infrastructure': {
            'instance_type': config['instance_type'],
            'initial_instance_count': config['initial_instance_count'],
            'auto_scaling': config['auto_scaling'],
            'monitoring': config.get('monitoring', True),
            'data_capture': config.get('data_capture', True)
        },
        'deployment_steps': [
            {
                'step': 1,
                'action': 'create_endpoint_config',
                'description': 'Create SageMaker endpoint configuration',
                'parameters': {
                    'config_name': config['config_name'],
                    'model_name': config['model_name'],
                    'instance_type': config['instance_type'],
                    'initial_instance_count': config['initial_instance_count']
                }
            },
            {
                'step': 2,
                'action': 'create_endpoint',
                'description': 'Create SageMaker endpoint',
                'parameters': {
                    'endpoint_name': config['endpoint_name'],
                    'config_name': config['config_name']
                }
            },
            {
                'step': 3,
                'action': 'setup_auto_scaling',
                'description': 'Configure auto-scaling policies',
                'parameters': config['auto_scaling']
            },
            {
                'step': 4,
                'action': 'create_cloudwatch_alarms',
                'description': 'Set up monitoring and alerting',
                'parameters': {
                    'endpoint_name': config['endpoint_name']
                }
            },
            {
                'step': 5,
                'action': 'validate_deployment',
                'description': 'Run deployment validation tests',
                'parameters': {
                    'endpoint_name': config['endpoint_name']
                }
            }
        ],
        'validation_tests': [
            'health_check',
            'latency_test',
            'load_test',
            'failover_test'
        ],
        'rollback_plan': {
            'previous_config': None,
            'rollback_steps': [
                'Stop traffic to new endpoint',
                'Restore previous endpoint configuration',
                'Validate rollback success',
                'Clean up failed deployment resources'
            ]
        }
    }
    
    # Save deployment plan
    plan_file = f"deployment_plan_{config['endpoint_name']}_{config['timestamp']}.json"
    with open(plan_file, 'w') as f:
        json.dump(plan, f, indent=2)
    
    logger.info(f"Deployment plan created: {plan_file}")
    return plan_file