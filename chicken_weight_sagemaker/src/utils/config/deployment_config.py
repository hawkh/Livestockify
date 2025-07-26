"""
SageMaker deployment configuration management.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List


@dataclass
class SageMakerConfig:
    """SageMaker endpoint configuration."""
    
    # Endpoint configuration
    endpoint_name: str = "chicken-weight-estimator"
    endpoint_config_name: str = "chicken-weight-estimator-config"
    model_name: str = "chicken-weight-model"
    
    # Instance configuration
    instance_type: str = "ml.g4dn.xlarge"
    initial_instance_count: int = 1
    max_instance_count: int = 5
    min_instance_count: int = 1
    
    # Auto-scaling configuration
    enable_auto_scaling: bool = True
    target_invocations_per_instance: int = 100
    scale_in_cooldown: int = 300  # seconds
    scale_out_cooldown: int = 60   # seconds
    
    # Container configuration
    container_image_uri: str = ""  # Will be set during deployment
    model_data_url: str = ""       # S3 URL for model artifacts
    
    # Environment variables for container
    environment: Dict[str, str] = None
    
    def __post_init__(self):
        if self.environment is None:
            self.environment = {
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/code",
                "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                "SAGEMAKER_REGION": "us-east-1"
            }


@dataclass
class ContainerConfig:
    """Docker container configuration."""
    
    # Base image
    base_image: str = "pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime"
    
    # Container registry
    ecr_repository: str = "chicken-weight-estimator"
    image_tag: str = "latest"
    
    # Build configuration
    dockerfile_path: str = "docker/Dockerfile"
    build_context: str = "."
    
    # Resource limits
    memory_limit: str = "8Gi"
    cpu_limit: str = "4"
    gpu_limit: int = 1
    
    # Health check
    health_check_path: str = "/ping"
    health_check_timeout: int = 30
    health_check_interval: int = 30
    health_check_retries: int = 3


@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration."""
    
    # CloudWatch configuration
    enable_cloudwatch: bool = True
    log_group_name: str = "/aws/sagemaker/Endpoints/chicken-weight-estimator"
    log_retention_days: int = 30
    
    # Custom metrics
    enable_custom_metrics: bool = True
    metrics_namespace: str = "ChickenWeightEstimator"
    
    # Alarms
    enable_alarms: bool = True
    error_rate_threshold: float = 0.05  # 5%
    latency_threshold_ms: float = 1000.0
    
    # Performance monitoring
    track_inference_time: bool = True
    track_detection_accuracy: bool = True
    track_weight_confidence: bool = True
    
    # Cost monitoring
    track_instance_usage: bool = True
    cost_alert_threshold: float = 100.0  # USD per day


@dataclass
class SecurityConfig:
    """Security configuration."""
    
    # IAM roles
    execution_role_arn: str = ""
    
    # VPC configuration
    enable_vpc: bool = False
    vpc_config: Optional[Dict[str, Any]] = None
    
    # Encryption
    enable_encryption: bool = True
    kms_key_id: Optional[str] = None
    
    # Network isolation
    enable_network_isolation: bool = False


@dataclass
class DeploymentConfig:
    """Complete deployment configuration."""
    
    sagemaker: SageMakerConfig
    container: ContainerConfig
    monitoring: MonitoringConfig
    security: SecurityConfig
    
    # AWS configuration
    aws_region: str = "us-east-1"
    s3_bucket: str = "chicken-weight-model-artifacts"
    s3_prefix: str = "models"
    
    # Deployment settings
    deployment_timeout: int = 600  # seconds
    update_strategy: str = "blue_green"  # blue_green, rolling
    
    def __init__(
        self,
        sagemaker_config: Optional[Dict[str, Any]] = None,
        container_config: Optional[Dict[str, Any]] = None,
        monitoring_config: Optional[Dict[str, Any]] = None,
        security_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        # Initialize sub-configs
        self.sagemaker = SageMakerConfig(**(sagemaker_config or {}))
        self.container = ContainerConfig(**(container_config or {}))
        self.monitoring = MonitoringConfig(**(monitoring_config or {}))
        self.security = SecurityConfig(**(security_config or {}))
        
        # Set other attributes
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DeploymentConfig':
        """Create DeploymentConfig from dictionary."""
        return cls(
            sagemaker_config=config_dict.get('sagemaker', {}),
            container_config=config_dict.get('container', {}),
            monitoring_config=config_dict.get('monitoring', {}),
            security_config=config_dict.get('security', {}),
            **{k: v for k, v in config_dict.items() 
               if k not in ['sagemaker', 'container', 'monitoring', 'security']}
        )
    
    def validate(self) -> bool:
        """Validate deployment configuration."""
        # Validate SageMaker config
        if not self.sagemaker.endpoint_name:
            raise ValueError("Endpoint name is required")
        
        if self.sagemaker.initial_instance_count <= 0:
            raise ValueError("Initial instance count must be positive")
        
        if self.sagemaker.min_instance_count > self.sagemaker.max_instance_count:
            raise ValueError("Min instance count cannot exceed max instance count")
        
        # Validate container config
        if not self.container.ecr_repository:
            raise ValueError("ECR repository is required")
        
        # Validate monitoring config
        if self.monitoring.error_rate_threshold < 0 or self.monitoring.error_rate_threshold > 1:
            raise ValueError("Error rate threshold must be between 0 and 1")
        
        # Validate security config
        if not self.security.execution_role_arn:
            raise ValueError("Execution role ARN is required")
        
        return True