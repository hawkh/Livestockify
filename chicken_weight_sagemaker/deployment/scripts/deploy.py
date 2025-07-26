"""
SageMaker deployment script for chicken weight estimation model.
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
import json
import os
import argparse
import tarfile
import shutil
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChickenWeightSageMakerDeployer:
    """Deploy chicken weight estimation model to SageMaker."""
    
    def __init__(self, role_arn: str, region: str = 'us-east-1'):
        self.role_arn = role_arn
        self.region = region
        self.sagemaker_session = sagemaker.Session(boto3.Session(region_name=region))
        self.s3_client = boto3.client('s3', region_name=region)
        self.bucket = self.sagemaker_session.default_bucket()
        
        logger.info(f"Initialized deployer for region {region}")
        logger.info(f"Using S3 bucket: {self.bucket}")
    
    def prepare_model_artifacts(self, model_dir: str, output_path: str = 'model.tar.gz'):
        """Package model artifacts for SageMaker deployment."""
        logger.info("Preparing model artifacts...")
        
        # Create temporary directory
        temp_dir = Path('temp_model_package')
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir()
        
        try:
            # Copy model files
            model_files = [
                'yolo_best.pt',
                'weight_nn.pt',
                'camera_config.yaml',
                'model_config.yaml'
            ]
            
            # Create model artifacts directory
            artifacts_dir = temp_dir / 'model_artifacts'
            artifacts_dir.mkdir()
            
            for file_name in model_files:
                src_path = Path(model_dir) / file_name
                if src_path.exists():
                    shutil.copy2(src_path, artifacts_dir / file_name)
                    logger.info(f"Copied {file_name}")
                else:
                    logger.warning(f"Model file not found: {file_name}")
            
            # Copy source code
            src_dir = Path('src')
            if src_dir.exists():
                dst_dir = temp_dir / 'code' / 'src'
                shutil.copytree(src_dir, dst_dir)
                logger.info("Copied source code")
            
            # Copy inference script
            inference_script = Path('src/inference/sagemaker_handler.py')
            if inference_script.exists():
                shutil.copy2(inference_script, temp_dir / 'code' / 'inference.py')
                logger.info("Copied inference script")
            
            # Create requirements.txt
            requirements_content = """
torch>=2.0.1
torchvision>=0.15.2
ultralytics>=8.0.0
opencv-python>=4.5.0
numpy>=1.21.0
Pillow>=8.3.0
flask>=2.3.0
pyyaml>=6.0
scikit-learn>=1.3.0
scipy>=1.10.0
filterpy>=1.4.5
psutil>=5.9.0
""".strip()
            
            with open(temp_dir / 'code' / 'requirements.txt', 'w') as f:
                f.write(requirements_content)
            
            # Create tar.gz archive
            with tarfile.open(output_path, 'w:gz') as tar:
                tar.add(temp_dir, arcname='.')
            
            logger.info(f"Model artifacts packaged: {output_path}")
            return output_path
            
        finally:
            # Cleanup
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def upload_model_to_s3(self, model_path: str, prefix: str = 'chicken-weight-model'):
        """Upload model artifacts to S3."""
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        s3_key = f"{prefix}/models/{timestamp}/model.tar.gz"
        
        logger.info(f"Uploading model to s3://{self.bucket}/{s3_key}")
        
        self.s3_client.upload_file(model_path, self.bucket, s3_key)
        s3_uri = f"s3://{self.bucket}/{s3_key}"
        
        logger.info(f"Model uploaded to: {s3_uri}")
        return s3_uri
    
    def deploy_model(
        self, 
        model_uri: str, 
        model_name: str = 'chicken-weight-model',
        instance_type: str = 'ml.g4dn.xlarge',
        endpoint_name: str = None
    ):
        """Deploy model to SageMaker endpoint."""
        if endpoint_name is None:
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            endpoint_name = f"{model_name}-endpoint-{timestamp}"
        
        logger.info(f"Deploying model to endpoint: {endpoint_name}")
        
        # Create PyTorch model
        pytorch_model = PyTorchModel(
            model_data=model_uri,
            role=self.role_arn,
            entry_point='inference.py',
            source_dir=None,  # Code is packaged in model.tar.gz
            framework_version='2.0.1',
            py_version='py310',
            sagemaker_session=self.sagemaker_session,
            env={
                'SAGEMAKER_PROGRAM': 'inference.py',
                'SAGEMAKER_SUBMIT_DIRECTORY': model_uri,
                'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
                'PYTHONUNBUFFERED': 'TRUE'
            }
        )
        
        # Deploy with configuration
        predictor = pytorch_model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer(),
            wait=True
        )
        
        logger.info(f"Model deployed successfully to endpoint: {endpoint_name}")
        
        # Configure auto-scaling
        self.configure_autoscaling(endpoint_name)
        
        # Setup monitoring
        self.setup_monitoring(endpoint_name)
        
        return predictor
    
    def configure_autoscaling(
        self, 
        endpoint_name: str, 
        min_instances: int = 1, 
        max_instances: int = 4,
        target_invocations: int = 100
    ):
        """Configure auto-scaling for the endpoint."""
        logger.info(f"Configuring auto-scaling for {endpoint_name}")
        
        autoscaling_client = boto3.client('application-autoscaling', region_name=self.region)
        
        try:
            # Register scalable target
            autoscaling_client.register_scalable_target(
                ServiceNamespace='sagemaker',
                ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
                ScalableDimension='sagemaker:variant:DesiredInstanceCount',
                MinCapacity=min_instances,
                MaxCapacity=max_instances
            )
            
            # Create scaling policy
            policy_name = f'{endpoint_name}-scaling-policy'
            autoscaling_client.put_scaling_policy(
                PolicyName=policy_name,
                ServiceNamespace='sagemaker',
                ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
                ScalableDimension='sagemaker:variant:DesiredInstanceCount',
                PolicyType='TargetTrackingScaling',
                TargetTrackingScalingPolicyConfiguration={
                    'TargetValue': float(target_invocations),
                    'PredefinedMetricSpecification': {
                        'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
                    },
                    'ScaleInCooldown': 300,  # 5 minutes
                    'ScaleOutCooldown': 60   # 1 minute
                }
            )
            
            logger.info(f"Auto-scaling configured: {min_instances}-{max_instances} instances")
            
        except Exception as e:
            logger.error(f"Failed to configure auto-scaling: {str(e)}")
    
    def setup_monitoring(self, endpoint_name: str):
        """Setup CloudWatch monitoring and alarms."""
        logger.info(f"Setting up monitoring for {endpoint_name}")
        
        cloudwatch = boto3.client('cloudwatch', region_name=self.region)
        
        # Define alarms
        alarms = [
            {
                'AlarmName': f'{endpoint_name}-high-latency',
                'MetricName': 'ModelLatency',
                'Threshold': 5000.0,  # 5 seconds
                'ComparisonOperator': 'GreaterThanThreshold',
                'AlarmDescription': 'High model latency detected'
            },
            {
                'AlarmName': f'{endpoint_name}-high-error-rate',
                'MetricName': 'ModelInvocation4XXErrors',
                'Threshold': 10.0,
                'ComparisonOperator': 'GreaterThanThreshold',
                'AlarmDescription': 'High error rate detected'
            },
            {
                'AlarmName': f'{endpoint_name}-low-invocations',
                'MetricName': 'Invocations',
                'Threshold': 1.0,
                'ComparisonOperator': 'LessThanThreshold',
                'AlarmDescription': 'Low invocation rate - possible issues'
            }
        ]
        
        for alarm_config in alarms:
            try:
                cloudwatch.put_metric_alarm(
                    **alarm_config,
                    Namespace='AWS/SageMaker',
                    Statistic='Average',
                    Period=300,
                    EvaluationPeriods=2,
                    Dimensions=[
                        {'Name': 'EndpointName', 'Value': endpoint_name},
                        {'Name': 'VariantName', 'Value': 'AllTraffic'}
                    ]
                )
                logger.info(f"Created alarm: {alarm_config['AlarmName']}")
                
            except Exception as e:
                logger.error(f"Failed to create alarm {alarm_config['AlarmName']}: {str(e)}")
    
    def test_endpoint(self, predictor):
        """Test the deployed endpoint."""
        logger.info("Testing deployed endpoint...")
        
        # Create test data
        test_request = {
            "stream_data": {
                "frame": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
                "camera_id": "test_camera",
                "frame_sequence": 1,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        try:
            response = predictor.predict(test_request)
            logger.info("Test successful!")
            logger.info(f"Response: {json.dumps(response, indent=2)}")
            return True
            
        except Exception as e:
            logger.error(f"Test failed: {str(e)}")
            return False


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description='Deploy Chicken Weight Estimation Model to SageMaker')
    parser.add_argument('--role-arn', required=True, help='SageMaker execution role ARN')
    parser.add_argument('--model-dir', default='./model_artifacts', help='Directory containing model files')
    parser.add_argument('--instance-type', default='ml.g4dn.xlarge', help='SageMaker instance type')
    parser.add_argument('--min-instances', type=int, default=1, help='Minimum number of instances')
    parser.add_argument('--max-instances', type=int, default=4, help='Maximum number of instances')
    parser.add_argument('--endpoint-name', help='Custom endpoint name')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    
    args = parser.parse_args()
    
    try:
        # Initialize deployer
        deployer = ChickenWeightSageMakerDeployer(
            role_arn=args.role_arn,
            region=args.region
        )
        
        # Prepare model artifacts
        model_archive = deployer.prepare_model_artifacts(args.model_dir)
        
        # Upload to S3
        model_uri = deployer.upload_model_to_s3(model_archive)
        
        # Deploy model
        predictor = deployer.deploy_model(
            model_uri=model_uri,
            instance_type=args.instance_type,
            endpoint_name=args.endpoint_name
        )
        
        # Test endpoint
        test_success = deployer.test_endpoint(predictor)
        
        # Cleanup local files
        if os.path.exists(model_archive):
            os.remove(model_archive)
        
        print("\n" + "="*60)
        print("🎉 DEPLOYMENT COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Endpoint name: {predictor.endpoint_name}")
        print(f"Instance type: {args.instance_type}")
        print(f"Auto-scaling: {args.min_instances}-{args.max_instances} instances")
        print(f"Test result: {'✅ PASSED' if test_success else '❌ FAILED'}")
        print("\nTo invoke the endpoint:")
        print(f"aws sagemaker-runtime invoke-endpoint \\")
        print(f"  --endpoint-name {predictor.endpoint_name} \\")
        print(f"  --content-type application/json \\")
        print(f"  --body file://test_request.json \\")
        print(f"  response.json")
        
        return 0
        
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        return 1


if __name__ == '__main__':
    exit(main())