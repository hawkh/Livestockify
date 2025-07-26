#!/usr/bin/env python3
"""
Script to update SageMaker endpoint with new model version.
"""

import boto3
import sagemaker
import argparse
import logging
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EndpointUpdater:
    """Update SageMaker endpoint with new model version."""
    
    def __init__(self, region: str = 'us-east-1'):
        self.region = region
        self.sagemaker_client = boto3.client('sagemaker', region_name=region)
        self.sagemaker_session = sagemaker.Session(boto3.Session(region_name=region))
        
        logger.info(f"Initialized endpoint updater for region {region}")
    
    def create_new_model(self, model_uri: str, model_name: str, role_arn: str,
                        image_uri: Optional[str] = None) -> str:
        """Create a new model version."""
        logger.info(f"Creating new model: {model_name}")
        
        if image_uri is None:
            # Use default PyTorch container
            image_uri = sagemaker.image_uris.retrieve(
                framework="pytorch",
                region=self.region,
                version="2.0.1",
                py_version="py310",
                instance_type="ml.g4dn.xlarge",
                accelerator_type=None,
                image_scope="inference"
            )
        
        try:
            response = self.sagemaker_client.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    'Image': image_uri,
                    'ModelDataUrl': model_uri,
                    'Environment': {
                        'SAGEMAKER_PROGRAM': 'inference.py',
                        'SAGEMAKER_SUBMIT_DIRECTORY': model_uri,
                        'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
                        'PYTHONUNBUFFERED': 'TRUE'
                    }
                },
                ExecutionRoleArn=role_arn
            )
            
            logger.info(f"Model created successfully: {model_name}")
            return model_name
            
        except Exception as e:
            logger.error(f"Failed to create model: {str(e)}")
            raise
    
    def create_new_endpoint_config(self, model_name: str, config_name: str,
                                 instance_type: str = 'ml.g4dn.xlarge',
                                 initial_instance_count: int = 1) -> str:
        """Create a new endpoint configuration."""
        logger.info(f"Creating new endpoint configuration: {config_name}")
        
        try:
            response = self.sagemaker_client.create_endpoint_config(
                EndpointConfigName=config_name,
                ProductionVariants=[
                    {
                        'VariantName': 'AllTraffic',
                        'ModelName': model_name,
                        'InitialInstanceCount': initial_instance_count,
                        'InstanceType': instance_type,
                        'InitialVariantWeight': 1.0
                    }
                ],
                DataCaptureConfig={
                    'EnableCapture': True,
                    'InitialSamplingPercentage': 10,
                    'DestinationS3Uri': f's3://{self.sagemaker_session.default_bucket()}/chicken-weight-data-capture',
                    'CaptureOptions': [
                        {'CaptureMode': 'Input'},
                        {'CaptureMode': 'Output'}
                    ]
                }
            )
            
            logger.info(f"Endpoint configuration created: {config_name}")
            return config_name
            
        except Exception as e:
            logger.error(f"Failed to create endpoint configuration: {str(e)}")
            raise
    
    def update_endpoint(self, endpoint_name: str, new_config_name: str,
                       wait_for_completion: bool = True) -> bool:
        """Update endpoint with new configuration."""
        logger.info(f"Updating endpoint {endpoint_name} with config {new_config_name}")
        
        try:
            # Start the update
            response = self.sagemaker_client.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=new_config_name
            )
            
            logger.info(f"Endpoint update initiated: {endpoint_name}")
            
            if wait_for_completion:
                logger.info("Waiting for endpoint update to complete...")
                
                # Wait for update to complete
                waiter = self.sagemaker_client.get_waiter('endpoint_in_service')
                waiter.wait(
                    EndpointName=endpoint_name,
                    WaiterConfig={
                        'Delay': 30,
                        'MaxAttempts': 60  # 30 minutes max
                    }
                )
                
                logger.info(f"Endpoint update completed successfully: {endpoint_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update endpoint: {str(e)}")
            raise
    
    def blue_green_deployment(self, endpoint_name: str, new_model_uri: str,
                            role_arn: str, traffic_shift_percentage: int = 10,
                            wait_minutes: int = 10) -> Dict[str, Any]:
        """Perform blue-green deployment with traffic shifting."""
        logger.info(f"Starting blue-green deployment for {endpoint_name}")
        
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        
        try:
            # Get current endpoint configuration
            current_endpoint = self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            current_config_name = current_endpoint['EndpointConfigName']
            
            current_config = self.sagemaker_client.describe_endpoint_config(
                EndpointConfigName=current_config_name
            )
            
            # Create new model
            new_model_name = f"chicken-weight-model-{timestamp}"
            self.create_new_model(new_model_uri, new_model_name, role_arn)
            
            # Create new endpoint configuration with traffic splitting
            new_config_name = f"chicken-weight-config-{timestamp}"
            
            # Get current variant configuration
            current_variant = current_config['ProductionVariants'][0]
            
            # Create configuration with both variants
            production_variants = [
                {
                    'VariantName': 'Blue',
                    'ModelName': current_variant['ModelName'],
                    'InitialInstanceCount': current_variant['InitialInstanceCount'],
                    'InstanceType': current_variant['InstanceType'],
                    'InitialVariantWeight': 100 - traffic_shift_percentage
                },
                {
                    'VariantName': 'Green',
                    'ModelName': new_model_name,
                    'InitialInstanceCount': current_variant['InitialInstanceCount'],
                    'InstanceType': current_variant['InstanceType'],
                    'InitialVariantWeight': traffic_shift_percentage
                }
            ]
            
            response = self.sagemaker_client.create_endpoint_config(
                EndpointConfigName=new_config_name,
                ProductionVariants=production_variants,
                DataCaptureConfig=current_config.get('DataCaptureConfig', {})
            )
            
            # Update endpoint
            self.update_endpoint(endpoint_name, new_config_name, wait_for_completion=True)
            
            logger.info(f"Blue-green deployment initiated with {traffic_shift_percentage}% traffic to new model")
            logger.info(f"Waiting {wait_minutes} minutes before full cutover...")
            
            # Wait for monitoring period
            time.sleep(wait_minutes * 60)
            
            # Create final configuration with 100% traffic to new model
            final_config_name = f"chicken-weight-config-final-{timestamp}"
            
            final_variants = [
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': new_model_name,
                    'InitialInstanceCount': current_variant['InitialInstanceCount'],
                    'InstanceType': current_variant['InstanceType'],
                    'InitialVariantWeight': 1.0
                }
            ]
            
            response = self.sagemaker_client.create_endpoint_config(
                EndpointConfigName=final_config_name,
                ProductionVariants=final_variants,
                DataCaptureConfig=current_config.get('DataCaptureConfig', {})
            )
            
            # Final update
            self.update_endpoint(endpoint_name, final_config_name, wait_for_completion=True)
            
            deployment_info = {
                'endpoint_name': endpoint_name,
                'old_model': current_variant['ModelName'],
                'new_model': new_model_name,
                'old_config': current_config_name,
                'new_config': final_config_name,
                'traffic_shift_percentage': traffic_shift_percentage,
                'wait_minutes': wait_minutes,
                'deployment_time': datetime.now().isoformat()
            }
            
            logger.info("Blue-green deployment completed successfully!")
            return deployment_info
            
        except Exception as e:
            logger.error(f"Blue-green deployment failed: {str(e)}")
            raise
    
    def rollback_endpoint(self, endpoint_name: str, target_config_name: str) -> bool:
        """Rollback endpoint to previous configuration."""
        logger.info(f"Rolling back endpoint {endpoint_name} to config {target_config_name}")
        
        try:
            # Update endpoint to previous configuration
            response = self.sagemaker_client.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=target_config_name
            )
            
            # Wait for rollback to complete
            waiter = self.sagemaker_client.get_waiter('endpoint_in_service')
            waiter.wait(
                EndpointName=endpoint_name,
                WaiterConfig={
                    'Delay': 30,
                    'MaxAttempts': 60
                }
            )
            
            logger.info(f"Rollback completed successfully: {endpoint_name}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {str(e)}")
            raise
    
    def get_endpoint_history(self, endpoint_name: str) -> List[Dict[str, Any]]:
        """Get endpoint configuration history."""
        try:
            # Get current endpoint
            endpoint = self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            
            # List all endpoint configurations (simplified - in practice you'd track this)
            configs = []
            
            # Get current configuration details
            current_config = self.sagemaker_client.describe_endpoint_config(
                EndpointConfigName=endpoint['EndpointConfigName']
            )
            
            config_info = {
                'config_name': endpoint['EndpointConfigName'],
                'creation_time': current_config['CreationTime'],
                'models': [variant['ModelName'] for variant in current_config['ProductionVariants']],
                'instance_types': [variant['InstanceType'] for variant in current_config['ProductionVariants']],
                'is_current': True
            }
            
            configs.append(config_info)
            
            return configs
            
        except Exception as e:
            logger.error(f"Failed to get endpoint history: {str(e)}")
            return []
    
    def validate_new_model(self, endpoint_name: str, test_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validate new model deployment with test payload."""
        logger.info(f"Validating model deployment for {endpoint_name}")
        
        try:
            # Test endpoint
            sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=self.region)
            
            start_time = time.time()
            response = sagemaker_runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=json.dumps(test_payload)
            )
            end_time = time.time()
            
            # Parse response
            result = json.loads(response['Body'].read().decode())
            
            validation_result = {
                'success': True,
                'response_time_ms': (end_time - start_time) * 1000,
                'response_keys': list(result.keys()),
                'detections_count': len(result.get('detections', [])),
                'tracks_count': len(result.get('tracks', [])),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Model validation successful: {validation_result}")
            return validation_result
            
        except Exception as e:
            logger.error(f"Model validation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


def main():
    """Main update function."""
    parser = argparse.ArgumentParser(description='Update SageMaker endpoint')
    parser.add_argument('--endpoint-name', required=True, help='SageMaker endpoint name')
    parser.add_argument('--model-uri', required=True, help='S3 URI of new model artifacts')
    parser.add_argument('--role-arn', required=True, help='SageMaker execution role ARN')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--deployment-type', choices=['direct', 'blue-green'], 
                       default='blue-green', help='Deployment strategy')
    parser.add_argument('--traffic-percentage', type=int, default=10, 
                       help='Initial traffic percentage for blue-green deployment')
    parser.add_argument('--wait-minutes', type=int, default=10, 
                       help='Wait time before full cutover in blue-green deployment')
    parser.add_argument('--instance-type', default='ml.g4dn.xlarge', help='Instance type')
    parser.add_argument('--rollback-config', help='Configuration name to rollback to')
    parser.add_argument('--validate-only', action='store_true', help='Only validate current endpoint')
    
    args = parser.parse_args()
    
    # Initialize updater
    updater = EndpointUpdater(region=args.region)
    
    try:
        if args.validate_only:
            # Validate current endpoint
            test_payload = {
                "stream_data": {
                    "frame": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
                    "camera_id": "validation_test",
                    "frame_sequence": 1,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            result = updater.validate_new_model(args.endpoint_name, test_payload)
            print(json.dumps(result, indent=2, default=str))
            
        elif args.rollback_config:
            # Rollback to specified configuration
            success = updater.rollback_endpoint(args.endpoint_name, args.rollback_config)
            if success:
                print(f"✅ Successfully rolled back {args.endpoint_name} to {args.rollback_config}")
            else:
                print(f"❌ Failed to rollback {args.endpoint_name}")
                
        elif args.deployment_type == 'blue-green':
            # Blue-green deployment
            deployment_info = updater.blue_green_deployment(
                endpoint_name=args.endpoint_name,
                new_model_uri=args.model_uri,
                role_arn=args.role_arn,
                traffic_shift_percentage=args.traffic_percentage,
                wait_minutes=args.wait_minutes
            )
            
            print("\n" + "="*60)
            print("🚀 BLUE-GREEN DEPLOYMENT COMPLETED!")
            print("="*60)
            print(f"Endpoint: {deployment_info['endpoint_name']}")
            print(f"Old Model: {deployment_info['old_model']}")
            print(f"New Model: {deployment_info['new_model']}")
            print(f"Traffic Shift: {deployment_info['traffic_shift_percentage']}%")
            print(f"Wait Time: {deployment_info['wait_minutes']} minutes")
            print(f"Deployment Time: {deployment_info['deployment_time']}")
            
        else:
            # Direct deployment
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            model_name = f"chicken-weight-model-{timestamp}"
            config_name = f"chicken-weight-config-{timestamp}"
            
            # Create new model and configuration
            updater.create_new_model(args.model_uri, model_name, args.role_arn)
            updater.create_new_endpoint_config(model_name, config_name, args.instance_type)
            
            # Update endpoint
            success = updater.update_endpoint(args.endpoint_name, config_name)
            
            if success:
                print(f"✅ Successfully updated {args.endpoint_name} with new model")
                print(f"Model: {model_name}")
                print(f"Config: {config_name}")
            else:
                print(f"❌ Failed to update {args.endpoint_name}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Update failed: {str(e)}")
        return 1


if __name__ == '__main__':
    exit(main())