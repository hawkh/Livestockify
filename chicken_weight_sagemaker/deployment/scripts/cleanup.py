#!/usr/bin/env python3
"""
Cleanup script for SageMaker resources.
"""

import boto3
import argparse
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SageMakerResourceCleaner:
    """Clean up SageMaker resources to avoid unnecessary costs."""
    
    def __init__(self, region: str = 'us-east-1', dry_run: bool = True):
        self.region = region
        self.dry_run = dry_run
        self.sagemaker_client = boto3.client('sagemaker', region_name=region)
        self.s3_client = boto3.client('s3', region_name=region)
        self.autoscaling_client = boto3.client('application-autoscaling', region_name=region)
        self.cloudwatch_client = boto3.client('cloudwatch', region_name=region)
        
        logger.info(f"Initialized cleanup for region {region} (dry_run={dry_run})")
    
    def list_chicken_weight_resources(self) -> Dict[str, List[str]]:
        """List all chicken weight estimation related resources."""
        resources = {
            'endpoints': [],
            'endpoint_configs': [],
            'models': [],
            'training_jobs': [],
            'processing_jobs': []
        }
        
        # List endpoints
        try:
            paginator = self.sagemaker_client.get_paginator('list_endpoints')
            for page in paginator.paginate():
                for endpoint in page['Endpoints']:
                    if 'chicken-weight' in endpoint['EndpointName'].lower():
                        resources['endpoints'].append(endpoint['EndpointName'])
        except Exception as e:
            logger.error(f"Error listing endpoints: {str(e)}")
        
        # List endpoint configurations
        try:
            paginator = self.sagemaker_client.get_paginator('list_endpoint_configs')
            for page in paginator.paginate():
                for config in page['EndpointConfigs']:
                    if 'chicken-weight' in config['EndpointConfigName'].lower():
                        resources['endpoint_configs'].append(config['EndpointConfigName'])
        except Exception as e:
            logger.error(f"Error listing endpoint configs: {str(e)}")
        
        # List models
        try:
            paginator = self.sagemaker_client.get_paginator('list_models')
            for page in paginator.paginate():
                for model in page['Models']:
                    if 'chicken-weight' in model['ModelName'].lower():
                        resources['models'].append(model['ModelName'])
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
        
        # List training jobs
        try:
            paginator = self.sagemaker_client.get_paginator('list_training_jobs')
            for page in paginator.paginate():
                for job in page['TrainingJobSummaries']:
                    if 'chicken-weight' in job['TrainingJobName'].lower():
                        resources['training_jobs'].append(job['TrainingJobName'])
        except Exception as e:
            logger.error(f"Error listing training jobs: {str(e)}")
        
        return resources
    
    def delete_endpoint(self, endpoint_name: str) -> bool:
        """Delete a SageMaker endpoint."""
        try:
            if self.dry_run:
                logger.info(f"[DRY RUN] Would delete endpoint: {endpoint_name}")
                return True
            
            logger.info(f"Deleting endpoint: {endpoint_name}")
            
            # Remove auto-scaling configuration first
            self._remove_autoscaling_config(endpoint_name)
            
            # Delete endpoint
            self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            
            # Wait for deletion
            waiter = self.sagemaker_client.get_waiter('endpoint_deleted')
            waiter.wait(EndpointName=endpoint_name)
            
            logger.info(f"Successfully deleted endpoint: {endpoint_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete endpoint {endpoint_name}: {str(e)}")
            return False
    
    def delete_endpoint_config(self, config_name: str) -> bool:
        """Delete a SageMaker endpoint configuration."""
        try:
            if self.dry_run:
                logger.info(f"[DRY RUN] Would delete endpoint config: {config_name}")
                return True
            
            logger.info(f"Deleting endpoint config: {config_name}")
            self.sagemaker_client.delete_endpoint_config(EndpointConfigName=config_name)
            logger.info(f"Successfully deleted endpoint config: {config_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete endpoint config {config_name}: {str(e)}")
            return False
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a SageMaker model."""
        try:
            if self.dry_run:
                logger.info(f"[DRY RUN] Would delete model: {model_name}")
                return True
            
            logger.info(f"Deleting model: {model_name}")
            self.sagemaker_client.delete_model(ModelName=model_name)
            logger.info(f"Successfully deleted model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model {model_name}: {str(e)}")
            return False
    
    def _remove_autoscaling_config(self, endpoint_name: str):
        """Remove auto-scaling configuration for an endpoint."""
        try:
            # List scaling policies
            response = self.autoscaling_client.describe_scaling_policies(
                ServiceNamespace='sagemaker',
                ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic'
            )
            
            # Delete scaling policies
            for policy in response.get('ScalingPolicies', []):\n                if not self.dry_run:
                    self.autoscaling_client.delete_scaling_policy(
                        PolicyName=policy['PolicyName'],
                        ServiceNamespace='sagemaker',
                        ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
                        ScalableDimension='sagemaker:variant:DesiredInstanceCount'
                    )
                    logger.info(f\"Deleted scaling policy: {policy['PolicyName']}\")
            
            # Deregister scalable target
            if not self.dry_run:
                self.autoscaling_client.deregister_scalable_target(
                    ServiceNamespace='sagemaker',
                    ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
                    ScalableDimension='sagemaker:variant:DesiredInstanceCount'
                )
                logger.info(f\"Deregistered scalable target for {endpoint_name}\")
                
        except Exception as e:
            logger.warning(f\"Failed to remove auto-scaling config for {endpoint_name}: {str(e)}\")
    
    def delete_cloudwatch_alarms(self, endpoint_name: str) -> bool:
        \"\"\"Delete CloudWatch alarms for an endpoint.\"\"\"
        try:
            # List alarms for the endpoint
            alarm_names = [
                f'{endpoint_name}-high-latency',
                f'{endpoint_name}-high-error-rate',
                f'{endpoint_name}-low-invocations'
            ]
            
            if self.dry_run:
                logger.info(f\"[DRY RUN] Would delete alarms: {alarm_names}\")
                return True
            
            # Delete alarms
            if alarm_names:
                self.cloudwatch_client.delete_alarms(AlarmNames=alarm_names)
                logger.info(f\"Deleted CloudWatch alarms for {endpoint_name}\")
            
            return True
            
        except Exception as e:
            logger.error(f\"Failed to delete CloudWatch alarms for {endpoint_name}: {str(e)}\")
            return False
    
    def cleanup_s3_artifacts(self, bucket_name: str, prefix: str = 'chicken-weight-model', 
                           older_than_days: int = 7) -> bool:
        \"\"\"Clean up old model artifacts from S3.\"\"\"
        try:
            cutoff_date = datetime.now() - timedelta(days=older_than_days)
            
            # List objects
            paginator = self.s3_client.get_paginator('list_objects_v2')
            objects_to_delete = []
            
            for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
                for obj in page.get('Contents', []):
                    if obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                        objects_to_delete.append({'Key': obj['Key']})
            
            if not objects_to_delete:
                logger.info(f\"No S3 objects older than {older_than_days} days found\")
                return True
            
            if self.dry_run:
                logger.info(f\"[DRY RUN] Would delete {len(objects_to_delete)} S3 objects\")
                for obj in objects_to_delete[:5]:  # Show first 5
                    logger.info(f\"  - {obj['Key']}\")
                if len(objects_to_delete) > 5:
                    logger.info(f\"  ... and {len(objects_to_delete) - 5} more\")
                return True
            
            # Delete objects in batches
            batch_size = 1000
            for i in range(0, len(objects_to_delete), batch_size):
                batch = objects_to_delete[i:i + batch_size]
                self.s3_client.delete_objects(
                    Bucket=bucket_name,
                    Delete={'Objects': batch}
                )
                logger.info(f\"Deleted batch of {len(batch)} S3 objects\")
            
            logger.info(f\"Successfully deleted {len(objects_to_delete)} S3 objects\")
            return True
            
        except Exception as e:
            logger.error(f\"Failed to cleanup S3 artifacts: {str(e)}\")
            return False
    
    def cleanup_all_resources(self, include_s3: bool = False, s3_bucket: str = None,
                            s3_older_than_days: int = 7) -> Dict[str, int]:
        \"\"\"Clean up all chicken weight estimation resources.\"\"\"
        logger.info(\"Starting comprehensive cleanup...\")
        
        # Get all resources
        resources = self.list_chicken_weight_resources()
        
        # Track cleanup results
        results = {
            'endpoints_deleted': 0,
            'endpoint_configs_deleted': 0,
            'models_deleted': 0,
            's3_objects_deleted': 0,
            'alarms_deleted': 0
        }
        
        # Delete endpoints (this will also handle auto-scaling and alarms)
        for endpoint_name in resources['endpoints']:
            if self.delete_endpoint(endpoint_name):
                results['endpoints_deleted'] += 1
                
            # Delete associated CloudWatch alarms
            if self.delete_cloudwatch_alarms(endpoint_name):
                results['alarms_deleted'] += 1
        
        # Delete endpoint configurations
        for config_name in resources['endpoint_configs']:
            if self.delete_endpoint_config(config_name):
                results['endpoint_configs_deleted'] += 1
        
        # Delete models
        for model_name in resources['models']:
            if self.delete_model(model_name):
                results['models_deleted'] += 1
        
        # Clean up S3 artifacts if requested
        if include_s3 and s3_bucket:
            if self.cleanup_s3_artifacts(s3_bucket, older_than_days=s3_older_than_days):
                results['s3_objects_deleted'] = 1  # Placeholder
        
        logger.info(\"Cleanup completed!\")
        return results
    
    def estimate_cost_savings(self, resources: Dict[str, List[str]]) -> Dict[str, float]:
        \"\"\"Estimate potential cost savings from cleanup.\"\"\"
        # Rough cost estimates (USD per hour)
        instance_costs = {
            'ml.g4dn.xlarge': 0.736,
            'ml.g4dn.2xlarge': 1.472,
            'ml.m5.large': 0.115,
            'ml.m5.xlarge': 0.230
        }
        
        total_hourly_cost = 0.0
        endpoint_details = []
        
        for endpoint_name in resources['endpoints']:
            try:
                # Get endpoint configuration
                response = self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
                config_name = response['EndpointConfigName']
                
                config_response = self.sagemaker_client.describe_endpoint_config(
                    EndpointConfigName=config_name
                )
                
                for variant in config_response['ProductionVariants']:
                    instance_type = variant['InstanceType']
                    instance_count = variant['InitialInstanceCount']
                    
                    hourly_cost = instance_costs.get(instance_type, 0.5) * instance_count
                    total_hourly_cost += hourly_cost
                    
                    endpoint_details.append({
                        'endpoint': endpoint_name,
                        'instance_type': instance_type,
                        'instance_count': instance_count,
                        'hourly_cost': hourly_cost
                    })
                    
            except Exception as e:
                logger.warning(f\"Could not get cost info for {endpoint_name}: {str(e)}\")
        
        return {
            'hourly_savings': total_hourly_cost,
            'daily_savings': total_hourly_cost * 24,
            'monthly_savings': total_hourly_cost * 24 * 30,
            'endpoint_details': endpoint_details
        }


def main():
    \"\"\"Main cleanup function.\"\"\"
    parser = argparse.ArgumentParser(description='Clean up SageMaker resources')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted without actually deleting')
    parser.add_argument('--include-s3', action='store_true', help='Also clean up S3 artifacts')
    parser.add_argument('--s3-bucket', help='S3 bucket name for artifact cleanup')
    parser.add_argument('--s3-older-than-days', type=int, default=7, help='Delete S3 objects older than N days')
    parser.add_argument('--endpoint-name', help='Clean up specific endpoint only')
    parser.add_argument('--show-costs', action='store_true', help='Show estimated cost savings')
    
    args = parser.parse_args()
    
    # Initialize cleaner
    cleaner = SageMakerResourceCleaner(region=args.region, dry_run=args.dry_run)
    
    try:
        if args.endpoint_name:
            # Clean up specific endpoint
            logger.info(f\"Cleaning up specific endpoint: {args.endpoint_name}\")
            
            success = cleaner.delete_endpoint(args.endpoint_name)
            cleaner.delete_cloudwatch_alarms(args.endpoint_name)
            
            if success:
                print(f\"✅ Successfully cleaned up endpoint: {args.endpoint_name}\")
            else:
                print(f\"❌ Failed to clean up endpoint: {args.endpoint_name}\")
                
        else:
            # Get all resources
            resources = cleaner.list_chicken_weight_resources()
            
            # Show what will be cleaned up
            print(\"\\n\" + \"=\"*60)
            print(\"🧹 SAGEMAKER RESOURCE CLEANUP\")
            print(\"=\"*60)
            print(f\"Region: {args.region}\")
            print(f\"Dry Run: {args.dry_run}\")
            print()
            
            for resource_type, items in resources.items():
                if items:
                    print(f\"{resource_type.replace('_', ' ').title()}: {len(items)}\")
                    for item in items:
                        print(f\"  - {item}\")
                    print()
            
            # Show cost estimates if requested
            if args.show_costs:
                cost_info = cleaner.estimate_cost_savings(resources)
                print(\"💰 ESTIMATED COST SAVINGS:\")
                print(f\"  Hourly: ${cost_info['hourly_savings']:.2f}\")
                print(f\"  Daily: ${cost_info['daily_savings']:.2f}\")
                print(f\"  Monthly: ${cost_info['monthly_savings']:.2f}\")
                print()
            
            # Confirm before proceeding (unless dry run)
            if not args.dry_run:
                total_resources = sum(len(items) for items in resources.values())
                if total_resources == 0:
                    print(\"No resources found to clean up.\")
                    return 0
                
                response = input(f\"Are you sure you want to delete {total_resources} resources? (yes/no): \")
                if response.lower() != 'yes':
                    print(\"Cleanup cancelled.\")
                    return 0
            
            # Perform cleanup
            results = cleaner.cleanup_all_resources(
                include_s3=args.include_s3,
                s3_bucket=args.s3_bucket,
                s3_older_than_days=args.s3_older_than_days
            )
            
            # Show results
            print(\"\\n\" + \"=\"*60)
            print(\"🎉 CLEANUP COMPLETED!\")
            print(\"=\"*60)
            for resource_type, count in results.items():
                if count > 0:
                    print(f\"{resource_type.replace('_', ' ').title()}: {count}\")
            
            if args.dry_run:
                print(\"\\n⚠️  This was a dry run. No resources were actually deleted.\")
                print(\"Run without --dry-run to perform actual cleanup.\")
        
        return 0
        
    except Exception as e:
        logger.error(f\"Cleanup failed: {str(e)}\")
        return 1


if __name__ == '__main__':
    exit(main())