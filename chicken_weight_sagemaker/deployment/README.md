# SageMaker Deployment Guide

This directory contains scripts and configurations for deploying the chicken weight estimation model to Amazon SageMaker.

## Directory Structure

```
deployment/
├── scripts/
│   ├── deploy.py           # Main deployment script
│   ├── cleanup.py          # Resource cleanup script
│   ├── monitor.py          # Monitoring and health checks
│   └── update_endpoint.py  # Endpoint update and rollback
├── config/
│   └── deployment_config.yaml  # Deployment configuration
└── README.md              # This file
```

## Prerequisites

1. **AWS CLI configured** with appropriate credentials
2. **Python dependencies** installed:
   ```bash
   pip install boto3 sagemaker matplotlib pandas
   ```
3. **IAM Role** with SageMaker execution permissions
4. **Model artifacts** prepared and available

## Quick Start

### 1. Deploy Model

```bash
python deployment/scripts/deploy.py \
    --role-arn arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole \
    --model-dir ./model_artifacts \
    --instance-type ml.g4dn.xlarge \
    --endpoint-name chicken-weight-endpoint
```

### 2. Monitor Endpoint

```bash
# Check endpoint status
python deployment/scripts/monitor.py \
    --endpoint-name chicken-weight-endpoint \
    --action status

# Generate performance report
python deployment/scripts/monitor.py \
    --endpoint-name chicken-weight-endpoint \
    --action report \
    --hours 24
```

### 3. Update Model

```bash
# Blue-green deployment
python deployment/scripts/update_endpoint.py \
    --endpoint-name chicken-weight-endpoint \
    --model-uri s3://your-bucket/new-model/model.tar.gz \
    --role-arn arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole \
    --deployment-type blue-green
```

### 4. Cleanup Resources

```bash
# Dry run to see what would be deleted
python deployment/scripts/cleanup.py --dry-run

# Actually delete resources
python deployment/scripts/cleanup.py
```

## Detailed Usage

### Deployment Script (deploy.py)

The main deployment script handles the complete deployment process:

**Basic deployment:**
```bash
python deployment/scripts/deploy.py \
    --role-arn arn:aws:iam::123456789012:role/SageMakerExecutionRole \
    --model-dir ./model_artifacts
```

**Advanced deployment with custom settings:**
```bash
python deployment/scripts/deploy.py \
    --role-arn arn:aws:iam::123456789012:role/SageMakerExecutionRole \
    --model-dir ./model_artifacts \
    --instance-type ml.g4dn.2xlarge \
    --min-instances 2 \
    --max-instances 8 \
    --endpoint-name my-chicken-endpoint \
    --region us-west-2
```

**Parameters:**
- `--role-arn`: SageMaker execution role ARN (required)
- `--model-dir`: Directory containing model files (default: ./model_artifacts)
- `--instance-type`: EC2 instance type (default: ml.g4dn.xlarge)
- `--min-instances`: Minimum auto-scaling instances (default: 1)
- `--max-instances`: Maximum auto-scaling instances (default: 4)
- `--endpoint-name`: Custom endpoint name (auto-generated if not provided)
- `--region`: AWS region (default: us-east-1)

### Monitoring Script (monitor.py)

Monitor endpoint performance and health:

**Check endpoint status:**
```bash
python deployment/scripts/monitor.py \
    --endpoint-name chicken-weight-endpoint \
    --action status
```

**Run health check:**
```bash
python deployment/scripts/monitor.py \
    --endpoint-name chicken-weight-endpoint \
    --action health
```

**Get metrics:**
```bash
python deployment/scripts/monitor.py \
    --endpoint-name chicken-weight-endpoint \
    --action metrics \
    --hours 24
```

**Generate comprehensive report:**
```bash
python deployment/scripts/monitor.py \
    --endpoint-name chicken-weight-endpoint \
    --action report \
    --hours 24 \
    --output-dir ./reports
```

**Continuous monitoring:**
```bash
python deployment/scripts/monitor.py \
    --endpoint-name chicken-weight-endpoint \
    --action monitor \
    --interval 5 \
    --duration 2
```

### Update Script (update_endpoint.py)

Update endpoints with new model versions:

**Blue-green deployment (recommended):**
```bash
python deployment/scripts/update_endpoint.py \
    --endpoint-name chicken-weight-endpoint \
    --model-uri s3://your-bucket/new-model/model.tar.gz \
    --role-arn arn:aws:iam::123456789012:role/SageMakerExecutionRole \
    --deployment-type blue-green \
    --traffic-percentage 10 \
    --wait-minutes 15
```

**Direct deployment:**
```bash
python deployment/scripts/update_endpoint.py \
    --endpoint-name chicken-weight-endpoint \
    --model-uri s3://your-bucket/new-model/model.tar.gz \
    --role-arn arn:aws:iam::123456789012:role/SageMakerExecutionRole \
    --deployment-type direct
```

**Rollback to previous version:**
```bash
python deployment/scripts/update_endpoint.py \
    --endpoint-name chicken-weight-endpoint \
    --rollback-config chicken-weight-config-20240126-143022
```

**Validate current deployment:**
```bash
python deployment/scripts/update_endpoint.py \
    --endpoint-name chicken-weight-endpoint \
    --validate-only
```

### Cleanup Script (cleanup.py)

Clean up SageMaker resources to avoid unnecessary costs:

**Dry run (recommended first):**
```bash
python deployment/scripts/cleanup.py --dry-run
```

**Show cost estimates:**
```bash
python deployment/scripts/cleanup.py --dry-run --show-costs
```

**Clean up all resources:**
```bash
python deployment/scripts/cleanup.py
```

**Clean up specific endpoint:**
```bash
python deployment/scripts/cleanup.py --endpoint-name chicken-weight-endpoint
```

**Include S3 cleanup:**
```bash
python deployment/scripts/cleanup.py \
    --include-s3 \
    --s3-bucket your-sagemaker-bucket \
    --s3-older-than-days 7
```

## Configuration

### Deployment Configuration (deployment_config.yaml)

The deployment configuration file contains default settings for:

- Model configuration (framework, versions)
- Instance configuration (types, counts)
- Auto-scaling settings
- Monitoring and alerting
- Security settings
- Regional configurations

You can customize these settings by editing `deployment/config/deployment_config.yaml`.

### Environment Variables

The deployment scripts support these environment variables:

- `AWS_DEFAULT_REGION`: Default AWS region
- `SAGEMAKER_ROLE_ARN`: Default SageMaker execution role
- `MODEL_BUCKET`: Default S3 bucket for model artifacts

## Model Artifacts Structure

Your model directory should contain:

```
model_artifacts/
├── yolo_best.pt              # YOLO detection model
├── weight_nn.pt              # Weight estimation neural network
├── camera_config.yaml        # Camera configuration
└── model_config.yaml         # Model configuration
```

## IAM Permissions

Your SageMaker execution role needs these permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:*",
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket",
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
                "cloudwatch:PutMetricData",
                "ecr:GetAuthorizationToken",
                "ecr:BatchCheckLayerAvailability",
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage"
            ],
            "Resource": "*"
        }
    ]
}
```

## Troubleshooting

### Common Issues

1. **Model loading errors**
   - Check model file paths in artifacts
   - Verify model compatibility with PyTorch version
   - Check inference script imports

2. **Endpoint creation failures**
   - Verify IAM role permissions
   - Check instance type availability in region
   - Ensure model artifacts are accessible

3. **High latency**
   - Consider using GPU instances (ml.g4dn.*)
   - Optimize model inference code
   - Enable model caching

4. **Auto-scaling issues**
   - Check CloudWatch metrics
   - Verify scaling policies
   - Monitor instance limits

### Logs and Debugging

- **CloudWatch Logs**: `/aws/sagemaker/Endpoints/{endpoint-name}`
- **Model logs**: Check inference script logging
- **Deployment logs**: Script output and CloudWatch

### Getting Help

1. Check AWS SageMaker documentation
2. Review CloudWatch logs and metrics
3. Use the monitoring script for diagnostics
4. Check model artifacts and configuration files

## Cost Optimization

1. **Use appropriate instance types**
   - Start with ml.g4dn.xlarge for GPU inference
   - Use ml.m5.large for CPU-only inference

2. **Configure auto-scaling**
   - Set appropriate min/max instances
   - Monitor invocation patterns

3. **Regular cleanup**
   - Use cleanup script to remove unused resources
   - Monitor S3 storage costs

4. **Consider spot instances** (for development)
   - Enable in deployment configuration
   - Not recommended for production

## Security Best Practices

1. **Use VPC endpoints** for private communication
2. **Enable encryption** at rest and in transit
3. **Implement proper IAM policies** with least privilege
4. **Monitor access** with CloudTrail
5. **Use data capture** for audit trails

## Next Steps

After successful deployment:

1. **Set up monitoring dashboards** in CloudWatch
2. **Configure alerting** for critical metrics
3. **Implement CI/CD pipeline** for model updates
4. **Set up data capture analysis** for model drift detection
5. **Plan for disaster recovery** and backup strategies