terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  backend "s3" {
    bucket         = "chicken-weight-estimator-terraform-state"
    key            = "terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "chicken-weight-estimator-terraform-locks"
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "ChickenWeightEstimation"
      Environment = var.environment
      ManagedBy   = "Terraform"
      Owner       = var.owner
    }
  }
}

# Data sources
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# Local values
locals {
  account_id = data.aws_caller_identity.current.account_id
  region     = data.aws_region.current.name
  
  common_tags = {
    Project     = "ChickenWeightEstimation"
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

# ECR Repository
resource "aws_ecr_repository" "chicken_weight_estimator" {
  name                 = "chicken-weight-estimator"
  image_tag_mutability = "MUTABLE"
  
  image_scanning_configuration {
    scan_on_push = true
  }
  
  encryption_configuration {
    encryption_type = "AES256"
  }
  
  lifecycle_policy {
    policy = jsonencode({
      rules = [
        {
          rulePriority = 1
          description  = "Keep last 10 production images"
          selection = {
            tagStatus     = "tagged"
            tagPrefixList = ["v"]
            countType     = "imageCountMoreThan"
            countNumber   = 10
          }
          action = {
            type = "expire"
          }
        },
        {
          rulePriority = 2
          description  = "Keep last 5 development images"
          selection = {
            tagStatus     = "tagged"
            tagPrefixList = ["develop", "feature"]
            countType     = "imageCountMoreThan"
            countNumber   = 5
          }
          action = {
            type = "expire"
          }
        },
        {
          rulePriority = 3
          description  = "Delete untagged images older than 1 day"
          selection = {
            tagStatus   = "untagged"
            countType   = "sinceImagePushed"
            countUnit   = "days"
            countNumber = 1
          }
          action = {
            type = "expire"
          }
        }
      ]
    })
  }
}

# ECR Repository Policy
resource "aws_ecr_repository_policy" "chicken_weight_estimator_policy" {
  repository = aws_ecr_repository.chicken_weight_estimator.name
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowSageMakerAccess"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
      }
    ]
  })
}

# S3 Bucket for model artifacts
resource "aws_s3_bucket" "model_artifacts" {
  bucket = "chicken-weight-estimator-models-${local.account_id}-${var.environment}"
}

resource "aws_s3_bucket_versioning" "model_artifacts_versioning" {
  bucket = aws_s3_bucket.model_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "model_artifacts_encryption" {
  bucket = aws_s3_bucket.model_artifacts.id
  
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_s3_bucket_public_access_block" "model_artifacts_pab" {
  bucket = aws_s3_bucket.model_artifacts.id
  
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# S3 Bucket for data and logs
resource "aws_s3_bucket" "data_bucket" {
  bucket = "chicken-weight-estimator-data-${local.account_id}-${var.environment}"
}

resource "aws_s3_bucket_versioning" "data_bucket_versioning" {
  bucket = aws_s3_bucket.data_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "data_bucket_encryption" {
  bucket = aws_s3_bucket.data_bucket.id
  
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "data_bucket_lifecycle" {
  bucket = aws_s3_bucket.data_bucket.id
  
  rule {
    id     = "log_lifecycle"
    status = "Enabled"
    
    filter {
      prefix = "logs/"
    }
    
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
    
    transition {
      days          = 90
      storage_class = "GLACIER"
    }
    
    expiration {
      days = 365
    }
  }
  
  rule {
    id     = "temp_data_cleanup"
    status = "Enabled"
    
    filter {
      prefix = "temp/"
    }
    
    expiration {
      days = 7
    }
  }
}

# IAM Role for SageMaker
resource "aws_iam_role" "sagemaker_execution_role" {
  name = "ChickenWeightEstimator-SageMaker-${var.environment}"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "sagemaker_execution_policy" {
  name = "ChickenWeightEstimator-SageMaker-Policy-${var.environment}"
  role = aws_iam_role.sagemaker_execution_role.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.model_artifacts.arn,
          "${aws_s3_bucket.model_artifacts.arn}/*",
          aws_s3_bucket.data_bucket.arn,
          "${aws_s3_bucket.data_bucket.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:GetAuthorizationToken"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "cloudwatch:GetMetricStatistics",
          "cloudwatch:ListMetrics"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "arn:aws:logs:*:*:log-group:/aws/sagemaker/*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_execution_role_policy" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# VPC for SageMaker (optional, for enhanced security)
resource "aws_vpc" "sagemaker_vpc" {
  count = var.use_vpc ? 1 : 0
  
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = merge(local.common_tags, {
    Name = "ChickenWeightEstimator-VPC-${var.environment}"
  })
}

resource "aws_subnet" "sagemaker_subnet" {
  count = var.use_vpc ? 2 : 0
  
  vpc_id            = aws_vpc.sagemaker_vpc[0].id
  cidr_block        = "10.0.${count.index + 1}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  tags = merge(local.common_tags, {
    Name = "ChickenWeightEstimator-Subnet-${count.index + 1}-${var.environment}"
  })
}

data "aws_availability_zones" "available" {
  state = "available"
}

resource "aws_security_group" "sagemaker_sg" {
  count = var.use_vpc ? 1 : 0
  
  name_prefix = "ChickenWeightEstimator-SG-${var.environment}"
  vpc_id      = aws_vpc.sagemaker_vpc[0].id
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = merge(local.common_tags, {
    Name = "ChickenWeightEstimator-SecurityGroup-${var.environment}"
  })
}

# SNS Topic for alerts
resource "aws_sns_topic" "alerts" {
  name = "chicken-weight-estimator-alerts-${var.environment}"
  
  tags = local.common_tags
}

resource "aws_sns_topic_subscription" "email_alerts" {
  count = length(var.alert_email_addresses)
  
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email_addresses[count.index]
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "sagemaker_logs" {
  name              = "/aws/sagemaker/chicken-weight-estimator-${var.environment}"
  retention_in_days = var.log_retention_days
  
  tags = local.common_tags
}

# KMS Key for encryption
resource "aws_kms_key" "chicken_weight_estimator" {
  description             = "KMS key for Chicken Weight Estimator ${var.environment}"
  deletion_window_in_days = 7
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${local.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "Allow SageMaker Service"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
        Action = [
          "kms:Decrypt",
          "kms:GenerateDataKey"
        ]
        Resource = "*"
      }
    ]
  })
  
  tags = local.common_tags
}

resource "aws_kms_alias" "chicken_weight_estimator" {
  name          = "alias/chicken-weight-estimator-${var.environment}"
  target_key_id = aws_kms_key.chicken_weight_estimator.key_id
}

# Application Auto Scaling for SageMaker endpoint
resource "aws_appautoscaling_target" "sagemaker_target" {
  count = var.enable_auto_scaling ? 1 : 0
  
  max_capacity       = var.max_capacity
  min_capacity       = var.min_capacity
  resource_id        = "endpoint/${var.endpoint_name}/variant/AllTraffic"
  scalable_dimension = "sagemaker:variant:DesiredInstanceCount"
  service_namespace  = "sagemaker"
}

resource "aws_appautoscaling_policy" "sagemaker_scaling_policy" {
  count = var.enable_auto_scaling ? 1 : 0
  
  name               = "chicken-weight-estimator-scaling-policy-${var.environment}"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.sagemaker_target[0].resource_id
  scalable_dimension = aws_appautoscaling_target.sagemaker_target[0].scalable_dimension
  service_namespace  = aws_appautoscaling_target.sagemaker_target[0].service_namespace
  
  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "SageMakerVariantInvocationsPerInstance"
    }
    target_value = var.target_invocations_per_instance
  }
}

# CloudWatch Dashboard
resource "aws_cloudwatch_dashboard" "chicken_weight_estimator" {
  dashboard_name = "ChickenWeightEstimator-${var.environment}"
  
  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        
        properties = {
          metrics = [
            ["AWS/SageMaker", "Invocations", "EndpointName", var.endpoint_name],
            [".", "InvocationErrors", ".", "."],
            [".", "ModelLatency", ".", "."],
            [".", "OverheadLatency", ".", "."]
          ]
          view    = "timeSeries"
          stacked = false
          region  = local.region
          title   = "SageMaker Endpoint Metrics"
          period  = 300
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 12
        height = 6
        
        properties = {
          metrics = [
            ["ChickenWeightEstimation", "ProcessingTime"],
            [".", "DetectionCount"],
            [".", "AverageWeight"],
            [".", "InferenceErrors"]
          ]
          view    = "timeSeries"
          stacked = false
          region  = local.region
          title   = "Application Metrics"
          period  = 300
        }
      }
    ]
  })
}

# Outputs
output "ecr_repository_url" {
  description = "ECR repository URL"
  value       = aws_ecr_repository.chicken_weight_estimator.repository_url
}

output "model_artifacts_bucket" {
  description = "S3 bucket for model artifacts"
  value       = aws_s3_bucket.model_artifacts.bucket
}

output "data_bucket" {
  description = "S3 bucket for data and logs"
  value       = aws_s3_bucket.data_bucket.bucket
}

output "sagemaker_execution_role_arn" {
  description = "SageMaker execution role ARN"
  value       = aws_iam_role.sagemaker_execution_role.arn
}

output "sns_topic_arn" {
  description = "SNS topic ARN for alerts"
  value       = aws_sns_topic.alerts.arn
}

output "kms_key_id" {
  description = "KMS key ID for encryption"
  value       = aws_kms_key.chicken_weight_estimator.key_id
}

output "vpc_id" {
  description = "VPC ID (if created)"
  value       = var.use_vpc ? aws_vpc.sagemaker_vpc[0].id : null
}

output "subnet_ids" {
  description = "Subnet IDs (if VPC is used)"
  value       = var.use_vpc ? aws_subnet.sagemaker_subnet[*].id : null
}

output "security_group_id" {
  description = "Security group ID (if VPC is used)"
  value       = var.use_vpc ? aws_security_group.sagemaker_sg[0].id : null
}