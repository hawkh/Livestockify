variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (e.g., dev, staging, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "owner" {
  description = "Owner of the resources"
  type        = string
  default     = "ChickenWeightEstimationTeam"
}

variable "endpoint_name" {
  description = "SageMaker endpoint name"
  type        = string
  default     = "chicken-weight-estimator"
}

variable "use_vpc" {
  description = "Whether to create and use a VPC for SageMaker"
  type        = bool
  default     = false
}

variable "enable_auto_scaling" {
  description = "Whether to enable auto scaling for the SageMaker endpoint"
  type        = bool
  default     = false
}

variable "min_capacity" {
  description = "Minimum number of instances for auto scaling"
  type        = number
  default     = 1
  validation {
    condition     = var.min_capacity >= 1
    error_message = "Minimum capacity must be at least 1."
  }
}

variable "max_capacity" {
  description = "Maximum number of instances for auto scaling"
  type        = number
  default     = 10
  validation {
    condition     = var.max_capacity >= var.min_capacity
    error_message = "Maximum capacity must be greater than or equal to minimum capacity."
  }
}

variable "target_invocations_per_instance" {
  description = "Target number of invocations per instance for auto scaling"
  type        = number
  default     = 1000
}

variable "alert_email_addresses" {
  description = "List of email addresses to receive alerts"
  type        = list(string)
  default     = []
}

variable "log_retention_days" {
  description = "Number of days to retain CloudWatch logs"
  type        = number
  default     = 30
  validation {
    condition     = contains([1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1827, 3653], var.log_retention_days)
    error_message = "Log retention days must be a valid CloudWatch retention period."
  }
}

variable "instance_type" {
  description = "SageMaker instance type"
  type        = string
  default     = "ml.m5.large"
  validation {
    condition = can(regex("^ml\\.", var.instance_type))
    error_message = "Instance type must be a valid SageMaker instance type (starting with 'ml.')."
  }
}

variable "instance_count" {
  description = "Number of instances for the SageMaker endpoint"
  type        = number
  default     = 1
  validation {
    condition     = var.instance_count >= 1
    error_message = "Instance count must be at least 1."
  }
}

variable "model_data_url" {
  description = "S3 URL for model artifacts"
  type        = string
  default     = ""
}

variable "image_uri" {
  description = "ECR image URI for the container"
  type        = string
  default     = ""
}

variable "enable_data_capture" {
  description = "Whether to enable data capture for the endpoint"
  type        = bool
  default     = true
}

variable "data_capture_percentage" {
  description = "Percentage of requests to capture for data capture"
  type        = number
  default     = 100
  validation {
    condition     = var.data_capture_percentage >= 0 && var.data_capture_percentage <= 100
    error_message = "Data capture percentage must be between 0 and 100."
  }
}

variable "enable_model_monitoring" {
  description = "Whether to enable model monitoring"
  type        = bool
  default     = true
}

variable "monitoring_schedule_name" {
  description = "Name for the model monitoring schedule"
  type        = string
  default     = "chicken-weight-estimator-monitoring"
}

variable "baseline_job_name" {
  description = "Name for the baseline job"
  type        = string
  default     = "chicken-weight-estimator-baseline"
}

variable "enable_explainability" {
  description = "Whether to enable model explainability"
  type        = bool
  default     = false
}

variable "enable_bias_detection" {
  description = "Whether to enable bias detection"
  type        = bool
  default     = false
}

variable "tags" {
  description = "Additional tags to apply to resources"
  type        = map(string)
  default     = {}
}

# Advanced configuration variables
variable "volume_size_gb" {
  description = "Size of the EBS volume for SageMaker instances (in GB)"
  type        = number
  default     = 30
  validation {
    condition     = var.volume_size_gb >= 1 && var.volume_size_gb <= 16384
    error_message = "Volume size must be between 1 and 16384 GB."
  }
}

variable "max_concurrent_transforms" {
  description = "Maximum number of concurrent transforms for batch transform jobs"
  type        = number
  default     = 0
}

variable "max_payload_mb" {
  description = "Maximum payload size in MB for batch transform"
  type        = number
  default     = 6
}

variable "batch_strategy" {
  description = "Batch strategy for batch transform jobs"
  type        = string
  default     = "MultiRecord"
  validation {
    condition     = contains(["MultiRecord", "SingleRecord"], var.batch_strategy)
    error_message = "Batch strategy must be either 'MultiRecord' or 'SingleRecord'."
  }
}

variable "enable_network_isolation" {
  description = "Whether to enable network isolation for training and inference"
  type        = bool
  default     = false
}

variable "enable_inter_container_traffic_encryption" {
  description = "Whether to enable encryption for inter-container traffic"
  type        = bool
  default     = false
}

variable "kms_key_id" {
  description = "KMS key ID for encryption (if not provided, a new key will be created)"
  type        = string
  default     = ""
}

variable "subnets" {
  description = "List of subnet IDs for VPC configuration"
  type        = list(string)
  default     = []
}

variable "security_group_ids" {
  description = "List of security group IDs for VPC configuration"
  type        = list(string)
  default     = []
}

variable "enable_cloudwatch_metrics" {
  description = "Whether to enable detailed CloudWatch metrics"
  type        = bool
  default     = true
}

variable "cloudwatch_metrics_namespace" {
  description = "CloudWatch metrics namespace"
  type        = string
  default     = "ChickenWeightEstimation"
}

variable "enable_xray_tracing" {
  description = "Whether to enable AWS X-Ray tracing"
  type        = bool
  default     = false
}

variable "async_inference_config" {
  description = "Configuration for async inference"
  type = object({
    output_path                = string
    max_concurrent_invocations = optional(number, 4)
    client_config = optional(object({
      max_concurrent_invocations_per_instance = optional(number, 4)
    }), {})
    notification_config = optional(object({
      success_topic = optional(string)
      error_topic   = optional(string)
    }), {})
  })
  default = null
}

variable "serverless_config" {
  description = "Configuration for serverless inference"
  type = object({
    memory_size_mb      = number
    max_concurrency     = number
    provisioned_concurrency = optional(number, 0)
  })
  default = null
  validation {
    condition = var.serverless_config == null || (
      var.serverless_config.memory_size_mb >= 1024 && 
      var.serverless_config.memory_size_mb <= 6144 &&
      var.serverless_config.max_concurrency >= 1 &&
      var.serverless_config.max_concurrency <= 1000
    )
    error_message = "Serverless config: memory must be 1024-6144 MB, max_concurrency must be 1-1000."
  }
}

variable "multi_model_config" {
  description = "Configuration for multi-model endpoints"
  type = object({
    model_cache_setting = optional(string, "Enabled")
  })
  default = null
}

variable "shadow_production_variants" {
  description = "Configuration for shadow production variants"
  type = list(object({
    variant_name                    = string
    model_name                     = string
    initial_instance_count         = number
    instance_type                  = string
    initial_variant_weight         = optional(number, 0)
    accelerator_type              = optional(string)
    core_dump_config              = optional(object({
      destination_s3_uri = string
      kms_key_id        = optional(string)
    }))
    serverless_config = optional(object({
      memory_size_mb      = number
      max_concurrency     = number
      provisioned_concurrency = optional(number, 0)
    }))
  }))
  default = []
}

variable "data_capture_config" {
  description = "Advanced data capture configuration"
  type = object({
    enable_capture                = bool
    initial_sampling_percentage   = number
    destination_s3_uri           = string
    kms_key_id                   = optional(string)
    capture_options = list(object({
      capture_mode = string
    }))
    capture_content_type_header = optional(object({
      csv_content_types  = optional(list(string))
      json_content_types = optional(list(string))
    }))
  })
  default = null
}