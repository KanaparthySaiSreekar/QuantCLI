# QuantCLI Production Infrastructure
# Multi-AZ, highly available, secure deployment

terraform {
  required_version = ">= 1.5"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
  }
  
  backend "s3" {
    bucket         = "quantcli-terraform-state"
    key            = "prod/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "quantcli-terraform-locks"
    kms_key_id     = "arn:aws:kms:us-east-1:ACCOUNT:key/KEY_ID"
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Environment = "production"
      Project     = "QuantCLI"
      ManagedBy   = "Terraform"
      CostCenter  = "Trading"
    }
  }
}

# VPC with private subnets (no public IPs)
module "vpc" {
  source = "../../modules/vpc"
  
  name               = "quantcli-prod-vpc"
  cidr               = "10.0.0.0/16"
  azs                = ["us-east-1a", "us-east-1b", "us-east-1c"]
  private_subnets    = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  database_subnets   = ["10.0.11.0/24", "10.0.12.0/24", "10.0.13.0/24"]
  
  enable_nat_gateway = true
  enable_vpn_gateway = true
  enable_dns_hostnames = true
  
  # VPC Flow Logs for security monitoring
  enable_flow_log = true
  flow_log_destination_type = "s3"
  flow_log_destination_arn = aws_s3_bucket.vpc_flow_logs.arn
  
  tags = {
    "kubernetes.io/cluster/${local.cluster_name}" = "shared"
  }
}

# EKS Cluster with node groups
module "eks" {
  source = "../../modules/eks"
  
  cluster_name    = local.cluster_name
  cluster_version = "1.28"
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  # Cluster endpoint access - private only
  cluster_endpoint_private_access = true
  cluster_endpoint_public_access  = false
  
  # Enable security features
  cluster_enabled_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
  
  # IRSA for pod-level IAM
  enable_irsa = true
  
  # Node groups
  eks_managed_node_groups = {
    # Trading engine nodes - high priority, low latency
    trading = {
      name = "trading-nodes"
      
      instance_types = ["c6i.2xlarge"]  # Compute optimized
      capacity_type  = "ON_DEMAND"
      
      min_size     = 3
      max_size     = 10
      desired_size = 3
      
      labels = {
        workload = "trading"
      }
      
      taints = [{
        key    = "trading"
        value  = "true"
        effect = "NoSchedule"
      }]
      
      # Enable IMDSv2
      metadata_options = {
        http_tokens = "required"
      }
    }
    
    # Model inference nodes - GPU enabled
    inference = {
      name = "inference-nodes"
      
      instance_types = ["g4dn.xlarge"]  # GPU instances
      capacity_type  = "SPOT"  # Cost optimization
      
      min_size     = 0
      max_size     = 5
      desired_size = 2
      
      labels = {
        workload = "inference"
      }
    }
    
    # Backtesting nodes - burstable
    backtesting = {
      name = "backtesting-nodes"
      
      instance_types = ["c6i.4xlarge"]
      capacity_type  = "SPOT"
      
      min_size     = 0
      max_size     = 20
      desired_size = 0  # Scale to zero
      
      labels = {
        workload = "backtesting"
      }
    }
  }
  
  # Fargate for serverless workloads
  fargate_profiles = {
    logging = {
      name = "logging-profile"
      selectors = [
        {
          namespace = "logging"
        }
      ]
    }
  }
}

# RDS Aurora PostgreSQL (TimescaleDB extension)
module "rds" {
  source = "../../modules/rds"
  
  identifier = "quantcli-prod-timescaledb"
  engine     = "aurora-postgresql"
  engine_version = "15.3"
  
  instance_class = "db.r6g.2xlarge"
  instances      = {
    1 = {}
    2 = {}
    3 = {}
  }
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.database_subnets
  
  # High availability
  availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]
  
  # Security
  storage_encrypted = true
  kms_key_id       = aws_kms_key.rds.arn
  
  # Performance
  performance_insights_enabled = true
  monitoring_interval         = 30
  
  # Backups
  backup_retention_period = 30
  preferred_backup_window = "03:00-04:00"
  
  # Enable IAM authentication
  iam_database_authentication_enabled = true
  
  # Enable deletion protection
  deletion_protection = true
  
  # Parameter group for TimescaleDB
  db_cluster_parameter_group_name = aws_rds_cluster_parameter_group.timescaledb.name
}

# ElastiCache Redis Cluster
module "redis" {
  source = "../../modules/redis"
  
  cluster_id           = "quantcli-prod-redis"
  engine_version       = "7.0"
  node_type            = "cache.r6g.xlarge"
  num_cache_nodes      = 3
  
  vpc_id               = module.vpc.vpc_id
  subnet_ids           = module.vpc.private_subnets
  
  # Multi-AZ with automatic failover
  automatic_failover_enabled = true
  multi_az_enabled          = true
  
  # Security
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token_enabled         = true
  
  # Performance
  parameter_group_name = aws_elasticache_parameter_group.redis.name
}

# MSK (Managed Kafka)
module "kafka" {
  source = "../../modules/kafka"
  
  cluster_name = "quantcli-prod-kafka"
  kafka_version = "3.5.1"
  
  number_of_broker_nodes = 3
  broker_instance_type   = "kafka.m5.xlarge"
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  # Storage
  storage_size = 1000  # GB per broker
  
  # Security
  encryption_in_transit_client_broker = "TLS"
  encryption_at_rest_kms_key_arn     = aws_kms_key.kafka.arn
  
  # Enable enhanced monitoring
  enhanced_monitoring = "PER_TOPIC_PER_BROKER"
  
  # Enable client authentication
  client_authentication = {
    sasl = {
      scram = true
    }
  }
}

# S3 Buckets for data storage
resource "aws_s3_bucket" "data_lake" {
  bucket = "quantcli-prod-datalake"
  
  tags = {
    Name = "QuantCLI Data Lake"
  }
}

resource "aws_s3_bucket_versioning" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.s3.arn
    }
  }
}

# S3 bucket for audit logs (immutable)
resource "aws_s3_bucket" "audit_logs" {
  bucket = "quantcli-prod-audit-logs"
}

resource "aws_s3_bucket_object_lock_configuration" "audit_logs" {
  bucket = aws_s3_bucket.audit_logs.id
  
  rule {
    default_retention {
      mode = "GOVERNANCE"
      years = 7  # Regulatory requirement
    }
  }
}

# KMS keys for encryption
resource "aws_kms_key" "rds" {
  description             = "KMS key for RDS encryption"
  deletion_window_in_days = 30
  enable_key_rotation     = true
}

resource "aws_kms_key" "kafka" {
  description             = "KMS key for Kafka encryption"
  deletion_window_in_days = 30
  enable_key_rotation     = true
}

resource "aws_kms_key" "s3" {
  description             = "KMS key for S3 encryption"
  deletion_window_in_days = 30
  enable_key_rotation     = true
}

# VPN for secure access
resource "aws_vpn_gateway" "main" {
  vpc_id = module.vpc.vpc_id
  
  tags = {
    Name = "quantcli-prod-vpn"
  }
}

# Bastion host for SSH access (in private subnet with Session Manager)
resource "aws_instance" "bastion" {
  ami           = data.aws_ami.amazon_linux_2.id
  instance_type = "t3.micro"
  subnet_id     = module.vpc.private_subnets[0]
  
  iam_instance_profile = aws_iam_instance_profile.bastion.name
  
  # No public IP
  associate_public_ip_address = false
  
  # Session Manager for SSH
  metadata_options {
    http_tokens = "required"  # IMDSv2
  }
  
  tags = {
    Name = "quantcli-bastion"
  }
}

# Secrets in AWS Secrets Manager
resource "aws_secretsmanager_secret" "api_keys" {
  name = "quantcli/prod/api-keys"
  
  kms_key_id = aws_kms_key.secrets.arn
  
  rotation_rules {
    automatically_after_days = 30
  }
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "trading_engine" {
  name              = "/quantcli/prod/trading-engine"
  retention_in_days = 90
  kms_key_id        = aws_kms_key.cloudwatch.arn
}

# CloudWatch Alarms
resource "aws_cloudwatch_metric_alarm" "high_latency" {
  alarm_name          = "quantcli-high-latency"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "Latency"
  namespace           = "QuantCLI"
  period              = "300"
  statistic           = "Average"
  threshold           = "500"  # 500ms
  alarm_description   = "Trading engine latency is too high"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

# SNS Topic for alerts
resource "aws_sns_topic" "alerts" {
  name = "quantcli-alerts"
  
  kms_master_key_id = aws_kms_key.sns.arn
}

# Outputs
output "eks_cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "rds_endpoint" {
  value = module.rds.cluster_endpoint
}

output "redis_endpoint" {
  value = module.redis.configuration_endpoint
}

output "kafka_bootstrap_brokers" {
  value = module.kafka.bootstrap_brokers_tls
}

locals {
  cluster_name = "quantcli-prod-eks"
}
