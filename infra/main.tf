# =============================================================================
# MISSION 9 - AWS EMR INFRASTRUCTURE (RGPD Compliant - EU Region)
# =============================================================================
# Usage:
#   1. Copier .env.example vers .env et remplir les credentials
#   2. docker compose --profile deploy run --rm terraform init
#   3. docker compose --profile deploy run --rm terraform apply -auto-approve
#   4. docker compose --profile deploy run --rm terraform destroy -auto-approve
# =============================================================================

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }
}

# RGPD: EU Region
provider "aws" {
  region = "eu-west-1"  # Ireland - ou eu-west-3 pour Paris
}

variable "project_name" {
  default = "mission9"
}

# =============================================================================
# EC2 KEY PAIR - Créée automatiquement
# =============================================================================
resource "tls_private_key" "emr" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "aws_key_pair" "emr" {
  key_name   = "${var.project_name}-key"
  public_key = tls_private_key.emr.public_key_openssh
}

# Sauvegarde la clé privée localement
resource "local_file" "private_key" {
  content         = tls_private_key.emr.private_key_pem
  filename        = "/keys/${var.project_name}-key.pem"
  file_permission = "0400"
}

# =============================================================================
# S3 BUCKET - Stockage des données et résultats
# =============================================================================
resource "aws_s3_bucket" "data" {
  bucket_prefix = "${var.project_name}-data-"
  force_destroy = true  # Permet de détruire même si non vide

  tags = {
    Project = var.project_name
    RGPD    = "EU-compliant"
  }
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket = aws_s3_bucket.data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Bootstrap script: install packages + download dataset
resource "aws_s3_object" "bootstrap" {
  bucket  = aws_s3_bucket.data.id
  key     = "scripts/bootstrap-emr.sh"
  content = <<-EOF
    #!/bin/bash
    set -e
    
    # Install Python packages on ALL nodes (including urllib3 fix for OpenSSL 1.0.2k)
    sudo pip3 install pandas pillow tensorflow pyarrow 'urllib3<2.0' 'requests<2.32' boto3
    
    # Install Java and configure PySpark in JupyterHub container (master only)
    if grep -q "isMaster.*true" /mnt/var/lib/info/instance.json 2>/dev/null; then
      echo "Master node - installing Java in JupyterHub container..."
      
      # Wait for JupyterHub container to be running
      sleep 60
      
      # Get JupyterHub container ID
      CONTAINER_ID=$(sudo docker ps --filter "ancestor=*jupyterhub*" -q 2>/dev/null | head -1)
      if [ -z "$CONTAINER_ID" ]; then
        CONTAINER_ID=$(sudo docker ps --format '{{.Names}}' | grep -i jupyter | head -1)
      fi
      
      # Install Java and Python packages in the JupyterHub container
      if [ -n "$CONTAINER_ID" ]; then
        echo "Found JupyterHub container: $CONTAINER_ID"
        sudo docker exec $CONTAINER_ID bash -c "apt-get update && apt-get install -y default-jdk" || true
        sudo docker exec $CONTAINER_ID bash -c "echo 'export JAVA_HOME=/usr/lib/jvm/default-java' >> /etc/profile"
        sudo docker exec $CONTAINER_ID bash -c "echo 'export SPARK_HOME=/usr/lib/spark' >> /etc/profile"
        # Fix urllib3/OpenSSL compatibility + install boto3
        sudo docker exec $CONTAINER_ID bash -c "pip install 'urllib3<2.0' 'requests<2.32' boto3" || true
        echo "Java and Python packages installed in JupyterHub container!"
      else
        echo "JupyterHub container not found, trying alternative..."
        # Install in conda environment directly
        sudo /emr/notebook-env/bin/pip install findspark || true
      fi
      
      # Download FULL dataset
      echo "Downloading FULL dataset..."
      cd /tmp
      curl -L -o fruits.zip "https://s3.eu-west-1.amazonaws.com/course.oc-static.com/projects/Data_Scientist_P8/fruits.zip"
      unzip -q fruits.zip
      
      echo "Contents of /tmp after unzip:"
      ls -la /tmp/
      
      # Upload the FULL dataset (both folders)
      echo "Uploading fruits-360_dataset to S3..."
      if [ -d "/tmp/fruits-360_dataset" ]; then
        aws s3 sync "/tmp/fruits-360_dataset" s3://${aws_s3_bucket.data.id}/fruits-360_dataset/ --quiet
        echo "fruits-360_dataset uploaded!"
      fi
      
      echo "Uploading fruits-360-original-size to S3..."
      if [ -d "/tmp/fruits-360-original-size" ]; then
        aws s3 sync "/tmp/fruits-360-original-size" s3://${aws_s3_bucket.data.id}/fruits-360-original-size/ --quiet
        echo "fruits-360-original-size uploaded!"
      fi
      
      # Count total files uploaded
      TOTAL=$(find /tmp/fruits-360* -type f 2>/dev/null | wc -l)
      echo "Total files uploaded: $TOTAL"
      
      rm -rf fruits.zip /tmp/fruits*
      echo "Dataset upload complete!"
    fi
  EOF
}

# =============================================================================
# IAM ROLES - Permissions pour EMR
# =============================================================================
resource "aws_iam_role" "emr_service" {
  name = "${var.project_name}-emr-service-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "elasticmapreduce.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "emr_service" {
  role       = aws_iam_role.emr_service.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonElasticMapReduceRole"
}

resource "aws_iam_role" "emr_ec2" {
  name = "${var.project_name}-emr-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "emr_ec2" {
  role       = aws_iam_role.emr_ec2.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonElasticMapReduceforEC2Role"
}

resource "aws_iam_instance_profile" "emr_ec2" {
  name = "${var.project_name}-emr-ec2-profile"
  role = aws_iam_role.emr_ec2.name
}

# =============================================================================
# VPC & SECURITY GROUPS - Secure network configuration
# =============================================================================

# Get default VPC
data "aws_vpc" "default" {
  default = true
}

# Get subnet in default VPC
data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# Security Group for EMR Master - Only JupyterHub (9443) public
resource "aws_security_group" "emr_master" {
  name        = "${var.project_name}-emr-master-v2"
  description = "Security group for EMR master node"
  vpc_id      = data.aws_vpc.default.id

  # JupyterHub - Port 9443 ONLY public port allowed by EMR
  ingress {
    description = "JupyterHub HTTPS"
    from_port   = 9443
    to_port     = 9443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # All traffic from slaves (internal)
  ingress {
    description = "All traffic from slave nodes"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    self        = true
  }

  # All outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "${var.project_name}-emr-master-v2"
    Project = var.project_name
  }
}

# Security Group for EMR Slaves - Only internal traffic
resource "aws_security_group" "emr_slave" {
  name        = "${var.project_name}-emr-slave-v2"
  description = "Security group for EMR slave nodes"
  vpc_id      = data.aws_vpc.default.id

  # All traffic from master and other slaves
  ingress {
    description     = "All traffic from master"
    from_port       = 0
    to_port         = 0
    protocol        = "-1"
    security_groups = [aws_security_group.emr_master.id]
  }

  ingress {
    description = "All traffic from other slaves"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    self        = true
  }

  # All outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "${var.project_name}-emr-slave-v2"
    Project = var.project_name
  }
}

# Allow master to talk to slaves
resource "aws_security_group_rule" "master_to_slave" {
  type                     = "ingress"
  from_port                = 0
  to_port                  = 0
  protocol                 = "-1"
  source_security_group_id = aws_security_group.emr_slave.id
  security_group_id        = aws_security_group.emr_master.id
}

# =============================================================================
# EMR CLUSTER - Spark + JupyterHub
# =============================================================================
resource "aws_emr_cluster" "spark" {
  name          = "${var.project_name}-cluster"
  release_label = "emr-6.15.0"
  applications  = ["Spark", "JupyterHub", "Hadoop"]

  service_role = aws_iam_role.emr_service.arn

  ec2_attributes {
    instance_profile                  = aws_iam_instance_profile.emr_ec2.arn
    key_name                          = aws_key_pair.emr.key_name
    subnet_id                         = data.aws_subnets.default.ids[0]
    emr_managed_master_security_group = aws_security_group.emr_master.id
    emr_managed_slave_security_group  = aws_security_group.emr_slave.id
  }

  # Master node
  master_instance_group {
    instance_type  = "m5.xlarge"
    instance_count = 1
    name           = "Master"
  }

  # Worker nodes
  core_instance_group {
    instance_type  = "m5.xlarge"
    instance_count = 2
    name           = "Core"
  }

  # Bootstrap: installe tensorflow, pandas, pillow
  bootstrap_action {
    name = "Install Python packages"
    path = "s3://${aws_s3_bucket.data.id}/scripts/bootstrap-emr.sh"
  }

  # Auto-terminate après 1h d'inactivité (économies!)
  auto_termination_policy {
    idle_timeout = 3600
  }

  # Logs vers S3
  log_uri = "s3://${aws_s3_bucket.data.id}/logs/"

  configurations_json = jsonencode([
    {
      Classification = "spark-defaults"
      Properties = {
        "spark.sql.parquet.writeLegacyFormat" = "true"
      }
    }
  ])

  tags = {
    Project = var.project_name
    RGPD    = "EU-compliant"
  }

  depends_on = [aws_s3_object.bootstrap]
}

# =============================================================================
# OUTPUTS - Infos de connexion
# =============================================================================
output "s3_bucket" {
  value       = aws_s3_bucket.data.id
  description = "S3 bucket pour les données"
}

output "emr_master_dns" {
  value       = aws_emr_cluster.spark.master_public_dns
  description = "DNS du master EMR (pour SSH tunnel)"
}

output "emr_cluster_id" {
  value       = aws_emr_cluster.spark.id
  description = "ID du cluster EMR"
}

output "jupyterhub_url" {
  value       = "https://${aws_emr_cluster.spark.master_public_dns}:9443"
  description = "URL JupyterHub (après SSH tunnel)"
}

output "ssh_command" {
  value       = "ssh -i keys/mission9-key.pem hadoop@${aws_emr_cluster.spark.master_public_dns}"
  description = "SSH to master node"
}

# =============================================================================
# STATUS JSON - Pour le dashboard
# =============================================================================
resource "local_file" "status_json" {
  filename = "${path.module}/status.json"
  content = jsonencode({
    timestamp       = timestamp()
    region          = "eu-west-1"
    project         = var.project_name
    s3_bucket       = aws_s3_bucket.data.id
    s3_arn          = aws_s3_bucket.data.arn
    emr_cluster_id  = aws_emr_cluster.spark.id
    emr_state       = "STARTING"
    emr_master_dns  = aws_emr_cluster.spark.master_public_dns
    jupyterhub_url  = "https://${aws_emr_cluster.spark.master_public_dns}:9443"
    ssh_command     = "ssh -i keys/mission9-key.pem hadoop@${aws_emr_cluster.spark.master_public_dns}"
  })
}
