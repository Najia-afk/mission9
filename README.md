# Mission 9 - Big Data Cloud with AWS EMR

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![PySpark](https://img.shields.io/badge/PySpark-3.4-orange.svg)](https://spark.apache.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![AWS EMR](https://img.shields.io/badge/AWS_EMR-6.15-yellow.svg)](https://aws.amazon.com/emr/)
[![Terraform](https://img.shields.io/badge/Terraform-1.5-purple.svg)](https://www.terraform.io/)
[![Docker](https://img.shields.io/badge/Docker-24.0+-blue.svg)](https://www.docker.com/)

Distributed image feature extraction pipeline using PySpark and TensorFlow on AWS EMR.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              LOCAL MACHINE                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  Docker Compose                                                          │    │
│  │  ├── terraform (v1.6)     ─── terraform apply ───┐                      │    │
│  │  └── aws-cli (v2)         ─── aws emr/s3 ────────┼──────────────────┐   │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │ AWS API
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     AWS CLOUD (eu-west-1 - GDPR COMPLIANT)                       │
│                                                                                  │
│  ┌────────────────────┐    ┌────────────────────────────────────────────────┐   │
│  │  📦 S3 Bucket       │    │  ⚡ EMR Cluster (emr-6.15.0)                   │   │
│  │  mission9-data-*   │    │                                                │   │
│  │                    │◄───│  ┌──────────────────────────────────────────┐  │   │
│  │  ├── fruits-360/   │    │  │  Master (m5.xlarge)                      │  │   │
│  │  │   ├── Training/ │    │  │  ├── JupyterHub :9443 (HTTPS)            │  │   │
│  │  │   └── Test/     │    │  │  └── Spark Driver                        │  │   │
│  │  ├── Results/      │    │  └──────────────────────────────────────────┘  │   │
│  │  ├── Results_PCA/  │    │  ┌──────────────────┐ ┌──────────────────┐     │   │
│  │  └── Results_CSV/  │    │  │  Worker 1        │ │  Worker 2        │     │   │
│  └────────────────────┘    │  │  Spark Executor  │ │  Spark Executor  │     │   │
│                            │  │  TensorFlow      │ │  TensorFlow      │     │   │
│  ┌────────────────────┐    │  └──────────────────┘ └──────────────────┘     │   │
│  │  🔐 IAM Roles       │    └────────────────────────────────────────────────┘   │
│  │  🛡️ Security Groups│                                                         │
│  │    (port 9443 only)│                                                         │
│  └────────────────────┘                                                         │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Configure AWS Credentials

```bash
cp .env.example .env
# Edit .env with your AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
```

### 2. Deploy to AWS (One Command)

```bash
# Initialize and deploy infrastructure
docker compose --profile deploy run --rm terraform init
docker compose --profile deploy run --rm terraform apply -auto-approve
```

**The bootstrap script automatically:**
- ✅ Installs TensorFlow, Pandas, Pillow on all nodes
- ✅ Installs Java in JupyterHub container (for PySpark)
- ✅ Downloads & uploads Fruits-360 dataset (~90K images) to S3

### 3. Access JupyterHub

After ~15 minutes, get the URL:
```bash
docker compose --profile deploy run --rm terraform output jupyterhub_url
# → https://ec2-xx-xx-xx-xx.eu-west-1.compute.amazonaws.com:9443
```

**Credentials:** `jovyan` / `jupyter`

### 4. Run the Notebook

1. Upload `notebooks/mission9_emr.ipynb` to JupyterHub
2. Click **Run All** cells
3. Wait for processing (~10-20 min for full dataset)

### 5. Cleanup (Important!)

```bash
# Destroy all AWS resources to avoid charges
docker compose --profile deploy run --rm terraform destroy -auto-approve
```

## 📊 Pipeline Overview

```
┌─────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────┐    ┌──────────┐
│ S3      │───▶│ Spark Read  │───▶│ MobileNetV2 │───▶│ PCA │───▶│ S3       │
│ Images  │    │ binaryFile  │    │ 1280 feat   │    │ 50  │    │ Parquet  │
│ 90K     │    │ DataFrame   │    │ broadcast   │    │ dim │    │ + CSV    │
└─────────┘    └─────────────┘    └─────────────┘    └─────┘    └──────────┘
```

## 📁 Project Structure

```
mission9/
├── 📓 notebooks/
│   └── mission9_emr.ipynb      # Main PySpark notebook (run on EMR)
├── 🏗️ infra/
│   └── main.tf                  # Terraform infrastructure (S3, EMR, IAM)
├── 🎨 presentation/
│   ├── generate_ppt.py          # PowerPoint generator (AWS theme)
│   └── generate_diagrams.py     # Architecture diagrams
├── 📊 reports/
│   └── skill_grid.md            # Competencies validation (9/9 ✅)
├── 🐳 docker-compose.yml        # Local dev + deploy services
├── 📦 Dockerfile                # Python + Spark + TensorFlow
└── 📋 requirements.txt          # Dependencies
```

## 🛡️ GDPR Compliance

| Requirement | Implementation |
|-------------|----------------|
| **Region** | `eu-west-1` (Ireland) - EU territory |
| **Data Residency** | All data stored & processed in EU |
| **S3 Security** | Public access blocked |
| **Network** | Only port 9443 (JupyterHub) exposed |

## 📜 License & Credits

### Dataset
**Fruits-360** dataset by Horea Muresan & Mihai Oltean  
Babeș-Bolyai University, Cluj-Napoca, Romania

> Horea Muresan, Mihai Oltean, *Fruit recognition from images using deep learning*,  
> Acta Univ. Sapientiae, Informatica Vol. 10, Issue 1, pp. 26-42, 2018.

- 📄 Paper: [DOI: 10.2478/ausi-2018-0002](https://doi.org/10.2478/ausi-2018-0002)
- 📦 Dataset: [Kaggle - Fruits-360](https://www.kaggle.com/datasets/moltean/fruits)
- 🔗 GitHub: [Horea94/Fruit-Images-Dataset](https://github.com/Horea94/Fruit-Images-Dataset)

### Pre-trained Model
**MobileNetV2** - Google Inc. (Apache 2.0 License)
