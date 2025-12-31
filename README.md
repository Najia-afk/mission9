# Mission 9 - Big Data Image Processing with PySpark

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5-red.svg)](https://pytorch.org/)
[![PySpark 3.5](https://img.shields.io/badge/PySpark-3.5-orange.svg)](https://spark.apache.org/)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)

Distributed image feature extraction and PCA pipeline for the Fruits-360 dataset using PySpark and PyTorch GPU acceleration.

## 🎯 Project Overview

This project implements a scalable Big Data pipeline for image classification preprocessing:

- **Feature Extraction**: MobileNetV2 transfer learning (1280-dim features)
- **Dimensionality Reduction**: PCA compression to 50 components
- **Distributed Processing**: PySpark for horizontal scaling
- **GPU Acceleration**: PyTorch with CUDA 12.4 support
- **Experiment Tracking**: MLflow integration

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Local Development (Docker)                    │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────┐       │
│  │  Images  │───▶│  PySpark     │───▶│  PyTorch GPU    │       │
│  │  (S3/Local)   │  DataFrames  │    │  MobileNetV2    │       │
│  └──────────┘    └──────────────┘    └────────┬────────┘       │
│                                               │                 │
│                                               ▼                 │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────┐       │
│  │  Output  │◀───│  PCA         │◀───│  Features       │       │
│  │  (Parquet)    │  Reduction   │    │  (1280-dim)     │       │
│  └──────────┘    └──────────────┘    └─────────────────┘       │
├─────────────────────────────────────────────────────────────────┤
│  Jupyter: http://localhost:8889                                 │
│  Spark UI: http://localhost:4049                                │
│  MLflow: http://localhost:5009                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
mission9/
├── Dockerfile                 # CUDA 12.4 + Python 3.12 + Spark 3.5
├── docker-compose.yml         # Services: notebook, mlflow
├── requirements.txt           # Python dependencies
├── .gitignore
├── .dockerignore
├── README.md
│
├── notebooks/
│   └── mission9.ipynb         # Main processing notebook
│
├── src/
│   ├── __init__.py
│   ├── feature_extractor.py   # MobileNetV2 GPU extraction
│   └── spark_pipeline.py      # PySpark processing functions
│
├── scripts/
│   └── subset_data.py         # Dataset scaling utility
│
├── dataset/                   # Git-ignored, download separately
│   └── fruits-360_dataset/
│       └── fruits-360/
│           ├── Training/      # 67,692 images
│           └── Test/          # 22,688 images
│
├── data/                      # Output results
│   └── Results/
│
└── mlruns/                    # MLflow experiments
```

## 🚀 Quick Start

### Prerequisites

1. **Docker Desktop** with WSL2 backend (Windows) or native (Linux/Mac)
2. **NVIDIA Container Toolkit** for GPU support
3. **NVIDIA Driver** ≥ 525.60.13 (CUDA 12.x compatible)

#### Install NVIDIA Container Toolkit (Linux/WSL2)

```bash
# Add NVIDIA package repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mission9
   ```

2. **Download the dataset**
   
   Download from [Kaggle Fruits-360](https://www.kaggle.com/datasets/moltean/fruits):
   ```bash
   # Using Kaggle CLI
   kaggle datasets download -d moltean/fruits -p dataset/
   unzip dataset/fruits.zip -d dataset/fruits-360_dataset/
   ```

3. **Start the services**
   ```bash
   docker-compose up -d --build
   ```

4. **Access the interfaces**
   - **Jupyter Lab**: http://localhost:8889
   - **MLflow UI**: http://localhost:5009
   - **Spark UI**: http://localhost:4049 (when running)

### Running the Pipeline

1. Open Jupyter Lab at http://localhost:8889
2. Navigate to `notebooks/mission9.ipynb`
3. Run all cells to execute the full pipeline

## 📊 Dataset Scaling

The project includes a utility for progressive dataset scaling:

```bash
# Enter the container
docker-compose exec notebook bash

# Generate subsets for testing
python scripts/subset_data.py --percentage 1    # ~900 images
python scripts/subset_data.py --percentage 5    # ~4,500 images
python scripts/subset_data.py --percentage 10   # ~9,000 images
python scripts/subset_data.py --percentage 25   # ~22,500 images
python scripts/subset_data.py --percentage 50   # ~45,000 images

# Generate all subsets
python scripts/subset_data.py --all

# List existing subsets
python scripts/subset_data.py --list
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SPARK_DRIVER_MEMORY` | 8g | Spark driver memory |
| `SPARK_EXECUTOR_MEMORY` | 8g | Spark executor memory |
| `MLFLOW_TRACKING_URI` | http://mlflow:5000 | MLflow server URI |
| `NVIDIA_VISIBLE_DEVICES` | all | GPU devices to use |

### Port Mapping

| Service | Container Port | Host Port |
|---------|---------------|-----------|
| Jupyter Lab | 8888 | **8889** |
| Spark UI | 4040 | **4049** |
| MLflow | 5000 | **5009** |

## 📈 Pipeline Outputs

### Parquet Files
```
data/Results/
├── training_pca/           # Training features (partitioned by label)
│   ├── label=Apple Braeburn/
│   ├── label=Banana/
│   └── ...
└── test_pca/               # Test features (partitioned by label)
```

### CSV Files
```
data/Results/
├── training_pca_csv/       # Training features (single CSV)
└── test_pca_csv/           # Test features (single CSV)
```

## 🔬 Module Reference

### Feature Extractor

```python
from src.feature_extractor import MobileNetV2Extractor

extractor = MobileNetV2Extractor(device='cuda', batch_size=32)
features = extractor.extract(image)           # Single image: (1280,)
features = extractor.extract_batch(images)    # Batch: (N, 1280)
```

### Spark Pipeline

```python
from src.spark_pipeline import SparkImagePipeline

pipeline = SparkImagePipeline()
df = pipeline.load_images("path/to/images")
df_features = pipeline.extract_features(df)
df_pca, model = pipeline.apply_pca(df_features, n_components=50)
pipeline.save_pca_results(df_pca, "output/path")
```

## ☁️ Cloud Deployment

### AWS EMR

1. Upload images to S3 (EU region for RGPD)
2. Create EMR cluster with GPU instances
3. Submit PySpark job with S3 paths

```python
# S3 configuration
CONFIG = {
    'DATASET_PATH': 's3a://bucket/fruits-360/',
    'OUTPUT_PATH': 's3a://bucket/results/',
}
```

### Databricks

1. Create Databricks workspace (EU region)
2. Upload notebook to workspace
3. Create GPU cluster with appropriate runtime
4. Mount S3 bucket or upload data

## 📋 RGPD Compliance

This project is designed for RGPD compliance:

- ✅ Data stored on EU servers (configure S3 region: eu-west-1)
- ✅ Processing on EU-based cloud infrastructure
- ✅ No personal data in fruit images
- ✅ Encrypted storage (enable S3 encryption)
- ✅ VPC isolation for network security

## 🛠️ Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### Out of Memory

Reduce batch size or driver memory:
```python
CONFIG['BATCH_SIZE'] = 16
CONFIG['SPARK_DRIVER_MEMORY'] = '4g'
```

### Spark UI Not Loading

Wait for Spark session to start, then access http://localhost:4049

## 📝 License

This project is for educational purposes as part of the OpenClassrooms Data Engineer program.

## 🙏 Acknowledgments

- [Fruits-360 Dataset](https://www.kaggle.com/datasets/moltean/fruits) by Mihai Oltean
- [PyTorch](https://pytorch.org/) for deep learning
- [Apache Spark](https://spark.apache.org/) for distributed computing
- [MLflow](https://mlflow.org/) for experiment tracking
