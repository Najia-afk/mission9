# =============================================================================
# Mission9 - PySpark + PyTorch GPU Environment
# CUDA 12.8 + Python 3.12 + Spark 3.5 + PyTorch 2.5
# Optimized for RTX 5070 local training
# =============================================================================

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Set locale
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# =============================================================================
# System Dependencies
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    # Python 3.12 from deadsnakes PPA
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    # Java (required for Spark)
    openjdk-17-jdk-headless \
    # Build tools
    build-essential \
    curl \
    wget \
    git \
    # Image processing libraries
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set Python 3.12 as default and install pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# =============================================================================
# Java Environment (for Spark)
# =============================================================================
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# =============================================================================
# Spark Installation
# =============================================================================
ENV SPARK_VERSION=3.5.3
ENV HADOOP_VERSION=3
ENV SPARK_HOME=/opt/spark
ENV PATH="${SPARK_HOME}/bin:${SPARK_HOME}/sbin:${PATH}"
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

RUN curl -fsSL "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" \
    | tar -xz -C /opt/ \
    && mv /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} ${SPARK_HOME}

# =============================================================================
# Python Dependencies
# =============================================================================
WORKDIR /app

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# =============================================================================
# Environment Configuration
# =============================================================================
ENV PYTHONPATH=/app/src:${PYTHONPATH}
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

# Spark configuration for GPU workloads
ENV SPARK_DRIVER_MEMORY=8g
ENV SPARK_EXECUTOR_MEMORY=8g

# =============================================================================
# Expose Ports
# =============================================================================
# 8888: Jupyter Lab
# 4040: Spark UI
EXPOSE 8888 4040

# =============================================================================
# Default Command
# =============================================================================
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--ServerApp.disable_check_xsrf=True"]
