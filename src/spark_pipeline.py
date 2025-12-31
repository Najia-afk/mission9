"""
=============================================================================
Mission9 - PySpark Image Processing Pipeline
=============================================================================
Distributed image processing with PySpark for feature extraction and PCA.
Integrates with MLflow for experiment tracking.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType, BinaryType, FloatType, IntegerType,
    StringType, StructField, StructType
)
from pyspark.ml.feature import PCA as SparkPCA
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import StandardScaler

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from .feature_extractor import MobileNetV2Extractor


# =============================================================================
# Spark Image Pipeline
# =============================================================================
class SparkImagePipeline:
    """
    PySpark pipeline for distributed image feature extraction and PCA.
    
    Features:
    - Binary file reading for images
    - Distributed feature extraction with PyTorch MobileNetV2
    - PCA dimensionality reduction
    - Parquet output for results
    - MLflow tracking integration
    
    Example:
        >>> pipeline = SparkImagePipeline()
        >>> df = pipeline.load_images("dataset/fruits-360/Training")
        >>> df_features = pipeline.extract_features(df)
        >>> df_pca = pipeline.apply_pca(df_features, n_components=50)
        >>> pipeline.save_results(df_pca, "data/Results")
    """
    
    def __init__(
        self,
        app_name: str = "Mission9_PySpark",
        master: str = "local[*]",
        driver_memory: str = "8g",
        executor_memory: str = "8g",
        enable_arrow: bool = True,
        mlflow_experiment: Optional[str] = "mission9_experiments"
    ):
        """
        Initialize Spark session and pipeline components.
        
        Args:
            app_name: Spark application name
            master: Spark master URL
            driver_memory: Driver memory allocation
            executor_memory: Executor memory allocation
            enable_arrow: Enable Arrow optimization for Pandas UDFs
            mlflow_experiment: MLflow experiment name (None to disable)
        """
        self.spark = self._create_spark_session(
            app_name, master, driver_memory, executor_memory, enable_arrow
        )
        self.sc = self.spark.sparkContext
        
        # Feature extractor (lazy initialization)
        self._extractor: Optional[MobileNetV2Extractor] = None
        self._broadcast_weights = None
        
        # MLflow setup
        self.mlflow_experiment = mlflow_experiment
        if mlflow_experiment and MLFLOW_AVAILABLE:
            mlflow.set_experiment(mlflow_experiment)
        
        # Pipeline state
        self._run_id: Optional[str] = None
        self._pca_model = None
        
        print(f"SparkImagePipeline initialized")
        print(f"  Spark UI: http://localhost:4040")
        print(f"  Parallelism: {self.sc.defaultParallelism}")
    
    def _create_spark_session(
        self,
        app_name: str,
        master: str,
        driver_memory: str,
        executor_memory: str,
        enable_arrow: bool
    ) -> SparkSession:
        """Create and configure Spark session."""
        builder = SparkSession.builder \
            .appName(app_name) \
            .master(master) \
            .config("spark.driver.memory", driver_memory) \
            .config("spark.executor.memory", executor_memory) \
            .config("spark.sql.parquet.writeLegacyFormat", "true")
        
        if enable_arrow:
            builder = builder \
                .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
                .config("spark.sql.execution.arrow.maxRecordsPerBatch", "1000")
        
        return builder.getOrCreate()
    
    @property
    def extractor(self) -> MobileNetV2Extractor:
        """Lazy initialization of feature extractor."""
        if self._extractor is None:
            self._extractor = MobileNetV2Extractor()
        return self._extractor
    
    # =========================================================================
    # Data Loading
    # =========================================================================
    def load_images(
        self,
        path: str,
        recursive: bool = True,
        path_glob_filter: str = "*.jpg"
    ) -> DataFrame:
        """
        Load images as binary files from a directory.
        
        Args:
            path: Path to image directory
            recursive: Search recursively in subdirectories
            path_glob_filter: Glob pattern for file filtering
            
        Returns:
            DataFrame with columns: path, modificationTime, length, content
        """
        df = self.spark.read.format("binaryFile") \
            .option("pathGlobFilter", path_glob_filter) \
            .option("recursiveFileLookup", str(recursive).lower()) \
            .load(path)
        
        # Add label column (extract from path)
        df = df.withColumn(
            "label",
            F.element_at(F.split(F.col("path"), "/"), -2)
        )
        
        # Add filename column
        df = df.withColumn(
            "filename",
            F.element_at(F.split(F.col("path"), "/"), -1)
        )
        
        return df
    
    def load_dataset(
        self,
        base_path: str,
        splits: List[str] = ["Training", "Test"]
    ) -> Dict[str, DataFrame]:
        """
        Load multiple dataset splits.
        
        Args:
            base_path: Base path to dataset
            splits: List of split names (subdirectories)
            
        Returns:
            Dictionary mapping split names to DataFrames
        """
        datasets = {}
        for split in splits:
            split_path = os.path.join(base_path, split)
            if os.path.exists(split_path):
                datasets[split] = self.load_images(split_path)
                count = datasets[split].count()
                print(f"Loaded {split}: {count:,} images")
            else:
                print(f"Warning: Split not found: {split_path}")
        
        return datasets
    
    # =========================================================================
    # Feature Extraction
    # =========================================================================
    def extract_features(
        self,
        df: DataFrame,
        batch_size: int = 32,
        num_partitions: Optional[int] = None
    ) -> DataFrame:
        """
        Extract features from images using MobileNetV2.
        
        Uses Pandas UDF for efficient batch processing on GPU.
        
        Args:
            df: DataFrame with 'content' column containing binary image data
            batch_size: Batch size for feature extraction
            num_partitions: Number of partitions for repartitioning
            
        Returns:
            DataFrame with added 'features' column (1280-dim vectors)
        """
        from pyspark.sql.functions import pandas_udf
        from pyspark.sql.types import ArrayType, FloatType
        
        # Repartition if needed for better parallelism
        if num_partitions:
            df = df.repartition(num_partitions)
        
        # Create feature extraction UDF
        extractor = self.extractor  # Capture for closure
        
        @pandas_udf(ArrayType(FloatType()))
        def extract_features_udf(content_series: pd.Series) -> pd.Series:
            """Pandas UDF for batch feature extraction."""
            # Process batch
            images = [bytes(img) for img in content_series]
            features = extractor.extract_batch(images)
            return pd.Series([f.tolist() for f in features])
        
        # Apply feature extraction
        df_features = df.withColumn("features", extract_features_udf(F.col("content")))
        
        return df_features
    
    def extract_features_with_progress(
        self,
        df: DataFrame,
        checkpoint_path: Optional[str] = None
    ) -> DataFrame:
        """
        Extract features with progress tracking and optional checkpointing.
        
        Args:
            df: Input DataFrame
            checkpoint_path: Path to save intermediate results
            
        Returns:
            DataFrame with features
        """
        total_count = df.count()
        print(f"Extracting features from {total_count:,} images...")
        
        # Extract features
        df_features = self.extract_features(df)
        
        # Checkpoint if path provided
        if checkpoint_path:
            df_features.write.mode("overwrite").parquet(checkpoint_path)
            df_features = self.spark.read.parquet(checkpoint_path)
            print(f"Checkpointed to: {checkpoint_path}")
        
        return df_features
    
    # =========================================================================
    # PCA Dimensionality Reduction
    # =========================================================================
    def features_to_vector(self, df: DataFrame) -> DataFrame:
        """Convert features array to Spark ML Vector type."""
        to_vector = F.udf(lambda x: Vectors.dense(x), VectorUDT())
        return df.withColumn("features_vector", to_vector(F.col("features")))
    
    def apply_pca(
        self,
        df: DataFrame,
        n_components: int = 50,
        standardize: bool = True
    ) -> Tuple[DataFrame, 'SparkPCA']:
        """
        Apply PCA dimensionality reduction to features.
        
        Args:
            df: DataFrame with 'features' column
            n_components: Number of PCA components
            standardize: Whether to standardize features before PCA
            
        Returns:
            Tuple of (transformed DataFrame, fitted PCA model)
        """
        # Convert to vector format
        df_vector = self.features_to_vector(df)
        
        # Standardize if requested
        input_col = "features_vector"
        if standardize:
            scaler = StandardScaler(
                inputCol="features_vector",
                outputCol="features_scaled",
                withStd=True,
                withMean=True
            )
            scaler_model = scaler.fit(df_vector)
            df_vector = scaler_model.transform(df_vector)
            input_col = "features_scaled"
        
        # Fit PCA
        pca = SparkPCA(k=n_components, inputCol=input_col, outputCol="pca_features")
        pca_model = pca.fit(df_vector)
        
        # Transform
        df_pca = pca_model.transform(df_vector)
        
        # Store model for later use
        self._pca_model = pca_model
        
        # Log explained variance
        explained_variance = pca_model.explainedVariance.toArray()
        cumulative_variance = np.cumsum(explained_variance)
        print(f"PCA: {n_components} components explain {cumulative_variance[-1]*100:.1f}% of variance")
        
        return df_pca, pca_model
    
    # =========================================================================
    # Results Saving
    # =========================================================================
    def save_features(
        self,
        df: DataFrame,
        output_path: str,
        format: str = "parquet",
        partition_by: Optional[str] = "label"
    ):
        """
        Save features to storage.
        
        Args:
            df: DataFrame with features
            output_path: Output path
            format: Output format ('parquet', 'csv')
            partition_by: Column to partition by
        """
        # Select relevant columns
        df_save = df.select("path", "label", "filename", "features")
        
        writer = df_save.write.mode("overwrite")
        
        if partition_by:
            writer = writer.partitionBy(partition_by)
        
        if format == "parquet":
            writer.parquet(output_path)
        elif format == "csv":
            # Convert features to string for CSV
            df_csv = df_save.withColumn("features", F.col("features").cast("string"))
            writer = df_csv.write.mode("overwrite")
            writer.csv(output_path, header=True)
        
        print(f"Saved features to: {output_path}")
    
    def save_pca_results(
        self,
        df: DataFrame,
        output_path: str,
        include_features: bool = False
    ):
        """
        Save PCA results to storage.
        
        Args:
            df: DataFrame with PCA features
            output_path: Output path
            include_features: Whether to include original features
        """
        # Convert PCA vector to array
        vector_to_array = F.udf(lambda v: v.toArray().tolist(), ArrayType(FloatType()))
        df_save = df.withColumn("pca_array", vector_to_array(F.col("pca_features")))
        
        # Select columns
        columns = ["path", "label", "filename", "pca_array"]
        if include_features:
            columns.append("features")
        
        df_save.select(*columns).write.mode("overwrite").parquet(output_path)
        print(f"Saved PCA results to: {output_path}")
    
    def export_to_csv(
        self,
        df: DataFrame,
        output_path: str,
        feature_col: str = "pca_features"
    ):
        """
        Export results to CSV format for cloud storage.
        
        Args:
            df: DataFrame with features
            output_path: Output CSV path
            feature_col: Name of feature column
        """
        # Convert vector to individual columns
        vector_to_array = F.udf(lambda v: v.toArray().tolist(), ArrayType(FloatType()))
        df_csv = df.withColumn("features_array", vector_to_array(F.col(feature_col)))
        
        # Explode array to columns
        n_features = len(df_csv.first()["features_array"])
        for i in range(n_features):
            df_csv = df_csv.withColumn(f"f_{i}", F.col("features_array")[i])
        
        # Select columns
        feature_cols = [f"f_{i}" for i in range(n_features)]
        df_csv.select("label", "filename", *feature_cols) \
            .coalesce(1) \
            .write.mode("overwrite") \
            .csv(output_path, header=True)
        
        print(f"Exported to CSV: {output_path}")
    
    # =========================================================================
    # MLflow Integration
    # =========================================================================
    def start_run(self, run_name: Optional[str] = None) -> str:
        """Start an MLflow run."""
        if not MLFLOW_AVAILABLE:
            print("MLflow not available")
            return None
        
        run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow.start_run(run_name=run_name)
        self._run_id = mlflow.active_run().info.run_id
        
        # Log Spark config
        mlflow.log_param("spark_master", self.spark.sparkContext.master)
        mlflow.log_param("spark_parallelism", self.sc.defaultParallelism)
        
        # Log extractor config
        self.extractor.log_to_mlflow()
        
        return self._run_id
    
    def end_run(self):
        """End the current MLflow run."""
        if MLFLOW_AVAILABLE and mlflow.active_run():
            mlflow.end_run()
            self._run_id = None
    
    def log_dataset_info(self, df: DataFrame, name: str = "dataset"):
        """Log dataset information to MLflow."""
        if not MLFLOW_AVAILABLE:
            return
        
        count = df.count()
        n_labels = df.select("label").distinct().count()
        
        mlflow.log_metric(f"{name}_count", count)
        mlflow.log_metric(f"{name}_classes", n_labels)
    
    def log_pca_info(self, pca_model, n_components: int):
        """Log PCA information to MLflow."""
        if not MLFLOW_AVAILABLE:
            return
        
        explained_variance = pca_model.explainedVariance.toArray()
        cumulative_variance = np.cumsum(explained_variance)
        
        mlflow.log_param("pca_n_components", n_components)
        mlflow.log_metric("pca_explained_variance", cumulative_variance[-1])
        
        # Log variance per component
        for i, var in enumerate(explained_variance[:10]):  # First 10
            mlflow.log_metric(f"pca_var_component_{i}", var)
    
    # =========================================================================
    # Cleanup
    # =========================================================================
    def stop(self):
        """Stop Spark session and cleanup."""
        if self._run_id:
            self.end_run()
        self.spark.stop()
        print("Spark session stopped")


# =============================================================================
# Convenience Functions
# =============================================================================
def create_pipeline(**kwargs) -> SparkImagePipeline:
    """Create a new SparkImagePipeline with default settings."""
    return SparkImagePipeline(**kwargs)


def run_full_pipeline(
    data_path: str,
    output_path: str,
    n_components: int = 50,
    splits: List[str] = ["Training", "Test"],
    run_name: Optional[str] = None
) -> Dict:
    """
    Run the complete feature extraction and PCA pipeline.
    
    Args:
        data_path: Path to dataset
        output_path: Path for output files
        n_components: Number of PCA components
        splits: Dataset splits to process
        run_name: MLflow run name
        
    Returns:
        Dictionary with results metadata
    """
    pipeline = SparkImagePipeline()
    results = {}
    
    try:
        # Start MLflow run
        pipeline.start_run(run_name)
        
        # Load datasets
        datasets = pipeline.load_dataset(data_path, splits)
        
        for split_name, df in datasets.items():
            print(f"\n{'='*60}")
            print(f"Processing {split_name}")
            print(f"{'='*60}")
            
            # Log dataset info
            pipeline.log_dataset_info(df, split_name)
            
            # Extract features
            df_features = pipeline.extract_features(df)
            
            # Apply PCA
            df_pca, pca_model = pipeline.apply_pca(df_features, n_components)
            
            # Log PCA info
            if split_name == splits[0]:  # Only for first split
                pipeline.log_pca_info(pca_model, n_components)
            
            # Save results
            split_output = os.path.join(output_path, split_name)
            pipeline.save_pca_results(df_pca, f"{split_output}_pca")
            pipeline.export_to_csv(df_pca, f"{split_output}_pca.csv")
            
            results[split_name] = {
                "count": df.count(),
                "pca_output": f"{split_output}_pca",
                "csv_output": f"{split_output}_pca.csv"
            }
        
        return results
        
    finally:
        pipeline.end_run()
        pipeline.stop()


# =============================================================================
# CLI for Testing
# =============================================================================
if __name__ == "__main__":
    print("Testing SparkImagePipeline...")
    
    # Initialize pipeline
    pipeline = SparkImagePipeline(driver_memory="4g", executor_memory="4g")
    
    print(f"\nSpark version: {pipeline.spark.version}")
    print(f"Parallelism: {pipeline.sc.defaultParallelism}")
    
    # Test with small dummy data if available
    test_path = "dataset/fruits-360_dataset/fruits-360/Training"
    if os.path.exists(test_path):
        print(f"\nLoading test data from: {test_path}")
        df = pipeline.load_images(test_path)
        print(f"Loaded {df.count()} images")
        df.show(5, truncate=50)
    else:
        print(f"\nTest data not found at: {test_path}")
        print("Run scripts/subset_data.py to create a test subset")
    
    pipeline.stop()
    print("\n✓ Pipeline test complete!")
