"""
=============================================================================
Mission9 - PyTorch Feature Extractor
=============================================================================
GPU-optimized feature extraction using MobileNetV2 for transfer learning.
Compatible with PySpark Pandas UDFs for distributed processing.
"""

import io
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


# =============================================================================
# Base Feature Extractor
# =============================================================================
class FeatureExtractor(ABC):
    """Abstract base class for feature extractors."""
    
    @abstractmethod
    def extract(self, image: Union[Image.Image, bytes, np.ndarray]) -> np.ndarray:
        """Extract features from a single image."""
        pass
    
    @abstractmethod
    def extract_batch(self, images: List[Union[Image.Image, bytes]]) -> np.ndarray:
        """Extract features from a batch of images."""
        pass
    
    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Return the dimension of extracted features."""
        pass


# =============================================================================
# MobileNetV2 Feature Extractor
# =============================================================================
class MobileNetV2Extractor(FeatureExtractor):
    """
    MobileNetV2-based feature extractor using PyTorch.
    
    Extracts features from the penultimate layer (1280-dim) for transfer learning.
    Optimized for GPU batch processing and compatible with PySpark Pandas UDFs.
    
    Attributes:
        device: torch device (cuda/cpu)
        model: MobileNetV2 model with classifier removed
        transform: Image preprocessing pipeline
        
    Example:
        >>> extractor = MobileNetV2Extractor(device='cuda')
        >>> features = extractor.extract(image)  # (1280,)
        >>> batch_features = extractor.extract_batch([img1, img2])  # (2, 1280)
    """
    
    _FEATURE_DIM = 1280  # MobileNetV2 penultimate layer dimension
    _INPUT_SIZE = (224, 224)  # Expected input size
    
    def __init__(
        self,
        device: Optional[str] = None,
        batch_size: int = 32,
        weights: str = "IMAGENET1K_V2"
    ):
        """
        Initialize MobileNetV2 feature extractor.
        
        Args:
            device: 'cuda', 'cpu', or None (auto-detect)
            batch_size: Batch size for batch processing
            weights: Pretrained weights to use
        """
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.batch_size = batch_size
        self._weights_name = weights
        
        # Load pretrained MobileNetV2
        self.model = self._build_model(weights)
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize(self._INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]    # ImageNet std
            )
        ])
        
        print(f"MobileNetV2Extractor initialized on {self.device}")
        if self.device.type == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def _build_model(self, weights: str) -> nn.Module:
        """Build MobileNetV2 model without final classifier."""
        # Load pretrained model
        if weights == "IMAGENET1K_V2":
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
        elif weights == "IMAGENET1K_V1":
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            model = models.mobilenet_v2(weights=None)
        
        # Remove classifier, keep feature extractor
        # MobileNetV2 structure: features -> avgpool -> classifier
        # We want output after avgpool (1280-dim)
        model.classifier = nn.Identity()
        
        return model
    
    @property
    def feature_dim(self) -> int:
        """Return feature dimension (1280 for MobileNetV2)."""
        return self._FEATURE_DIM
    
    def _preprocess_image(self, image: Union[Image.Image, bytes, np.ndarray]) -> torch.Tensor:
        """Preprocess a single image to tensor."""
        # Handle different input types
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Apply transforms
        return self.transform(image)
    
    def extract(self, image: Union[Image.Image, bytes, np.ndarray]) -> np.ndarray:
        """
        Extract features from a single image.
        
        Args:
            image: PIL Image, bytes, or numpy array
            
        Returns:
            numpy array of shape (1280,)
        """
        tensor = self._preprocess_image(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(tensor)
        
        return features.cpu().numpy().flatten()
    
    def extract_batch(
        self,
        images: List[Union[Image.Image, bytes, np.ndarray]]
    ) -> np.ndarray:
        """
        Extract features from a batch of images.
        
        Args:
            images: List of PIL Images, bytes, or numpy arrays
            
        Returns:
            numpy array of shape (n_images, 1280)
        """
        if not images:
            return np.empty((0, self._FEATURE_DIM))
        
        # Preprocess all images
        tensors = [self._preprocess_image(img) for img in images]
        batch = torch.stack(tensors).to(self.device)
        
        with torch.no_grad():
            features = self.model(batch)
        
        return features.cpu().numpy()
    
    def extract_batch_generator(
        self,
        images: List[Union[Image.Image, bytes, np.ndarray]]
    ) -> np.ndarray:
        """
        Extract features from images using batched processing.
        Memory-efficient for large datasets.
        
        Args:
            images: List of images to process
            
        Returns:
            numpy array of shape (n_images, 1280)
        """
        all_features = []
        
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            batch_features = self.extract_batch(batch_images)
            all_features.append(batch_features)
        
        return np.vstack(all_features) if all_features else np.empty((0, self._FEATURE_DIM))
    
    def get_model_info(self) -> dict:
        """Get model information for logging."""
        info = {
            "model_name": "MobileNetV2",
            "weights": self._weights_name,
            "feature_dim": self._FEATURE_DIM,
            "input_size": self._INPUT_SIZE,
            "device": str(self.device),
            "batch_size": self.batch_size,
        }
        
        if self.device.type == "cuda":
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        return info
    
    def log_to_mlflow(self):
        """Log model configuration to MLflow."""
        if not MLFLOW_AVAILABLE:
            print("MLflow not available for logging")
            return
        
        info = self.get_model_info()
        for key, value in info.items():
            mlflow.log_param(f"feature_extractor_{key}", value)


# =============================================================================
# Utility Functions for PySpark Integration
# =============================================================================
def create_extractor_broadcast(spark_context, device: str = "cuda") -> 'Broadcast':
    """
    Create a broadcast variable containing model weights for PySpark workers.
    
    This allows efficient distribution of the model to all workers
    without serializing the full model each time.
    
    Args:
        spark_context: PySpark SparkContext
        device: Device to use on workers
        
    Returns:
        Broadcast variable with model state dict
    """
    # Create model and get weights
    extractor = MobileNetV2Extractor(device="cpu")  # Load on CPU first
    weights = extractor.model.state_dict()
    
    # Broadcast weights to workers
    return spark_context.broadcast(weights)


def extract_features_udf_factory(broadcast_weights, device: str = "cuda"):
    """
    Factory function to create a Pandas UDF for feature extraction.
    
    Args:
        broadcast_weights: Broadcast variable with model weights
        device: Device to use for extraction
        
    Returns:
        Pandas UDF function for feature extraction
    """
    import pandas as pd
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import ArrayType, FloatType
    
    @pandas_udf(ArrayType(FloatType()))
    def extract_features(image_data: pd.Series) -> pd.Series:
        """Extract features from binary image data."""
        # Lazy initialization of model on worker
        if not hasattr(extract_features, "_extractor"):
            extractor = MobileNetV2Extractor(device=device)
            extractor.model.load_state_dict(broadcast_weights.value)
            extract_features._extractor = extractor
        
        extractor = extract_features._extractor
        
        # Process batch
        images = [bytes(img) for img in image_data]
        features = extractor.extract_batch(images)
        
        return pd.Series([f.tolist() for f in features])
    
    return extract_features


# =============================================================================
# CLI for Testing
# =============================================================================
if __name__ == "__main__":
    import sys
    
    print("Testing MobileNetV2Extractor...")
    
    # Initialize extractor
    extractor = MobileNetV2Extractor()
    print(f"\nModel info: {extractor.get_model_info()}")
    
    # Test with dummy image
    dummy_image = Image.new("RGB", (100, 100), color="red")
    features = extractor.extract(dummy_image)
    print(f"\nSingle image features shape: {features.shape}")
    print(f"Feature range: [{features.min():.4f}, {features.max():.4f}]")
    
    # Test batch extraction
    batch_images = [Image.new("RGB", (100, 100), color=c) for c in ["red", "green", "blue"]]
    batch_features = extractor.extract_batch(batch_images)
    print(f"\nBatch features shape: {batch_features.shape}")
    
    print("\n✓ Feature extractor test passed!")
