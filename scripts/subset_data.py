#!/usr/bin/env python3
"""
=============================================================================
Mission9 - Dataset Subset Generator
=============================================================================
Generate stratified subsets of the Fruits-360 dataset for progressive scaling.
Supports: 1%, 5%, 10%, 25%, 50%, 100% of the dataset.

Usage:
    python scripts/subset_data.py --percentage 10 --source dataset/fruits-360_dataset/fruits-360
    python scripts/subset_data.py --percentage 5 --split Training
    python scripts/subset_data.py --list  # Show available percentages
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================
AVAILABLE_PERCENTAGES = [1, 5, 10, 25, 50, 100]
DEFAULT_SOURCE = "dataset/fruits-360_dataset/fruits-360"
DEFAULT_OUTPUT = "dataset/subsets"
SPLITS = ["Training", "Test"]
RANDOM_SEED = 42


# =============================================================================
# Subset Generator Class
# =============================================================================
class DatasetSubsetGenerator:
    """Generate stratified subsets of the Fruits-360 dataset."""
    
    def __init__(
        self,
        source_path: str,
        output_path: str = DEFAULT_OUTPUT,
        seed: int = RANDOM_SEED
    ):
        self.source_path = Path(source_path)
        self.output_path = Path(output_path)
        self.seed = seed
        random.seed(seed)
        
        # Validate source exists
        if not self.source_path.exists():
            raise FileNotFoundError(f"Source dataset not found: {self.source_path}")
    
    def get_class_images(self, split: str) -> Dict[str, List[Path]]:
        """Get all images grouped by class for a given split."""
        split_path = self.source_path / split
        if not split_path.exists():
            raise FileNotFoundError(f"Split not found: {split_path}")
        
        class_images = {}
        for class_dir in sorted(split_path.iterdir()):
            if class_dir.is_dir():
                images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                if images:
                    class_images[class_dir.name] = sorted(images)
        
        return class_images
    
    def generate_subset(
        self,
        percentage: int,
        splits: List[str] = None,
        copy_files: bool = True,
        log_to_mlflow: bool = True
    ) -> Dict:
        """
        Generate a stratified subset of the dataset.
        
        Args:
            percentage: Percentage of data to include (1, 5, 10, 25, 50, 100)
            splits: List of splits to process ['Training', 'Test']
            copy_files: If True, copy files to output directory
            log_to_mlflow: If True, log metadata to MLflow
            
        Returns:
            Dictionary with subset metadata
        """
        if percentage not in AVAILABLE_PERCENTAGES:
            raise ValueError(f"Percentage must be one of {AVAILABLE_PERCENTAGES}")
        
        splits = splits or SPLITS
        subset_name = f"subset_{percentage}pct"
        subset_path = self.output_path / subset_name
        
        metadata = {
            "subset_name": subset_name,
            "percentage": percentage,
            "seed": self.seed,
            "source_path": str(self.source_path),
            "output_path": str(subset_path),
            "created_at": datetime.now().isoformat(),
            "splits": {}
        }
        
        print(f"\n{'='*60}")
        print(f"Generating {percentage}% subset: {subset_name}")
        print(f"{'='*60}")
        
        for split in splits:
            print(f"\nProcessing {split} split...")
            class_images = self.get_class_images(split)
            
            split_metadata = {
                "total_classes": len(class_images),
                "original_images": 0,
                "subset_images": 0,
                "classes": {}
            }
            
            # Progress bar if available
            iterator = tqdm(class_images.items()) if TQDM_AVAILABLE else class_images.items()
            
            for class_name, images in iterator:
                original_count = len(images)
                split_metadata["original_images"] += original_count
                
                # Calculate subset size (minimum 1 image per class)
                subset_count = max(1, int(original_count * percentage / 100))
                
                # Stratified sampling
                selected_images = random.sample(images, min(subset_count, original_count))
                split_metadata["subset_images"] += len(selected_images)
                
                # Copy files if requested
                if copy_files:
                    class_output = subset_path / split / class_name
                    class_output.mkdir(parents=True, exist_ok=True)
                    
                    for img_path in selected_images:
                        dst = class_output / img_path.name
                        if not dst.exists():
                            shutil.copy2(img_path, dst)
                
                split_metadata["classes"][class_name] = {
                    "original": original_count,
                    "subset": len(selected_images)
                }
            
            metadata["splits"][split] = split_metadata
            print(f"  Classes: {split_metadata['total_classes']}")
            print(f"  Original images: {split_metadata['original_images']:,}")
            print(f"  Subset images: {split_metadata['subset_images']:,}")
            print(f"  Actual percentage: {split_metadata['subset_images']/split_metadata['original_images']*100:.1f}%")
        
        # Log to MLflow if available and requested
        if log_to_mlflow and MLFLOW_AVAILABLE:
            self._log_to_mlflow(metadata)
        
        # Save metadata to JSON
        if copy_files:
            import json
            metadata_file = subset_path / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"\nMetadata saved to: {metadata_file}")
        
        return metadata
    
    def _log_to_mlflow(self, metadata: Dict):
        """Log subset generation to MLflow."""
        try:
            mlflow.set_experiment("mission9_data_subsets")
            
            with mlflow.start_run(run_name=metadata["subset_name"]):
                # Log parameters
                mlflow.log_param("percentage", metadata["percentage"])
                mlflow.log_param("seed", metadata["seed"])
                mlflow.log_param("source_path", metadata["source_path"])
                
                # Log metrics for each split
                for split, split_data in metadata["splits"].items():
                    mlflow.log_metric(f"{split.lower()}_classes", split_data["total_classes"])
                    mlflow.log_metric(f"{split.lower()}_original_images", split_data["original_images"])
                    mlflow.log_metric(f"{split.lower()}_subset_images", split_data["subset_images"])
                    mlflow.log_metric(
                        f"{split.lower()}_actual_pct",
                        split_data["subset_images"] / split_data["original_images"] * 100
                    )
                
                # Log metadata as artifact
                import json
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(metadata, f, indent=2)
                    mlflow.log_artifact(f.name, "metadata")
                
            print(f"\nMLflow run logged: {metadata['subset_name']}")
            
        except Exception as e:
            print(f"\nWarning: Could not log to MLflow: {e}")
    
    def generate_all_subsets(self, splits: List[str] = None):
        """Generate all available subset percentages."""
        for pct in AVAILABLE_PERCENTAGES:
            if pct < 100:  # Skip 100% as it's just a copy
                self.generate_subset(pct, splits=splits)
    
    def list_subsets(self) -> List[Dict]:
        """List all existing subsets with their metadata."""
        subsets = []
        
        if not self.output_path.exists():
            return subsets
        
        for subset_dir in sorted(self.output_path.iterdir()):
            if subset_dir.is_dir() and subset_dir.name.startswith("subset_"):
                metadata_file = subset_dir / "metadata.json"
                if metadata_file.exists():
                    import json
                    with open(metadata_file) as f:
                        subsets.append(json.load(f))
                else:
                    # Count images manually
                    subsets.append({
                        "subset_name": subset_dir.name,
                        "path": str(subset_dir),
                        "metadata": "not found"
                    })
        
        return subsets


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Generate stratified subsets of the Fruits-360 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/subset_data.py --percentage 10
  python scripts/subset_data.py --percentage 5 --split Training
  python scripts/subset_data.py --all
  python scripts/subset_data.py --list
        """
    )
    
    parser.add_argument(
        "--percentage", "-p",
        type=int,
        choices=AVAILABLE_PERCENTAGES,
        help=f"Percentage of data to include: {AVAILABLE_PERCENTAGES}"
    )
    
    parser.add_argument(
        "--source", "-s",
        type=str,
        default=DEFAULT_SOURCE,
        help=f"Source dataset path (default: {DEFAULT_SOURCE})"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output directory for subsets (default: {DEFAULT_OUTPUT})"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        choices=SPLITS,
        help="Process only specific split (Training or Test)"
    )
    
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Generate all subset percentages (1%%, 5%%, 10%%, 25%%, 50%%)"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List existing subsets"
    )
    
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for reproducibility (default: {RANDOM_SEED})"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without copying files"
    )
    
    args = parser.parse_args()
    
    # Initialize generator
    try:
        generator = DatasetSubsetGenerator(
            source_path=args.source,
            output_path=args.output,
            seed=args.seed
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"\nMake sure the dataset is downloaded to: {args.source}")
        print("Download from: https://www.kaggle.com/datasets/moltean/fruits")
        sys.exit(1)
    
    # List existing subsets
    if args.list:
        subsets = generator.list_subsets()
        if subsets:
            print("\nExisting subsets:")
            for s in subsets:
                print(f"  - {s['subset_name']}: {s.get('percentage', '?')}%")
        else:
            print("\nNo subsets found.")
        return
    
    # Determine splits to process
    splits = [args.split] if args.split else SPLITS
    
    # Generate subsets
    if args.all:
        generator.generate_all_subsets(splits=splits)
    elif args.percentage:
        generator.generate_subset(
            percentage=args.percentage,
            splits=splits,
            copy_files=not args.dry_run,
            log_to_mlflow=not args.no_mlflow
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
