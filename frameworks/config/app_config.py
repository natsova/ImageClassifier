# ============================================================
# frameworks/config/app_config.py 
# ============================================================
"""Enhanced configuration with validation and external loading."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import os
import yaml
import json


@dataclass
class AppConfig:
    """Application configuration with validation."""
    
    '''
    Python automatically generates an __init__:
        def __init__(self, dataset_path=Path("datasets"), etc):
            self.dataset_path = dataset_path
            etc

    - assigns values
    - does no validation
    - does no type conversion
    - does no side effects (like creating folders
    '''
    # Dataset settings
    dataset_path: Path = Path("datasets")
    categories: List[str] = field(default_factory=lambda: ["sky", "ocean", "umbrella", "dog", "book"])
    images_per_category: int = 200
    images_per_search: int = 50
    
    # Training settings
    batch_size: int = 32
    epochs: int = 10
    valid_pct: float = 0.2
    resize_size: int = 192
    seed: int = 42
    
    # Model settings
    model_path: Path = Path("models/classifier.pkl")
    architecture: str = "resnet18"
    metrics: List[str] = field(default_factory=lambda: ["error_rate"])
    
    # Download settings
    sleep_time: int = 2
    remove_duplicates: bool = True
    download_timeout: int = 30
    max_refill_rounds: int = 5
    max_retries: int = 3
    use_modifiers: bool = True
    
    # Performance settings
    num_workers: int = 4  # For parallel processing
    use_mixed_precision: bool = False
    
    def __post_init__(self):
        """Validate configuration."""

        '''
        __post_init__ is the official hook that runs after dataclass 
        initialisation, giving you a place to:

        - validate fields
        - convert input types
        - compute derived values
        - initialise directories
        - prevent invalid configurations

        It is the only safe way to add logic after the dataclass has done its field 
        assignment.
        '''
        # Convert String, etc paths to Path paths
        '''
        If someone passes a string:
            AppConfig(dataset_path="mydata")
        it becomes a Path("mydata").
        '''
        self.dataset_path = Path(self.dataset_path)
        self.model_path = Path(self.model_path)
        
        # Validation
        if not self.categories:
            raise ValueError("Categories cannot be empty")
        
        if self.images_per_category <= 0:
            raise ValueError("images_per_category must be positive")
        
        if not 0 < self.valid_pct < 1:
            raise ValueError("valid_pct must be between 0 and 1")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        
        # Create directory for the model file
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_yaml(cls, path: Path) -> 'AppConfig': # To call: config = AppConfig.from_yaml("config.yaml")
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @classmethod
    def from_json(cls, path: Path) -> 'AppConfig':
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
    
    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Load configuration from environment variables."""

        '''
        Loads configuration from environment variables such as:
        DATASET_PATH
        IMAGES_PER_CATEGORY
        EPOCHS
        etc.

        Fallbacks are provided if environment variables are missing.
        '''
        return cls(
            dataset_path=Path(os.getenv('DATASET_PATH', 'datasets')),
            categories=os.getenv('CATEGORIES', 'cat,dog,bird').split(','),
            images_per_category=int(os.getenv('IMAGES_PER_CATEGORY', '150')),
            epochs=int(os.getenv('EPOCHS', '10')),
            batch_size=int(os.getenv('BATCH_SIZE', '32')),
        )
    
    def to_yaml(self, path: Path):
        """Save configuration to YAML file."""
        data = {k: str(v) if isinstance(v, Path) else v 
                for k, v in self.__dict__.items()}
        with open(path, 'w') as f:
            yaml.dump(data, f)