# ============================================================
# interface_adapters/repositories/dataset_repo_fs.py 
# ============================================================
"""Enhanced filesystem repository with better error handling."""

from pathlib import Path
from typing import List, Set
import shutil
from domain.interfaces.repositories import ImageRepository
from domain.interfaces.logger import Logger
from domain.entities.image_item import ImageItem
from domain.entities.category import Category


class DatasetRepositoryFS:
    """Enhanced filesystem repository."""
    
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
    
    def __init__(self, base_path: Path, logger: Logger):
        self.base_path = Path(base_path)
        self.logger = logger
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save(self, image: ImageItem, category: Category) -> None:
        """Save image to category folder."""
        category_path = self.base_path / category.name
        category_path.mkdir(exist_ok=True)
        
        if not image.file_path.exists():
            raise FileNotFoundError(f"Image not found: {image.file_path}")
        
        # If image is not in correct location, move it
        target_path = category_path / image.file_path.name
        if image.file_path != target_path:
            shutil.copy2(image.file_path, target_path)
            self.logger.debug(f"Copied {image.file_path} to {target_path}")
    
    def load_all(self) -> List[ImageItem]:
        """Load all images with multiple format support."""
        images = []
        
        try:
            for category_dir in self.base_path.iterdir():
                if not category_dir.is_dir() or category_dir.name.startswith(('temp', '.')):
                    continue

                category = Category(category_dir.name)
                
                for img_path in category_dir.iterdir():
                    if img_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                        images.append(ImageItem(img_path, category))
            
            self.logger.info(f"Loaded {len(images)} images")
            return images
            
        except Exception as e:
            self.logger.error(f"Error loading images: {e}")
            raise
    
    def load_by_category(self, category: Category) -> List[ImageItem]:
        """Load images for specific category."""
        category_path = self.base_path / category.name
        
        if not category_path.exists():
            self.logger.warning(f"Category not found: {category.name}")
            return []
        
        images = []
        for img_path in category_path.iterdir():
            if img_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                images.append(ImageItem(img_path, category))
        
        return images
    
    def delete(self, image: ImageItem) -> None:
        """Remove image file."""
        try:
            if image.file_path.exists():
                image.file_path.unlink()
                self.logger.debug(f"Deleted: {image.file_path}")
        except Exception as e:
            self.logger.error(f"Failed to delete {image.file_path}: {e}")
            raise
    
    def list_categories(self) -> List[str]:
        """List all category folders."""
        categories = []
        for item in self.base_path.iterdir():
            if item.is_dir() and not item.name.startswith(('temp', '.')):
                categories.append(item.name)
        return sorted(categories)
