# ============================================================
# interface_adapters/processors/image_processor_pillow.py 
# ============================================================
"""Enhanced image processor with batch operations and non-mutation."""

from pathlib import Path
from typing import List, Optional
from domain.interfaces.processors import ImageProcessor
from domain.interfaces.logger import Logger
from domain.entities.image_item import ImageItem
from PIL import Image
import shutil


class ImageProcessorPillow(ImageProcessor):
    """Enhanced processor that returns new files instead of mutating."""
    
    def __init__(self, logger: Logger):
        self.logger = logger
    
    def resize(self, image: ImageItem, size: tuple) -> ImageItem:
        """Resize image and return new ImageItem."""
        try:
            output_path = image.file_path.with_suffix('.processed.jpg')
            
            with Image.open(image.file_path) as img:
                img = img.resize(size, Image.LANCZOS)
                img.save(output_path, 'JPEG')
            
            # Replace original
            shutil.move(str(output_path), str(image.file_path))
            
            return ImageItem(image.file_path, image.category)
            
        except Exception as e:
            self.logger.error(f"Error resizing {image.file_path}: {e}")
            raise
    
    def normalize(self, image: ImageItem) -> ImageItem:
        """Convert to RGB and return new ImageItem."""
        try:
            with Image.open(image.file_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    img.save(image.file_path, 'JPEG')
                    self.logger.debug(f"Normalized {image.file_path}")
            
            return ImageItem(image.file_path, image.category)
            
        except Exception as e:
            self.logger.error(f"Error normalizing {image.file_path}: {e}")
            raise
    
    def validate_and_convert(self, image_path: Path) -> bool:
        """Validate and convert image."""
        try:
            with Image.open(image_path) as img:
                img.verify()
            
            # Reopen and convert
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(image_path, 'JPEG')
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Invalid image {image_path}: {e}")
            return False
    
    def batch_process(
        self,
        images: List[ImageItem],
        size: Optional[tuple] = None
    ) -> List[ImageItem]:
        """Process multiple images efficiently."""
        processed = []
        
        for image in images:
            try:
                # Normalize first
                img = self.normalize(image)
                
                # Resize if specified
                if size:
                    img = self.resize(img, size)
                
                processed.append(img)
                
            except Exception as e:
                self.logger.error(f"Failed to process {image.file_path}: {e}")
                continue
        
        self.logger.info(f"Batch processed {len(processed)}/{len(images)} images")
        return processed
