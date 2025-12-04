# domain/interfaces/processors.py
"""
Interfaces for image processing operations.
Pure abstractions with no dependency on Pillow or filesystem code.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
from domain.entities.image_item import ImageItem


class ImageProcessor(ABC):
    """
    Interface for image transformations (resize, normalize, validation).
    Implementations (Pillow, OpenCV) must return ImageItem objects.
    """

    @abstractmethod
    def resize(self, image: ImageItem, size: tuple) -> ImageItem:
        """
        Resize the image to the given (width, height).
        Returns a new ImageItem describing the processed file.
        """
        pass

    @abstractmethod
    def normalize(self, image: ImageItem) -> ImageItem:
        """
        Normalize image format (e.g., convert to RGB).
        Returns a new ImageItem.
        """
        pass

    @abstractmethod
    def validate_and_convert(self, image_path: Path) -> bool:
        """
        Validate that the file is a readable image.
        Returns True if the image is valid and converted if needed.
        """
        pass

    @abstractmethod
    def batch_process(
        self,
        images: List[ImageItem],
        size: Optional[tuple] = None
    ) -> List[ImageItem]:
        """
        Perform processing on a batch of images.
        Returns the list of processed ImageItem objects.
        """
        pass
