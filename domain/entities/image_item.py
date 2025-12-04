# domain/entities/image_item.py
"""
Domain entity representing an image and its associated category.
Pure data model with no framework or I/O logic.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageItem:
    """
    Simple immutable domain entity describing an image.
    Contains the location of the image file and its assigned category.
    """

    file_path: Path
    category: str

    def __post_init__(self):
        # Normalise file path
        object.__setattr__(self, "file_path", Path(self.file_path))
