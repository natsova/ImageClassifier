# not sure about this
# domain/interfaces/repositories.py
"""
Repository interfaces for image storage and retrieval.
Defines stable abstractions for infrastructure adapters.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from domain.entities.image_item import ImageItem
from domain.entities.category import Category


class ImageRepository(ABC):
    """Interface for image persistence operations."""

    @abstractmethod
    def save(self, image: ImageItem, category: Category) -> None:
        """
        Persist an image under the given category.
        Implementations may copy, move, or write files.
        """
        pass

    @abstractmethod
    def load_all(self) -> List[ImageItem]:
        """
        Load all stored images from all categories.
        """
        pass

    @abstractmethod
    def load_by_category(self, category: Category) -> List[ImageItem]:
        """
        Load all images from the specified category.
        """
        pass

    @abstractmethod
    def delete(self, image: ImageItem) -> None:
        """
        Remove the given image from storage.
        """
        pass

    @abstractmethod
    def list_categories(self) -> List[str]:
        """
        Return the list of all categories present in storage.
        """
        pass
