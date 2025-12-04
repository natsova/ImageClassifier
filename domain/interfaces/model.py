# ============================================================
# domain/interfaces/model.py
# ============================================================
"""Interface for machine learning models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from domain.entities.image_item import ImageItem
from domain.entities.category import Category


class Model(ABC):
    """Interface for ML model adapters."""

    @abstractmethod
    def build(self, dls) -> None:
        """
        Build the model using a dataset or DataLoaders.
        Backend-specific logic (e.g., FastAI Learner) is implemented in the adapter.
        """
        pass

    @abstractmethod
    def train(self, epochs: int) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save the trained model to disk."""
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load a trained model from disk."""
        pass

    @abstractmethod
    def predict(self, image: ImageItem) -> Category:
        """Predict the category of a single image."""
        pass

    @abstractmethod
    def predict_batch(self, images: List[ImageItem]) -> List[Category]:
        """Predict multiple images at once."""
        pass
