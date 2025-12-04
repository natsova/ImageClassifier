# ============================================================
# domain/services/model_service.py
# ============================================================
"""Domain service for model-related operations."""

from typing import List
from domain.entities.image_item import ImageItem
from domain.entities.category import Category
from domain.value_objects.training_metrics import TrainingMetrics
from domain.interfaces.model import Model


class ModelService:
    """Domain-level logic for managing ML models."""

    def __init__(self, model: Model):
        """
        Initialize with a model that implements Model interface.
        This allows any ML backend to be used (FastAI, PyTorch, etc.).
        """
        self.model = model

    def train_model(self, dls, epochs: int = 10) -> None:
        """Train the model on given DataLoaders or dataset."""
        self.model.build(dls)
        self.model.train(epochs)

    def save_model(self, path: str) -> None:
        """Save the trained model to disk."""
        self.model.save(path)

    def load_model(self, path: str) -> None:
        """Load a model from disk."""
        self.model.load(path)

    def predict_single(self, image: ImageItem) -> Category:
        """Predict the category of a single image."""
        return self.model.predict(image)

    def predict_batch(self, images: List[ImageItem]) -> List[Category]:
        """Predict multiple images at once."""
        return self.model.predict_batch(images)

    def evaluate(self, images: List[ImageItem], true_labels: List[Category]) -> TrainingMetrics:
        """
        Evaluate model predictions and compute metrics.
        Returns a domain value object TrainingMetrics.
        """
        predictions = self.predict_batch(images)
        total = len(predictions)
        correct = sum(p.name == t.name for p, t in zip(predictions, true_labels))

        accuracy = correct / total if total > 0 else 0.0
        error_rate = 1 - accuracy

        metrics = TrainingMetrics(metrics={
            "accuracy": accuracy,
            "error_rate": error_rate
        })

        return metrics
