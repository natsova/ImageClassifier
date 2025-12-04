# ============================================================
# use_cases/training/train_model.py
# ============================================================
"""Use case for training a machine learning model."""

from pathlib import Path
from typing import List
from domain.entities.image_item import ImageItem
from domain.entities.category import Category
from domain.services.model_service import ModelService
from domain.services.dataset_service import DatasetService
from domain.interfaces.repositories import ImageRepository
from domain.interfaces.model import Model
from domain.interfaces.logger import Logger


class TrainModelUseCase:
    """
    Use case to train a model:
    - Load images from repository
    - Build DataLoaders (or pass dataset to model)
    - Train the model
    - Save trained model
    """

    def __init__(
        self,
        model_service: ModelService,
        dataloader_factory,
        model_adapter_factory,
        logger: Logger
    ):
        self.model_service = model_service
        self.dataloader_factory = dataloader_factory
        self.model_adapter_factory = model_adapter_factory
        self.logger = logger

    def execute(
        self,
        epochs: int = 10,
        model_save_path: Path = Path("models/classifier.pkl")
    ) -> dict:
        """
        Execute the training use case.

        Returns a summary dictionary:
        {
            "total_images": int,
            "categories": List[str],
            "metrics": TrainingMetrics
        }
        """
        # Step 1: Load all images from repository
        images = self.repository.load_all()
        total_images = len(images)
        categories = self.repository.list_categories()
        self.logger.info(f"Loaded {total_images} images across {len(categories)} categories.")

        if total_images == 0:
            raise ValueError("No images found for training.")

        # Step 2: Build dataset / dataloaders
        # Note: For FastAI, you'd normally convert to a DataLoaders object here.
        # In this use case, we pass images directly to the model adapter, keeping it framework-agnostic.
        dls = self._prepare_dataloaders(images, categories)

        # Step 3: Train the model
        self.model_service.train_model(dls, epochs=epochs)

        # Step 4: Save the trained model
        self.model_service.save_model(model_save_path)
        self.logger.info(f"Model saved at {model_save_path}")

        # Step 5: Evaluate on training data (optional)
        metrics = self.model_service.evaluate(images, [Category(img.category) for img in images])

        return {
            "total_images": total_images,
            "categories": categories,
            "metrics": metrics
        }

    def _prepare_dataloaders(self, images: List[ImageItem], categories: List[str]):
        """
        Convert domain images into a dataset or DataLoaders object for the model.

        This is framework-dependent, so the actual implementation can be
        delegated to the model adapter or a helper function.
        """
        # Placeholder: for FastAI, you would create a DataBlock/DataLoaders here
        # For a framework-agnostic approach, return images directly
        return images
