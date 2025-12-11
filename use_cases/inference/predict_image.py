# ============================================================
# use_cases/inference/predict_image.py
# ============================================================
"""Use case for predicting image categories."""

from pathlib import Path
from typing import List
from domain.entities.image_item import ImageItem
from domain.entities.category import Category
from domain.interfaces.model import Model
from domain.interfaces.logger import Logger
from domain.interfaces.processors import ImageProcessor


class PredictImageUseCase:
    """
    Use case to predict the category of a single image or batch of images:
    - Normalize / preprocess the image
    - Pass it to the model for prediction
    - Return the predicted category
    """

    def __init__(
        self,
        model: Model,
        processor: ImageProcessor,
        logger: Logger
    ):
        self.model = model
        self.processor = processor
        self.logger = logger

    def execute(self, image_paths: List[Path]) -> List[Category]:
        """
        Predict the categories for a list of image file paths.

        Returns a list of Category entities in the same order as input.
        """
        predictions = []

        for path in image_paths:
            try:
                # Step 1: Wrap path in ImageItem and preprocess
                img_item = ImageItem(file_path=path, category=Category("unknown"))
                img_item = self.processor.normalize(img_item)

                # Step 2: Predict category
                category = self.model.predict(img_item)
                predictions.append(category)

                self.logger.info(f"Predicted {category.name} for {path}")

            except Exception as e:
                self.logger.error(f"Prediction failed for {path}: {e}")
                predictions.append(Category(name="unknown"))

        return predictions
