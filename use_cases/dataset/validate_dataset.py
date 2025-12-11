# ============================================================
# use_cases/dataset/validate_dataset.py
# ============================================================
"""Use case for validating a dataset."""

from typing import Dict
from domain.entities.category import Category
from domain.entities.image_item import ImageItem
from domain.services.dataset_service import DatasetService
from domain.interfaces.repositories import ImageRepository
from domain.interfaces.processors import ImageProcessor
from domain.interfaces.logger import Logger


class ValidateDatasetUseCase:
    """
    Validate a dataset by checking:
    - All images are valid (can be opened and normalized)
    - All categories exist
    - Optionally resize/normalize images
    """

    def __init__(
        self,
        dataset_service: DatasetService,
        processor: ImageProcessor,
        logger: Logger
    ):
        self.dataset_service = dataset_service
        self.processor = processor
        self.logger = logger

    def execute(self) -> Dict[str, int]:
        """
        Validate all images in the repository.

        Returns a summary dictionary:
        {
            "total_images": int,
            "valid_images": int,
            "invalid_images": int
        }
        """
        total_images = 0
        valid_images = 0
        invalid_images = 0

        categories = self.dataset_service.list_categories()
        self.logger.info(f"Found categories: {categories}")

        for cat_name in categories:
            images = self.dataset_service.get_images_by_category(cat_name)
            total_images += len(images)

            for image in images:
                if self.processor.validate_and_convert(image.file_path):
                    valid_images += 1
                else:
                    invalid_images += 1
                    self.logger.warning(f"Invalid image: {image.file_path}")

        self.logger.info(
            f"Dataset validation complete: "
            f"{valid_images}/{total_images} images valid, {invalid_images} invalid."
        )

        return {
            "total_images": total_images,
            "valid_images": valid_images,
            "invalid_images": invalid_images
        }
