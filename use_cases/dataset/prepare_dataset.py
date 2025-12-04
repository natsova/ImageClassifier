# ============================================================
# use_cases/dataset/prepare_dataset.py
# ============================================================
"""Use case for preparing the dataset with debug logging."""

from pathlib import Path
from typing import List, Dict
from domain.entities.image_item import ImageItem
from domain.entities.category import Category
from domain.interfaces.downloaders import ImageDownloader
from domain.interfaces.processors import ImageProcessor
from domain.interfaces.repositories import ImageRepository
from domain.services.dataset_service import DatasetService
from domain.interfaces.logger import Logger


class PrepareDatasetUseCase:
    """
    Use case to prepare dataset:
    - create folders per category
    - download images
    - process (normalize, resize)
    - save to repository
    """

    def __init__(
        self,
        dataset_service: DatasetService,
        downloader: ImageDownloader,
        processor: ImageProcessor,
        repository: ImageRepository,
        logger: Logger
    ):
        self.dataset_service = dataset_service
        self.downloader = downloader
        self.processor = processor
        self.repository = repository
        self.logger = logger

    def execute(
        self,
        categories: List[str],
        images_per_category: int,
        images_per_search: int
    ) -> Dict[str, any]:
        """
        Prepare dataset end-to-end.
        Returns summary dictionary with:
            - total_downloaded
            - total_valid
            - duplicates_removed
            - corrupted_removed
            - categories: {category_name: count}
        """
        total_downloaded = 0
        total_valid = 0
        duplicates_removed = 0
        corrupted_removed = 0
        category_counts = {}

        for cat_name in categories:
            category = Category(name=cat_name)
            category_path = Path(self.repository.base_path) / cat_name
            category_path.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"Preparing category: {cat_name}")
            print(f"[DEBUG] Preparing category folder: {category_path}")

            # Download images
            downloaded_paths = self.downloader.search_and_download(
                query=cat_name,
                limit=images_per_search,
                output_dir=category_path
            )

            if not downloaded_paths:
                self.logger.warning(f"No images downloaded for {cat_name}. Check your downloader!")
                print(f"[DEBUG] Downloader returned 0 images for category '{cat_name}'")
                continue

            self.logger.debug(f"Downloaded {len(downloaded_paths)} images for {cat_name}")
            total_downloaded += len(downloaded_paths)

            # Process and save valid images
            valid_count = 0
            for path in downloaded_paths:
                if self.processor.validate_and_convert(path):
                    image_item = ImageItem(file_path=path, category=cat_name)
                    image_item = self.processor.normalize(image_item)
                    image_item = self.processor.resize(image_item, (192, 192))
                    self.repository.save(image_item, category)
                    valid_count += 1
                else:
                    corrupted_removed += 1
                    self.logger.warning(f"Invalid image skipped: {path}")
                    print(f"[DEBUG] Invalid image skipped: {path}")

            total_valid += valid_count
            category_counts[cat_name] = valid_count
            print(f"[DEBUG] Category '{cat_name}' - valid images: {valid_count}")

        return {
            "total_downloaded": total_downloaded,
            "total_valid": total_valid,
            "duplicates_removed": duplicates_removed,
            "corrupted_removed": corrupted_removed,
            "categories": category_counts
        }
