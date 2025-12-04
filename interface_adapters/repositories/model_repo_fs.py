from domain.interfaces.repositories import ImageRepository
from domain.interfaces.logger import Logger
from pathlib import Path
import shutil
from domain.entities.image_item import ImageItem
from domain.entities.category import Category

class ModelRepositoryFS(ImageRepository):
    """Concrete implementation for storing models (images, predictions, etc.)"""

    SUPPORTED_EXTENSIONS = {'.pkl', '.pt'}

    def __init__(self, base_path: Path, logger: Logger):
        self.base_path = Path(base_path)
        self.logger = logger
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(self, image: ImageItem, category: Category) -> None:
        # implement saving logic for your model if needed
        pass

    def load_all(self) -> list[ImageItem]:
        # implement logic to load all items
        return []

    def load_by_category(self, category: Category) -> list[ImageItem]:
        # implement logic to load items by category
        return []

    def delete(self, image: ImageItem) -> None:
        # implement delete logic
        pass

    def list_categories(self) -> list[str]:
        # implement listing categories
        return []
