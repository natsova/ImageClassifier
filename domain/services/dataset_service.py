from domain.interfaces.repositories import ImageRepository

class DatasetService:
    """Domain-level operations on datasets."""

    def __init__(self, repository: ImageRepository):
        self.repository = repository

    def get_images(self) -> list:
        return self.repository.load_all()

    def filter_by_category(self, category_name: str) -> list:
        return [img for img in self.get_images() if img.category == category_name]

    def validate_images(self) -> list:
        valid_images = []
        for img in self.get_images():
            if img.is_valid():
                valid_images.append(img)
        return valid_images

    def compute_category_distribution(self) -> dict:
        distribution = {}
        for img in self.get_images():
            cat = img.category
            distribution[cat] = distribution.get(cat, 0) + 1
        return distribution
    
    def list_categories(self) -> list:
        """Return list of unique category names in the dataset."""
        return list(self.compute_category_distribution().keys())

    def get_images_by_category(self, category_name: str) -> list:
        """Return all images in a given category."""
        return [img for img in self.get_images() if img.category == category_name]