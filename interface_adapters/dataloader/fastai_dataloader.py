# ============================================================
# interface_adapters/dataloader/fastai_dataloader.py
# ============================================================
"""FastAI dataloader - based on your create_dataloader function."""

from pathlib import Path
from fastai.vision.all import (
    DataBlock, ImageBlock, CategoryBlock,
    get_image_files, RandomSplitter, parent_label, Resize
)


class FastAIDataLoader:  
    def __init__(self, dataset_path: Path):
        self.dataset_path = Path(dataset_path)
    
    def create_dataloader(
        self,
        batch_size: int = 32,
        valid_pct: float = 0.2,
        resize_size: int = 192
    ):

        # Check if dataset path exists
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")

        # Get all image files
        files = get_image_files(self.dataset_path)
        if len(files) == 0:
            raise ValueError(
                f"No images found in dataset path {self.dataset_path}. "
                "Ensure you have downloaded/prepared the dataset and each category "
                "is in its own subfolder."
            )

        # Ensure there are category subfolders
        categories = [f.name for f in self.dataset_path.iterdir() if f.is_dir()]
        if len(categories) == 0:
            raise ValueError(
                f"No category subfolders found in {self.dataset_path}. "
                "Create one subfolder per category with images inside."
            )
        
        # Create and return FastAI DataLoader.
        dls = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_items=get_image_files,
            splitter=RandomSplitter(valid_pct=valid_pct, seed=42),
            get_y=parent_label,
            item_tfms=[Resize(resize_size, method='squish')]
        ).dataloaders(self.dataset_path, bs=batch_size)
        
        return dls
    
    def check_dataloader(self, dls) -> dict:
        all_files = get_image_files(self.dataset_path)
        
        stats = {
            'total_files': len(all_files),
            'train_size': len(dls.train_ds),
            'valid_size': len(dls.valid_ds),
            'vocab': list(dls.vocab),
            'example_files': [str(f) for f in all_files[:3]]
        }
        
        return stats
