# ============================================================
# frameworks/di/container.py
# ============================================================
"""Dependency injection container"""

from frameworks.config.app_config import AppConfig
from interface_adapters.repositories.dataset_repo_fs import DatasetRepositoryFS
from interface_adapters.repositories.model_repo_fs import ModelRepositoryFS
from interface_adapters.downloaders.bing_downloader import BingDownloader
from interface_adapters.processors.image_processor_pillow import ImageProcessorPillow
from interface_adapters.dataloader.fastai_dataloader import FastAIDataLoader
from interface_adapters.model.fastai_model_adapter import FastAIModelAdapter
from domain.services.dataset_service import DatasetService
from domain.services.model_service import ModelService
from use_cases.dataset.prepare_dataset import PrepareDatasetUseCase
from use_cases.dataset.validate_dataset import ValidateDatasetUseCase
from use_cases.training.train_model import TrainModelUseCase
from use_cases.inference.predict_image import PredictImageUseCase
from pathlib import Path
from interface_adapters.logging.python_logger import PythonLogger

class Container:
    """Dependency injection container."""
    
    def __init__(self, config: AppConfig):
        self.config = config

        # Logger
        self.logger = PythonLogger(name="container")

        # Repositories
        # Manages reading/writing dataset images on disk
        self.image_repo = DatasetRepositoryFS(
            base_path=config.dataset_path,
            logger=self.logger
        )
        self.model_repo = ModelRepositoryFS(
            base_path=Path("models"),
            logger=self.logger
        )

        # Adapters (interfaces to external systems)
        self.downloader = BingDownloader(
            logger=self.logger,
            sleep_time=config.sleep_time,
            timeout=config.download_timeout,
            remove_duplicates=config.remove_duplicates,
            max_refill_rounds=config.max_refill_rounds,
            max_retries=config.max_retries,
            use_modifiers=config.use_modifiers
        )
        self.processor = ImageProcessorPillow(logger=self.logger) # Handles image resizing, format conversion, etc. 

        # Services
        # Contain business logic and act as an intermediary between repositories and use cases.
        self.dataset_service = DatasetService(self.image_repo)
        self.model_service = ModelService(self.model_repo)

        # Lazy-loaded / lazy initialisation
        '''
        Instance variables of the Container class. "None" indicates that the object doesn’t exist yet.
        Avoids expensive operations until they’re actually needed.
        Underscore means "private, don’t access directly outside the class" - use getter methods to access.
        '''
        self._dataloader = None    # Will eventually hold a FastAIDataLoader object.
        self._model_adapter = None    # Will eventually hold a FastAIModelAdapter object.

        # Use case
        self.prepare_dataset_uc = PrepareDatasetUseCase(
            dataset_service=self.dataset_service,
            repository=self.image_repo,
            downloader=self.downloader,
            processor=self.processor,
            logger=self.logger
        )
        # Use case
        self.validate_dataset_uc = ValidateDatasetUseCase(
            dataset_service=self.dataset_service,
            processor=self.processor,
            logger=self.logger
        )
        # Use case
        self.train_model_uc = TrainModelUseCase(
            model_service=self.model_service,
            dataloader_factory=self.get_dataloader,
            model_adapter_factory=self.get_model_adapter,
            logger=self.logger
        )
        # Use case
        self.predict_image_uc = PredictImageUseCase(
            model=self.get_model_adapter, # Passes the function, not the object.
            processor=self.processor,
            logger=self.logger
        )
    
    def get_dataloader(self) -> FastAIDataLoader:
        """Lazy load dataloader."""
        if self._dataloader is None:
            self._dataloader = FastAIDataLoader(self.config.dataset_path) # Create object
        return self._dataloader
    
    def get_model_adapter(self) -> FastAIModelAdapter:
        """Lazy load model adapter with dataloader."""
        if self._model_adapter is None:
            dataloader = self.get_dataloader()
            dls = dataloader.create_dataloader(
                batch_size=self.config.batch_size,
                valid_pct=self.config.valid_pct,
                resize_size=self.config.resize_size
            )
            self._model_adapter = FastAIModelAdapter(dls)    # Create object
        return self._model_adapter