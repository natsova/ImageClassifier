# domain/interfaces/downloaders.py
"""
Interfaces for image downloading services.
Pure abstractions with no external library or network code.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List


class ImageDownloader(ABC):
    """
    Interface for remote image downloaders (Bing, Google, custom APIs).
    Implementations must return raw downloaded file paths.
    """

    @abstractmethod
    def search_and_download(
        self,
        query: str,
        limit: int,
        output_dir: Path
    ) -> List[Path]:
        """
        Search for images and download them into the given directory.
        Returns a list of absolute paths to downloaded files.
        """
        pass
