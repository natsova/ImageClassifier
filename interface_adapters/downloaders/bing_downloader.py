# ============================================================
# interface_adapters/downloaders/bing_downloader.py 
# ============================================================
"""Enhanced Bing downloader with retry logic and better separation."""

from pathlib import Path
from typing import List, Optional
import time
import random
import shutil
import socket
from domain.interfaces.downloaders import ImageDownloader
from domain.interfaces.logger import Logger
from bing_image_downloader import downloader as bing_downloader


class DownloadError(Exception):
    """Custom exception for download errors."""
    pass


class BingDownloader(ImageDownloader):
    """Enhanced Bing downloader with retry and better error handling."""
    
    def __init__(
        self,
        logger: Logger,
        sleep_time: int = 2,
        timeout: int = 30,
        max_retries: int = 3,
        use_modifiers: bool = True
    ):
        self.logger = logger
        self.sleep_time = sleep_time
        self.timeout = timeout
        self.max_retries = max_retries
        self.use_modifiers = use_modifiers
        
        self._modifiers = [
            "high quality", "hdr", "aesthetic", "macro", "film",
            "close up", "dawn", "dusk", "natural light", "4k"
        ]
    
    def search_and_download(
        self,
        query: str,
        limit: int,
        output_dir: Path
    ) -> List[Path]:
        """
        Download images with retry logic.
        Returns list of RAW downloaded paths (no processing).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        temp_dir = output_dir / "temp_download"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        downloaded_paths = []
        
        # Try multiple randomized queries
        for attempt in range(3):
            query_to_use = self._randomize_query(query) if self.use_modifiers else query
            self.logger.debug(f"Query attempt {attempt + 1}: {query_to_use}")
            
            try:
                paths = self._download_with_retry(query_to_use, limit, temp_dir)
                downloaded_paths.extend(paths)
                
            except DownloadError as e:
                self.logger.warning(f"Download failed: {e}")
                continue
        
        # Cleanup temp directory
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()
        
        self.logger.info(f"Downloaded {len(downloaded_paths)} images for '{query}'")
        return downloaded_paths
    
    def _download_with_retry(
        self,
        query: str,
        limit: int,
        temp_dir: Path
    ) -> List[Path]:
        """Download with exponential backoff retry."""
        last_exception = None
        
        for retry in range(self.max_retries):
            try:
                bing_downloader.download(
                    query,
                    limit=limit,
                    output_dir=str(temp_dir),
                    adult_filter_off=True,
                    force_replace=False,
                    timeout=self.timeout,
                    verbose=False
                )
                
                # Wait before checking results
                time.sleep(self.sleep_time)
                
                # Collect downloaded files
                query_folder = temp_dir / query
                if not query_folder.exists():
                    self.logger.warning(f"No folder created for '{query}'")
                    return []
                
                downloaded = list(query_folder.glob("*.*"))
                
                # Move files to parent and cleanup
                moved_paths = []
                for file_path in downloaded:
                    new_path = temp_dir.parent / file_path.name
                    shutil.move(str(file_path), str(new_path))
                    moved_paths.append(new_path)
                
                # Remove query folder
                shutil.rmtree(query_folder, ignore_errors=True)
                
                return moved_paths
                
            except (TimeoutError, socket.timeout) as e:
                last_exception = e
                wait_time = self.sleep_time * (2 ** retry)
                self.logger.warning(
                    f"Timeout on retry {retry + 1}/{self.max_retries}. "
                    f"Waiting {wait_time}s..."
                )
                time.sleep(wait_time)
                
            except Exception as e:
                last_exception = e
                self.logger.error(f"Unexpected error: {e}")
                break
        
        raise DownloadError(f"Failed after {self.max_retries} retries: {last_exception}")
    
    def _randomize_query(self, base: str) -> str:
        """Add random modifiers to query."""
        return f"{base} {random.choice(self._modifiers)}"
