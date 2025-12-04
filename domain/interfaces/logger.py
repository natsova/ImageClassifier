# ============================================================
# domain/interfaces/logger.py
# ============================================================
"""Logger interface for dependency injection."""

from abc import ABC, abstractmethod
from typing import Any


class Logger(ABC):
    """Interface for logging operations in the domain."""

    @abstractmethod
    def info(self, message: str, **kwargs: Any) -> None:
        """Log an informational message."""
        pass

    @abstractmethod
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning message."""
        pass

    @abstractmethod
    def error(self, message: str, **kwargs: Any) -> None:
        """Log an error message."""
        pass

    @abstractmethod
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log a debug message."""
        pass
