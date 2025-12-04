# domain/entities/category.py
"""
Domain entity representing an image category.
Pure data model with no filesystem or framework logic.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Category:
    """
    Represents a high-level category (e.g., 'dog', 'ocean', 'umbrella').
    Immutable and used to strongly type repository operations.
    """

    name: str

    def __post_init__(self):
        normalized = self.name.strip().lower()
        object.__setattr__(self, "name", normalized)
