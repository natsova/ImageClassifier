# ============================================================
# domain/value_objects/training_metrics.py
# ============================================================
"""Value object for training metrics."""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class TrainingMetrics:
    """Immutable representation of model training metrics."""
    
    epoch: int
    train_loss: float
    valid_loss: float
    error_rate: float

    def as_dict(self):
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "valid_loss": self.valid_loss,
            "error_rate": self.error_rate
        }

    def __str__(self):
        return (
            f"Epoch {self.epoch}: "
            f"train_loss={self.train_loss:.4f}, "
            f"valid_loss={self.valid_loss:.4f}, "
            f"error_rate={self.error_rate:.4f}"
        )