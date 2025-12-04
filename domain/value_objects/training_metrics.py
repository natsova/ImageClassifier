# ============================================================
# domain/value_objects/training_metrics.py
# ============================================================
"""Value object for training metrics."""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class TrainingMetrics:
    """Immutable representation of model training metrics."""
    
    metrics: Dict[str, float]

    def __post_init__(self):
        # Ensure all values are floats between 0 and 1
        for name, value in self.metrics.items():
            if not isinstance(value, (float, int)):
                raise TypeError(f"Metric '{name}' must be a number, got {type(value)}")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Metric '{name}' must be between 0 and 1, got {value}")

    def get(self, metric_name: str) -> float:
        """Get a metric by name."""
        if metric_name not in self.metrics:
            raise KeyError(f"Metric '{metric_name}' not found")
        return self.metrics[metric_name]

    def as_dict(self) -> Dict[str, float]:
        """Return metrics as a plain dictionary."""
        return dict(self.metrics)

    def __str__(self):
        return ", ".join(f"{k}: {v:.4f}" for k, v in self.metrics.items())
