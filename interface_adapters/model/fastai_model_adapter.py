# ============================================================
# interface_adapters/model/fastai_model_adapter.py
# ============================================================
"""FastAI model adapter - based on your training code."""

from pathlib import Path
from fastai.vision.all import (
    vision_learner, resnet18, resnet34, resnet50,
    error_rate, load_learner, ClassificationInterpretation
)
from domain.value_objects.training_metrics import TrainingMetrics
from typing import List
import matplotlib.pyplot as plt


class FastAIModelAdapter:
    """Adapter for FastAI model operations - based on your code."""
    
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.learner = None
    
    def build(self, architecture="resnet18"):
        """Build model with specified architecture."""
        arch_map = {
            'resnet18': resnet18,
            'resnet34': resnet34,
            'resnet50': resnet50
        }
        
        arch = arch_map.get(architecture, resnet18)
        
        self.learner = vision_learner(
            self.dataloader,
            arch,
            metrics=error_rate
        )
        
        print(f"Model built with {architecture}")
        return self.learner
    
    def train(self, epochs: int) -> List[TrainingMetrics]:
        """Train model - based on your learn.fine_tune(10)."""
        if self.learner is None:
            raise ValueError("Model must be built before training")
        
        print(f"🏋️  Fine-tuning for {epochs} epochs...")
        self.learner.fine_tune(epochs)
        
        # Extract metrics from recorder
        metrics = []
        recorder = self.learner.recorder
        
        for i in range(len(recorder.values)):
            metrics.append(TrainingMetrics(
                epoch=i + 1,
                train_loss=float(recorder.values[i][0]),
                valid_loss=float(recorder.values[i][1]),
                error_rate=float(recorder.values[i][2])
            ))
        
        return metrics
    
    def save(self, path: Path):
        """Save trained model using FastAI export."""
        if self.learner is None:
            raise ValueError("No model to save")
        
        path.parent.mkdir(parents=True, exist_ok=True)
        self.learner.export(path)
        print(f"Model saved to: {path}")
    
    def load(self, path: Path):
        """Load saved model."""
        self.learner = load_learner(path)
        print(f"Model loaded from: {path}")
    
    def predict(self, image_path: Path) -> dict:
        """Predict category for image."""
        if self.learner is None:
            raise ValueError("Model must be loaded before prediction")
        
        pred_class, pred_idx, probs = self.learner.predict(image_path)
        
        return {
            'category': str(pred_class),
            'confidence': float(probs[pred_idx]),
            'all_probabilities': {
                str(self.learner.dls.vocab[i]): float(prob)
                for i, prob in enumerate(probs)
            }
        }
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix - based on your code."""
        if self.learner is None:
            raise ValueError("Model must be trained before plotting")
        
        interp = ClassificationInterpretation.from_learner(self.learner)
        interp.plot_confusion_matrix()
        plt.show()
    
    def show_results(self, n: int = 9):
        """Show prediction results."""
        if self.learner is None:
            raise ValueError("Model must be trained")
        
        self.learner.show_results(max_n=n)
