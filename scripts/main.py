# scripts/main.py

import sys
from pathlib import Path

# Add project root to sys.path for VS Code
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.config import Config
from core.dataset_manager import DatasetManager
#from core.dataloader import create_dataloader
from core.model_manager import ModelManager  # if you have this

def main():
    # ---------------- Dataset setup ----------------
    config = Config(categories=["sky", "ocean", "umbrella", "dog", "book"])
    
    dataset = DatasetManager(config)
    dataset.prepare_full_dataset()  # downloads, cleans, refills, displays samples
    
    # ---------------- Dataloader -------------------
    dls = create_dataloader(config)
    
    # ---------------- Model workflow ----------------
    model = ModelManager(dls)
    model.build()
    model.train(epochs=10)
    model.save("imageclassifier.pkl")
    
    # ---------------- Example inference -------------
    pred, probs = model.predict_url(
        "https://example.com/dog.jpg"
    )
    print(pred, probs)

if __name__ == "__main__":
    main()
