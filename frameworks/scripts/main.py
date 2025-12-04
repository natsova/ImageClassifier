# ============================================================
# frameworks/scripts/main.py
# To run:   python3 -m frameworks.scripts.main
# ============================================================
"""Main entry point for image classifier workflow."""

#from pathlib import Path
from frameworks.config.app_config import AppConfig
from frameworks.di.container import Container
#from collections import Counter


def main():
    print("Starting Image Classifier")
    print()

    # 1. Initialise configuration
    config = AppConfig()

    print(f"Dataset path: {config.dataset_path}")
    print(f"Categories: {', '.join(config.categories)}")
    print(f"Target: {config.images_per_category} images per category\n")

    # 2. Create dependency injection container (factory that knows what depends on what).
    # Wires dependencies (repositories, services, use cases) so that you can 
    # just call methods on the container without manually constructing each dependency.
    container = Container(config)

    # 3. Prepare dataset
    print("STEP 1: Preparing dataset...")
    print("-" * 70)
    dataset_results = container.prepare_dataset_uc.execute(
        categories=config.categories,
        images_per_category=config.images_per_category,
        images_per_search=config.images_per_search
    )

    print(f"\nDataset preparation complete:")
    print(f"   Total downloaded: {dataset_results['total_downloaded']}")
    print(f"   Total valid: {dataset_results['total_valid']}")
    print(f"   Duplicates removed: {dataset_results['duplicates_removed']}")
    print(f"   Corrupted removed: {dataset_results['corrupted_removed']}")
    for cat, count in dataset_results['categories'].items():
        print(f"   - {cat}: {count} images")
    print()

    # 4. Validate dataset
    print("STEP 2: Validating dataset...")
    print("-" * 70)
    validation_report = container.validate_dataset_uc.execute()
    print(f"Validation complete: {validation_report['valid_images']}/{validation_report['total_images']} valid\n")

    if validation_report['invalid_images'] > 0:
        print("Some images are invalid. Consider cleaning dataset before training.\n")

    # 5. Create dataloader
    print("STEP 3: Creating dataloader...")
    print("-" * 70)
    dataloader = container.get_dataloader()
    dls = dataloader.create_dataloader(
        batch_size=config.batch_size,
        valid_pct=config.valid_pct,
        resize_size=config.resize_size
    )

    stats = dataloader.check_dataloader(dls)
    print(f"Total image files: {stats['total_files']}")
    print(f"Train dataset size: {stats['train_size']}")
    print(f"Validation dataset size: {stats['valid_size']}")
    print(f"Categories: {', '.join(stats['vocab'])}\n")

    # 6. Build model
    print("STEP 4: Building model...")
    print("-" * 70)
    model_adapter = container.get_model_adapter()
    model_adapter.build(architecture=config.architecture)
    print()

    # 7. Train model
    print(f"STEP 5: Training for {config.epochs} epochs...")
    print("-" * 70)
    metrics = model_adapter.train(config.epochs)
    final = metrics[-1]

    print(f"\nTraining complete! Final metrics:")
    print(f"   Epoch: {final.epoch}")
    print(f"   Train Loss: {final.train_loss:.4f}")
    print(f"   Valid Loss: {final.valid_loss:.4f}")
    print(f"   Error Rate: {final.error_rate:.4f}")
    print(f"   Accuracy: {(1 - final.error_rate) * 100:.2f}%\n")

    # 8. Save model
    print("STEP 6: Saving model...")
    print("-" * 70)
    model_adapter.save(config.model_path)
    print()

    # 9. Plot confusion matrix
    print("STEP 7: Generating confusion matrix...")
    print("-" * 70)
    model_adapter.plot_confusion_matrix()
    print()

    # 10. Example prediction
    print("STEP 8: Testing prediction...")
    print("-" * 70)
    first_image = next(config.dataset_path.rglob("*.jpg"))
    result = model_adapter.predict(first_image)
    print(f"Test image: {first_image.name}")
    print(f"True category: {first_image.parent.name}")
    print(f"Predicted: {result['category']}")
    print(f"Confidence: {result['confidence']:.2%}\n")

    print("Top 3 predictions:")
    sorted_probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]
    for cat, prob in sorted_probs:
        print(f"  {cat}: {prob:.2%}")
    print()

    # Final summary
    print("="*70)
    print("COMPLETE! Your image classifier is ready.")
    print("="*70)
    print()
    print("Files created:")
    print(f"   Dataset: {config.dataset_path}/")
    print(f"   Model: {config.model_path}")
    print()
    print("Next steps:")
    print("   - Use the model for predictions")
    print("   - Add more categories")
    print("   - Experiment with different architectures")
    print("   - Deploy as an API")
    print()


if __name__ == "__main__":
    main()
