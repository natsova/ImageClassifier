# ============================================================
# frameworks/cli/prepare.py 
# ============================================================
"""CLI command for dataset preparation."""

import argparse
from pathlib import Path
from frameworks.config.app_config import AppConfig
from frameworks.di.container import Container


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset')
    parser.add_argument('--config', type=Path, help='Config file path')
    parser.add_argument('--categories', nargs='+', help='Categories to download')
    parser.add_argument('--images', type=int, help='Images per category')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = AppConfig.from_yaml(args.config)
    else:
        config = AppConfig()
    
    # Override with CLI args
    if args.categories:
        config.categories = args.categories
    if args.images:
        config.images_per_category = args.images
    
    # Run preparation
    container = Container(config)
    results = container.prepare_dataset_uc.execute(
        categories=config.categories,
        images_per_category=config.images_per_category,
        images_per_search=config.images_per_search
    )
    
    print(f"Downloaded {results['total_valid']} images")


if __name__ == '__main__':
    main()


print(" Improved implementation created!")
print(" Key improvements:")
print("   1. Configuration validation & external loading (YAML/JSON/ENV)")
print("   2. Logger interface for dependency injection")
print("   3. Enhanced repositories with better error handling")
print("   4. Retry logic with exponential backoff in downloader")
print("   5. Non-mutating image processor with batch operations")
print("   6. CLI command structure for better modularity")
print("   7. Proper exception handling throughout")
print("   8. Support for multiple image formats")
print("   9. Better separation of concerns")
print("   10. Production-ready patterns")