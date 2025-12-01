#!/usr/bin/env python3
"""Download datasets for LM thinking token prediction research."""

import os
import json
from pathlib import Path

try:
    from datasets import load_dataset
    print("✓ datasets library available")
except ImportError:
    print("ERROR: Please install datasets library:")
    print("  pip install datasets")
    exit(1)

def download_and_save_dataset(dataset_name, dataset_id, config=None, split=None):
    """Download and save a HuggingFace dataset."""
    print(f"\n{'='*60}")
    print(f"Downloading: {dataset_name}")
    print(f"{'='*60}")

    save_path = Path("datasets") / dataset_name
    save_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load dataset
        if config:
            dataset = load_dataset(dataset_id, config, trust_remote_code=True)
        else:
            dataset = load_dataset(dataset_id, trust_remote_code=True)

        # Save to disk
        dataset.save_to_disk(str(save_path))
        print(f"✓ Saved to: {save_path}")

        # Print dataset info
        print(f"\nDataset splits:")
        for split_name, split_data in dataset.items():
            print(f"  - {split_name}: {len(split_data)} examples")

        # Save sample examples
        samples_dir = save_path / "samples"
        samples_dir.mkdir(exist_ok=True)

        # Save first 5 examples from train/validation split
        for split_name in ['train', 'validation', 'test']:
            if split_name in dataset:
                samples = []
                for i, example in enumerate(dataset[split_name]):
                    if i >= 5:
                        break
                    samples.append(example)

                sample_file = samples_dir / f"{split_name}_samples.json"
                with open(sample_file, 'w') as f:
                    json.dump(samples, f, indent=2, default=str)
                print(f"✓ Saved {len(samples)} {split_name} samples to: {sample_file}")

        return True

    except Exception as e:
        print(f"✗ Error downloading {dataset_name}: {e}")
        return False

def main():
    print("="*60)
    print("Downloading Datasets for LM Thinking Token Prediction")
    print("="*60)

    datasets_to_download = [
        {
            "name": "gsm8k",
            "id": "openai/gsm8k",
            "config": "main",
            "description": "Grade School Math 8K - math word problems requiring multi-step reasoning"
        },
        {
            "name": "math",
            "id": "hendrycks/competition_math",
            "config": None,
            "description": "Competition MATH - challenging mathematics problems with solutions"
        },
        {
            "name": "hotpotqa",
            "id": "hotpotqa/hotpot_qa",
            "config": "distractor",
            "description": "HotpotQA - multi-hop question answering"
        },
        {
            "name": "arc_challenge",
            "id": "allenai/ai2_arc",
            "config": "ARC-Challenge",
            "description": "AI2 Reasoning Challenge - science questions requiring reasoning"
        }
    ]

    results = {}
    for dataset_info in datasets_to_download:
        success = download_and_save_dataset(
            dataset_info["name"],
            dataset_info["id"],
            dataset_info.get("config")
        )
        results[dataset_info["name"]] = success

    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {name}")

    print("\nAll datasets downloaded!")
    print("See datasets/README.md for usage instructions")

if __name__ == "__main__":
    main()
