# Downloaded Datasets

This directory contains datasets for the research project on predicting LM thinking tokens. Data files are locally available but excluded from git via .gitignore.

## Datasets

1. **GSM8K** - 8,792 grade school math problems (gsm8k/)
2. **MATH-500** - 500 competition math problems (math/)
3. **HotpotQA** - 97,852 multi-hop QA examples (hotpotqa/)
4. **ARC-Challenge** - 2,590 science reasoning questions (arc_challenge/)

All datasets are in HuggingFace format and ready to use.

## Quick Start

Load a dataset:
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/gsm8k")
```

See sample data in each dataset's `samples/` subdirectory.

For complete documentation, download instructions, and usage examples, see the main workspace documentation files.
