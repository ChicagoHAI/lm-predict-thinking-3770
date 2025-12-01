# Cloned Code Repositories

This directory contains code repositories relevant to chain-of-thought reasoning and thinking token prediction research.

## Repositories

### 1. Chain-of-Thought Hub
**Location:** `code/chain-of-thought-hub/`
**URL:** https://github.com/FranxYao/chain-of-thought-hub
**Purpose:** Benchmarking large language models' complex reasoning ability with chain-of-thought prompting

**Key Features:**
- GSM8K evaluation scripts and prompts
- Multiple prompt templates (hardest, standard, etc.)
- Benchmarking infrastructure for CoT reasoning
- Performance comparison across different LLMs

**Key Files:**
- `gsm8k/lib_prompt/` - Contains prompt templates for GSM8K
- `gsm8k/` - GSM8K-specific evaluation code
- Various benchmark scripts

**How to Use:**
```bash
cd code/chain-of-thought-hub
# See repository README for setup instructions
```

**Relevance:**
Provides established evaluation infrastructure for testing chain-of-thought reasoning on GSM8K and other benchmarks. Can be adapted to measure thinking token usage and prediction accuracy.

---

### 2. Auto-CoT
**Location:** `code/auto-cot/`
**URL:** https://github.com/amazon-science/auto-cot
**Purpose:** Official implementation of "Automatic Chain of Thought Prompting in Large Language Models"

**Key Features:**
- Automated demonstration selection for CoT prompting
- Reduces manual effort in prompt design
- Implements clustering-based demonstration selection
- Evaluation on multiple reasoning benchmarks

**Key Files:**
- `run_gsm8k.py` - GSM8K evaluation script
- `run_demo.py` - Demo for automatic CoT generation
- Various utility scripts

**Dependencies:**
- PyTorch 1.8.2+
- CUDA support recommended
- Transformers library

**How to Use:**
```bash
cd code/auto-cot
# Install dependencies
pip install -r requirements.txt
# Run GSM8K evaluation
python run_gsm8k.py
```

**Relevance:**
Demonstrates how to automatically generate CoT demonstrations. Useful for understanding how reasoning chains vary in length and complexity, which relates to thinking token prediction.

---

### 3. ThoughtSource
**Location:** `code/ThoughtSource/`
**URL:** https://github.com/OpenBioLink/ThoughtSource
**Purpose:** Central, open resource for data and tools related to chain-of-thought reasoning

**Key Features:**
- Comprehensive dataset collection with reasoning chains
- Tools for generating reasoning chains with various LLMs
- Evaluation frameworks for CoT performance
- GSM8K with reasoning chains included

**Key Components:**
- Dataset collection with annotated reasoning chains
- Evaluation tools for reasoning quality
- Model inference scripts
- Reasoning chain analysis tools

**Key Files:**
- `libs/` - Core library code
- `dataset/` - Dataset loading and processing
- `evaluation/` - Evaluation scripts

**How to Use:**
```bash
cd code/ThoughtSource
# See repository README for setup and usage
```

**Relevance:**
Provides extensive resources for analyzing reasoning chains, including length, complexity, and structure. Directly applicable to studying thinking token patterns and prediction.

---

## Setup Instructions

### General Setup

1. **Clone repositories** (already done):
   ```bash
   cd code
   ls -la
   ```

2. **Install common dependencies**:
   ```bash
   # Create virtual environment (recommended)
   python3 -m venv venv
   source venv/bin/activate

   # Install common packages
   pip install torch transformers datasets accelerate
   ```

3. **Repository-specific setup**:
   Each repository has its own README with specific installation instructions. Refer to:
   - `code/chain-of-thought-hub/README.md`
   - `code/auto-cot/README.md`
   - `code/ThoughtSource/README.md`

## Usage Recommendations for Research

### For Testing Thinking Token Prediction:

1. **Start with chain-of-thought-hub:**
   - Use existing GSM8K evaluation scripts
   - Modify to capture intermediate token counts
   - Track reasoning chain lengths

2. **Leverage Auto-CoT for automation:**
   - Automatic generation of diverse reasoning chains
   - Study length variation across problems
   - Analyze demonstration selection impact

3. **Use ThoughtSource for analysis:**
   - Access pre-generated reasoning chains
   - Compare reasoning patterns
   - Evaluate prediction accuracy

## Adaptation Ideas

To adapt these repositories for thinking token prediction research:

### 1. Token Counting Instrumentation
```python
# Add to inference loops
def count_thinking_tokens(model_output, problem):
    """Count tokens in reasoning chain before final answer."""
    # Split reasoning from answer
    reasoning, answer = split_cot(model_output)

    # Count tokens in reasoning portion
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    thinking_tokens = len(tokenizer.encode(reasoning))

    return {
        'problem': problem,
        'thinking_tokens': thinking_tokens,
        'total_tokens': len(tokenizer.encode(model_output)),
        'reasoning': reasoning
    }
```

### 2. Prediction Task Setup
```python
# Train model to predict thinking tokens
def predict_thinking_tokens(problem, model):
    """Predict number of thinking tokens before generation."""
    # Option 1: Use meta-prompt
    meta_prompt = f"How many reasoning steps will this problem require? {problem}"
    prediction = model.generate(meta_prompt)

    # Option 2: Train predictor model
    # Features: problem length, difficulty, domain
    features = extract_features(problem)
    predicted_tokens = predictor_model(features)

    return predicted_tokens
```

### 3. Evaluation Framework
```python
def evaluate_prediction_accuracy(dataset, model, predictor):
    """Evaluate how well model predicts its own thinking tokens."""
    results = []

    for problem in dataset:
        # Predict token count
        predicted = predictor.predict(problem)

        # Generate actual response
        response = model.generate(problem)
        actual = count_thinking_tokens(response, problem)

        # Compare
        error = abs(predicted - actual['thinking_tokens'])
        results.append({
            'problem': problem,
            'predicted': predicted,
            'actual': actual['thinking_tokens'],
            'error': error,
            'relative_error': error / actual['thinking_tokens']
        })

    return analyze_results(results)
```

## Key Files to Examine

For understanding CoT implementation details:

1. **chain-of-thought-hub:**
   - `gsm8k/lib_prompt/prompt_hardest.txt` - Example prompts
   - GSM8K evaluation scripts

2. **auto-cot:**
   - `run_gsm8k.py` - End-to-end evaluation
   - Demonstration selection logic

3. **ThoughtSource:**
   - Dataset loading utilities
   - Reasoning chain parsing code

## Notes

- All repositories cloned on 2025-11-30
- Check individual READMEs for latest updates
- Some repositories may require API keys for model access
- GPU recommended for running evaluations

## Potential Extensions

1. **Meta-learning predictor**: Train a model to predict token counts based on problem features
2. **Adaptive generation**: Adjust generation parameters based on predicted complexity
3. **Efficiency optimization**: Skip unnecessary thinking tokens when predicted count is low
4. **Cost estimation**: Use predictions to estimate inference costs before generation

## Citation

If using these repositories, cite the original papers:

**Chain-of-Thought Hub:**
See repository for citation information

**Auto-CoT:**
```
@inproceedings{zhang2022automatic,
  title={Automatic Chain of Thought Prompting in Large Language Models},
  author={Zhang, Zhuosheng and Zhang, Aston and Li, Mu and Smola, Alex},
  booktitle={ICLR},
  year={2023}
}
```

**ThoughtSource:**
See repository for citation information
