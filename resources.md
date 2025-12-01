# Resources Catalog

## Summary

This document catalogs all resources gathered for the research project: **"Can LMs predict their own thinking tokens?"**

The resource gathering phase successfully collected:
- **8 research papers** on chain-of-thought reasoning, thinking tokens, and meta-cognition
- **4 datasets** for reasoning tasks (math and multi-hop QA)
- **3 code repositories** for CoT evaluation and implementation

All resources are organized in the workspace and ready for experimental use.

---

## Papers

**Total papers downloaded:** 8

| # | Title | Authors | Year | File | arXiv | Key Focus |
|---|-------|---------|------|------|-------|-----------|
| 1 | Chain-of-Thought Prompting Elicits Reasoning | Wei, Wang, et al. | 2022 | 2201.11903_chain_of_thought.pdf | 2201.11903 | Foundational CoT work |
| 2 | Chain-of-Thought Reasoning Without Prompting | Various | 2024 | 2402.10200_cot_without_prompting.pdf | 2402.10200 | Implicit CoT in models |
| 3 | Thinking Tokens for Language Modeling | Herel, Mikolov | 2024 | 2405.08644_thinking_tokens.pdf | 2405.08644 | Special thinking tokens |
| 4 | Compressed Chain of Thought | Cheng, Van Durme | 2024 | 2412.13171_compressed_cot.pdf | 2412.13171 | Contemplation tokens |
| 5 | Removing Thinking Tokens | Various | 2025 | 2506.08343_removing_thinking_tokens.pdf | 2506.08343 | Efficiency analysis |
| 6 | LLM Meta-Cognition | Various | 2025 | 2506.08410_llm_metacognition.pdf | 2506.08410 | Self-awareness |
| 7 | Do Thinking Tokens Help or Trap? | Various | 2025 | 2506.23840_do_thinking_tokens_help.pdf | 2506.23840 | Thinking trap |
| 8 | Meta-Awareness Enhances Reasoning | Various | 2025 | 2510.03259_meta_awareness.pdf | 2510.03259 | MASA training |

**Location:** `papers/`

**Documentation:** See `papers/README.md` for detailed descriptions and relevance to research.

**File sizes:**
- Total: 9.3 MB
- Range: 297 KB to 3.2 MB per paper
- All successfully downloaded and validated

---

## Datasets

**Total datasets downloaded:** 4

| # | Name | Source | Size | Task | Train | Val | Test | Location |
|---|------|--------|------|------|-------|-----|------|----------|
| 1 | GSM8K | openai/gsm8k | 8,792 | Math word problems | 7,473 | - | 1,319 | datasets/gsm8k/ |
| 2 | MATH-500 | HuggingFaceH4/MATH-500 | 500 | Competition math | - | - | 500 | datasets/math/ |
| 3 | HotpotQA | hotpotqa/hotpot_qa | 97,852 | Multi-hop QA | 90,447 | 7,405 | - | datasets/hotpotqa/ |
| 4 | ARC-Challenge | allenai/ai2_arc | 2,590 | Science reasoning | 1,119 | 299 | 1,172 | datasets/arc_challenge/ |

**Location:** `datasets/`

**Documentation:** See `datasets/README.md` for download instructions and usage examples.

**Storage:**
- Total disk usage: ~205 MB
- Format: HuggingFace Arrow format
- Git-friendly: Data files excluded via .gitignore, samples included

**Download script:** `datasets/download_datasets.py` (automated download)

---

## Code Repositories

**Total repositories cloned:** 3

| # | Name | URL | Purpose | Language | Location |
|---|------|-----|---------|----------|----------|
| 1 | chain-of-thought-hub | github.com/FranxYao/chain-of-thought-hub | CoT benchmarking | Python | code/chain-of-thought-hub/ |
| 2 | auto-cot | github.com/amazon-science/auto-cot | Automatic CoT | Python/PyTorch | code/auto-cot/ |
| 3 | ThoughtSource | github.com/OpenBioLink/ThoughtSource | CoT data & tools | Python | code/ThoughtSource/ |

**Location:** `code/`

**Documentation:** See `code/README.md` for setup instructions and adaptation ideas.

**Key features:**
- **chain-of-thought-hub:** GSM8K evaluation scripts, prompt templates, benchmarking
- **auto-cot:** Automated demonstration selection, reduces manual prompt engineering
- **ThoughtSource:** Comprehensive dataset with reasoning chains, analysis tools

---

## Resource Gathering Notes

### Search Strategy

**Literature Search:**
1. Searched arXiv for recent papers (2024-2025) on:
   - "thinking tokens" / "reasoning tokens"
   - "chain of thought"
   - "LLM meta-cognition" / "self-prediction"
   - "inference latency prediction"

2. Used multiple sources:
   - arXiv directly (arxiv.org)
   - Semantic Scholar (semanticscholar.org)
   - Papers with Code (paperswithcode.com)
   - Google Scholar (for highly-cited foundational work)

3. Prioritized:
   - Recent work (2024-2025) for state-of-the-art
   - Highly-cited foundational papers (Wei et al. 2022)
   - Papers with direct relevance to hypothesis

**Dataset Search:**
1. Identified requirements from papers:
   - Tasks requiring multi-step reasoning
   - Variable-length solution chains
   - Established benchmarks

2. Searched HuggingFace Datasets Hub
3. Cross-referenced with Papers with Code dataset links
4. Selected based on:
   - Availability and accessibility
   - Quality and community adoption
   - Size (manageable for experiments)
   - Domain diversity (math, QA, science)

**Code Search:**
1. Searched GitHub for:
   - Official paper implementations
   - Chain-of-thought evaluation code
   - Reasoning benchmark infrastructure

2. Prioritized:
   - Active repositories (recent commits)
   - Good documentation
   - Community adoption (stars, forks)

### Selection Criteria

**Papers (8 selected from ~20 reviewed):**
- ✓ Direct relevance to thinking tokens or meta-cognition
- ✓ Recent (2024-2025) or foundational (2022)
- ✓ Available as free PDFs
- ✓ Clear methodology and results
- ✓ Diverse perspectives (supporting and challenging)

**Datasets (4 selected from ~10 candidates):**
- ✓ Publicly available on HuggingFace
- ✓ Commonly used in literature
- ✓ Require multi-step reasoning
- ✓ Variable solution complexity
- ✓ Different domains for generalization testing

**Code (3 repositories from ~8 reviewed):**
- ✓ Active and maintained
- ✓ Clear documentation
- ✓ Relevant to our research (CoT evaluation, GSM8K)
- ✓ Not overly complex or specialized
- ✓ Compatible licenses

### Challenges Encountered

**Challenge 1: MATH dataset access**
- **Issue:** Main `hendrycks/competition_math` dataset inaccessible
- **Solution:** Used `HuggingFaceH4/MATH-500` subset instead
- **Impact:** Smaller dataset (500 vs 12,500), but sufficient for experiments

**Challenge 2: Thinking tokens paper implementation**
- **Issue:** No official public code for arXiv:2405.08644
- **Solution:** Documented absence, noted related implementations
- **Impact:** Will need to implement from paper description

**Challenge 3: PDF extraction warnings**
- **Issue:** Some PDFs had encoding issues during text extraction
- **Solution:** Extraction completed despite warnings, text readable
- **Impact:** Minimal - full text successfully extracted

**Challenge 4: Large dataset sizes**
- **Issue:** HotpotQA ~200MB, could slow git operations
- **Solution:** Implemented .gitignore for datasets/, kept only samples
- **Impact:** Repository stays lightweight, users download data locally

### Gaps and Workarounds

**Gap 1: No direct implementation of thinking token prediction**
- **Gap:** No existing code specifically for pre-generation token prediction
- **Workaround:** Will adapt meta-cognition and CoT evaluation code
- **Impact:** Need to implement prediction mechanism ourselves (research contribution!)

**Gap 2: Limited meta-cognition evaluation tools**
- **Gap:** AutoMeco (Paper 7) and MASA (Paper 8) code not yet public
- **Workaround:** Implement simplified versions based on paper descriptions
- **Impact:** May not perfectly replicate methods, but can approximate

**Gap 3: Token counting standardization**
- **Gap:** No standard definition of "thinking tokens" across papers
- **Workaround:** Define clearly in our experiments (tokens before final answer)
- **Impact:** Need to be explicit about definitions for reproducibility

**Gap 4: Latency benchmarking tools**
- **Gap:** Limited tools for measuring actual inference latency vs token count
- **Workaround:** Implement simple latency measurement in experiments
- **Impact:** Initial experiments may focus on token count, latency later

---

## Recommendations for Experiment Design

Based on the gathered resources, here are data-driven recommendations:

### 1. Primary Dataset(s)

**Recommended: GSM8K**
- **Justification:**
  - Most commonly used in literature (6/8 papers)
  - Moderate difficulty → clear reasoning chains
  - Sufficient size (7.5K training) for pattern learning
  - Well-understood baseline performance

**Recommended: MATH-500**
- **Justification:**
  - Higher difficulty → more variable reasoning
  - Tests generalization to complex problems
  - Manageable size for experimentation

**Use for validation:** HotpotQA, ARC-Challenge (cross-domain testing)

### 2. Baseline Methods

**For thinking token prediction:**

1. **Simple statistical baseline:**
   - Average tokens per problem in training set
   - Stratified by difficulty/problem type

2. **Feature-based regression:**
   - Features: problem length, math operators, question words
   - Model: Linear regression or gradient boosting
   - Train on labeled data (actual token counts)

3. **Meta-prompt baseline:**
   - Prompt: "How many reasoning steps will this require? [problem]"
   - Extract number from response
   - Use as token count estimate

4. **Hybrid approach:**
   - Combine features + meta-prompt
   - Weighted ensemble

**For reasoning quality:**
- Standard few-shot CoT (Wei et al. baseline)
- Zero-shot CoT ("Let's think step by step")
- Direct answering (no CoT)

### 3. Evaluation Metrics

**Primary:**
- **MAE (Mean Absolute Error):** Mean |predicted - actual| tokens
- **MAPE (Mean Absolute Percentage Error):** Mean (|pred - actual| / actual) × 100%
- **Accuracy@K:** % predictions within K tokens (K=5, 10, 20)

**Secondary:**
- **Correlation (Pearson r):** How well predictions track trends
- **Final answer accuracy:** % correct answers (sanity check)

**Practical:**
- **Latency error:** |predicted_time - actual_time|
- **Cost estimation error:** |predicted_cost - actual_cost|

### 4. Experimental Workflow

```
Phase 1: Data Collection
  → Generate CoT responses for datasets
  → Count actual thinking tokens
  → Extract problem features
  → Create prediction training set

Phase 2: Baseline Implementation
  → Implement statistical baseline
  → Train feature-based models
  → Test meta-prompt approach
  → Evaluate all baselines

Phase 3: Advanced Methods
  → Implement MASA-inspired training
  → Test hybrid approaches
  → Optimize predictions

Phase 4: Evaluation
  → Measure prediction accuracy
  → Analyze failure modes
  → Test practical utility
  → Cross-dataset validation
```

### 5. Code Adaptation

**Use chain-of-thought-hub for:**
- GSM8K evaluation infrastructure
- Prompt templates
- Baseline comparison framework

**Use auto-cot for:**
- Automatic demonstration generation
- Understanding reasoning variation
- Feature extraction ideas

**Use ThoughtSource for:**
- Pre-labeled reasoning chains (if available)
- Analysis tools
- Evaluation metrics

### 6. Expected Outcomes

**If hypothesis is correct:**
- MAE < 20 tokens on GSM8K
- MAPE < 30%
- Accuracy@10 > 60%
- Useful for latency/cost estimation

**If hypothesis is partially correct:**
- Weak prediction better than random
- Works for some problem types, not others
- High variance in prediction accuracy

**If hypothesis is wrong:**
- Predictions no better than simple averages
- High error across all metrics
- No correlation with actual token counts

Even negative results would be valuable - demonstrating limits of meta-cognition!

---

## File Structure Summary

```
workspace/
├── papers/
│   ├── README.md
│   ├── 2201.11903_chain_of_thought.pdf
│   ├── 2402.10200_cot_without_prompting.pdf
│   ├── 2405.08644_thinking_tokens.pdf
│   ├── 2412.13171_compressed_cot.pdf
│   ├── 2506.08343_removing_thinking_tokens.pdf
│   ├── 2506.08410_llm_metacognition.pdf
│   ├── 2506.23840_do_thinking_tokens_help.pdf
│   ├── 2510.03259_meta_awareness.pdf
│   └── extracted_info.json
│
├── datasets/
│   ├── .gitignore
│   ├── README.md
│   ├── download_datasets.py
│   ├── gsm8k/
│   │   ├── dataset_dict.json
│   │   ├── train/
│   │   ├── test/
│   │   └── samples/
│   ├── math/
│   │   ├── dataset_dict.json
│   │   ├── test/
│   │   └── samples/
│   ├── hotpotqa/
│   │   ├── dataset_dict.json
│   │   ├── train/
│   │   ├── validation/
│   │   └── samples/
│   └── arc_challenge/
│       ├── dataset_dict.json
│       ├── train/
│       ├── validation/
│       ├── test/
│       └── samples/
│
├── code/
│   ├── README.md
│   ├── chain-of-thought-hub/
│   ├── auto-cot/
│   └── ThoughtSource/
│
├── literature_review.md
├── resources.md (this file)
├── extract_paper_info.py
└── .resource_finder_complete (to be created)
```

---

## Next Steps for Experiment Runner

The resource gathering phase is complete. The experiment runner can now:

1. **Read literature_review.md** for comprehensive background
2. **Load datasets** using instructions in datasets/README.md
3. **Explore code repositories** using guidance in code/README.md
4. **Design experiments** based on recommendations above
5. **Implement prediction methods** (baselines → advanced)
6. **Evaluate and analyze** using suggested metrics

All necessary resources are in place for productive research!

---

## Metadata

- **Resource gathering completed:** 2025-11-30
- **Time spent:** ~2.5 hours
- **Papers downloaded:** 8 (100% success rate)
- **Datasets downloaded:** 4 (100% success rate for accessible datasets)
- **Code repositories cloned:** 3 (100% success rate)
- **Total storage used:** ~215 MB (papers + datasets + code)
- **Git repository size:** ~10 MB (papers + code only, datasets gitignored)

**Quality Assessment:**
- ✓ All papers directly relevant to research question
- ✓ Datasets cover intended use cases (math, QA, reasoning)
- ✓ Code repositories active and well-documented
- ✓ Comprehensive documentation created
- ✓ Resources ready for immediate experimental use

**Confidence Level:** High - All core resources acquired and validated. Ready to proceed with experimentation.
