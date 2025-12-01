# Downloaded Papers

This directory contains research papers relevant to the question: "Can LMs predict their own thinking tokens?"

## Papers List

### 1. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
**File:** 2201.11903_chain_of_thought.pdf
**Authors:** Jason Wei, Xuezhi Wang, et al. (Google Research)
**Year:** 2022
**arXiv:** https://arxiv.org/abs/2201.11903
**Why relevant:** Seminal work that established chain-of-thought reasoning in LLMs. Shows that generating intermediate reasoning steps (thinking tokens) significantly improves complex reasoning abilities. Foundational for understanding what "thinking tokens" are.

### 2. Thinking Tokens for Language Modeling
**File:** 2405.08644_thinking_tokens.pdf
**Authors:** Various
**Year:** 2024
**arXiv:** https://arxiv.org/abs/2405.08644
**Why relevant:** Directly addresses the concept of "thinking tokens" - special tokens that allow language models to perform more calculations when encountering complex problems. Highly relevant to understanding whether models can predict their own thinking token usage.

### 3. Compressed Chain of Thought: Efficient Reasoning Through Dense Representations
**File:** 2412.13171_compressed_cot.pdf
**Authors:** Jeffrey Cheng, Benjamin Van Durme
**Year:** 2024
**arXiv:** https://arxiv.org/abs/2412.13171
**Why relevant:** Introduces "contemplation tokens" as compressed representations of reasoning chains. Addresses the efficiency vs. reasoning trade-off. Relevant for understanding the variable-length nature of thinking tokens and potential for prediction.

### 4. Chain-of-Thought Reasoning Without Prompting
**File:** 2402.10200_cot_without_prompting.pdf
**Authors:** Various
**Year:** 2024
**arXiv:** https://arxiv.org/abs/2402.10200
**Why relevant:** Shows CoT reasoning can be elicited by altering decoding process rather than prompting. Suggests reasoning paths are inherently present in model predictions, which relates to whether models can predict their own reasoning token usage.

### 5. Wait, We Don't Need to "Wait"! Removing Thinking Tokens Improves Reasoning Efficiency
**File:** 2506.08343_removing_thinking_tokens.pdf
**Authors:** Various
**Year:** 2025
**arXiv:** https://arxiv.org/abs/2506.08343
**Why relevant:** Examines whether explicit thinking tokens (like "Wait", "Hmm") are necessary for reasoning. Proposes NoWait approach that reduces CoT length by 27%-51%. Directly relevant to understanding the necessity and predictability of thinking tokens.

### 6. Do Thinking Tokens Help or Trap? Towards More Efficient Large Reasoning Model
**File:** 2506.23840_do_thinking_tokens_help.pdf
**Authors:** Various
**Year:** 2025
**arXiv:** https://arxiv.org/abs/2506.23840
**Why relevant:** Critically examines thinking tokens and identifies the "thinking trap" where they may hinder reasoning within constrained token budgets. Essential for understanding when/whether models should predict and use thinking tokens.

### 7. Large Language Models Have Intrinsic Meta-Cognition, but Need a Good Lens
**File:** 2506.08410_llm_metacognition.pdf
**Authors:** Various
**Year:** 2025
**arXiv:** https://arxiv.org/abs/2506.08410
**Why relevant:** Examines meta-cognitive abilities of LLMs - their self-awareness of their own reasoning processes. Directly relevant to whether models can predict their own thinking token usage. Proposes AutoMeco evaluation framework.

### 8. Meta-Awareness Enhances Reasoning Models: Self-Alignment Reinforcement Learning
**File:** 2510.03259_meta_awareness.pdf
**Authors:** Various
**Year:** 2025
**arXiv:** https://arxiv.org/abs/2510.03259
**Why relevant:** Explores meta-awareness - the ability to "know how to think" - and shows that models can be trained to better predict and align their reasoning processes. Introduces MASA training pipeline. Highly relevant to predicting thinking token count.

## Research Themes

### Thinking Tokens Defined
Across papers, thinking/reasoning tokens refer to:
- Intermediate reasoning steps in chain-of-thought
- Special tokens for extended computation (e.g., "Wait", "Hmm", "Therefore")
- Hidden/contemplation tokens that compress reasoning chains
- Tokens generated during the "thinking" phase (as in OpenAI o1)

### Key Questions Addressed
1. **What are thinking tokens?** (Papers 1, 2, 3)
2. **Are they necessary?** (Papers 4, 5, 6)
3. **Can models be self-aware about their reasoning?** (Papers 7, 8)
4. **Can thinking be compressed/predicted?** (Papers 3, 5)

### Relevance to Our Hypothesis
Our hypothesis is that LMs can predict the number of thinking tokens they will generate. The papers suggest:
- **Supporting evidence:** Meta-cognition papers (7, 8) show models have some self-awareness
- **Challenges:** Variable token usage (6), implicit reasoning (4), and token budget constraints (5, 6)
- **Potential approaches:** Meta-awareness training (8), proxy models for prediction, compressed representations (3)

## Citation Information

For citation information and full metadata, refer to the individual arXiv pages linked above.

## Notes

All papers downloaded on 2025-11-30. Papers focus on:
- Chain-of-thought reasoning mechanisms
- Meta-cognitive capabilities of LLMs
- Efficiency vs. reasoning trade-offs
- Token-level analysis of reasoning processes
