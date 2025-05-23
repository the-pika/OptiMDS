# OptiMDS
A LLM and multi-objective genetic algorithm framework to optimize the readability of medical discharge papers.


# 🧠 Readability Optimization for Medical Summaries using LLaMA 3.1 and NSGA-II

This repository contains a two-stage framework to improve the **readability** of medical discharge summaries generated by large language models (LLMs), while preserving clinical accuracy. The approach combines **abstractive summarization** using the **LLaMA 3.1 Instruct model** with **multi-objective optimization** via **NSGA-II**.

---

## 🔍 Project Overview

- **Summarization Module**: Uses the LLaMA 3.1 Instruct model (via Hugging Face) to generate a high-quality paragraph-format summary from medical discharge papers.
- **Optimization Module**: Applies NSGA-II to optimize:
  - ✅ Readability (minimize Flesch-Kincaid Grade Level - FKGL)
  - ✅ Semantic fidelity (minimize the number of word replacements)

Two optimization strategies are implemented:
- `NSGA_wordnet.py`: Uses WordNet-based synonyms
- `NSGA_word2vec.py`: Uses Word2Vec-based semantic similarity

---

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/readability-optimization-medical-llm.git
   cd readability-optimization-medical-llm

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
  
💡 Note: Ensure you have a Hugging Face token set up for LLaMA model access.

