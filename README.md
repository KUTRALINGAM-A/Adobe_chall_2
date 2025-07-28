# 📘 Enhanced Persona-Driven Analyzer

This project is a **comprehensive PDF analyzer** that extracts the most relevant sections from multiple PDF documents based on a specified **persona** and **job to be done**. It intelligently combines heuristics with **LLM support (TinyLlama)** to deliver **high-quality summaries** tailored to user needs.

---

## 🚀 Features

- 🧠 **Persona-Aware Relevance**: Extracts sections most relevant to a specific persona's role and goal.
- 📄 **Smart PDF Parsing**: Uses `pdfplumber` and `PyMuPDF (fitz)` to ensure maximum text extraction accuracy.
- 🔍 **Heuristic + LLM Fusion**: Combines rule-based analysis with LLM (via `llama-cpp-python`) for improved accuracy.
- 🧾 **Balanced Section Selection**: Selects exactly **10 sections** with balanced coverage across documents.
- 🧵 **Thread-safe LLM Usage**: Supports multithreaded execution without race conditions.
- 📝 **Cleaned Titles**: Automatic correction of noisy OCR titles for better readability.
- 📚 **Support for up to 100 pages per PDF**.
- ⚙️ Runs on local CPU, **no cloud dependency**.

## Requirements

### Before You Start

> **📌 Important:**  
> This project uses a **large model file** tracked using **Git LFS**.  
> Make sure you have [Git LFS](https://git-lfs.com/) installed **before cloning this repo**:

```bash
git lfs install
git clone https://github.com/yourusername/enhanced_persona_analyzer.git
