
# 🧠 LangChain Agent Streamlit App

This repository provides a simple but powerful LangChain-based AI assistant capable of interacting with multiple HTML documents. It consists of a backend agent implemented in a Jupyter notebook, and a front-end web interface built using Streamlit. A sample notebook demonstrating basic Python operations is also included for illustration.

---

## 📁 Repository Overview

```
├── agent.ipynb             # Core logic for processing and querying documents using LangChain
├── app.py                  # Streamlit app providing a user interface for the agent
├── sample_notebook.ipynb   # Example notebook demonstrating basic arithmetic in Python
```

---

## 📘 File Descriptions

### `agent.ipynb` - The AI Agent Core

This notebook is the heart of the system and includes the following capabilities:
- Loads and processes a directory of HTML documents.
- Uses `LangChain` for document embedding, storage (via FAISS), and retrieval.
- Integrates with an LLM (e.g., Ollama) for answering user questions.
- Constructs a retriever chain with history awareness for contextual responses.

> 💡 This notebook essentially converts a collection of documents into a queryable knowledge base.

---

### `app.py` - The Streamlit Frontend

This Python script runs a Streamlit app that acts as the frontend for the AI agent. Features include:
- A user interface for asking questions to the document-based agent.
- Real-time response generation powered by the logic defined in `agent.ipynb`.
- Streamlined and clean UX using Streamlit components.
- Includes environment and dependency management for seamless execution.

Run the app locally with:

```bash
streamlit run app.py
```

---

### `sample_notebook.ipynb` - Python Arithmetic Demo

This is a standalone educational notebook that:
- Demonstrates fundamental arithmetic operations in Python.
- Prints results of operations like addition, subtraction, multiplication, division, modulus, etc.
- Useful as a basic template or for testing notebook parsing.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/langchain-agent-app.git
cd langchain-agent-app
```

### 2. Install requirements

Use pip or your preferred environment manager:

```bash
pip install -r requirements.txt
```

> Make sure dependencies like `LangChain`, `FAISS`, `Streamlit`, and `Ollama` are correctly installed.

### 3. Start the App

```bash
streamlit run app.py
```

---

## 📌 Requirements

- Python 3.8+
- streamlit
- langchain
- faiss-cpu
- huggingface embeddings
- ollama
- torch

> Use `pip install -r requirements.txt` to install all dependencies.

---

## 🤖 Powered By

- [LangChain](https://www.langchain.com/)
- [Streamlit](https://streamlit.io/)
- [Ollama](https://ollama.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
