# Rag Pipeline Simple

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg) ![License](https://img.shields.io/badge/License-MIT-green.svg) ![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

## 📖 Description

A tutorial to create a RAG pipeline from scratch. Will be implemented step by step to serve as an application demonstrating LLM pipeline building skills. This is an active work in progress.

## ✨ Features

- Step-by-step RAG implementation
- Document processing and indexing
- Vector database integration
- Query processing and retrieval
- Response generation
- Performance evaluation tools

## 🛠️ Tech Stack

- **Primary Language:** Python
- **Framework:** Ai/Ml
- **Additional Tools:** N/A

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/shivajipanam/rag-pipeline-simple.git
cd rag-pipeline-simple

# Install dependencies
pip install -r requirements.txt

# Install vector database
pip install chromadb

# Set up environment
python setup.py
``````bash
# Process documents
python process_documents.py --input docs/

# Build vector index
python build_index.py

# Query the RAG system
python query.py --question "What is machine learning?"

# Evaluate performance
python evaluate.py
```