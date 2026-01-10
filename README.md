# Intelligent RAG-powered Complaint Chatbot

A RAG (Retrieval-Augmented Generation) system for analyzing customer complaints from the CFPB dataset.

## Features
- **Exploratory Data Analysis**: Scripts for processing large CFPB datasets.
- **Vector Store**: ChromaDB integration with ONNX-based embeddings.
- **RAG Pipeline**: Semantic search and LLM-based answer generation using Flan-T5.
- **Interactive UI**: Gradio-powered web interface.

## Project Structure
- `src/`: Core logic and processing scripts.
- `data/`: (Ignored/Local) Dataset storage.
- `vector_store/`: (Ignored/Local) Local vector database.
- `app.py`: Main entry point for the Gradio UI.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   python app.py
   ```
