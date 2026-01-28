# Djezzy Chatbot (LLM + RAG)

This project implements a **question-answering chatbot** for **Djezzy** using a **Large Language Model (LLM)** combined with a **Retrieval-Augmented Generation (RAG)** approach.  
The entire system is implemented in **a single Jupyter Notebook** as a proof of concept.

The chatbot answers user questions by retrieving relevant information from a document corpus and using this context to generate accurate and grounded responses.

---

## Project Content

This repository contains a single file:

- `chatbot_rag.ipynb` — Jupyter Notebook implementing the full LLM + RAG pipeline

---

## Description

The notebook demonstrates an end-to-end **LLM + RAG workflow**, including:
- Loading and preprocessing textual documents
- Chunking text for efficient retrieval
- Generating semantic embeddings
- Performing similarity search over a vector index
- Injecting retrieved context into the LLM prompt
- Generating final answers based on retrieved evidence

This approach improves answer reliability and reduces hallucinations compared to a standalone LLM.

---

## Technologies Used

- **Language:** Python  
- **Environment:** Jupyter Notebook  
- **LLM:** Transformer-based language model  
- **Embeddings:** Sentence embeddings  
- **Vector Search:** FAISS (or equivalent similarity search)  
- **Libraries:** PyTorch, Transformers, NumPy, Pandas  

---


