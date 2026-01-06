# RAG-based Machine Learning Textbook Chatbot

This project is a Retrieval-Augmented Generation (RAG) based chatbot that answers questions using a Machine Learning textbook as its knowledge source.

Instead of relying on the language model’s internal knowledge, the chatbot first retrieves relevant content from the textbook and then generates answers based only on that information. This helps reduce hallucinations and improves accuracy.

---

## Project Overview

Large Language Models often produce confident but incorrect answers when they lack domain-specific knowledge.  
This project addresses that issue by implementing a RAG pipeline where a Machine Learning textbook acts as the single source of truth.

The system retrieves relevant sections from the book using semantic search and then uses a local LLM to generate grounded responses.

---

## Concepts Used

- Retrieval-Augmented Generation (RAG)
- Text chunking and overlap
- Embeddings and semantic similarity
- Vector databases
- Local LLM inference
- LangChain for orchestration

---

## Architecture Flow

PDF Textbook  
→ Text Extraction  
→ Chunking  
→ Embeddings  
→ Vector Database  

User Question  
→ Embedding  
→ Similarity Search  
→ Retrieved Context  
→ LLM  
→ Final Answer

---

## Tech Stack

- Python  
- LangChain  
- ChromaDB  
- Sentence Transformers  
- Ollama  
- LLaMA 3  

---

## Project Structure

