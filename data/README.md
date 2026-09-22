# Dataset Documentation - Computer Science & Software Engineering Corpus

This repository contains educational documents covering fundamental and advanced topics in Computer Science and Software Engineering.

## Domain Overview
The document assistant is designed to serve university students, software engineers, and researchers by answering questions grounded in core academic computer science materials.

## Included Source Documents (`data/raw/`)

| Document Name | Topic Area | Key Concepts Covered |
| :--- | :--- | :--- |
| `cs101_operating_systems_concurrency.pdf` | Operating Systems | Process Lifecycle, Threads, CPU Scheduling (Round Robin, CFS), Semaphores & Mutexes, Deadlock Coffman Conditions, Banker's Algorithm. |
| `cs201_database_indexing_and_acid.pdf` | Database Systems | B+ Tree Indexing, Clustered vs Non-Clustered Indexes, ACID Properties, Write-Ahead Logging (WAL), Transaction Isolation Levels, Phantom Reads. |
| `cs301_computer_networking_and_protocols.pdf` | Computer Networks | TCP/IP Model, 3-Way Handshake & 4-Way Termination, Congestion Control, HTTP/1.1 vs HTTP/2 Multiplexing, HTTP/3 (QUIC over UDP). |
| `cs401_deep_learning_and_transformers.pdf` | Machine Learning / AI | Scaled Dot-Product Attention (Q, K, V), Multi-Head Attention, Positional Encodings, Transformer Architecture, Decoder-Only LLMs. |

## Data Ingestion & OCR Verification
- All PDF files in `data/raw/` are digital-native text documents verified with `pypdf` extraction.
- None of the documents require external OCR engines because all text layers are directly extractable.
- If scanned or image-only PDFs are added in the future, OCR preprocessing should be integrated before chunking.

## How to Regenerate or Add Documents
1. To regenerate the academic PDFs:
   ```bash
   python data/generate_dataset.py
   ```
2. Place any additional `.pdf` or `.txt` files directly into `data/raw/`.
3. Run `notebooks/rag_pipeline.ipynb` or run the ingestion script to rebuild the ChromaDB vector store.
