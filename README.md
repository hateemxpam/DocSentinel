# 🛡️ DocSentinel — Intelligent Policy Compliance Assistant

DocSentinel is an advanced Retrieval-Augmented Generation (RAG) system designed to accurately parse, retrieve, and evaluate complex compliance policies (like GDPR or the EU AI Act). 

Unlike standard RAG tutorials, DocSentinel implements a production-grade architecture featuring hybrid retrieval, cross-encoder reranking, and a dedicated **Trust Layer** to actively prevent LLM hallucination.

## ✨ Key Features

* **Robust Ingestion Pipeline**: Extracts plain text and tables from complex legal PDFs using PyMuPDF and LangChain.
* **Hybrid Retrieval (RRF)**: Fuses traditional keyword search (BM25) with dense semantic search (Qdrant vector database) using Reciprocal Rank Fusion to ensure high recall.
* **Precision Reranking**: Utilizes a local `ms-marco` cross-encoder model to score and re-order hybrid candidates based on deep token-level interactions.
* **Anti-Hallucination Trust Layer**: Automatically blocks the LLM from generating an answer if the source material isn't confident enough. Scores chunks based on:
  * **Freshness** (Time decay based on ingestion date)
  * **Consistency** (Overlap between lexical and semantic retrieval)
  * **Intrinsic Trust** (Document-level metadata scores)
* **Dual-Database Architecture**: Fast vector search via Qdrant, backed by robust metadata storage in PostgreSQL.

## 🏗️ Architecture

1. Ingestion: PDF -> Chunking -> all-MiniLM-L6-v2 Embeddings -> Qdrant / Postgres
2. Retrieval: User Query -> BM25 + Semantic Search -> RRF Merge -> Cross-Encoder Rerank
3. Trust: Candidate Chunks -> Freshness & Consistency Scoring -> Quality Gate
4. Generation (WIP): Passed Chunks -> Groq LLM -> User Output

## 🚀 Tech Stack

* **Language**: Python
* **Vector Database**: Qdrant (Docker)
* **Relational Database**: PostgreSQL (Docker)
* **Embeddings & Reranking**: HuggingFace (`all-MiniLM-L6-v2`, `ms-marco-MiniLM-L-6-v2`)
* **Frameworks**: LangChain, PyMuPDF, `rank_bm25`

## ⚙️ Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/hateemxpam/DocSentinel.git
   cd DocSentinel
   ```

2. **Start the Databases**
   ```bash
   docker run -d -p 6333:6333 qdrant/qdrant
   docker run -d --name docsentinel-pg -p 5432:5432 -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=docsentinel postgres:15
   ```

3. **Install Dependencies**
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate
   pip install -r requirements.txt
   ```

4. **Run Ingestion**
   Populate your `data/raw/` folder with PDFs and run:
   ```bash
   python main.py
   ```