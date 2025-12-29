# Mini RAG System (Interactive)

A lightweight, end-to-end Retrieval-Augmented Generation (RAG) pipeline designed for speed and interactivity. This system allows users to chat with a document corpus using a high-performance LLM (Groq) and local embeddings (Ollama).

## 🧠 Architecture Decisions & Reasoning

* **Inference Engine (LLM): Groq (`openai/gpt-oss-120b`)**
  * *Reasoning*: Switched from local inference to Groq to leverage LPU (Language Processing Unit) technology. This minimizes *Time-to-First-Token*, ensuring the interactive chat experience is seamless (latency < 1s) compared to CPU-based local execution.

* **Embeddings: Ollama (`mxbai-embed-large`)**
  * *Reasoning*: Chosen over default models for its superior performance on the MTEB (Massive Text Embedding Benchmark), providing higher relevance in retrieval while keeping data processing local and private.

* **Vector Store: Chroma (In-memory)**
  * *Reasoning*: Optimized for rapid prototyping and ease of setup. It eliminates the need for external database infrastructure for this specific test scope.

* **Evaluation Strategy: LLM-as-a-Judge (Faithfulness Check)**
  * *Reasoning*: Instead of relying solely on similarity scores, the system employs a secondary LLM call to verify if the generated answer is strictly grounded in the retrieved context, effectively detecting hallucinations.

## 🛠️ Prerequisites

1. **Python 3.10+**
2. **Ollama**: Ensure Ollama is installed and running locally.
3. **Pull Embedding Model**:

    ```bash
    ollama pull mxbai-embed-large
    ```

## ⚙️ Installation & Setup

1. **Install Dependencies**:

    ```bash
    pip install langchain langchain-community langchain-chroma langchain-ollama langchain-groq python-dotenv bs4
    ```

2. **Environment Configuration**:
    Create a `.env` file in the project root to store your credentials securely:

    ```env
    GROQ_API_KEY=your_groq_api_key_here
    ```

## 🚀 How to Run

1. Start the pipeline:

    ```bash
    python src/main.py
    ```

2. **Interaction**:
    * The system will ingest the sample data (automatically chunked and embedded).
    * Once ready, you will enter an **interactive loop**.
    * Type your questions directly into the terminal.
    * Type `exit` or `quit` to stop the program.

## 📂 Project Structure

```text
/project_root
  ├── src/
  │    ├── main.py       # Entry point: Handles UI loop and orchestration
  │    ├── pipeline.py   # Core logic: Ingestion, Chunking, Retrieval Chain
  │    └── eval.py       # Evaluation logic: Faithfulness scoring
  ├── .env               # API Keys (Not committed)
  └── README.md          # Documentation
