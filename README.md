# 🤖 RAG-Based Chatbot System

A **Retrieval-Augmented Generation (RAG) based AI chatbot** that allows users to ask questions about documents and receive answers based on the information contained in those documents.

The system combines **document retrieval, embeddings, vector search, and a Large Language Model (LLM)** to provide context-aware responses while reducing hallucinations.

---

## 📌 Project Overview

Traditional AI chatbots may generate answers that are not related to the user's documents.

This project solves that problem using **Retrieval-Augmented Generation (RAG)**.

The system:

1. Loads a PDF/document.
2. Extracts the text.
3. Splits the document into smaller chunks.
4. Converts the chunks into vector embeddings.
5. Stores the embeddings in a vector database.
6. Retrieves the most relevant information when the user asks a question.
7. Sends the retrieved context to an LLM.
8. Generates an answer based on the retrieved document content.

### Basic Architecture

```text
                ┌─────────────────┐
                │   PDF / Document│
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Text Extraction │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Text Chunking   │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │   Embeddings    │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Vector Database │
                └────────┬────────┘
                         │
                  User Question
                         │
                         ▼
                ┌─────────────────┐
                │ Similarity      │
                │ Search          │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │      LLM        │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ AI Generated    │
                │ Answer          │
                └─────────────────┘
```

---

## 🚀 Features

* 📄 PDF document processing
* 🔎 Semantic document search
* 🧠 Retrieval-Augmented Generation
* 💬 Question answering over documents
* 🧩 Document chunking
* 🔢 Vector embeddings
* 🤖 LLM-based response generation
* 💻 Local AI/LLM support
* ⚡ GPU acceleration support when available
* 🔐 Can be configured to run without sending document data to external AI APIs

---



## 🛠️ Technologies Used

* **Python** – Core programming language
* **Streamlit** – Web-based chatbot interface
* **LangChain** – Document processing and RAG pipeline integration
* **PyMuPDF** – PDF document loading and text extraction
* **Hugging Face** – Embedding and reranking models
* **BAAI/bge-base-en-v1.5** – Text embedding model
* **BAAI/bge-reranker-base** – Document reranking model
* **FAISS** – Vector database for semantic similarity search
* **BM25** – Keyword-based document retrieval
* **Reciprocal Rank Fusion (RRF)** – Combines vector and BM25 retrieval results
* **Ollama** – Local LLM inference
* **Granite 4.2 8B** – Large Language Model used for response generation
* **PyTorch** – GPU computation and model acceleration
* **CUDA** – GPU acceleration for embeddings and reranking
* **Sentence Transformers** – Cross-encoder reranking


---

## 📂 Project Structure

```text
RAGBOT/
│
├── app.py
├── README.md
├── LICENSE
├── .gitignore
│
├── doc/
│   └── Everything_About_AI_Resource_Guide.pdf
│
└── ...
```

---

## ⚙️ How It Works

### 1. Document Loading

The application reads the PDF document and extracts its text.

### 2. Text Chunking

Large documents are divided into smaller chunks so that relevant sections can be efficiently retrieved.

### 3. Embedding Generation

Each chunk is converted into a numerical vector representation called an **embedding**.

### 4. Vector Storage

The embeddings are stored in a vector database/index.

### 5. User Query

The user enters a question about the document.

### 6. Retrieval

The system performs semantic similarity search and retrieves the most relevant document chunks.

### 7. Generation

The retrieved information is provided to the LLM as context.

The LLM then generates the final response.

---

## 💡 Example

### User Question

```text
What are the important AI concepts mentioned in the document?
```

### RAG Pipeline

```text
Question
   ↓
Embedding
   ↓
Similarity Search
   ↓
Relevant Document Chunks
   ↓
LLM
   ↓
Context-Aware Answer
```

---

## 🎯 Objective

The main objective of this project is to build an AI chatbot capable of **answering questions from a specific knowledge base instead of relying only on the model's pre-trained knowledge**.

This approach can be useful for:

* 📚 Educational documents
* 📖 Research papers
* 🏢 Company documentation
* 📋 Technical manuals
* 🧑‍💻 Developer documentation
* 📄 Large PDF collections
* 🎓 Academic projects

---

## 🔮 Future Improvements

Possible future improvements include:

* Support for multiple PDFs
* Web-based user interface
* Conversation memory
* Source citations for retrieved information
* Better document ranking
* Hybrid keyword + semantic search
* OCR support for scanned PDFs
* Multi-language document support
* Streaming responses
* Advanced GPU optimization
* Cloud deployment
* Authentication and user management

---

## 📈 Learning Outcomes

Through this project, the following concepts were explored:

* Natural Language Processing
* Large Language Models
* Retrieval-Augmented Generation
* Semantic Search
* Vector Embeddings
* Document Processing
* Prompt Engineering
* AI Application Development
* Local LLM Deployment
* GPU-accelerated AI inference

---

## 👨‍💻 Author

**Vighnesh Ganji**

GitHub: `https://github.com/vg123-sys`

---

## 📜 License

This project is licensed under the MIT License.
