# GraphRAG

A RAG (Retrieval-Augmented Generation) system using LangGraph, Milvus vector database, and OpenAI/Cohere for intelligent document Q&A.

## Overview

GraphRAG implements a graph-based workflow that stores documents in a vector database, retrieves relevant context using hybrid search, and generates responses using LLMs. The system supports multiple document collections organized by namespaces.

## Features

- **Document Storage**: Vector database with Milvus for scalable document storage
- **Hybrid Search**: Combines dense embeddings + sparse BM25 vectors for better retrieval
- **Reranking**: Optional Cohere reranker to improve search result quality
- **Graph Workflow**: LangGraph orchestrates the retrieval → generation pipeline
- **Multi-namespace**: Organize documents by project/collection for better isolation
- **Summarization**: Auto-generate collection summaries for context awareness
- **Conversational Memory**: Supports both short-term (session) and long-term memory so the system can maintain context across multiple turns and sessions. This improves coherence for follow-up questions, enables limited personalization, and helps the agent remember important facts or user preferences when appropriate.

## Architecture

```
Query → [Store] → Retrieve → [Reranker] → [Agent] → Generate → Response
          ↓                                   ↑
       Milvus DB                         LangGraph
```

The workflow:
1. **Store documents** with embeddings in Milvus
2. **Retrieve** relevant docs using hybrid search (dense + sparse)
3. **Rerank** results with Cohere (optional)
4. **Generate** response using LLM with retrieved context

1. Install dependencies:
```bash
poetry install
```

2. Set environment variables:
```bash
OPENAI_API_KEY=your_key
COHERE_API_KEY=your_key  # Optional
```

3. Start Milvus:
```bash
docker run -d -p 19530:19530 milvusdb/milvus:latest
```

## Usage

```python
from graphrag.store import Store
from graphrag.agent import GraphRAG
from langchain_core.documents import Document

# Initialize store
store = Store(
    uri="http://localhost:19530",
    database="my_db",
    collection="docs",
    k=4
)

# Add documents
docs = [Document(
    page_content="Your content",
    metadata={"namespace": "project1", "page_start": 1, "path": "doc.pdf"}
)]
store.add(docs)

# Create agent and query
agent = GraphRAG(store=store, llm="gpt-4o-mini", rerank=True)
result = agent.run("Your question here")

print(f"Answer: {result['response']}")
print(f"Sources: {len(result['context'])} documents")
```

## Core Components

### Store (`graphrag.store.Store`)
Manages document storage and retrieval:
- Stores documents with OpenAI embeddings in Milvus
- Supports similarity search with score thresholds
- Generates collection summaries for better context
- Query by metadata filters (namespace, page range, etc.)

### Agent (`graphrag.agent.GraphRAG`)
LangGraph-based workflow orchestration:
- Takes user queries and retrieves relevant documents
- Generates responses using specified LLM (GPT, Claude, etc.)
- Returns complete state with query, context, and response
- Configurable retrieval parameters and reranking
- Maintains conversational memory (short-term/session and optional long-term storage). Memory provides recent-turn context for coherent multi-turn dialogue and can persist selected information across sessions to improve continuity and personalization. Memory usage is integrated into retrieval and generation to surface relevant prior exchanges or saved facts when producing answers.

### Reranker (`graphrag.reranker.CohereReranker`)
Optional component for improving retrieval quality:
- Reorders retrieved documents by relevance to query
- Reduces noise and improves answer quality
- Configurable top-N results and models

## Document Metadata

```python
metadata = {
    "namespace": "project_name",  # Required
    "page_start": 1,
    "page_end": 1, 
    "path": "document.pdf"
}
```

## API Reference

### Store Methods
```python
# Document management
store.add(docs)                    # Add documents to collection
store.drop_collection()            # Remove entire collection

# Retrieval
store.retrieve(query, score=False) # Basic similarity search
store.retrieve_with_reranker(query) # Search + reranking
store.query('namespace == "proj"')  # Metadata filtering

# Utilities
store.summarize(model="gpt-4o")     # Generate collection summary
```

### Agent Methods
```python
# Main workflow
result = agent.run(query)          # Process query end-to-end

# Result structure
{
    "query": "user question",
    "context": [Document, ...],    # Retrieved documents
    "response": "generated answer"
}
```

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Required for embeddings and LLM
- `COHERE_API_KEY`: Optional, for reranking functionality

### Store Parameters
- `uri`: Milvus server URI (default: localhost:19530)
- `database/collection`: Database and collection names
- `k`: Number of documents to retrieve (default: 4)
- `embedding_model`: OpenAI model (default: text-embedding-3-small)

### Agent Parameters
- `llm`: Model name (gpt-4o-mini, claude-3-sonnet, etc.)
- `rerank`: Enable/disable Cohere reranking (default: False)
