# Copilot Instructions

## Project Overview

This project implements an AI agent orchestrated with **LangGraph**. It uses **Milvus** as the vector database for semantic retrieval and **Cohere** as the reranker to improve the quality of retrieved results before passing them to the agent.

---

## Core Stack

- **LangGraph** — agent orchestration and graph state management
- **Milvus** (`pymilvus`) — vector database for storing and querying embeddings
- **Cohere** — reranking of retrieved documents via the Cohere Rerank API

---

## Key Principles

### LangGraph
- The agent is structured as a **stateful graph**. All data flowing between nodes must go through the graph state.
- Each node is responsible for a single, well-defined step. Do not mix retrieval, reranking, or generation logic inside the same node.
- Routing logic and edge definitions belong in the graph definition file, not inside nodes.
- Always return only the state keys a node modifies — do not return the full state.

### Milvus
- Milvus is used exclusively for **vector similarity search**. It is the first step in the retrieval pipeline.
- Retrieve a reasonably large set of candidates (e.g. top-k = 20–50) to give the reranker enough material to work with.
- The Milvus client connection should be initialized once and reused — do not reconnect on every query.

### Cohere Reranker
- Cohere Rerank is applied **after** Milvus retrieval, taking the candidate documents and the original query as input.
- The reranker reduces the candidate set down to the most relevant documents (e.g. top 3–5) that will be passed further into the agent pipeline.
- Always pass the full list of candidates in a **single API call** — do not call the reranker in a loop.
- Preserve the relevance score returned by Cohere alongside each document for traceability.

---

## Retrieval Pipeline Order

```
User Query → Milvus (broad retrieval, top-k candidates) → Cohere Rerank (narrow to top-n) → Agent / Generation
```

---

## General Guidelines

- Never hardcode API keys or connection strings. Use environment variables.
- Handle the case where Milvus returns zero results explicitly — do not pass empty context downstream.
- Keep node functions pure and side-effect free where possible.
- Use type hints throughout the codebase.
