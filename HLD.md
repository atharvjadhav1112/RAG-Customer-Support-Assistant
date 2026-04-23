# High-Level Design (HLD) Document
## Retrieval-Augmented Generation Customer Support Assistant

---

### 1. System Overview
**Problem Definition:**
Modern customer support teams struggle with high volumes of repetitive inquiries. Traditional chatbots often fail to understand nuance or provide accurate information based on the company's specific knowledge base, leading to frustrating customer experiences.

**Scope of the System:**
The system is an AI-powered customer support assistant utilizing Retrieval-Augmented Generation (RAG). It ingests static knowledge bases (PDFs), retrieves highly relevant document chunks using vector search, and generates contextual answers using a Large Language Model (LLM). Crucially, the system utilizes a state-graph to route queries conditionally based on intent, and includes a Human-in-the-Loop (HITL) escalation mechanism for queries where model confidence is lacking.

---

### 2. Architecture Diagram

The system operates across two primary pipelines:

**1. Document Ingestion Pipeline**
User Interface / API -> Document Loader (PyPDF/pdfplumber) -> Text Chunker -> Embedding System -> Vector Database (ChromaDB)

**2. Retrieval & Generation Pipeline (LangGraph Workflow)**
User Interface / API -> Workflow Orchestrator (LangGraph) Which Executes:
  [Input Node] -> [Intent Router Node]
    |-> (If Greeting) -> [Greeting Generator] -> Output
    |-> (If FAQ/Technical) -> [Retrieval Node] -> [LLM Processing Layer] -> [Confidence Evaluator]
          |-> (If High Confidence) -> Output
          |-> (If Low Confidence or Escalate Intent) -> [HITL System Node] -> Human Agent

---

### 3. Component Description
- **Document Loader:** Implemented via `PDFIngestionService`. Uses `PyPDF` as the primary extractor, falling back to `pdfplumber` to handle complex layouts and tabular data.
- **Chunking Strategy:** Employs a sentence-aware chunker that divides documents into overlapping segments (default 500 words, 50-word overlap) to preserve semantic continuity.
- **Embedding Model:** A flexible strategy that defaults to `SentenceTransformers (all-MiniLM-L6-v2)`, scales to OpenAI `text-embedding-ada-002`, or falls back to offline `TF-IDF` if necessary.
- **Vector Store:** Uses `ChromaDB` configured for persistent local storage, mapping embeddings to source chunks.
- **Retriever:** Fetches top-K relevant chunks using Maximal Marginal Relevance (MMR) approximation to balance relevance and diversity. 
- **LLM:** Pluggable backend supporting direct OpenAI API calls (`gpt-4o-mini`) or a rule-based offline smart-mock generator that extracts highly concordant sentences for zero-cost operation.
- **Graph Workflow Engine:** Powered by `LangGraph` (`StateGraph`), representing the processing pipeline as distinct, pure-function nodes modifying a central immutable state.
- **Routing Layer:** Intent-based keyword classification and evaluation rules that dynamically alter the graph execution path.
- **HITL Module:** A JSON-backed, session-aware ticketing queue that suspends the AI interaction and notifies human agents when escalation criteria are met.

---

### 4. Data Flow
1. **Query Lifecycle:** The user sends a POST request with a text query.
2. The `input_node` normalizes the text. 
3. The `intent_router_node` checks for trigger phrases (e.g., complaint, escalate).
4. The router directs the query. For knowledge inquiries, it proceeds to the `retrieval_node`.
5. The `retrieval_node` queries ChromaDB and formats top chunks into a single context string.
6. The `llm_node` builds a grounded prompt combining the query, context, and chat history, then invokes the model.
7. The `confidence_eval_node` assesses the retrieval density and LLM uncertainty markers.
8. Based on confidence, the route diverges either to the user (`output_node`) or to the `hitl_node` which issues a ticket ticket ID.

---

### 5. Technology Choices
- **ChromaDB:** Chosen for its open-source, serverless nature. It embeds seamlessly within Python applications without requiring separate database orchestration, simplifying local deployment while maintaining performance.
- **LangGraph:** Selected over linear chaining because customer support flows are inherently cyclical and condition-heavy (e.g., evaluating confidence dynamically and routing backward or forward).
- **FastAPI:** Provides high-performance, asynchronous request handling, and automatic documentation formatting.

---

### 6. Scalability Considerations
- **Handling Large Documents:** Chunking streams data efficiently. The vector store inherently scales to millions of vectors. 
- **Increasing Query Load:** FastAPI's asynchronous architecture handles concurrent non-blocking IO requests efficiently.
- **Latency Concerns:** Utilizing local sentence transformers reduces network latency. MMR computation occurs in-memory post-retrieval to optimize time constraints. 
