# Low-Level Design (LLD) Document
## Retrieval-Augmented Generation Customer Support Assistant

---

### 1. Module-Level Design

- **Document Processing Module (`ingestion.py`):** Encapsulates the `PDFIngestionService`. It exposes an `.extract()` method handling binary payloads or file paths, leveraging robust failover extraction techniques.
- **Chunking Module (`chunking.py`):** Responsible for taking raw `PageDocument` elements and segmenting them into indexed arrays of text, preserving strict token/word limitation bounds while retaining a configured overlap buffer.
- **Embedding Module (`embedding.py`):** Implements an Abstract Base Class (ABC) providing structural guarantees across the `LocalEmbedding`, `OpenAIEmbedding`, and `TFIDFEmbedding` sub-services.
- **Vector Storage / Retrieval Module (`retriever.py`):** Acts as the mediator to `chromadb.PersistentClient()`. It executes standard queries and applies filtering variables corresponding to session metadata.
- **Query Processing Module (`rag_pipeline.py`):** Orchestrates intent detection through regex matching, performs confidence computation formulas on retrieved content, and maps formatted contexts dynamically to rigorous system prompt structures.
- **Graph Execution Module (`graph.py`):** Uses LangGraph to compile the `StateGraph`. Central hub maintaining the mapping matrix of pure node functions and their conditional edges.
- **HITL Module (`hitl.py`):** Employs simple persistent dicts mapping string schemas to the filesystem containing active tickets, resolving, and monitoring timeouts.

---

### 2. Data Structures

- **Document Representation:**
  ```python
  @dataclass
  class PageDocument:
      source: str
      page_num: int
      text: str
  ```
- **Chunk Format:**
  Represented as `RetrievedChunk` which extends fundamental chunk data to include `score` metrics extracted from vector distance properties.
- **Query-Response Schema:**
  Managed via Pydantic (`models/schemas.py`).
  Inputs validate strictly: `{"query": str, "session_id": str}`
  Outputs conform to: `{"answer": str, "confidence": float, "needs_hitl": bool, "sources": list}`
- **State Object for Graph:**
  `GraphState` is a `TypedDict`. State persistence guarantees that nodes only emit partial dicts of keys they change, merging back seamlessly to properties like `intent`, `context_string`, `llm_result`, and `routing_reason`.

---

### 3. Workflow Design (LangGraph)

- **Nodes:** 
  1. `input_node` (Sanitization)
  2. `intent_router_node` (Classification)
  3. `retrieval_node` (DB fetch)
  4. `llm_node` (Generative synthesis)
  5. `confidence_eval_node` (Metric checks)
  6. `hitl_node` (Escalation queue)
  7. `greeting_node` (Short-circuit generator)
  8. `output_node` (Final payload assembler)
- **Edges:**
  `input_node` → `intent_router_node` → (Conditional Edge by Intent) → `greeting_node` | `retrieval_node` | `hitl_node`. 
  After generation: `llm_node` → `confidence_eval_node` → (Conditional Edge by Score) → `output_node` | `hitl_node`.
- **State Flow:**
  Data is hydrated through sequential dict-merging.

---

### 4. Conditional Routing Logic

- **Intent Recognition Criteria:**
  Identifies trigger words leveraging regular expressions (e.g. FAQ intent matches "how, when, where"; Escalate matches "manager, legal, human").
- **Answer Generation vs Escalation Criteria:**
  Routing defaults to generating an answer. Escalation conditionally supersedes if the query triggers the `escalate` path, or if post-LLM processing hits a trigger point.
- **Low Confidence & Missing Context:**
  Confidence is calculated as `0.6 * top_score + 0.4 * avg_score - penalty`. If score < 0.50 (or 0.55 for Technical intent), route diverts dynamically to HITL. Missing chunks trigger an immediate automatic 0.0 limit.

---

### 5. HITL Design

- **When escalation is triggered:** If `routing_reason` exists, transition into `hitl_node`. 
- **What happens after escalation:** Execution halts for user response context. A persistent ticket ID is generated locally in `hitl_queue.json`, and the bot informs the user of a timeline.
- **How human response is integrated:** Agents can resolve via `POST /hitl/{id}/resolve`. The API allows the human payload substitution to resolve the active state.

---

### 6. API / Interface Design

- **Endpoints:**
  - `POST /upload`: Mutli-part form upload for knowledge instantiation.
  - `POST /query`: Core agent query interface via JSON body.
  - `POST /hitl/{id}/resolve`: Intervention endpoint.
- **Interaction Flow:**
  Stateless REST calls with pseudo-session consistency via a user-assigned `session_id`.

---

### 7. Error Handling

- **Missing Data:** Ingestion ignores null payloads. Query nodes gracefully handle blank strings using placeholder inputs.
- **No Relevant Chunks Found:** Context resolves to defined fallback strings, immediately punishing confidence multipliers and tripping the safety HITL condition organically.
- **LLM Failure:** Enclosed within `try/except`. Any generation failure triggers the fallback `_mock_call` method, ensuring uptime and returning deterministic sentence extraction if the external provider is down.
