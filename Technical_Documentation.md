# Technical Documentation
## Retrieval-Augmented Generation Customer Support Assistant

---

### 1. Introduction
**What is RAG?**
Retrieval-Augmented Generation (RAG) is an architectural approach that couples the vast language generation capabilities of Large Language Models (LLMs) with targeted, external data stores. Instead of relying solely on the pre-trained weights of the model, RAG actively searches a localized knowledge base during run-time, appending relevant text to the AI’s generative prompt.
**Why it is needed?**
For enterprise customer support, LLMs hallucinate policies, pricing, and specific instructional manuals. RAG effectively grounds the model to verifiable reality.
**Use Case Overview:**
This project deploys a support chatbot answering customer operations questions precisely scoped to internal PDF guidelines, escalating automatically when confused.

---

### 2. System Architecture Explanation
The High-Level Architecture centers natively on an ingestion pipeline transforming unstructured PDFs into normalized vector representations. These representations persist securely within ChromaDB. When a query is initiated over the FastAPI REST interface, LangGraph assumes centralized control. It determines if semantic vector search is necessary, probes the DB, formats structured instructions, queries OpenAI, and mathematically quantifies its own confidence prior to exposing the response to the frontend client. Component relations are strictly defined via stateless service classes interacting across decoupled logic models.

---

### 3. Design Decisions
- **Chunk Size Choice:** Configured to an aggressive ~500 word span with a 50-word sliding overlap buffer (`setting.CHUNK_SIZE`, `setting.CHUNK_OVERLAP`). This bounds response coherence, ensuring chunk granularity isn't so tiny it loses noun subjects, while remaining small enough to permit `top-k=5` fetching without blowing contextual token limits.
- **Embedding Strategy:** Abstracting the base class permits hardware-agnostic operation. Production can utilize 1536-dimensional Ada-002 models, whereas CI/CD or low-resource offline nodes safely downgrade to robust MinLM or TF-IDF.
- **Retrieval Approach:** Employs explicit K-distance parameters based on contextual classification (e.g. technical queries expand search radius parameters dynamically).
- **Prompt Design Logic:** System prompts stringently employ negation logic bounds (e.g., "NEVER invent facts..."). Iterative history (T-3) is injected to establish conversational context windows seamlessly.

---

### 4. Workflow Explanation
The `graph.py` core establishes the `LangGraph` topology. 
State is exclusively tracked through a globally typed dict `GraphState`. 
- **Node Responsibilities:** Nodes execute singularly on atomic actions (e.g., `retrieval_node` only fetches content, `confidence_eval_node` purely processes numerical output logic without mutation of intent structures). 
- **State Transitions:** Edges evaluate the dict. Explicit mutator mappings control logical flows preventing deadlocks. If `state["needs_hitl"]` is flipped dynamically based on evaluating the node score below the threshold variable, edge routing organically switches out the endpoint response mapping.

---

### 5. Conditional Logic
- **Intent Detection:** Applied via lightweight, offline Regular Expressions in `rag_pipeline.py`. Pattern clusters immediately sort interactions prior to vector resource consumption.
- **Routing Decisions:** LangGraph conditionally checks thresholds. 0.72 standard threshold; 0.80 for more risky technical documentation queries.

---

### 6. HITL Implementation
- **Role of Human Intervention:** Acting as the fail-safe backstop preventing dangerous hallucinations or unresolved critical user anger.
- **Benefits and Limitations:** Increases response accuracy guarantees significantly. The key limitation is that synchronous user waiting scales inversely with available workforce queues. As implemented, the file-lock queue serves as simplified persistence but would require database clustering for concurrent multi-agent architectures in production.

---

### 7. Challenges & Trade-offs
- **Retrieval Accuracy vs Speed:** Increased contextual fetch values (`K>10`) generate expansive multi-doc validation matrices but critically augment processing time and token computation costs.
- **Chunk Size vs Context Quality:** Micro chunks map perfectly in cosine space but lack procedural instruction flow. A 500-unit cap strikes the desired trade-off bounds.
- **Cost vs Performance:** Fallback methods (`smart-mock`) effectively drop operational costs to zero when necessary, trading conversational fluidity for explicit document snippet extraction.

---

### 8. Testing Strategy
Our internal `test_all.py` runs over 30 isolated integration protocols.
- **Approach:** Validates chunk boundary edge conditions, tests intent routers utilizing synthetic matrix data, and probes HITL endpoint resilience mechanisms.
- **Sample Queries:** Assesses boundary inputs like short-text (`"Hello"` -> forces greeting bypass) vs complex ambiguous inputs indicating missing constraints to trigger the Escalation graph nodes flawlessly.

---

### 9. Future Enhancements
- **Multi-Document Support:** Expanding metadata schemas to track extensive file libraries and allow session-directed document limitation scope filters.
- **Feedback Loop:** Instantiating user `thumbs_up/down` endpoints that recursively alter MMR priority retrieval weights dynamically over long-term usage.
- **Deployment:** Transitioning JSON queue stores into Postgres architectures and deploying graph nodes over asynchronous micro-service clusters using Docker/Kubernetes combinations.
