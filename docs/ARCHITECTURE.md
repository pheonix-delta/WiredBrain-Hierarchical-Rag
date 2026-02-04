# WiredBrain Architecture Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [3-Stage Intent Routing](#3-stage-intent-routing)
3. [Hierarchical Addressing](#hierarchical-addressing)
4. [Hybrid Retrieval System](#hybrid-retrieval-system)
5. [Code Usage Examples](#code-usage-examples)

---

## System Overview

WiredBrain is a hierarchical RAG system that solves the "lost in the middle" problem through intelligent query routing and context reduction.

**The Core Problem:**
- Traditional RAG: 693K chunks → Vector search → Top 20 → LLM (context collision, slow)
- WiredBrain: 693K chunks → Intent routing → Gate filtering → 20 chunks → LLM (fast, accurate)

**Key Innovation:** Reduce search space by 99.997% BEFORE vector search using hierarchical addressing.

---

## 3-Stage Intent Routing

WiredBrain uses a **3-stage fallback mechanism** to route queries to the correct knowledge gate:

### Stage 1: SetFit Neural Classification (Primary)
**Model:** SetFit with sentence-transformers/all-MiniLM-L6-v2  
**Accuracy:** 76.67%  
**Latency:** <50ms on CPU  
**Confidence Threshold:** 0.7

```python
# Example: SetFit routing
from src.addressing.gate_router import GateRouter

router = GateRouter()
query = "Explain LQR controller design for quadrotors"

# Stage 1: SetFit classification
result = router.classify(query)
# Output: {
#   "gate": "MATH-CTRL",
#   "confidence": 0.89,
#   "method": "setfit"
# }
```

**How SetFit Works:**
1. Encode query using sentence transformer (768-dim embedding)
2. Few-shot classification head predicts gate (13 classes)
3. Returns gate + confidence score

**Training Data:**
- 13 gates (classes)
- ~100 examples per gate
- 80/20 train/validation split
- Trained on domain-specific queries

### Stage 2: Keyword Matching (Fallback 1)
**Trigger:** SetFit confidence < 0.7  
**Method:** TF-IDF keyword matching against gate taxonomies

```python
# Example: Keyword fallback
query = "STM32 ADC configuration"

# Stage 1 fails (low confidence)
setfit_result = {"gate": "GENERAL", "confidence": 0.45}

# Stage 2: Keyword matching
keyword_result = router.keyword_fallback(query)
# Output: {
#   "gate": "HARD-SPEC",
#   "confidence": 0.82,
#   "method": "keyword",
#   "matched_terms": ["STM32", "ADC", "configuration"]
# }
```

**How Keyword Matching Works:**
1. Extract keywords from query (TF-IDF)
2. Match against gate taxonomy keywords
3. Score by keyword overlap + frequency
4. Return highest-scoring gate

### Stage 3: Semantic Similarity (Fallback 2)
**Trigger:** Keyword matching confidence < 0.6  
**Method:** Cosine similarity between query embedding and gate descriptions

```python
# Example: Semantic fallback
query = "quantum entanglement applications"

# Stage 1 & 2 fail
# Stage 3: Semantic similarity
semantic_result = router.semantic_fallback(query)
# Output: {
#   "gate": "PHYS-QUANT",
#   "confidence": 0.71,
#   "method": "semantic"
# }
```

**How Semantic Similarity Works:**
1. Encode query using sentence transformer
2. Compare with pre-computed gate description embeddings
3. Return gate with highest cosine similarity

### Fallback Chain Visualization

```
User Query
    ↓
┌─────────────────────────────────────┐
│ Stage 1: SetFit Classification      │
│ Confidence > 0.7?                   │
└─────────────────────────────────────┘
    ↓ YES (76.67% of queries)
  [GATE]
    ↓ NO (23.33% of queries)
┌─────────────────────────────────────┐
│ Stage 2: Keyword Matching           │
│ Confidence > 0.6?                   │
└─────────────────────────────────────┘
    ↓ YES (18% of queries)
  [GATE]
    ↓ NO (5.33% of queries)
┌─────────────────────────────────────┐
│ Stage 3: Semantic Similarity        │
│ Always returns a gate               │
└─────────────────────────────────────┘
    ↓
  [GATE]
```

**Coverage:**
- Stage 1 (SetFit): 76.67% of queries
- Stage 2 (Keywords): 18% of queries
- Stage 3 (Semantic): 5.33% of queries
- **Total Success Rate: 100%** (always routes to a gate)

---

## Hierarchical Addressing

Once the gate is determined, WiredBrain uses a 4-level hierarchy to narrow down the search space:

```
Gate (13 options)
  ↓
Branch (~5-10 per gate)
  ↓
Topic (~10-20 per branch)
  ↓
Level (4 options: Beginner, Intermediate, Advanced, Expert)
```

### Example: Full Routing Pipeline

```python
from src.addressing.gate_router import GateRouter
from src.retrieval.hybrid_retriever_v2 import HybridRetriever

# Initialize
router = GateRouter()
retriever = HybridRetriever()

# User query
query = "Design an LQR controller for a quadrotor with 6 DOF"

# Step 1: Route to gate (3-stage fallback)
gate_result = router.route(query)
# Output: {
#   "gate": "MATH-CTRL",
#   "confidence": 0.89,
#   "method": "setfit"
# }

# Step 2: Hierarchical filtering
filtered_chunks = retriever.filter_by_hierarchy(
    gate="MATH-CTRL",
    branch="Control Theory",  # Auto-detected or user-specified
    topic="LQR Design",       # Auto-detected or user-specified
    level="Advanced"          # Auto-detected from query complexity
)
# Reduced from 693,313 to ~2,000 chunks

# Step 3: Vector search within filtered space
results = retriever.retrieve(
    query=query,
    gate="MATH-CTRL",
    top_k=20
)
# Final: 20 most relevant chunks
```

### Search Space Reduction

| Stage | Chunks | Reduction |
|-------|--------|-----------|
| Initial | 693,313 | - |
| After Gate Filtering | ~53,000 | 92.4% |
| After Branch Filtering | ~5,000 | 99.3% |
| After Topic Filtering | ~500 | 99.9% |
| After Vector Search | 20 | 99.997% |

**This is how we solve the "lost in the middle" problem!**

---

## Hybrid Retrieval System

WiredBrain combines three retrieval methods with learned fusion weights:

### 1. Vector Search (Qdrant)
```python
# Vector search within gate
vector_results = retriever.vector_search(
    query="LQR controller design",
    gate="MATH-CTRL",
    top_k=50
)
# Returns: [(chunk_id, score, content), ...]
```

**Details:**
- Embedding model: sentence-transformers/all-mpnet-base-v2 (768-dim)
- Index: HNSW (M=16, ef_construct=100)
- Distance: Cosine similarity
- Latency: <100ms at 693K scale

### 2. Graph Traversal (PostgreSQL)
```python
# Graph-based retrieval
graph_results = retriever.graph_search(
    entity="LQR",
    depth=2,
    max_neighbors=50
)
# Returns: Related chunks via entity relationships
```

**Details:**
- 172,683 entities
- 688,642 relationships (avg 3.99 per entity)
- Traversal: BFS with depth limit
- Enriches results with prerequisites and related concepts

### 3. Quality Scoring
```python
# Re-rank by quality
quality_scores = retriever.score_quality(chunks)
# Quality = 0.4×completeness + 0.3×readability + 0.2×informativeness + 0.1×structure
```

### Fusion Ranking

```python
# Combine all three methods
final_score = 0.5 × vector_score + 0.3 × graph_score + 0.2 × quality_score

# Get top-K
top_k_results = retriever.fuse_and_rank(
    vector_results=vector_results,
    graph_results=graph_results,
    quality_scores=quality_scores,
    k=20
)
```

**Weights optimized on 500 validation queries to maximize NDCG@20**

---

## Code Usage Examples

### Example 1: Basic Query

```python
from src.addressing.gate_router import GateRouter
from src.retrieval.hybrid_retriever_v2 import HybridRetriever

# Initialize
router = GateRouter()
retriever = HybridRetriever(
    qdrant_url="localhost:6333",
    postgres_url="postgresql://localhost:5432/wiredbrain"
)

# Query
query = "How does the A* algorithm work?"

# Route to gate
gate = router.route(query)["gate"]  # "AV-NAV"

# Retrieve
results = retriever.retrieve(query, gate=gate, top_k=20)

# Results structure:
for result in results:
    print(f"Gate: {result['gate']}")
    print(f"Branch: {result['branch']}")
    print(f"Topic: {result['topic']}")
    print(f"Content: {result['content'][:200]}...")
    print(f"Quality: {result['quality_score']}")
    print(f"Entities: {result['entities']}")
    print("---")
```

### Example 2: Multi-Stage Routing with Fallback

```python
# Query that might trigger fallback
query = "quantum computing basics"

# Detailed routing with fallback info
routing_result = router.route_detailed(query)

print(f"Final Gate: {routing_result['gate']}")
print(f"Method Used: {routing_result['method']}")  # 'setfit', 'keyword', or 'semantic'
print(f"Confidence: {routing_result['confidence']}")
print(f"Fallback Chain: {routing_result['fallback_chain']}")
# Output: ['setfit_failed', 'keyword_success']
```

### Example 3: Knowledge Graph Exploration

```python
from src.retrieval.hybrid_retriever_v2 import HybridRetriever

retriever = HybridRetriever()

# Find related entities
related = retriever.get_related_entities(
    entity="LQR",
    relation_types=["USES", "REQUIRES", "IS_A"],
    max_depth=2
)

# Output:
# {
#   "LQR": {
#     "USES": ["Riccati Equation", "State Space Representation"],
#     "REQUIRES": ["Linear Algebra", "Controllability"],
#     "IS_A": ["Optimal Control"]
#   }
# }
```

### Example 4: Full Pipeline with TRM Engine

```python
from src.retrieval.trm_engine_v2 import TRMEngine

# Initialize Transparent Reasoning Module
trm = TRMEngine()

# Query with reasoning
query = "Design an LQR controller for a quadrotor"

# TRM provides x/y/z stream reasoning
reasoning_result = trm.reason(query)

print(f"X-Stream (What): {reasoning_result['x_stream']}")
# "LQR controller design for quadrotor system"

print(f"Y-Stream (Why): {reasoning_result['y_stream']}")
# "Optimal control for multi-input system with stability guarantees"

print(f"Z-Stream (How): {reasoning_result['z_stream']}")
# "Solve Riccati equation for Q/R matrices, compute feedback gain K"

# Retrieve with reasoning context
results = trm.retrieve_with_reasoning(query, top_k=20)
```

---

## Performance Metrics

### Routing Performance
- **SetFit Accuracy:** 76.67%
- **Overall Routing Success:** 100% (with fallback)
- **Avg Routing Latency:** 48ms (Stage 1), 120ms (Stage 2), 85ms (Stage 3)

### Retrieval Performance
- **Latency:** 98ms for top-20 at 693K scale
- **NDCG@20:** 0.842
- **Speedup vs Flat Search:** 13× (1,300ms → 98ms)

### Ablation Study
| Component Removed | Latency Impact | NDCG Impact |
|-------------------|----------------|-------------|
| Hierarchical Filtering | +1,202ms (13×) | -0.044 |
| Graph Traversal | -3ms | -0.031 |
| Quality Scoring | 0ms | -0.017 |
| SetFit Routing | +147ms (2.5×) | -0.079 |

**Conclusion:** Hierarchical filtering + SetFit routing are the most critical components.

---

## Next Steps

1. **Try the Examples:** Run the code snippets above
2. **Explore the Data:** Check `data/samples/sample_data.json`
3. **Read the Paper:** See `docs/WiredBrain_Research_Paper.pdf` for full details
4. **Customize:** Modify gate definitions in `src/addressing/gate_definitions.py`

---

## Questions?

**Contact:**
- Email: 251030181@juitsolan.in, devcoder29cse@gmail.com
- Paper: [WiredBrain_Research_Paper.pdf](../WiredBrain_Research_Paper.pdf)
