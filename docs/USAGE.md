# WiredBrain Usage Guide

## Quick Start (5 Minutes)

### Prerequisites
```bash
# Install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start databases (Docker)
docker-compose up -d
```

### Basic Query Example
```python
from src.addressing.gate_router import GateRouter
from src.retrieval.hybrid_retriever_v2 import HybridRetriever

# Initialize
router = GateRouter()
retriever = HybridRetriever()

# Query
query = "Explain LQR controller design"
gate = router.route(query)["gate"]
results = retriever.retrieve(query, gate=gate, top_k=20)

# Print results
for r in results:
    print(f"{r['gate']}/{r['branch']}/{r['topic']}")
    print(r['content'][:200])
    print("---")
```

---

## Understanding the 3-Stage Routing

WiredBrain uses a **fallback chain** to ensure 100% routing success:

### Stage 1: SetFit (Primary) - 76.67% Success
```python
router = GateRouter()
result = router.classify_setfit(query)

if result['confidence'] > 0.7:
    gate = result['gate']  # Use SetFit result
else:
    # Fall back to Stage 2
```

### Stage 2: Keyword Matching - 18% Success  
```python
result = router.classify_keywords(query)

if result['confidence'] > 0.6:
    gate = result['gate']  # Use keyword result
else:
    # Fall back to Stage 3
```

### Stage 3: Semantic Similarity - 5.33% Success
```python
result = router.classify_semantic(query)
gate = result['gate']  # Always succeeds
```

### Automatic Fallback (Recommended)
```python
# This handles all 3 stages automatically
result = router.route(query)
print(f"Gate: {result['gate']}")
print(f"Method: {result['method']}")  # 'setfit', 'keyword', or 'semantic'
print(f"Confidence: {result['confidence']}")
```

---

## SetFit Training (Optional)

If you want to retrain SetFit for your own gates:

### 1. Prepare Training Data
```python
# Format: List of (text, label) pairs
training_data = [
    ("Design an LQR controller", "MATH-CTRL"),
    ("STM32 ADC configuration", "HARD-SPEC"),
    ("A* path planning algorithm", "AV-NAV"),
    # ... ~100 examples per gate
]
```

### 2. Train SetFit
```python
from setfit import SetFitModel, SetFitTrainer
from datasets import Dataset

# Create dataset
dataset = Dataset.from_dict({
    "text": [x[0] for x in training_data],
    "label": [x[1] for x in training_data]
})

# Initialize model
model = SetFitModel.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2",
    labels=["MATH-CTRL", "HARD-SPEC", "AV-NAV", ...]  # All 13 gates
)

# Train
trainer = SetFitTrainer(
    model=model,
    train_dataset=dataset,
    eval_dataset=dataset,  # Use proper validation split
    num_epochs=3
)
trainer.train()

# Save
model.save_pretrained("models/setfit_gate_classifier")
```

### 3. Use Custom Model
```python
router = GateRouter(setfit_model_path="models/setfit_gate_classifier")
```

---

## Hierarchical Filtering

### Automatic Filtering (Recommended)
```python
# Retriever automatically filters by gate
results = retriever.retrieve(
    query="LQR controller design",
    gate="MATH-CTRL",  # Only search within this gate
    top_k=20
)
```

### Manual Filtering
```python
# Filter by multiple levels
results = retriever.filter_by_hierarchy(
    gate="MATH-CTRL",
    branch="Control Theory",
    topic="LQR Design",
    level="Advanced"
)
# Returns: ~500 chunks instead of 693K
```

---

## Hybrid Retrieval

### Vector Search Only
```python
results = retriever.vector_search(
    query="LQR controller",
    gate="MATH-CTRL",
    top_k=50
)
```

### Graph Search Only
```python
results = retriever.graph_search(
    entity="LQR",
    depth=2,
    max_neighbors=50
)
```

### Hybrid (Recommended)
```python
# Combines vector + graph + quality
results = retriever.retrieve(
    query="LQR controller",
    gate="MATH-CTRL",
    top_k=20,
    use_hybrid=True  # Default
)
```

---

## Knowledge Graph Exploration

### Find Related Entities
```python
related = retriever.get_related_entities(
    entity="LQR",
    relation_types=["USES", "REQUIRES"],
    max_depth=2
)
```

### Traverse Graph
```python
path = retriever.find_path(
    start_entity="LQR",
    end_entity="Riccati Equation",
    max_depth=3
)
```

---

## TRM Reasoning Engine

### Basic Reasoning
```python
from src.retrieval.trm_engine_v2 import TRMEngine

trm = TRMEngine()
reasoning = trm.reason(query="Design LQR controller")

print(f"What: {reasoning['x_stream']}")
print(f"Why: {reasoning['y_stream']}")
print(f"How: {reasoning['z_stream']}")
```

### Retrieve with Reasoning
```python
results = trm.retrieve_with_reasoning(
    query="Design LQR controller",
    top_k=20
)
# Returns chunks enriched with reasoning context
```

---

## Performance Tuning

### Adjust Fusion Weights
```python
retriever = HybridRetriever(
    fusion_weights={
        'vector': 0.5,  # Default
        'graph': 0.3,   # Default
        'quality': 0.2  # Default
    }
)
```

### Adjust Confidence Thresholds
```python
router = GateRouter(
    setfit_threshold=0.7,    # Default
    keyword_threshold=0.6,   # Default
    semantic_threshold=0.5   # Default
)
```

---

## Troubleshooting

### Low Routing Confidence
```python
result = router.route_detailed(query)
print(f"Fallback chain: {result['fallback_chain']}")
# Example: ['setfit_failed', 'keyword_success']
```

### No Results Found
```python
# Try broader gate
results = retriever.retrieve(query, gate="GENERAL", top_k=20)

# Or search all gates (slow)
results = retriever.retrieve_all_gates(query, top_k=20)
```

### Slow Retrieval
```python
# Use hierarchical filtering
results = retriever.retrieve(
    query=query,
    gate="MATH-CTRL",
    branch="Control Theory",  # Narrows search space
    top_k=20
)
```

---

## Advanced: Pipeline Processing

### Run Full Pipeline
```bash
# Stage 1: Download data
python src/pipeline/stage1_acquisition.py --config config.yaml

# Stage 2: Deduplicate
python src/pipeline/stage2_deduplication.py --input data/raw --output data/dedup

# Stage 3: Clean
python src/pipeline/stage3_cleaning.py --input data/dedup --output data/clean

# Stage 4: Classify
python src/pipeline/stage4_classification.py --input data/clean --output data/classified

# Stage 4.5: Extract KG
python src/pipeline/stage4_5_kg_extraction.py --input data/classified --output data/kg

# Stage 6: Populate DBs
python src/pipeline/stage6_db_population.py --input data/kg
```

---

## Next Steps

1. Read [ARCHITECTURE.md](ARCHITECTURE.md) for system design
2. Check [sample_data.json](../data/samples/sample_data.json) for examples
3. See [WiredBrain_Research_Paper.pdf](WiredBrain_Research_Paper.pdf) for full details

**Questions?** Email: 251030181@juitsolan.in, devcoder29cse@gmail.com
