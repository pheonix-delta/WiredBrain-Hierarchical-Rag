# SetFit Gate Classification Training Guide

## Overview

SetFit (Sentence Transformer Fine-Tuning) is used for **Stage 1** of the 3-stage routing fallback. It achieves 76.67% accuracy with <50ms latency.

---

## Why SetFit?

1. **Few-Shot Learning:** Works with ~100 examples per class (vs 1000s for traditional fine-tuning)
2. **Fast Inference:** <50ms on CPU (no GPU needed)
3. **Small Model:** 22M parameters (sentence-transformers/all-MiniLM-L6-v2)
4. **High Accuracy:** 76.67% on 13-class gate classification

---

## Training Data Format

```python
# training_data.json
[
    {
        "text": "Design an LQR controller for a quadrotor",
        "label": "MATH-CTRL"
    },
    {
        "text": "STM32F4 ADC configuration and DMA setup",
        "label": "HARD-SPEC"
    },
    {
        "text": "A* algorithm implementation for path planning",
        "label": "AV-NAV"
    },
    # ... ~100 examples per gate (1300 total for 13 gates)
]
```

### Data Collection Tips

1. **Diverse Queries:** Include beginner, intermediate, and advanced queries
2. **Natural Language:** Use how users actually ask questions
3. **Domain-Specific:** Include technical jargon and acronyms
4. **Balanced:** ~100 examples per gate (min 50, max 150)

---

## Training Script

```python
from setfit import SetFitModel, SetFitTrainer
from datasets import Dataset
import json

# 1. Load training data
with open('training_data.json') as f:
    data = json.load(f)

# 2. Create dataset
dataset = Dataset.from_dict({
    "text": [x['text'] for x in data],
    "label": [x['label'] for x in data]
})

# 3. Split train/validation (80/20)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# 4. Initialize SetFit model
model = SetFitModel.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2",
    labels=[
        "GENERAL", "MATH-CTRL", "HARD-SPEC", "SYS-OPS",
        "CHEM-BIO", "OLYMPIAD", "SPACE-AERO", "CODE-GEN",
        "PHYS-DYN", "TELEM-LOG", "AV-NAV", "PHYS-QUANT", "CS-AI"
    ]
)

# 5. Create trainer
trainer = SetFitTrainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    num_epochs=3,
    batch_size=16,
    num_iterations=20  # Few-shot iterations
)

# 6. Train
trainer.train()

# 7. Evaluate
metrics = trainer.evaluate()
print(f"Accuracy: {metrics['accuracy']:.4f}")

# 8. Save model
model.save_pretrained("models/setfit_gate_classifier")
```

---

## Evaluation

```python
from setfit import SetFitModel

# Load model
model = SetFitModel.from_pretrained("models/setfit_gate_classifier")

# Test queries
test_queries = [
    "Explain LQR controller design",
    "STM32 timer configuration",
    "Quantum entanglement basics"
]

# Predict
for query in test_queries:
    prediction = model.predict([query])[0]
    probs = model.predict_proba([query])[0]
    confidence = max(probs)
    
    print(f"Query: {query}")
    print(f"Gate: {prediction}")
    print(f"Confidence: {confidence:.3f}")
    print("---")
```

---

## Integration with WiredBrain

```python
from src.addressing.gate_router import GateRouter

# Use custom SetFit model
router = GateRouter(setfit_model_path="models/setfit_gate_classifier")

# Route query
result = router.route("Design an LQR controller")
print(f"Gate: {result['gate']}")
print(f"Confidence: {result['confidence']}")
```

---

## Performance Tuning

### 1. Adjust Confidence Threshold
```python
router = GateRouter(setfit_threshold=0.75)  # Default: 0.7
```

### 2. Increase Training Data
- More examples per gate → Higher accuracy
- Target: 100-150 examples per gate

### 3. Fine-Tune Hyperparameters
```python
trainer = SetFitTrainer(
    model=model,
    train_dataset=dataset['train'],
    num_epochs=5,        # Default: 3
    batch_size=32,       # Default: 16
    num_iterations=30    # Default: 20
)
```

---

## Current Performance (WiredBrain)

| Metric | Value |
|--------|-------|
| **Accuracy** | 76.67% |
| **Latency** | <50ms (CPU) |
| **Training Data** | ~1300 examples (13 gates) |
| **Model Size** | 22M parameters |
| **Success Rate** | 76.67% (Stage 1 only) |
| **Overall Success** | 100% (with 3-stage fallback) |

---

## Fallback Chain

SetFit is **Stage 1** of the 3-stage routing:

```
Query → SetFit (76.67% success)
    ↓ (if confidence < 0.7)
Keyword Matching (18% success)
    ↓ (if confidence < 0.6)
Semantic Similarity (5.33% success)
    ↓
100% Success Rate
```

---

## Common Issues

### Issue: Low Accuracy
**Solution:** Add more training examples, especially for confused gates

### Issue: Slow Inference
**Solution:** Use smaller model or GPU acceleration

### Issue: Imbalanced Classes
**Solution:** Ensure ~100 examples per gate, use class weights

---

## Next Steps

1. Collect domain-specific training data
2. Train SetFit model
3. Evaluate on validation set
4. Integrate with GateRouter
5. Monitor performance in production

**Questions?** Email: 251030181@juitsolan.in, devcoder29cse@gmail.com
