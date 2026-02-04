"""
Neural Router for WiredBrain - TRAINED MODEL VERSION
Loads the trained SetFit model (76.67% accuracy) for gate classification
"""

import logging
from typing import Dict, List, Optional
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class NeuralRouter:
    """
    ML-based gate router using trained SetFit model.
    
    Loads the model trained with train_neural_router.py (76.67% accuracy)
    Falls back to embedding-based routing if trained model not available.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize neural router with trained SetFit model.
        
        Args:
            model_path: Path to trained model (default: backend/gates/trained_neural_router)
        """
        self.gates = [
            "ROBO-CORE",    # 0
            "PHYS-DYN",     # 1
            "MATH-PROVE",   # 2
            "CODE-GEN",     # 3
            "DATA-TLM",     # 4
            "ACAD-MOTIV",   # 5
            "ELC-HW",       # 6
            "SPACE-AERO",   # 7
            "SEA-SUB",      # 8
            "BIO-MECH",     # 9
            "SFT-SEC"       # 10
        ]
        
        # Default model path
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__),
                "trained_neural_router"
            )
        
        # Try to load trained SetFit model
        self.model = None
        self.use_setfit = False
        
        if os.path.exists(model_path):
            try:
                from setfit import SetFitModel
                logger.info(f"Loading trained SetFit model from: {model_path}")
                self.model = SetFitModel.from_pretrained(model_path)
                self.use_setfit = True
                logger.info(f"✅ Trained SetFit model loaded (76.67% accuracy)")
            except ImportError:
                logger.warning("SetFit not installed. Install with: pip install setfit")
            except Exception as e:
                logger.warning(f"Failed to load SetFit model: {e}")
        else:
            logger.warning(f"Trained model not found at: {model_path}")
        
        # Fallback to embedding-based routing if SetFit not available
        if not self.use_setfit:
            logger.info("Using fallback embedding-based routing")
            from sentence_transformers import SentenceTransformer
            import numpy as np
            self.model = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")
            self.gate_embeddings = self._load_gate_embeddings()
            self.np = np
        
        logger.info(f"✅ Neural Router initialized (SetFit: {self.use_setfit})")
    
    def _load_gate_embeddings(self):
        """Load or create gate embeddings for fallback mode."""
        import numpy as np
        
        cache_path = Path(__file__).parent / "gate_embeddings.npy"
        
        if cache_path.exists():
            logger.info("Loading cached gate embeddings")
            return np.load(cache_path)
        
        # Create embeddings from example queries
        gate_examples = {
            "ROBO-CORE": ["ROS2 workspace", "URDF file", "robot nodes"],
            "PHYS-DYN": ["torque calculation", "kinematics", "dynamics"],
            "MATH-PROVE": ["prove theorem", "eigenvalue", "differential equation"],
            "CODE-GEN": ["write function", "implement algorithm", "generate code"],
            "DATA-TLM": ["parse ulog", "telemetry data", "flight logs"],
            "ACAD-MOTIV": ["feeling stuck", "need help", "motivation"],
            "ELC-HW": ["resistor", "transistor", "circuit design"],
            "SPACE-AERO": ["orbital velocity", "rocket propulsion", "satellite"],
            "SEA-SUB": ["submarine", "underwater", "sonar"],
            "BIO-MECH": ["biomechanics", "prosthetic", "gait analysis"],
            "SFT-SEC": ["vulnerability", "security", "exploit"]
        }
        
        embeddings = []
        for gate in self.gates:
            examples = gate_examples[gate]
            example_embeddings = self.model.encode(examples)
            gate_embedding = np.mean(example_embeddings, axis=0)
            embeddings.append(gate_embedding)
        
        embeddings = np.array(embeddings)
        np.save(cache_path, embeddings)
        logger.info(f"Created gate embeddings: {embeddings.shape}")
        
        return embeddings
    
    def predict(self, query: str) -> Dict:
        """
        Predict the best gate for a query.
        
        Args:
            query: User query text
            
        Returns:
            {
                "gate": "MATH-PROVE",
                "gate_id": 2,
                "confidence": 0.87,
                "secondary_gates": ["PHYS-DYN"],
                "all_scores": {...}
            }
        """
        if self.use_setfit:
            return self._predict_setfit(query)
        else:
            return self._predict_embedding(query)
    
    def _predict_setfit(self, query: str) -> Dict:
        """Predict using trained SetFit model."""
        # SetFit returns class label (0-10)
        prediction = self.model.predict([query])[0]
        gate_id = int(prediction)
        
        # Get probabilities if available
        try:
            probs = self.model.predict_proba([query])[0]
            confidence = float(probs[gate_id])
            
            # Get secondary gates (top 3)
            top_3_indices = self.np.argsort(probs)[-3:][::-1]
            secondary_gates = [
                self.gates[idx] for idx in top_3_indices[1:]
                if probs[idx] > 0.1
            ]
            
            all_scores = {
                gate: float(prob)
                for gate, prob in zip(self.gates, probs)
            }
        except:
            # If predict_proba not available, use simple confidence
            confidence = 0.85
            secondary_gates = []
            all_scores = {gate: 0.0 for gate in self.gates}
            all_scores[self.gates[gate_id]] = confidence
        
        return {
            "gate": self.gates[gate_id],
            "gate_id": gate_id,
            "confidence": confidence,
            "secondary_gates": secondary_gates,
            "all_scores": all_scores
        }
    
    def _predict_embedding(self, query: str) -> Dict:
        """Predict using embedding similarity (fallback)."""
        # Encode query
        query_embedding = self.model.encode([query])[0]
        
        # Calculate cosine similarity
        similarities = self.np.dot(self.gate_embeddings, query_embedding) / (
            self.np.linalg.norm(self.gate_embeddings, axis=1) * self.np.linalg.norm(query_embedding)
        )
        
        # Softmax to get probabilities
        exp_similarities = self.np.exp(similarities - self.np.max(similarities))
        probabilities = exp_similarities / exp_similarities.sum()
        
        # Get best gate
        best_gate_idx = int(self.np.argmax(probabilities))
        confidence = float(probabilities[best_gate_idx])
        
        # Get secondary gates
        top_3_indices = self.np.argsort(probabilities)[-3:][::-1]
        secondary_gates = [
            self.gates[idx] for idx in top_3_indices[1:]
            if probabilities[idx] > 0.1
        ]
        
        return {
            "gate": self.gates[best_gate_idx],
            "gate_id": best_gate_idx,
            "confidence": confidence,
            "secondary_gates": secondary_gates,
            "all_scores": {
                gate: float(prob)
                for gate, prob in zip(self.gates, probabilities)
            }
        }
    
    def predict_multi_gate(self, query: str, threshold: float = 0.15) -> List[Dict]:
        """Predict multiple gates if query spans domains."""
        result = self.predict(query)
        
        multi_gates = []
        for gate, prob in result["all_scores"].items():
            if prob >= threshold:
                gate_id = self.gates.index(gate)
                multi_gates.append({
                    "gate": gate,
                    "gate_id": gate_id,
                    "probability": prob
                })
        
        multi_gates.sort(key=lambda x: x["probability"], reverse=True)
        
        if len(multi_gates) > 1:
            logger.info(f"Multi-gate query: {[g['gate'] for g in multi_gates]}")
        
        return multi_gates


# Singleton instance
_neural_router = None

def get_neural_router() -> NeuralRouter:
    """Get or create the global neural router instance."""
    global _neural_router
    if _neural_router is None:
        _neural_router = NeuralRouter()
    return _neural_router


if __name__ == "__main__":
    # Test the neural router
    router = NeuralRouter()
    
    test_queries = [
        "Calculate the eigenvalue of a matrix",
        "How do I configure a ROS2 workspace?",
        "Write a Python function for binary search",
        "Explain how a transistor works",
        "What is orbital mechanics?",
    ]
    
    print("=" * 80)
    print("NEURAL ROUTER TEST")
    print(f"Using SetFit: {router.use_setfit}")
    print("=" * 80)
    
    for query in test_queries:
        result = router.predict(query)
        print(f"\nQuery: {query}")
        print(f"Gate: {result['gate']} ({result['confidence']:.2%})")
        if result['secondary_gates']:
            print(f"Secondary: {result['secondary_gates']}")
