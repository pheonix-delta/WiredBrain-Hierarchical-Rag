"""
Gate Router for Axiom Nexus
Implements smart routing logic with optimization for simple queries
"""

import re
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
from .gate_definitions import Gate, get_all_gates

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Result of gate routing decision"""
    primary_gate: Gate
    secondary_gates: List[Gate]
    confidence: float
    reasoning: str
    skip_vector_search: bool
    use_graph_walk: bool


class GateRouter:
    """
    Smart router that assigns queries to appropriate gates
    Optimized to skip expensive operations for simple queries
    """
    
    def __init__(self):
        self.gates = get_all_gates()
        
        # CRITICAL: Load Neural Router for ML-based routing (76.67% accuracy)
        try:
            from backend.gates.neural_router import NeuralRouter
            self.neural_router = NeuralRouter()
            logger.info("✅ Neural Router loaded (SetFit model)")
        except Exception as e:
            logger.warning(f"Neural Router unavailable, using keyword-only routing: {e}")
            self.neural_router = None
        
        # Smart router patterns (from gates.yaml)
        self.skip_vector_patterns = [
            r"^(hi|hello|hey|thanks|thank you)",
            r"^(yes|no|okay|ok|sure)",
            r"^what is",
        ]
        
        self.force_vector_patterns = [
            r"how to|how do i",
            r"why|explain",
            r"debug|error|issue|problem",
        ]
        
        self.simple_query_max_words = 5
        self.force_graph_walk_min_complexity = 3
    
    def route(self, query: str, use_smart_optimization: bool = True) -> RoutingDecision:
        """
        Route a query to the appropriate gate(s)
        
        PRIORITY:
        1. Neural Router (SetFit - 76.67% accuracy, <50ms)
        2. Keyword matching (fallback)
        
        Args:
            query: User query text
            use_smart_optimization: Enable smart routing optimizations
        
        Returns:
            RoutingDecision with gate assignments and optimization flags
        """
        # Step 1: Check if we should skip expensive operations
        skip_vector = False
        use_graph = True
        
        if use_smart_optimization:
            skip_vector = self._should_skip_vector_search(query)
            use_graph = self._should_use_graph_walk(query)
        
        # Step 2: Try Neural Router first (FAST & ACCURATE)
        if self.neural_router:
            try:
                ml_result = self.neural_router.predict(query)
                
                # Use neural prediction if confidence > 0.6
                if ml_result['confidence'] > 0.6:
                    primary_gate = next(
                        (g for g in self.gates if g.name == ml_result['gate']),
                        self.gates[0]
                    )
                    
                    secondary_gates = [
                        next((g for g in self.gates if g.name == sec_gate), None)
                        for sec_gate in ml_result.get('secondary_gates', [])
                    ]
                    secondary_gates = [g for g in secondary_gates if g is not None]
                    
                    return RoutingDecision(
                        primary_gate=primary_gate,
                        secondary_gates=secondary_gates,
                        confidence=ml_result['confidence'],
                        reasoning=f"Neural Router: {ml_result['confidence']:.2%} confidence",
                        skip_vector_search=skip_vector,
                        use_graph_walk=use_graph
                    )
            except Exception as e:
                logger.warning(f"Neural routing failed, falling back to keywords: {e}")
        
        # Step 3: Fallback to keyword-based gate matching
        gate_scores = self._score_gates_by_keywords(query)
        
        # Step 4: Determine primary and secondary gates
        if not gate_scores:
            # Fallback to ACAD-MOTIV (personality gate)
            primary_gate = next((g for g in self.gates if g.name == "ACAD-MOTIV"), self.gates[0])
            secondary_gates = []
            confidence = 0.3
            reasoning = "No keyword match - fallback to personality gate"
        else:
            # Sort by score
            sorted_gates = sorted(gate_scores.items(), key=lambda x: x[1], reverse=True)
            
            primary_gate = sorted_gates[0][0]
            confidence = sorted_gates[0][1]
            
            # Secondary gates are those with score > 0.3 and not primary
            secondary_gates = [
                gate for gate, score in sorted_gates[1:] 
                if score > 0.3
            ][:2]  # Max 2 secondary gates
            
            reasoning = f"Keyword match: {confidence:.2f} confidence"
        
        return RoutingDecision(
            primary_gate=primary_gate,
            secondary_gates=secondary_gates,
            confidence=confidence,
            reasoning=reasoning,
            skip_vector_search=skip_vector,
            use_graph_walk=use_graph
        )
    
    def _should_skip_vector_search(self, query: str) -> bool:
        """
        Determine if we should skip expensive vector search
        for simple queries that can be answered directly
        """
        query_lower = query.lower().strip()
        
        # Check skip patterns
        for pattern in self.skip_vector_patterns:
            if re.match(pattern, query_lower):
                logger.debug(f"Skipping vector search - matched pattern: {pattern}")
                return True
        
        # Check force patterns (force vector search)
        for pattern in self.force_vector_patterns:
            if re.search(pattern, query_lower):
                logger.debug(f"Forcing vector search - matched pattern: {pattern}")
                return False
        
        # Simple heuristic: very short queries can skip vector search
        word_count = len(query.split())
        if word_count <= self.simple_query_max_words:
            logger.debug(f"Skipping vector search - simple query ({word_count} words)")
            return True
        
        return False
    
    def _should_use_graph_walk(self, query: str) -> bool:
        """
        Determine if graph walk is needed based on query complexity
        """
        # Always use graph walk for complex queries
        complexity = self._estimate_query_complexity(query)
        return complexity >= self.force_graph_walk_min_complexity
    
    def _estimate_query_complexity(self, query: str) -> int:
        """
        Estimate query complexity on scale 1-10
        
        Factors:
        - Length
        - Multiple questions
        - Technical terms
        - Procedural keywords
        """
        complexity = 1
        
        words = query.split()
        word_count = len(words)
        
        # Length factor
        if word_count > 20:
            complexity += 3
        elif word_count > 10:
            complexity += 2
        elif word_count > 5:
            complexity += 1
        
        # Multiple questions
        question_marks = query.count('?')
        if question_marks > 1:
            complexity += 2
        elif question_marks == 1:
            complexity += 1
        
        # Technical/procedural keywords
        complex_keywords = [
            'how', 'why', 'debug', 'error', 'implement', 'integrate',
            'optimize', 'configure', 'troubleshoot', 'explain'
        ]
        matched_keywords = sum(1 for kw in complex_keywords if kw in query.lower())
        complexity += min(matched_keywords, 3)
        
        return min(complexity, 10)
    
    def _score_gates_by_keywords(self, query: str) -> dict:
        """
        Score each gate based on keyword matching
        
        Returns:
            Dict[Gate, float] - gate to confidence score mapping
        """
        query_lower = query.lower()
        scores = {}
        
        for gate in self.gates:
            # Count keyword matches
            matches = sum(1 for kw in gate.keywords if kw in query_lower)
            
            if matches > 0:
                # Score based on match ratio and absolute matches
                score = min(matches / len(gate.keywords) + (matches * 0.1), 1.0)
                scores[gate] = score
        
        return scores
    
    def get_gate_model(self, gate: Gate) -> Tuple[str, str]:
        """
        Get the model and provider for a gate
        
        Returns:
            Tuple[model_name, provider]
        """
        return (gate.specialist_model, gate.provider)
