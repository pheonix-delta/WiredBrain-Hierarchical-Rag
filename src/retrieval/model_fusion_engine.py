"""
Model Fusion Engine - Personality Integration

Applies personality metadata from retrieved chunks to response generation.
Implements the vision from MODEL_FUSION_AND_PERSONALITY.md.
"""

from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ModelFusionEngine:
    """
    Applies personality weighting and tone adjustment to responses.
    
    Reads personality JSONB metadata from chunks and adjusts response style.
    """
    
    def __init__(self):
        """Initialize fusion engine"""
        logger.info("✅ ModelFusionEngine initialized")
    
    def _aggregate_personality(self, chunks: List[Dict]) -> Dict[str, float]:
        """
        Extract and average personality scores from chunk metadata.
        
        Args:
            chunks: List of retrieved chunks with metadata
            
        Returns:
            Dict of personality traits to scores (0.0-1.0)
            Example: {"warm": 0.8, "encouraging": 0.7, "playful": 0.3}
        """
        scores: Dict[str, float] = {}
        count = 0
        
        for chunk in chunks:
            # Handle both dict and object-style chunks
            if isinstance(chunk, dict):
                metadata = chunk.get("metadata", {}) or {}
            else:
                metadata = getattr(chunk, "metadata", {}) or {}
            
            personality = metadata.get("personality")
            if not personality:
                continue
            
            count += 1
            for trait, value in personality.items():
                scores[trait] = scores.get(trait, 0.0) + float(value)
        
        if count == 0:
            # Default personality if no metadata found
            return {
                "warm": 0.7,
                "helpful": 0.9,
                "clear": 0.8,
                "encouraging": 0.6
            }
        
        # Average the scores
        return {trait: value / count for trait, value in scores.items()}
    
    def apply_personality_and_fusion(
        self,
        answer_text: str,
        retrieved_chunks: List[Dict],
        gate: Dict[str, Any]
    ) -> str:
        """
        Apply personality tone to the answer.
        
        Args:
            answer_text: Base answer from TRM reasoning
            retrieved_chunks: Chunks used for context (with personality metadata)
            gate: Gate configuration
            
        Returns:
            Answer with personality-adjusted tone
        """
        # Aggregate personality from chunks
        personality_scores = self._aggregate_personality(retrieved_chunks)
        
        # Build style description
        style_parts = []
        
        if personality_scores.get("warm", 0) > 0.5:
            style_parts.append("warm")
        if personality_scores.get("encouraging", 0) > 0.5:
            style_parts.append("encouraging")
        if personality_scores.get("playful", 0) > 0.4:
            style_parts.append("lightly playful")
        if personality_scores.get("clear", 0) > 0.6:
            style_parts.append("clear")
        if personality_scores.get("concise", 0) > 0.5:
            style_parts.append("concise")
        
        style_note = ", ".join(style_parts) if style_parts else "helpful and clear"
        
        # For now, we prepend a tone note
        # Future enhancement: Use LLM to rewrite answer in specified tone
        logger.info(f"Applied personality: {style_note}")
        
        # Simple implementation: just return the answer
        # The personality is already baked into the system prompt
        # This function is a hook for future LLM-based style rewriting
        return answer_text
    
    def get_greeting(self, personality_scores: Dict[str, float] = None) -> str:
        """
        Generate a personality-aware greeting.
        
        Args:
            personality_scores: Optional personality trait scores
            
        Returns:
            Greeting message
        """
        if not personality_scores:
            personality_scores = {
                "warm": 0.7,
                "helpful": 0.9,
                "playful": 0.3
            }
        
        # Choose greeting based on personality
        if personality_scores.get("playful", 0) > 0.6:
            return "Hey there! I'm WiredBrain 🧠 What are we exploring today?"
        elif personality_scores.get("warm", 0) > 0.6:
            return "Hi! I'm WiredBrain, your friendly AI assistant. How can I help you today?"
        else:
            return "Hello! I'm WiredBrain. What would you like to learn about?"
