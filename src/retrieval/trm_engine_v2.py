#!/usr/bin/env python3
"""
WiredBrain V2: TRM (Tiny Reasoning Model) Engine
Implements x/y/z stream architecture with iterative refinement
"""

from typing import Dict, List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from groq import Groq
import logging
import numpy as np
import uuid
import os
from dotenv import load_dotenv

# Load environment variables from backend/.env
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5433'),  # WiredBrain uses 5433
    'database': os.getenv('DB_NAME_V2', 'wiredbrain'),  # FIXED: Use actual database with 693K chunks
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres123')  # FIXED: Actual password
}

class TRMEngineV2:
    """Tiny Reasoning Model with x/y/z streams"""
    
    def __init__(self, retriever):
        self.retriever = retriever  # HybridRetrieverV2 instance
        self._client = None  # Lazy-loaded Groq client
        
        # PostgreSQL for trace storage
        self.pg_conn = psycopg2.connect(**DB_CONFIG)
        
        logger.info("✅ TRM Engine V2 initialized")
    
    @property
    def client(self):
        """Lazy-load Groq client to ensure dotenv is loaded"""
        if self._client is None:
            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                logger.warning("GROQ_API_KEY not found, checking .env file...")
                from dotenv import load_dotenv
                load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
                api_key = os.getenv('GROQ_API_KEY')
            self._client = Groq(api_key=api_key)
            logger.info(f"Groq client initialized (key: {api_key[:10]}...)")
        return self._client
    
    def solve_with_trm(
        self,
        problem: str,
        gate_id: int,
        user_level: str = "intermediate",
        max_iterations: int = 10
    ) -> Dict:
        """
        Solve problem using TRM approach
        
        Returns:
            {
                "answer": final_answer,
                "reasoning_trace": [...],
                "iterations": n,
                "confidence": score,
                "verification": {...},
                "query_id": uuid
            }
        """
        
        # Generate unique query ID
        query_id = str(uuid.uuid4())
        
        # TRM streams
        x_stream = problem  # Immutable original problem
        y_stream = ""       # Current answer (updated each iteration)
        z_stream = []       # Reasoning trace (accumulated)
        
        iteration = 0
        
        logger.info(f"Starting TRM reasoning for query: {query_id}")
        
        while iteration < max_iterations:
            iteration += 1
            
            logger.info(f"Iteration {iteration}/{max_iterations}")
            
            # Step 1: Retrieve relevant context
            context = self.retriever.hybrid_retrieve(
                query=f"{x_stream}\nCurrent thinking: {y_stream}",
                gate_id=gate_id,
                user_level=user_level,
                top_k=5
            )
            
            if not context:
                logger.warning("No context retrieved, using general reasoning")
            
            # Step 2: Generate next reasoning step
            z_new = self.generate_reasoning_step(
                x_stream, y_stream, z_stream, context
            )
            z_stream.append(z_new)
            
            # Step 3: Synthesize current answer
            y_stream = self.synthesize_answer(
                x_stream, z_stream, context
            )
            
            # Step 4: Verify solution
            verification = self.verify_solution(
                y_stream, gate_id, context
            )
            
            # Store trace
            self.store_trace(
                query_id, iteration,
                x_stream, y_stream, z_stream,
                context, verification
            )
            
            if not verification["passed"]:
                # Backtrack
                z_stream.append(
                    f"❌ Verification failed: {verification['reason']}"
                )
                z_stream.append("🔄 Trying different approach...")
                continue
            
            # Step 5: Check confidence (Gaussian sampling)
            confident, variance = self.gaussian_confidence_check(
                y_stream, x_stream
            )
            
            if confident:
                logger.info(f"✅ Confident after {iteration} iterations")
                
                return {
                    "answer": y_stream,
                    "reasoning_trace": z_stream,
                    "iterations": iteration,
                    "confidence": 1.0 - variance,
                    "verification": verification,
                    "query_id": query_id,
                    "status": "success"
                }
        
        # Max iterations reached
        logger.warning(f"Max iterations ({max_iterations}) reached")
        
        return {
            "answer": y_stream,
            "reasoning_trace": z_stream,
            "iterations": max_iterations,
            "confidence": 0.5,
            "verification": verification,
            "query_id": query_id,
            "status": "uncertain"
        }
    
    def generate_reasoning_step(
        self,
        x_stream: str,
        y_stream: str,
        z_stream: List[str],
        context: List[Dict]
    ) -> str:
        """Generate next reasoning step using LLM"""
        
        # Format context
        context_str = "\n\n".join([
            f"[{c['node_type']}] {c['display_name']}\n{c.get('content_toon', '')}"
            for c in context[:3]
        ])
        
        # Format previous reasoning
        prev_reasoning = "\n".join([
            f"{i+1}. {step}"
            for i, step in enumerate(z_stream[-3:])  # Last 3 steps
        ])
        
        prompt = f"""You are solving this problem step-by-step.

PROBLEM:
{x_stream}

CURRENT ANSWER:
{y_stream or "[Not yet formulated]"}

PREVIOUS REASONING STEPS:
{prev_reasoning or "[None yet]"}

RELEVANT KNOWLEDGE:
{context_str or "[No specific knowledge retrieved]"}

Generate the NEXT reasoning step. Be specific and concise.
Focus on one logical step forward.

NEXT STEP:"""
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return f"[Error generating reasoning step: {e}]"
    
    def synthesize_answer(
        self,
        x_stream: str,
        z_stream: List[str],
        context: List[Dict]
    ) -> str:
        """Synthesize current answer from reasoning trace"""
        
        reasoning_summary = "\n".join([
            f"{i+1}. {step}"
            for i, step in enumerate(z_stream)
        ])
        
        prompt = f"""Based on this reasoning trace, synthesize the current answer.

PROBLEM:
{x_stream}

REASONING STEPS:
{reasoning_summary}

Provide a clear, concise answer. If solution is incomplete, state what's known so far.

ANSWER:"""
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # Lower temperature for synthesis
                max_tokens=300
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return f"[Error synthesizing answer: {e}]"
    
    def verify_solution(
        self,
        answer: str,
        gate_id: int,
        context: List[Dict]
    ) -> Dict:
        """Gate-specific verification"""
        
        # Get verifiers from context
        verifiers = []
        for node in context:
            verifiers.extend(node.get("verifiers", []))
        
        if not verifiers:
            return {
                "passed": True,
                "method": "none",
                "reason": "No verifiers available"
            }
        
        # Use first verifier
        verifier = verifiers[0]
        
        # TODO: Implement actual verification
        # For now, placeholder (always pass)
        
        return {
            "passed": True,
            "method": verifier.get("verification_method", "manual"),
            "verifier_used": verifier.get("display_name"),
            "reason": "Verification passed (placeholder)"
        }
    
    def gaussian_confidence_check(
        self,
        answer: str,
        problem: str,
        num_samples: int = 4
    ) -> tuple:
        """
        Multi-temperature sampling for confidence estimation
        
        Returns:
            (confident: bool, variance: float)
        """
        
        samples = []
        
        for temp in [0.2, 0.4, 0.6, 0.8]:
            # Re-generate answer at different temperature
            prompt = f"""Problem: {problem}

Provide a concise answer (one sentence).

Answer:"""
            
            try:
                response = self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temp,
                    max_tokens=100
                )
                
                sample_answer = response.choices[0].message.content.strip()
                
                # Calculate similarity to main answer (simple: word overlap)
                main_words = set(answer.lower().split())
                sample_words = set(sample_answer.lower().split())
                
                if main_words and sample_words:
                    similarity = len(main_words & sample_words) / len(main_words | sample_words)
                else:
                    similarity = 0.0
                
                samples.append(similarity)
            
            except Exception as e:
                logger.error(f"Gaussian sampling error: {e}")
                samples.append(0.5)  # Default
        
        mean = np.mean(samples)
        variance = np.var(samples)
        
        # High confidence if variance is low
        confident = variance < 0.05
        
        logger.info(f"Confidence: mean={mean:.3f}, variance={variance:.3f}, confident={confident}")
        
        return confident, variance
    
    def store_trace(
        self,
        query_id: str,
        iteration: int,
        x_stream: str,
        y_stream: str,
        z_stream: List[str],
        context: List[Dict],
        verification: Dict
    ):
        """Store reasoning trace in database"""
        
        cursor = self.pg_conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO reasoning_traces_v2 (
                    query_id, iteration,
                    x_stream, y_stream, z_stream,
                    nodes_used,
                    verification_passed, verification_details
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (query_id, iteration) DO NOTHING
            """, (
                query_id,
                iteration,
                x_stream,
                y_stream,
                z_stream,
                [c["node_id"] for c in context],
                verification["passed"],
                str(verification)
            ))
            
            self.pg_conn.commit()
        
        except Exception as e:
            logger.error(f"Error storing trace: {e}")
            self.pg_conn.rollback()
        
        finally:
            cursor.close()
    
    def close(self):
        """Cleanup"""
        if self.pg_conn:
            self.pg_conn.close()

# ============================================
# TEST TRM ENGINE
# ============================================

if __name__ == "__main__":
    from backend.retrieval_v2.hybrid_retriever_v2 import HybridRetrieverV2
    
    retriever = HybridRetrieverV2()
    trm = TRMEngineV2(retriever)
    
    try:
        # Test problem
        problem = """
A quadcopter exhibits oscillation in yaw axis during hover.
Using Ziegler-Nichols method, the ultimate gain Ku = 2.5 
and ultimate period Tu = 0.8 seconds were determined.
Calculate the PID controller gains (Kp, Ki, Kd).
        """
        
        print("="*60)
        print("🚀 TESTING TRM ENGINE V2")
        print("="*60)
        
        result = trm.solve_with_trm(
            problem=problem,
            gate_id=6,  # AV-NAV
            user_level="intermediate",
            max_iterations=5
        )
        
        print("\n" + "="*60)
        print("RESULT")
        print("="*60)
        print(f"Status: {result['status']}")
        print(f"Answer: {result['answer']}")
        print(f"Iterations: {result['iterations']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Verification: {result['verification']['passed']}")
        
        print("\n" + "="*60)
        print("REASONING TRACE")
        print("="*60)
        for i, step in enumerate(result['reasoning_trace'], 1):
            print(f"{i}. {step}")
        
        print("\n" + "="*60)
        print("✅ TRM TEST COMPLETE")
        print("="*60)
    
    finally:
        trm.close()
        retriever.close()
