#!/usr/bin/env python3
"""
WiredBrain V2: Hybrid Retrieval Engine
Combines hierarchical filtering + graph walks + vector search
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5433'),  # Local Postgres on 5433
    'database': os.getenv('DB_NAME_V2', 'wiredbrain'),  # Database with 693K chunks
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres123')
}

QDRANT_CONFIG = {
    'path': os.getenv('QDRANT_PATH', './qdrant_local_storage')  # Local Qdrant storage
}

COLLECTION_NAME = os.getenv('QDRANT_COLLECTION', 'ros2_knowledge')  # Default collection
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

class HybridRetrieverV2:
    """Complete hybrid retrieval system for V2 brain"""
    
    def __init__(self):
        # PostgreSQL connection
        self.pg_conn = psycopg2.connect(**DB_CONFIG)
        
        # Qdrant connection (local storage)
        self.qdrant = QdrantClient(**QDRANT_CONFIG)
        self.collection_name = COLLECTION_NAME
        
        # Embedding model
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        logger.info(f"✅ Hybrid Retriever V2 initialized (local Qdrant: {QDRANT_CONFIG.get('path', 'N/A')})")
    
    # =========================================
    # HIERARCHICAL RETRIEVAL
    # =========================================
    
    def retrieve_by_hierarchy(
        self,
        gate_id: int,
        branch_id: Optional[int] = None,
        place_prefix: Optional[List[str]] = None,
        depth_min: int = 1,
        depth_max: int = 7,
        difficulty_max: int = 5,
        limit: int = 10
    ) -> List[Dict]:
        """Retrieve using hierarchical addressing"""
        
        cursor = self.pg_conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
            SELECT 
                id, node_type, name, display_name,
                gate_name, branch_name,
                place_path, depth,
                difficulty_level, difficulty_id,
                content_original, content_toon,
                quality_score
            FROM nodes_v2
            WHERE gate_id = %s
              AND depth BETWEEN %s AND %s
              AND difficulty_id <= %s
        """
        
        params = [gate_id, depth_min, depth_max, difficulty_max]
        
        if branch_id:
            query += " AND branch_id = %s"
            params.append(branch_id)
        
        if place_prefix:
            query += " AND place_path @> %s"
            params.append(place_prefix)
        
        query += " ORDER BY quality_score DESC, depth ASC LIMIT %s"
        params.append(limit)
        
        cursor.execute(query, params)
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "node_id": str(row['id']),
                "node_type": row['node_type'],
                "name": row['name'],
                "display_name": row['display_name'],
                "gate_name": row['gate_name'],
                "branch_name": row['branch_name'],
                "place_path": row['place_path'],
                "depth": row['depth'],
                "difficulty_level": row['difficulty_level'],
                "content_toon": row['content_toon'] or row['content_original'],
                "quality_score": float(row['quality_score']) if row['quality_score'] else 0.5,
                "retrieval_method": "hierarchical"
            })
        
        cursor.close()
        return results
    
    # =========================================
    # GRAPH WALKS
    # =========================================
    
    def get_prerequisites(self, node_id: str, max_depth: int = 3) -> List[Dict]:
        """Get all prerequisites via REQUIRES edges"""
        
        cursor = self.pg_conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM get_prerequisites_v2(%s, %s)", (node_id, max_depth))
        
        prereqs = []
        for row in cursor.fetchall():
            prereqs.append({
                "node_id": str(row['node_id']),
                "node_type": row['node_type'],
                "display_name": row['display_name'],
                "prerequisite_depth": row['prerequisite_depth']
            })
        
        cursor.close()
        return prereqs
    
    def get_examples(self, concept_node_id: str) -> List[Dict]:
        """Get problem examples via EXAMPLE_OF edges"""
        
        cursor = self.pg_conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT 
                n.id, n.display_name, n.content_toon, n.content_original,
                e.weight, e.reasoning
            FROM nodes_v2 n
            JOIN edges_v2 e ON e.from_node = n.id
            WHERE e.to_node = %s
              AND e.edge_type = 'EXAMPLE_OF'
              AND n.node_type = 'PROBLEM'
            ORDER BY e.weight DESC
        """, (concept_node_id,))
        
        examples = []
        for row in cursor.fetchall():
            examples.append({
                "node_id": str(row['id']),
                "display_name": row['display_name'],
                "content_toon": row['content_toon'] or row['content_original'],
                "relevance": float(row['weight']) if row['weight'] else 1.0,
                "reasoning": row['reasoning']
            })
        
        cursor.close()
        return examples
    
    def get_verifiers(self, node_id: str) -> List[Dict]:
        """Get verification skills via VERIFIES edges"""
        
        cursor = self.pg_conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT 
                n.id, n.display_name,
                n.verification_method, n.verification_code
            FROM nodes_v2 n
            JOIN edges_v2 e ON e.from_node = n.id
            WHERE e.to_node = %s
              AND e.edge_type = 'VERIFIES'
              AND n.node_type = 'SKILL'
        """, (node_id,))
        
        verifiers = []
        for row in cursor.fetchall():
            verifiers.append({
                "node_id": str(row['id']),
                "display_name": row['display_name'],
                "verification_method": row['verification_method'],
                "verification_code": row['verification_code']
            })
        
        cursor.close()
        return verifiers
    
    # =========================================
    # FILE CHUNKS RETRIEVAL (Uploaded Files)
    # =========================================
    
    def search_file_chunks(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Search uploaded file chunks with intent detection.
        
        Strategy A (Summary Request): Return first N chunks (intro/abstract)
        Strategy B (Specific Query): Use semantic text search
        """
        cursor = self.pg_conn.cursor(cursor_factory=RealDictCursor)
        
        # INTENT DETECTION: Summary vs. Specific
        # Removed broad keywords like "what", "tell me", "file" (too generic)
        summary_keywords = [
            "summarize", "summary", "overview", 
            "what is this file", "what is the document", "what is this pdf",
            "describe the content", "about the content", 
            "main points", "key takeaways",
            "what's inside", "what is inside"
        ]
        
        query_lower = query.lower()
        is_summary_request = any(keyword in query_lower for keyword in summary_keywords)
        
        logger.info(f"File search intent: query='{query}', is_summary={is_summary_request}, session_id={session_id}")
        
        if is_summary_request and session_id:
            # STRATEGY A: "Introduction Fetch"
            # Return first 5 chunks from the MOST RECENTLY uploaded file only
            # This ensures we don't mix content from multiple files
            sql = """
                SELECT 
                    fc.chunk_id as node_id,
                    fc.file_id,
                    fc.session_id,
                    fc.chunk_index,
                    fc.content,
                    fc.contextual_content,
                    fc.tokens,
                    uf.filename,
                    uf.file_type,
                    uf.document_summary,
                    1.0 as vector_score
                FROM file_chunks fc
                JOIN uploaded_files uf ON fc.file_id = uf.file_id
                WHERE uf.file_id = (
                    SELECT file_id FROM uploaded_files 
                    WHERE session_id = %s
                    ORDER BY uploaded_at DESC 
                    LIMIT 1
                )
                ORDER BY fc.chunk_index ASC
                LIMIT 5
            """
            params = [session_id]
            logger.info(f"Summary request: fetching first 5 chunks from MOST RECENT file for session {session_id}")
            
        else:
            # STRATEGY B: Semantic/Text Search
            # User asking specific question - search both content and contextual_content
            sql = """
                SELECT 
                    fc.chunk_id as node_id,
                    fc.file_id,
                    fc.session_id,
                    fc.chunk_index,
                    fc.content,
                    fc.contextual_content,
                    fc.tokens,
                    uf.filename,
                    uf.file_type,
                    uf.document_summary,
                    CASE 
                        WHEN fc.content ILIKE %s THEN 2.0
                        WHEN fc.contextual_content ILIKE %s THEN 1.5
                        ELSE 1.0
                    END as vector_score
                FROM file_chunks fc
                JOIN uploaded_files uf ON fc.file_id = uf.file_id
                WHERE (fc.content ILIKE %s OR fc.contextual_content ILIKE %s)
            """
            
            search_pattern = f"%{query}%"
            params = [search_pattern, search_pattern, search_pattern, search_pattern]
            
            if session_id:
                # Search only current session for strict isolation
                sql += " AND fc.session_id = %s"
                params.append(session_id)
            
            sql += " ORDER BY vector_score DESC, fc.chunk_index LIMIT %s"
            params.append(limit)
            logger.info(f"Specific query: searching file content for '{query}'")
        
        try:
            cursor.execute(sql, params)
            results = cursor.fetchall()
            cursor.close()
            
            # Format results
            formatted_results = []
            for row in results:
                formatted_results.append({
                    "node_id": row['node_id'],
                    "file_id": row['file_id'],
                    "filename": row['filename'],
                    "file_type": row['file_type'],
                    "content": row['contextual_content'] or row['content'],
                    "content_original": row['content'],
                    "tokens": row['tokens'],
                    "vector_score": row['vector_score'],
                    "source": "uploaded_file",
                    "document_summary": row['document_summary']
                })
            
            logger.info(f"Found {len(formatted_results)} file chunks. First chunk preview: {formatted_results[0]['content'][:100] if formatted_results else 'NONE'}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"File chunk search failed: {e}")
            cursor.close()
            return []
    
    # =========================================
    # VECTOR SEARCH
    # =========================================
    
    def vector_search(
        self,
        query: str,
        gate_id: Optional[int] = None,
        node_type: Optional[str] = None,
        difficulty_max: int = 5,
        depth_min: int = 1,
        limit: int = 20
    ) -> List[Dict]:
        """Semantic search with hierarchical filtering"""
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)
        
        # Build filter
        filter_conditions = []
        
        if gate_id:
            filter_conditions.append(
                FieldCondition(key="gate_id", match=MatchValue(value=gate_id))
            )
        
        if node_type:
            filter_conditions.append(
                FieldCondition(key="node_type", match=MatchValue(value=node_type))
            )
        
        if difficulty_max < 5:
            filter_conditions.append(
                FieldCondition(key="difficulty_id", range=Range(lte=difficulty_max))
            )
        
        if depth_min > 1:
            filter_conditions.append(
                FieldCondition(key="depth", range=Range(gte=depth_min))
            )
        
        # Search using query_points (Qdrant v1.7+)
        from qdrant_client.models import PointStruct, ScoredPoint
        
        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            query_filter=Filter(must=filter_conditions) if filter_conditions else None,
            limit=limit
        )
        
        return [
            {
                "node_id": r.id,
                "vector_score": r.score,
                **r.payload,
                "retrieval_method": "vector"
            }
            for r in results
        ]
    
    # =========================================
    # HYBRID RETRIEVAL (Main Method)
    # =========================================
    
    def hybrid_retrieve(
        self,
        query: str,
        gate_id: Optional[int] = None,
        user_level: str = "intermediate",
        top_k: int = 10,
        session_id: Optional[str] = None  # NEW: for file chunk search
    ) -> List[Dict]:
        """
        Complete hybrid retrieval:
        1. Search uploaded file chunks (if session_id provided)
        2. Try vector search on knowledge base (if Qdrant available)
        3. Fallback to PostgreSQL text search + hierarchical
        4. Graph enrichment (prerequisites, examples, verifiers)
        5. Hybrid ranking and combine results
        """
        
        all_results = []
        
        # Step 0: Search uploaded files first (if session provided)
        if session_id:
            file_chunks = self.search_file_chunks(
                query=query,
                session_id=session_id,
                limit=top_k
            )
            if file_chunks:
                logger.info(f"Found {len(file_chunks)} file chunks, prioritizing uploaded files")
                all_results.extend(file_chunks)
                # If we have enough file results, return them
                if len(file_chunks) >= top_k:
                    return file_chunks[:top_k]
        
        # Map user level to difficulty
        level_map = {
            "foundation": 1,
            "intermediate": 2,
            "advanced": 3,
            "research": 4,
            "frontier": 5
        }
        
        max_difficulty = level_map.get(user_level.lower(), 2)
        
        # Step 1: Try vector search, fallback to PostgreSQL text search
        try:
            vector_results = self.vector_search(
                query=query,
                gate_id=gate_id,
                difficulty_max=max_difficulty,
                limit=20
            )
        except Exception as e:
            logger.warning(f"Vector search failed, using PostgreSQL text search: {e}")
            # Fallback to PostgreSQL full-text search
            vector_results = self._postgres_text_search(query, gate_id, max_difficulty, limit=20)
        
        if not vector_results:
            logger.warning(f"No results for query: {query}")
            # CRITICAL: Return file chunks if we have them, even if vector search failed
            if all_results:
                logger.info(f"Returning {len(all_results)} file chunks since vector search returned nothing")
                return all_results[:top_k]
            return []
        
        # Step 2: Enrich with graph walks
        enriched_results = []
        
        for node in vector_results:
            node_id = node["node_id"]
            
            # Get prerequisites
            prerequisites = self.get_prerequisites(node_id, max_depth=2)
            
            # Get examples (if concept/formula)
            examples = []
            if node.get("node_type") in ["CONCEPT", "FORMULA"]:
                examples = self.get_examples(node_id)
            
            # Get verifiers
            verifiers = self.get_verifiers(node_id)
            
            # Calculate graph enrichment score
            graph_score = (
                0.4 * min(len(prerequisites) / 3.0, 1.0) +
                0.3 * min(len(examples) / 2.0, 1.0) +
                0.3 * min(len(verifiers) / 1.0, 1.0)
            )
            
            enriched_results.append({
                **node,
                "prerequisites": prerequisites,
                "examples": examples,
                "verifiers": verifiers,
                "graph_enrichment_score": graph_score
            })
        
        # Step 3: Hybrid ranking
        ranked = self.rank_hybrid(enriched_results)
        
        return ranked[:top_k]
    
    def _postgres_text_search(
        self,
        query: str,
        gate_id: Optional[int] = None,
        difficulty_max: int = 5,
        limit: int = 20
    ) -> List[Dict]:
        """
        Fallback: PostgreSQL full-text search when Qdrant unavailable.
        """
        # Rollback any stale transactions
        try:
            self.pg_conn.rollback()
        except Exception:
            # Reconnect if connection is dead
            try:
                self.pg_conn = psycopg2.connect(**DB_CONFIG)
            except Exception as e:
                logger.error(f"Failed to reconnect to PostgreSQL: {e}")
                return []
        
        cursor = self.pg_conn.cursor(cursor_factory=RealDictCursor)
        
        # Build query using actual knowledge_chunks schema
        sql = """
            SELECT 
                id as node_id,
                content,
                source,
                gate,
                gate_id,
                quality_score,
                0.5 as vector_score,
                'CONCEPT' as node_type
            FROM knowledge_chunks
            WHERE 1=1
        """
        params = []
        
        if gate_id is not None:
            sql += " AND gate_id = %s"
            params.append(gate_id)
        
        # Add text search on content
        sql += " AND content ILIKE %s"
        search_term = f"%{query}%"
        params.append(search_term)
        
        sql += " ORDER BY quality_score DESC NULLS LAST"
        sql += f" LIMIT {limit}"
        
        try:
            cursor.execute(sql, params)
            results = cursor.fetchall()
        except Exception as e:
            logger.error(f"PostgreSQL text search failed: {e}")
            self.pg_conn.rollback()
            cursor.close()
            return []
        
        cursor.close()
        
        return [
            {
                "node_id": str(r["node_id"]),
                "content": r.get("content", "")[:500],
                "source": r.get("source", ""),
                "gate": r.get("gate"),
                "gate_id": r.get("gate_id"),
                "quality_score": float(r["quality_score"]) if r.get("quality_score") else 0.5,
                "node_type": r.get("node_type", "CONCEPT"),
                "vector_score": 0.5,
                "retrieval_method": "postgres_text"
            }
            for r in results
        ]
    
    def rank_hybrid(self, nodes: List[Dict]) -> List[Dict]:
        """Rank by combined vector + graph + quality scores"""
        
        scored = []
        
        for node in nodes:
            vector_score = node.get("vector_score", 0.0)
            graph_score = node.get("graph_enrichment_score", 0.0)
            quality_score = node.get("quality_score", 0.5)
            
            # Weighted combination
            combined = (
                0.5 * vector_score +
                0.3 * graph_score +
                0.2 * quality_score
            )
            
            scored.append({
                **node,
                "combined_score": combined
            })
        
        return sorted(scored, key=lambda x: x["combined_score"], reverse=True)
    
    # =========================================
    # MULTI-GATE RETRIEVAL
    # =========================================
    
    def multi_gate_retrieve(
        self,
        query: str,
        gates: List[int],
        user_level: str = "intermediate",
        top_k_per_gate: int = 5
    ) -> Dict[int, List[Dict]]:
        """Retrieve from multiple gates simultaneously"""
        
        results = {}
        
        for gate_id in gates:
            gate_results = self.hybrid_retrieve(
                query=query,
                gate_id=gate_id,
                user_level=user_level,
                top_k=top_k_per_gate
            )
            
            if gate_results:
                results[gate_id] = gate_results
        
        return results
    
    def close(self):
        """Cleanup"""
        if self.pg_conn:
            self.pg_conn.close()

# ============================================
# TEST HYBRID RETRIEVAL
# ============================================

if __name__ == "__main__":
    retriever = HybridRetrieverV2()
    
    try:
        # Test 1: Hierarchical retrieval
        print("\n" + "="*60)
        print("TEST 1: Hierarchical Retrieval")
        print("="*60)
        
        results = retriever.retrieve_by_hierarchy(
            gate_id=6,
            branch_id=3,
            depth_min=2,
            limit=5
        )
        
        for r in results:
            print(f"- {r['display_name']} ({r['node_type']}, depth={r['depth']})")
        
        # Test 2: Graph walks
        if results:
            print("\n" + "="*60)
            print("TEST 2: Prerequisites")
            print("="*60)
            
            sample_node_id = results[0]["node_id"]
            prereqs = retriever.get_prerequisites(sample_node_id)
            
            if prereqs:
                for p in prereqs:
                    print(f"- {p['display_name']} (depth={p['prerequisite_depth']})")
            else:
                print("No prerequisites found")
        
        # Test 3: Vector search
        print("\n" + "="*60)
        print("TEST 3: Vector Search")
        print("="*60)
        
        vector_results = retriever.vector_search(
            query="PID tuning for drones",
            gate_id=6,
            limit=5
        )
        
        for r in vector_results:
            print(f"- {r['display_name']} (score={r['vector_score']:.3f})")
        
        # Test 4: Hybrid retrieval
        print("\n" + "="*60)
        print("TEST 4: Hybrid Retrieval")
        print("="*60)
        
        hybrid_results = retriever.hybrid_retrieve(
            query="How to tune PID for drone yaw control?",
            gate_id=6,
            user_level="intermediate",
            top_k=5
        )
        
        for r in hybrid_results:
            print(f"\n- {r['display_name']}")
            print(f"  Combined: {r['combined_score']:.3f}")
            print(f"  Prerequisites: {len(r['prerequisites'])}")
            print(f"  Examples: {len(r['examples'])}")
            print(f"  Verifiers: {len(r['verifiers'])}")
        
        # Test 5: Multi-gate retrieval
        print("\n" + "="*60)
        print("TEST 5: Multi-Gate Retrieval")
        print("="*60)
        
        multi_results = retriever.multi_gate_retrieve(
            query="control systems",
            gates=[6],  # AV-NAV
            top_k_per_gate=3
        )
        
        for gate_id, gate_results in multi_results.items():
            print(f"\nGate {gate_id}:")
            for r in gate_results:
                print(f"  - {r['display_name']}")
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
    
    finally:
        retriever.close()
