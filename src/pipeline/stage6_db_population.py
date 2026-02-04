#!/usr/bin/env python3
"""
STAGE 6: Database Population (PRODUCTION)
- PostgreSQL with array schema + prerequisite edges
- Qdrant vector embeddings
- Cytoscape graph export

Tools: psycopg2, qdrant-client, sentence-transformers, networkx
Input: toon/*.toon OR labeled/*.jsonl
Output: PostgreSQL + Qdrant + Cytoscape JSON
"""

import os
import json
import msgpack
import lz4.frame
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check imports
try:
    import psycopg2
    from psycopg2.extras import execute_values
    PG_AVAILABLE = True
except ImportError:
    PG_AVAILABLE = False
    logger.warning("psycopg2 not installed")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("qdrant-client not installed")

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning("sentence-transformers not installed")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning("networkx not installed")

# Configuration
INPUT_DIR = Path("kg_ready")  # Use kg_ready/ for JSON, toon/ for compressed
TOON_DIR = Path("toon")
OUTPUT_DIR = Path("checkpoints")
OUTPUT_DIR.mkdir(exist_ok=True)

# Build DB config - support Unix socket (empty host) or TCP connection
def get_db_config():
    """Build PostgreSQL connection config, supporting Unix socket when host is empty"""
    config = {
        'database': os.getenv('DB_NAME', 'axiom_brain_prod'),
        'user': os.getenv('DB_USER', 'postgres'),
    }
    
    # Only add host/port if specified (empty = Unix socket)
    db_host = os.getenv('DB_HOST', '').strip()
    db_port = os.getenv('DB_PORT', '').strip()
    db_password = os.getenv('DB_PASSWORD', '').strip()
    
    if db_host:
        config['host'] = db_host
    if db_port:
        config['port'] = db_port
    if db_password:
        config['password'] = db_password
    
    return config

DB_CONFIG = get_db_config()

QDRANT_CONFIG = {
    'host': os.getenv('QDRANT_HOST', 'localhost'),
    'port': int(os.getenv('QDRANT_PORT', '6333'))
}

BATCH_SIZE = 500
COLLECTION_NAME = "axiom_knowledge"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

class ProductionDatabasePopulator:
    """Production database populator with Graph + Vector support"""
    
    def __init__(self):
        self.pg_conn = None
        self.qdrant = None
        self.embedding_model = None
        self.graph = None
        
        self.stats = {
            'total_chunks': 0,
            'pg_inserted': 0,
            'qdrant_inserted': 0,
            'edges_created': 0,
            'failed': 0
        }
    
    def connect_postgres(self):
        """Connect to PostgreSQL"""
        if not PG_AVAILABLE:
            logger.error("psycopg2 not available")
            return False
        
        try:
            logger.info(f"Connecting to PostgreSQL: {DB_CONFIG['database']}@{DB_CONFIG['host']}")
            self.pg_conn = psycopg2.connect(**DB_CONFIG)
            logger.info("  ✅ PostgreSQL connected")
            return True
        except Exception as e:
            logger.error(f"  ❌ PostgreSQL failed: {e}")
            return False
    
    def connect_qdrant(self):
        """Connect to Qdrant"""
        if not QDRANT_AVAILABLE:
            logger.warning("Qdrant not available, skipping vector search")
            return False
        
        try:
            logger.info(f"Connecting to Qdrant: {QDRANT_CONFIG['host']}:{QDRANT_CONFIG['port']}")
            self.qdrant = QdrantClient(**QDRANT_CONFIG)
            
            # Create collection
            self.qdrant.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
            )
            logger.info("  ✅ Qdrant connected + collection created")
            return True
        except Exception as e:
            logger.warning(f"  ⚠️ Qdrant failed: {e}")
            return False
    
    def load_embedding_model(self):
        """Load embedding model"""
        if not EMBEDDINGS_AVAILABLE:
            logger.warning("sentence-transformers not available")
            return False
        
        try:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("  ✅ Embedding model loaded")
            return True
        except Exception as e:
            logger.warning(f"  ⚠️ Embedding model failed: {e}")
            return False
    
    def init_graph(self):
        """Initialize NetworkX graph for Cytoscape export"""
        if NETWORKX_AVAILABLE:
            self.graph = nx.DiGraph()
            logger.info("  ✅ NetworkX graph initialized")
    
    def create_postgres_schema(self):
        """Create full PostgreSQL schema with pgvector + edges table"""
        logger.info("Creating PostgreSQL schema...")
        
        cur = self.pg_conn.cursor()
        
        # Try to create pgvector extension
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            self.pg_conn.commit()
            logger.info("  ✅ pgvector extension enabled")
            pgvector_available = True
        except Exception as e:
            logger.warning(f"  ⚠️ pgvector extension not available: {e}")
            logger.warning("  Install with: CREATE EXTENSION vector; (requires pgvector)")
            self.pg_conn.rollback()
            pgvector_available = False
        
        # Build schema with conditional pgvector column
        embedding_column = "embedding vector(384)," if pgvector_available else "-- embedding vector(384),  -- pgvector not installed"
        
        schema_sql = f"""
        -- Main knowledge chunks table
        DROP TABLE IF EXISTS prerequisite_edges CASCADE;
        DROP TABLE IF EXISTS knowledge_chunks CASCADE;
        
        CREATE TABLE knowledge_chunks (
            id SERIAL PRIMARY KEY,
            
            -- Content
            content_text TEXT,
            content_toon BYTEA,
            
            -- 4-level hierarchy (ARRAY FORMAT)
            gate_path TEXT[] NOT NULL,
            
            -- Numeric coordinates (fast indexing)
            gate_id INTEGER NOT NULL,
            branch_id INTEGER NOT NULL,
            topic_id INTEGER NOT NULL,
            level_id INTEGER NOT NULL,
            
            -- Metadata
            concepts TEXT[],
            prerequisites JSONB,
            confidence FLOAT,
            quality_score FLOAT,
            
            -- Vector embedding (pgvector)
            {embedding_column}
            
            -- Timestamps
            created_at TIMESTAMP DEFAULT NOW()
        );
        
        -- Prerequisite edges table (for graph traversal)
        CREATE TABLE prerequisite_edges (
            id SERIAL PRIMARY KEY,
            chunk_id INTEGER REFERENCES knowledge_chunks(id),
            prerequisite_chunk_id INTEGER REFERENCES knowledge_chunks(id),
            strength FLOAT DEFAULT 1.0,
            created_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(chunk_id, prerequisite_chunk_id)
        );
        
        -- Indexes for hybrid search
        CREATE INDEX idx_gate_path ON knowledge_chunks USING GIN (gate_path);
        CREATE INDEX idx_coordinates ON knowledge_chunks (gate_id, branch_id, topic_id, level_id);
        CREATE INDEX idx_concepts ON knowledge_chunks USING GIN (concepts);
        CREATE INDEX idx_gate_id ON knowledge_chunks (gate_id);
        CREATE INDEX idx_quality ON knowledge_chunks (quality_score) WHERE quality_score > 0.6;
        
        -- Graph indexes
        CREATE INDEX idx_prereq_chunk ON prerequisite_edges (chunk_id);
        CREATE INDEX idx_prereq_target ON prerequisite_edges (prerequisite_chunk_id);
        """
        
        # Add pgvector index if available
        if pgvector_available:
            schema_sql += """
        -- Vector index (pgvector)
        CREATE INDEX idx_embedding ON knowledge_chunks 
        USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
            """
        
        cur.execute(schema_sql)
        self.pg_conn.commit()
        cur.close()
        
        self.pgvector_available = pgvector_available
        logger.info("  ✅ PostgreSQL schema created with edges table")
    
    def load_labeled_chunks(self):
        """Load chunks from labeled/ directory"""
        logger.info("Loading labeled chunks...")
        
        chunks = []
        for jsonl_file in INPUT_DIR.glob("*.jsonl"):
            with open(jsonl_file, 'r') as f:
                for line in f:
                    try:
                        chunk = json.loads(line)
                        chunks.append(chunk)
                    except:
                        continue
        
        logger.info(f"  Loaded {len(chunks):,} chunks")
        return chunks
    
    def insert_to_postgres(self, chunks):
        """Insert chunks to PostgreSQL with optional embeddings"""
        logger.info("Inserting to PostgreSQL...")
        
        cur = self.pg_conn.cursor()
        
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i+BATCH_SIZE]
            
            # Generate embeddings if needed
            embeddings = []
            if hasattr(self, 'pgvector_available') and self.pgvector_available and self.embedding_model:
                texts = [c.get('content', c.get('text', ''))[:512] for c in batch]
                embeddings = self.embedding_model.encode(texts)
            
            values = []
            for idx, chunk in enumerate(batch):
                coords = chunk.get('coordinates', {})
                
                row = [
                    chunk.get('content', chunk.get('text', '')),
                    None,  # content_toon (for later)
                    chunk.get('gate_path', []),
                    coords.get('gate_id', 0),
                    coords.get('branch_id', 0),
                    coords.get('topic_id', 0),
                    coords.get('level_id', 0),
                    chunk.get('concepts', []),
                    json.dumps(chunk.get('prerequisites', [])),
                    chunk.get('confidence', 0.5),
                    chunk.get('quality_score', 0.5)
                ]
                
                # Add embedding if pgvector available
                if embeddings:
                    row.append(embeddings[idx].tolist())
                
                values.append(tuple(row))
            
            # Build SQL based on pgvector availability
            if hasattr(self, 'pgvector_available') and self.pgvector_available:
                sql = """
                    INSERT INTO knowledge_chunks 
                    (content_text, content_toon, gate_path, gate_id, branch_id, topic_id, level_id,
                     concepts, prerequisites, confidence, quality_score, embedding)
                    VALUES %s
                    RETURNING id
                """
            else:
                sql = """
                    INSERT INTO knowledge_chunks 
                    (content_text, content_toon, gate_path, gate_id, branch_id, topic_id, level_id,
                     concepts, prerequisites, confidence, quality_score)
                    VALUES %s
                    RETURNING id
                """
            
            execute_values(cur, sql, values)
            
            self.stats['pg_inserted'] += len(batch)
        
        self.pg_conn.commit()
        cur.close()
        
        logger.info(f"  ✅ Inserted {self.stats['pg_inserted']:,} chunks to PostgreSQL")
    
    def insert_to_qdrant(self, chunks):
        """Insert chunks with embeddings to Qdrant"""
        if not self.qdrant or not self.embedding_model:
            logger.warning("Skipping Qdrant (not configured)")
            return
        
        logger.info("Inserting to Qdrant with embeddings...")
        
        # Generate embeddings in batches
        texts = [c.get('content', c.get('text', ''))[:512] for c in chunks]
        
        for i in range(0, len(chunks), BATCH_SIZE):
            batch_texts = texts[i:i+BATCH_SIZE]
            batch_chunks = chunks[i:i+BATCH_SIZE]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(batch_texts)
            
            # Create points
            points = []
            for j, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings)):
                point = PointStruct(
                    id=i + j,
                    vector=embedding.tolist(),
                    payload={
                        "gate_path": chunk.get('gate_path', []),
                        "coordinates": chunk.get('coordinates', {}),
                        "concepts": chunk.get('concepts', []),
                        "content_preview": chunk.get('content', '')[:200]
                    }
                )
                points.append(point)
            
            # Upsert
            self.qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            self.stats['qdrant_inserted'] += len(points)
        
        logger.info(f"  ✅ Inserted {self.stats['qdrant_inserted']:,} vectors to Qdrant")
    
    def build_prerequisite_graph(self, chunks):
        """Build prerequisite edges in PostgreSQL + NetworkX"""
        logger.info("Building prerequisite graph...")
        
        cur = self.pg_conn.cursor()
        
        # Get all chunks with IDs
        cur.execute("SELECT id, gate_path FROM knowledge_chunks")
        chunk_map = {tuple(row[1]): row[0] for row in cur.fetchall()}
        
        edges_created = 0
        
        for chunk in chunks:
            chunk_path = tuple(chunk.get('gate_path', []))
            chunk_id = chunk_map.get(chunk_path)
            
            if not chunk_id:
                continue
            
            # Add node to graph
            if self.graph:
                self.graph.add_node(chunk_id, **{
                    'gate': chunk_path[0] if chunk_path else 'UNKNOWN',
                    'label': chunk_path[2] if len(chunk_path) > 2 else 'Unknown'
                })
            
            # Process prerequisites
            for prereq in chunk.get('prerequisites', []):
                if isinstance(prereq, list) and len(prereq) == 4:
                    prereq_tuple = tuple(prereq)
                    prereq_id = chunk_map.get(prereq_tuple)
                    
                    if prereq_id:
                        # Insert edge to PostgreSQL
                        try:
                            cur.execute("""
                                INSERT INTO prerequisite_edges (chunk_id, prerequisite_chunk_id)
                                VALUES (%s, %s)
                                ON CONFLICT DO NOTHING
                            """, (chunk_id, prereq_id))
                            edges_created += 1
                            
                            # Add to NetworkX graph
                            if self.graph:
                                self.graph.add_edge(prereq_id, chunk_id)
                        except:
                            continue
        
        self.pg_conn.commit()
        cur.close()
        
        self.stats['edges_created'] = edges_created
        logger.info(f"  ✅ Created {edges_created:,} prerequisite edges")
    
    def export_for_cytoscape(self):
        """Export graph to Cytoscape JSON format"""
        if not self.graph or len(self.graph.nodes) == 0:
            logger.warning("No graph to export")
            return
        
        logger.info("Exporting for Cytoscape...")
        
        cytoscape_data = {
            "elements": {
                "nodes": [
                    {
                        "data": {
                            "id": str(node),
                            "label": data.get('label', str(node)),
                            "gate": data.get('gate', 'UNKNOWN')
                        }
                    }
                    for node, data in self.graph.nodes(data=True)
                ],
                "edges": [
                    {
                        "data": {
                            "source": str(u),
                            "target": str(v)
                        }
                    }
                    for u, v in self.graph.edges()
                ]
            }
        }
        
        output_file = OUTPUT_DIR / "knowledge_graph_cytoscape.json"
        with open(output_file, 'w') as f:
            json.dump(cytoscape_data, f, indent=2)
        
        logger.info(f"  ✅ Exported {len(self.graph.nodes):,} nodes, {len(self.graph.edges):,} edges to {output_file}")
    
    def run(self):
        """Execute full population"""
        logger.info("=" * 80)
        logger.info("STAGE 6: DATABASE POPULATION (PRODUCTION)")
        logger.info("=" * 80)
        
        # Connect services
        pg_ok = self.connect_postgres()
        qdrant_ok = self.connect_qdrant()
        embed_ok = self.load_embedding_model()
        self.init_graph()
        
        if not pg_ok:
            logger.error("PostgreSQL required but not available")
            return
        
        # Create schema
        self.create_postgres_schema()
        
        # Load chunks
        chunks = self.load_labeled_chunks()
        self.stats['total_chunks'] = len(chunks)
        
        if not chunks:
            logger.error("No chunks found in labeled/")
            return
        
        # Insert to PostgreSQL
        self.insert_to_postgres(chunks)
        
        # Insert to Qdrant (if available)
        if qdrant_ok and embed_ok:
            self.insert_to_qdrant(chunks)
        
        # Build prerequisite graph
        self.build_prerequisite_graph(chunks)
        
        # Export for Cytoscape
        self.export_for_cytoscape()
        
        # Save stats
        stats = {
            "stage": 6,
            **self.stats,
            "database": DB_CONFIG['database'],
            "qdrant_collection": COLLECTION_NAME,
            "completed_at": datetime.now().isoformat()
        }
        
        with open(OUTPUT_DIR / "stage6_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("=" * 80)
        logger.info("STAGE 6 COMPLETE")
        logger.info(f"PostgreSQL: {self.stats['pg_inserted']:,} chunks")
        logger.info(f"Qdrant: {self.stats['qdrant_inserted']:,} vectors")
        logger.info(f"Graph: {self.stats['edges_created']:,} edges")
        logger.info("✅ PIPELINE COMPLETE!")
        logger.info("=" * 80)
    
    def close(self):
        """Cleanup"""
        if self.pg_conn:
            self.pg_conn.close()

def main():
    populator = ProductionDatabasePopulator()
    try:
        populator.run()
    finally:
        populator.close()

if __name__ == "__main__":
    main()
