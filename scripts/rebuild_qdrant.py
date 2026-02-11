#!/usr/bin/env python3
"""
Rebuild Qdrant collection from PostgreSQL knowledge_chunks
This creates the wiredbrain_knowledge collection with proper embeddings
"""

import psycopg2
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import logging
from tqdm import tqdm
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5433'),
    'database': os.getenv('DB_NAME_V2', 'wiredbrain'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres123')
}

COLLECTION_NAME = "wiredbrain_knowledge"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 100

def main():
    # Connect to PostgreSQL
    logger.info("Connecting to PostgreSQL...")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        logger.info("✅ Connected to PostgreSQL")
    except Exception as e:
        logger.error(f"❌ Failed to connect to PostgreSQL: {e}")
        return
    
    # Count total chunks
    cursor.execute("SELECT COUNT(*) FROM knowledge_chunks")
    total_chunks = cursor.fetchone()[0]
    logger.info(f"Found {total_chunks:,} chunks to process")
    
    if total_chunks == 0:
        logger.error("❌ No chunks found in knowledge_chunks table!")
        return
    
    # Initialize Qdrant
    logger.info("Initializing Qdrant...")
    qdrant_path = os.path.join(os.path.dirname(__file__), '..', 'qdrant_local_storage')
    client = QdrantClient(path=qdrant_path)
    
    # Delete old collection if exists
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info(f"Deleted old collection: {COLLECTION_NAME}")
    except:
        logger.info(f"No existing collection to delete")
    
    # Create new collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,  # bge-small-en-v1.5 dimension
            distance=Distance.COSINE
        )
    )
    logger.info(f"✅ Created collection: {COLLECTION_NAME}")
    
    # Load embedding model
    logger.info("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("✅ Model loaded")
    
    # Process in batches
    offset = 0
    processed = 0
    
    with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
        while offset < total_chunks:
            # Fetch batch
            cursor.execute(f"""
                SELECT id, content, gate, gate_id, quality_score
                FROM knowledge_chunks
                ORDER BY id
                LIMIT {BATCH_SIZE} OFFSET {offset}
            """)
            
            batch = cursor.fetchall()
            if not batch:
                break
            
            # Generate embeddings
            texts = [row[1][:512] if row[1] else "" for row in batch]  # Truncate to 512 chars
            embeddings = model.encode(texts, show_progress_bar=False)
            
            # Prepare points
            points = []
            for i, row in enumerate(batch):
                chunk_id, content, gate, gate_id, quality_score = row
                
                points.append(PointStruct(
                    id=chunk_id,
                    vector=embeddings[i].tolist(),
                    payload={
                        "gate": gate or "GENERAL",
                        "gate_id": gate_id or 0,
                        "quality_score": float(quality_score) if quality_score else 0.5,
                        "content": content[:200] if content else "",  # Preview
                        "node_type": "CONCEPT",  # Default
                        "difficulty_id": 2,  # Default intermediate
                        "depth": 3  # Default
                    }
                ))
            
            # Upload to Qdrant
            try:
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points
                )
            except Exception as e:
                logger.error(f"Failed to upsert batch at offset {offset}: {e}")
                continue
            
            processed += len(batch)
            offset += BATCH_SIZE
            pbar.update(len(batch))
    
    logger.info(f"✅ Processed {processed:,} chunks")
    
    # Verify
    collection_info = client.get_collection(COLLECTION_NAME)
    logger.info(f"✅ Collection '{COLLECTION_NAME}' has {collection_info.points_count:,} points")
    
    cursor.close()
    conn.close()
    
    logger.info("🎉 Qdrant rebuild complete!")

if __name__ == "__main__":
    main()
