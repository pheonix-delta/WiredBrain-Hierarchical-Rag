#!/usr/bin/env python3
"""
STAGE 4.5: KNOWLEDGE GRAPH EXTRACTION (InfraNodus-Style)

Extracts Subject → Relation → Object triplets for brain visualization.

WHAT THIS STAGE DOES:
1. NER (Named Entity Recognition) using GLiNER + spaCy
2. Relation Extraction using LLM (triplets)
3. Neo4j graph population
4. Community Detection (Louvain clustering)
5. Export for InfraNodus/Obsidian/Cytoscape

Input: labeled/*.jsonl (from Stage 4)
Output: 
  - Neo4j graph database
  - triplets/*.jsonl (raw triplets)
  - exports/brain_graph.json (for visualization)

Tools: GLiNER, spaCy, Neo4j, NetworkX, LLM APIs
"""

import os
import json
import time
import logging
import re
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
INPUT_DIR = PROJECT_DIR / "labeled"
TRIPLET_DIR = PROJECT_DIR / "triplets"
EXPORT_DIR = PROJECT_DIR / "exports"
TRIPLET_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Initialize tools
GLINER_AVAILABLE = False
SPACY_AVAILABLE = False
NEO4J_AVAILABLE = False
NETWORKX_AVAILABLE = False

try:
    from gliner import GLiNER
    GLINER_AVAILABLE = True
    logger.info("✅ GLiNER loaded")
except ImportError:
    logger.warning("⚠️ GLiNER not available: pip install gliner")

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
    logger.info("✅ spaCy loaded")
except:
    logger.warning("⚠️ spaCy not available")

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
    logger.info("✅ Neo4j driver loaded")
except ImportError:
    logger.warning("⚠️ Neo4j not available: pip install neo4j")

try:
    import networkx as nx
    from networkx.algorithms.community import louvain_communities
    NETWORKX_AVAILABLE = True
    logger.info("✅ NetworkX loaded with Louvain")
except ImportError:
    logger.warning("⚠️ NetworkX not available")

# LLM API Configuration (same as Stage 4)
WORKING_APIS = [
    {
        "name": "groq",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key": os.getenv("GROQ_API_KEY"),
        "model": "llama-3.3-70b-versatile",
        "timeout": 30
    },
    {
        "name": "ollama",
        "url": "http://localhost:11434/api/chat",
        "key": None,
        "model": "llama3.2:latest",
        "timeout": 60
    }
]

# Entity types for GLiNER (robotics/AI focused)
ENTITY_TYPES = [
    "concept", "technology", "algorithm", "framework", "language",
    "hardware", "sensor", "actuator", "protocol", "standard",
    "method", "technique", "tool", "library", "model",
    "system", "component", "process", "operation", "function"
]

# Relation types for extraction
RELATION_TYPES = [
    "is_used_for", "is_part_of", "requires", "enables",
    "implements", "extends", "controls", "processes",
    "communicates_with", "depends_on", "contains", "produces",
    "measures", "optimizes", "configures", "integrates_with"
]


class KnowledgeGraphExtractor:
    """Extracts knowledge graph triplets from classified chunks"""
    
    def __init__(self):
        self.gliner_model = None
        self.neo4j_driver = None
        self.graph = None
        self.api_index = 0
        
        self.stats = {
            'chunks_processed': 0,
            'entities_extracted': 0,
            'triplets_extracted': 0,
            'neo4j_nodes': 0,
            'neo4j_edges': 0,
            'communities_detected': 0
        }
        
        self._init_tools()
    
    def _init_tools(self):
        """Initialize NER and graph tools"""
        # GLiNER
        if GLINER_AVAILABLE:
            try:
                logger.info("Loading GLiNER model...")
                self.gliner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
                logger.info("  ✅ GLiNER model loaded")
            except Exception as e:
                logger.warning(f"  ⚠️ GLiNER model failed: {e}")
        
        # Neo4j
        if NEO4J_AVAILABLE:
            try:
                self.neo4j_driver = GraphDatabase.driver(
                    NEO4J_URI, 
                    auth=(NEO4J_USER, NEO4J_PASSWORD)
                )
                # Test connection
                with self.neo4j_driver.session() as session:
                    session.run("RETURN 1")
                logger.info("  ✅ Neo4j connected")
            except Exception as e:
                logger.warning(f"  ⚠️ Neo4j connection failed: {e}")
                self.neo4j_driver = None
        
        # NetworkX
        if NETWORKX_AVAILABLE:
            self.graph = nx.Graph()
            logger.info("  ✅ NetworkX graph initialized")
    
    def extract_entities_gliner(self, text):
        """Extract entities using GLiNER (domain-specific NER)"""
        if not self.gliner_model:
            return []
        
        try:
            # GLiNER extraction
            entities = self.gliner_model.predict_entities(
                text[:5000],  # Limit text length
                ENTITY_TYPES,
                threshold=0.5
            )
            
            extracted = []
            for ent in entities:
                extracted.append({
                    'text': ent['text'],
                    'type': ent['label'],
                    'score': ent['score'],
                    'source': 'gliner'
                })
            
            return extracted
        except Exception as e:
            logger.debug(f"GLiNER error: {e}")
            return []
    
    def extract_entities_spacy(self, text):
        """Extract entities using spaCy (general NER)"""
        if not SPACY_AVAILABLE:
            return []
        
        try:
            doc = nlp(text[:5000])
            extracted = []
            
            for ent in doc.ents:
                extracted.append({
                    'text': ent.text,
                    'type': ent.label_,
                    'score': 1.0,
                    'source': 'spacy'
                })
            
            # Also extract noun chunks (important concepts)
            for chunk in doc.noun_chunks:
                if len(chunk.text) > 3 and chunk.text.lower() not in ['the', 'a', 'an']:
                    extracted.append({
                        'text': chunk.text,
                        'type': 'CONCEPT',
                        'score': 0.7,
                        'source': 'spacy_noun_chunk'
                    })
            
            return extracted
        except Exception as e:
            logger.debug(f"spaCy error: {e}")
            return []
    
    def merge_entities(self, gliner_entities, spacy_entities):
        """Merge and deduplicate entities from both sources"""
        seen = set()
        merged = []
        
        # Prioritize GLiNER (domain-specific)
        for ent in gliner_entities:
            key = ent['text'].lower().strip()
            if key not in seen and len(key) > 2:
                seen.add(key)
                merged.append(ent)
        
        # Add spaCy entities if not already present
        for ent in spacy_entities:
            key = ent['text'].lower().strip()
            if key not in seen and len(key) > 2:
                seen.add(key)
                merged.append(ent)
        
        self.stats['entities_extracted'] += len(merged)
        return merged
    
    def call_llm(self, prompt):
        """Call LLM API for relation extraction"""
        for i in range(len(WORKING_APIS)):
            api = WORKING_APIS[self.api_index % len(WORKING_APIS)]
            self.api_index += 1
            
            if api['name'] != 'ollama' and not api['key']:
                continue
            
            try:
                headers = {"Content-Type": "application/json"}
                if api['key']:
                    headers["Authorization"] = f"Bearer {api['key']}"
                
                if api['name'] == 'ollama':
                    payload = {
                        "model": api['model'],
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False
                    }
                else:
                    payload = {
                        "model": api['model'],
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.2,
                        "max_tokens": 1000
                    }
                
                response = requests.post(
                    api['url'],
                    headers=headers,
                    json=payload,
                    timeout=api['timeout']
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if api['name'] == 'ollama':
                        return result['message']['content']
                    else:
                        return result['choices'][0]['message']['content']
            
            except Exception as e:
                logger.debug(f"LLM error ({api['name']}): {e}")
                continue
        
        return None
    
    def extract_triplets_llm(self, text, entities):
        """Extract Subject → Relation → Object triplets using LLM"""
        if not entities:
            return []
        
        entity_list = ", ".join([e['text'] for e in entities[:20]])
        relation_list = ", ".join(RELATION_TYPES)
        
        prompt = f"""Extract knowledge graph triplets from this text.

ENTITIES FOUND: {entity_list}

VALID RELATIONS: {relation_list}

TEXT:
{text[:2000]}

Output a JSON array of triplets. Each triplet has:
- subject: entity name (from entities list)
- relation: relationship type (from valid relations)
- object: entity name (from entities list)

Example output:
[
  {{"subject": "Python", "relation": "is_used_for", "object": "ROS2"}},
  {{"subject": "SLAM", "relation": "requires", "object": "LiDAR"}}
]

Rules:
- Only use entities from the list above
- Only use relations from the valid list
- Extract 3-10 triplets maximum
- Be specific and accurate

Output JSON only, no explanation:"""

        result = self.call_llm(prompt)
        
        if not result:
            return []
        
        try:
            # Parse JSON
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0]
            elif "```" in result:
                result = result.split("```")[1].split("```")[0]
            
            # Find JSON array
            match = re.search(r'\[.*\]', result, re.DOTALL)
            if match:
                triplets = json.loads(match.group())
                self.stats['triplets_extracted'] += len(triplets)
                return triplets
        except Exception as e:
            logger.debug(f"Triplet parse error: {e}")
        
        return []
    
    def add_to_neo4j(self, chunk_id, entities, triplets, gate_path):
        """Add nodes and edges to Neo4j"""
        if not self.neo4j_driver:
            return
        
        with self.neo4j_driver.session() as session:
            # Create entity nodes
            for ent in entities:
                try:
                    session.run("""
                        MERGE (e:Entity {name: $name})
                        SET e.type = $type,
                            e.last_seen = $timestamp
                    """, {
                        "name": ent['text'],
                        "type": ent['type'],
                        "timestamp": datetime.now().isoformat()
                    })
                    self.stats['neo4j_nodes'] += 1
                except:
                    continue
            
            # Create gate node (for hierarchy)
            if gate_path:
                gate_name = gate_path[0] if gate_path else "UNKNOWN"
                session.run("""
                    MERGE (g:Gate {name: $name})
                """, {"name": gate_name})
            
            # Create triplet edges
            for triplet in triplets:
                try:
                    subj = triplet.get('subject', '')
                    rel = triplet.get('relation', 'related_to')
                    obj = triplet.get('object', '')
                    
                    if subj and obj:
                        # Clean relation name for Neo4j
                        rel_clean = rel.upper().replace(' ', '_').replace('-', '_')
                        
                        session.run(f"""
                            MATCH (s:Entity {{name: $subject}})
                            MATCH (o:Entity {{name: $object}})
                            MERGE (s)-[r:{rel_clean}]->(o)
                            SET r.chunk_id = $chunk_id,
                                r.gate = $gate
                        """, {
                            "subject": subj,
                            "object": obj,
                            "chunk_id": chunk_id,
                            "gate": gate_path[0] if gate_path else "UNKNOWN"
                        })
                        self.stats['neo4j_edges'] += 1
                except Exception as e:
                    logger.debug(f"Neo4j edge error: {e}")
    
    def add_to_networkx(self, entities, triplets, gate_path):
        """Add to NetworkX graph for community detection"""
        if not self.graph:
            return
        
        gate = gate_path[0] if gate_path else "UNKNOWN"
        
        # Add entity nodes
        for ent in entities:
            self.graph.add_node(
                ent['text'],
                type=ent['type'],
                gate=gate
            )
        
        # Add triplet edges
        for triplet in triplets:
            subj = triplet.get('subject', '')
            obj = triplet.get('object', '')
            rel = triplet.get('relation', 'related')
            
            if subj and obj:
                self.graph.add_edge(subj, obj, relation=rel, gate=gate)
    
    def process_chunk(self, chunk, chunk_id):
        """Process single chunk for knowledge graph extraction"""
        self.stats['chunks_processed'] += 1
        
        text = chunk.get('content', '')
        gate_path = chunk.get('gate_path', [])
        existing_concepts = chunk.get('concepts', [])
        
        # Step 1: NER with both tools
        gliner_entities = self.extract_entities_gliner(text)
        spacy_entities = self.extract_entities_spacy(text)
        
        # Add existing concepts as entities
        for concept in existing_concepts:
            gliner_entities.append({
                'text': concept,
                'type': 'CONCEPT',
                'score': 1.0,
                'source': 'stage4_concepts'
            })
        
        # Merge entities
        entities = self.merge_entities(gliner_entities, spacy_entities)
        
        if not entities:
            return None
        
        # Step 2: Relation extraction with LLM
        triplets = self.extract_triplets_llm(text, entities)
        
        # Step 3: Add to Neo4j
        self.add_to_neo4j(chunk_id, entities, triplets, gate_path)
        
        # Step 4: Add to NetworkX
        self.add_to_networkx(entities, triplets, gate_path)
        
        # Return extraction result
        return {
            'chunk_id': chunk_id,
            'gate_path': gate_path,
            'entities': entities,
            'triplets': triplets,
            'extracted_at': datetime.now().isoformat()
        }
    
    def detect_communities(self):
        """Detect communities using Louvain algorithm (like InfraNodus colors)"""
        if not self.graph or len(self.graph.nodes) < 2:
            return {}
        
        logger.info("Detecting communities (Louvain)...")
        
        try:
            # Convert to undirected for community detection
            undirected = self.graph.to_undirected() if self.graph.is_directed() else self.graph
            
            # Louvain community detection
            communities = louvain_communities(undirected, seed=42)
            
            # Assign community IDs to nodes
            community_map = {}
            for i, community in enumerate(communities):
                for node in community:
                    community_map[node] = i
            
            # Update graph nodes
            for node, comm_id in community_map.items():
                self.graph.nodes[node]['community'] = comm_id
            
            self.stats['communities_detected'] = len(communities)
            logger.info(f"  ✅ Detected {len(communities)} communities")
            
            return community_map
        
        except Exception as e:
            logger.warning(f"  ⚠️ Community detection failed: {e}")
            return {}
    
    def export_for_visualization(self, community_map):
        """Export graph for InfraNodus/Obsidian/Cytoscape"""
        if not self.graph:
            return
        
        logger.info("Exporting for visualization...")
        
        # Color palette for communities (InfraNodus-style)
        colors = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
            "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
            "#F8B500", "#00CED1", "#FF69B4", "#32CD32", "#FFD700"
        ]
        
        # Build Cytoscape-compatible JSON
        nodes = []
        edges = []
        
        for node, data in self.graph.nodes(data=True):
            comm_id = community_map.get(node, 0)
            nodes.append({
                "data": {
                    "id": node,
                    "label": node,
                    "type": data.get('type', 'UNKNOWN'),
                    "gate": data.get('gate', 'UNKNOWN'),
                    "community": comm_id,
                    "color": colors[comm_id % len(colors)]
                }
            })
        
        for u, v, data in self.graph.edges(data=True):
            edges.append({
                "data": {
                    "source": u,
                    "target": v,
                    "relation": data.get('relation', 'related'),
                    "gate": data.get('gate', 'UNKNOWN')
                }
            })
        
        # Cytoscape format
        cytoscape_data = {
            "elements": {
                "nodes": nodes,
                "edges": edges
            },
            "metadata": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "communities": self.stats['communities_detected'],
                "generated_at": datetime.now().isoformat()
            }
        }
        
        output_file = EXPORT_DIR / "brain_graph_cytoscape.json"
        with open(output_file, 'w') as f:
            json.dump(cytoscape_data, f, indent=2)
        logger.info(f"  ✅ Saved: {output_file}")
        
        # InfraNodus/Obsidian format (nodes with backlinks)
        infranodus_data = {
            "nodes": [
                {
                    "id": node,
                    "label": f"[[{node}]]",  # Obsidian-style
                    "community": community_map.get(node, 0),
                    "degree": self.graph.degree(node),
                    "type": self.graph.nodes[node].get('type', 'UNKNOWN')
                }
                for node in self.graph.nodes()
            ],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "relation": data.get('relation', 'related')
                }
                for u, v, data in self.graph.edges(data=True)
            ],
            "communities": [
                {
                    "id": i,
                    "size": sum(1 for n, c in community_map.items() if c == i),
                    "color": colors[i % len(colors)]
                }
                for i in range(self.stats['communities_detected'])
            ]
        }
        
        output_file = EXPORT_DIR / "brain_graph_infranodus.json"
        with open(output_file, 'w') as f:
            json.dump(infranodus_data, f, indent=2)
        logger.info(f"  ✅ Saved: {output_file}")
        
        # GraphML for Gephi/other tools
        if NETWORKX_AVAILABLE:
            graphml_file = EXPORT_DIR / "brain_graph.graphml"
            nx.write_graphml(self.graph, graphml_file)
            logger.info(f"  ✅ Saved: {graphml_file}")
    
    def run(self, sample_size=None):
        """Run knowledge graph extraction"""
        logger.info("=" * 80)
        logger.info("STAGE 4.5: KNOWLEDGE GRAPH EXTRACTION")
        logger.info("Creating InfraNodus-style brain visualization")
        logger.info("=" * 80)
        
        # Check input
        if not INPUT_DIR.exists():
            logger.error(f"Input directory not found: {INPUT_DIR}")
            logger.error("Run Stage 4 first!")
            return
        
        # Load chunks
        input_files = list(INPUT_DIR.glob("*.jsonl"))
        if not input_files:
            logger.error(f"No files in {INPUT_DIR}")
            return
        
        logger.info(f"Processing {len(input_files)} files")
        
        all_extractions = []
        chunk_id = 0
        
        for input_file in input_files:
            logger.info(f"\n📄 {input_file.name}")
            
            with open(input_file, 'r') as f:
                lines = f.readlines()
            
            if sample_size:
                lines = lines[:sample_size]
            
            for line in tqdm(lines, desc="Extracting KG"):
                try:
                    chunk = json.loads(line)
                    result = self.process_chunk(chunk, chunk_id)
                    
                    if result:
                        all_extractions.append(result)
                    
                    chunk_id += 1
                    
                    # Rate limiting for LLM
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.debug(f"Error: {e}")
        
        # Save triplets
        triplets_file = TRIPLET_DIR / "all_triplets.jsonl"
        with open(triplets_file, 'w') as f:
            for extraction in all_extractions:
                f.write(json.dumps(extraction) + '\n')
        logger.info(f"\n✅ Saved triplets: {triplets_file}")
        
        # Detect communities
        community_map = self.detect_communities()
        
        # Export for visualization
        self.export_for_visualization(community_map)
        
        # Save stats
        stats = {
            "stage": "4.5",
            **self.stats,
            "completed_at": datetime.now().isoformat()
        }
        
        with open(EXPORT_DIR / "stage4_5_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 4.5 COMPLETE - KNOWLEDGE GRAPH READY!")
        logger.info("=" * 80)
        logger.info(f"📊 Chunks processed: {self.stats['chunks_processed']:,}")
        logger.info(f"🔍 Entities extracted: {self.stats['entities_extracted']:,}")
        logger.info(f"🔗 Triplets extracted: {self.stats['triplets_extracted']:,}")
        logger.info(f"🗄️ Neo4j nodes: {self.stats['neo4j_nodes']:,}")
        logger.info(f"🔀 Neo4j edges: {self.stats['neo4j_edges']:,}")
        logger.info(f"🎨 Communities: {self.stats['communities_detected']}")
        logger.info("")
        logger.info("📁 Exports:")
        logger.info(f"   - {EXPORT_DIR}/brain_graph_cytoscape.json (Cytoscape)")
        logger.info(f"   - {EXPORT_DIR}/brain_graph_infranodus.json (InfraNodus/Obsidian)")
        logger.info(f"   - {EXPORT_DIR}/brain_graph.graphml (Gephi)")
        logger.info("")
        logger.info("🧠 YOUR BRAIN IS READY FOR VISUALIZATION!")
        logger.info("=" * 80)
    
    def close(self):
        """Cleanup"""
        if self.neo4j_driver:
            self.neo4j_driver.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Stage 4.5: Knowledge Graph Extraction")
    parser.add_argument("--sample", type=int, help="Process only N chunks (for testing)")
    args = parser.parse_args()
    
    extractor = KnowledgeGraphExtractor()
    try:
        extractor.run(sample_size=args.sample)
    finally:
        extractor.close()


if __name__ == "__main__":
    main()
