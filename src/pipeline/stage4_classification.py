#!/usr/bin/env python3
"""
STAGE 4: Hierarchical AI Classification
- Classifies into 3 addresses: Gate, Room, Place
- Place has variable depth (1-7 layers)
- Difficulty separated from location
- Groq → MegaLLM → GLM → Ollama rotation
"""

import os
import json
import time
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import requests
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import threading
import shutil
from stage4_puter_extension import get_puter_models, smart_parse_response

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
INPUT_DIR = Path("cleaned_entities")  # Fixed: was "cleaned", now points to actual directory
OUTPUT_DIR = Path("labeled")
OUTPUT_DIR.mkdir(exist_ok=True)

TAXONOMY_FILE = Path("config/taxonomy_hierarchical.json")
CONFIDENCE_THRESHOLD = 0.7

# ============================================================================
# API Configuration - 6 Cloud APIs (OLLAMA REMOVED FOR SPEED)
# ============================================================================
# VERIFICATION DATE: 2024-12-24 18:30 IST
# PERFORMANCE UPDATE: Ollama removed - was causing 100x slowdown by blocking
# fast cloud APIs while processing on local CPU at ~1 chunk/min
# 
# GROQ MODELS (2 working):
#   ✅ llama-3.3-70b-versatile - Fast &amp; powerful, recommended
#   ✅ llama-3.1-8b-instant - Ultra fast, lowest latency
#
# MEGALLM MODELS (4 working, 2 excluded):
#   ✅ llama3.3-70b-instruct - Meta's best, works perfectly
#   ✅ deepseek-ai/deepseek-v3.1-terminus - Open source leader
#   ✅ deepseek-r1-distill-llama-70b - Reasoning specialist
#   ✅ openai-gpt-oss-20b - OpenAI open source
#   ❌ alibaba-qwen3-32b - EXCLUDED: Rate limit exceeded (429)
#   ❌ moonshotai/kimi-k2-instruct-0905 - EXCLUDED: Rate limit exceeded (429)
#
# ZHIPU/GLM MODELS (0 working, 1 excluded):
#   ❌ glm-4.5-flash - EXCLUDED: Rate limit (429 - Chinese API too many requests)
#   Note: May work later, but excluding for now due to immediate rate limit
#
# TOTAL: 6 cloud models (Groq: 2, MegaLLM: 4)
# ROTATION: Round-robin across all 6 APIs for maximum throughput
# ============================================================================

WORKING_APIS = [
    # ========== GROQ (2 models - VERIFIED WORKING) ==========
    {
        "name": "groq_llama33_70b",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key": os.getenv("GROQ_API_KEY"),
        "model": "llama-3.3-70b-versatile",
        "timeout": 25
    },
    {
        "name": "groq_llama31_8b",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key": os.getenv("GROQ_API_KEY"),
        "model": "llama-3.1-8b-instant",
        "timeout": 20
    },
    
    # ========== MEGALLM FREE TIER (4 models - TESTED & VERIFIED) ==========
    {
        "name": "mega_llama33_70b",
        "url": "https://ai.megallm.io/v1/chat/completions",
        "key": os.getenv("MEGALLM_API_KEY"),
        "model": "llama3.3-70b-instruct",
        "timeout": 25
    },
    {
        "name": "mega_deepseek_v3",
        "url": "https://ai.megallm.io/v1/chat/completions",
        "key": os.getenv("MEGALLM_API_KEY"),
        "model": "deepseek-ai/deepseek-v3.1-terminus",
        "timeout": 30
    },
    {
        "name": "mega_gpt_oss_20b",
        "url": "https://ai.megallm.io/v1/chat/completions",
        "key": os.getenv("MEGALLM_API_KEY"),
        "model": "openai-gpt-oss-20b",
        "timeout": 25
    },
    {
        "name": "mega_llama3_8b",
        "url": "https://ai.megallm.io/v1/chat/completions",
        "key": os.getenv("MEGALLM_API_KEY"),
        "model": "llama3-8b-instruct",
        "timeout": 20
    }
]

# GLM excluded due to rate limiting - uncomment if you have higher tier access
# {
#     "name": "zhipu_glm45_flash",
#     "url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
#     "key": os.getenv("GLM_API_KEY"),
#     "model": "glm-4.5-flash",
#     "timeout": 25
# }

# ============================================================================
# PUTER.JS MODELS - 500+ FREE AI MODELS FROM BROWSER AUTH!
# ============================================================================
try:
    PUTER_MODELS = get_puter_models()
    WORKING_APIS.extend(PUTER_MODELS)
    if PUTER_MODELS:
        logger.info(f"🚀 TURBO MODE: Added {len(PUTER_MODELS)} Puter.js models!")
except Exception as e:
    logger.info(f"ℹ️  Puter.js not available ({e}). Using standard APIs only.")


class HierarchicalClassifier:
    """
    Hierarchical classification with 3-address system:
    - Gate (house)
    - Room (branch)
    - Place (variable-depth location)
    """
    
    def __init__(self, taxonomy_file=TAXONOMY_FILE):
        # Load hierarchical taxonomies
        if taxonomy_file.exists():
            with open(taxonomy_file) as f:
                data = json.load(f)
                self.taxonomies = data.get('taxonomies', {})
                self.difficulty_map = data.get('difficulty_levels', {})
            logger.info(f"Loaded {len(self.taxonomies)} hierarchical taxonomies")
        else:
            logger.warning(f"Taxonomy file not found: {taxonomy_file}")
            logger.warning("Run: python scripts/utils/parse_hierarchical_taxonomies.py")
            self.taxonomies = {}
            self.difficulty_map = {}
        
        self.api_index = 0
        self.stats = {
            'total_processed': 0,
            'classified': 0,
            'failed': 0,
            'by_gate': {},
            'by_depth': {},
            'api_calls': {api['name']: 0 for api in WORKING_APIS}
        }
    
    def analyze_personality(self, text: str) -> dict:
        """
        Cross-cutting personality analysis for ALL content.
        Returns scores 0-1 for personality traits.
        """
        text_lower = text.lower()
        words = text_lower.split()
        
        # Helpful markers
        helpful_phrases = ['let me help', 'i can assist', 'here\'s how', 'you can', 
                          'try this', 'alternatively', 'consider', 'might want to']
        helpful_score = sum(1 for phrase in helpful_phrases if phrase in text_lower) / 3
        
        # Polite markers
        polite_markers = ['please', 'thank you', 'thanks', 'appreciate', 'kindly',
                         'would you', 'could you', 'may i', 'excuse me', 'sorry']
        polite_score = sum(1 for marker in polite_markers if marker in text_lower) / 4
        
        # Professional markers
        professional_words = ['regarding', 'furthermore', 'however', 'therefore',
                             'accordingly', 'nevertheless', 'consequently']
        professional_score = sum(1 for word in professional_words if word in text_lower) / 3
        
        # Casual markers
        casual_markers = ['gonna', 'wanna', 'yeah', 'cool', 'awesome', 'hey',
                         'lol', 'btw', 'fyi', 'basically', 'like', 'just']
        casual_score = sum(1 for marker in casual_markers if marker in text_lower) / 4
        
        # Technical jargon (inverse of casualness)
        tech_markers = ['algorithm', 'function', 'parameter', 'variable', 'class',
                       'method', 'implementation', 'configuration', 'optimization']
        technical_score = sum(1 for marker in tech_markers if marker in text_lower) / 4
        
        # Normalize to 0-1 range
        return {
            'helpful': min(helpful_score, 1.0),
            'polite': min(polite_score, 1.0),
            'professional': min(professional_score, 1.0),
            'casual': min(casual_score, 1.0),
            'technical': min(technical_score, 1.0)
        }
    
    def classify_gate(self, text):
        """Primary gate classification using comprehensive keywords extracted from taxonomies"""
        
        # Load comprehensive keywords from taxonomies (cached)
        if not hasattr(self, '_gate_keywords_cache'):
            keywords_file = Path("config/gate_keywords.json")
            if keywords_file.exists():
                with open(keywords_file) as f:
                    self._gate_keywords_cache = json.load(f)
                logger.info(f"Loaded comprehensive keywords for {len(self._gate_keywords_cache)} gates")
            else:
                # Fallback to basic keywords if extraction file not found
                log                                                                                                                                                                                               
                self._gate_keywords_cache = {
                    "MATH-CTRL": ["calculus", "algebra", "theorem", "proof", "matrix", "differential", "integral"],
                    "PHYS-QUANT": ["physics", "quantum", "momentum", "force", "velocity", "energy", "particle"],
                    "CHEM-BIO": ["chemistry", "biology", "molecule", "protein", "reaction", "cell", "enzyme"],
                    "CODE-GEN": ["algorithm", "function", "class", "array", "loop", "recursion", "programming"],
                    "AV-NAV": ["robot", "slam", "navigation", "autonomous", "lidar", "path planning", "localization"],
                    "SPACE-AERO": ["rocket", "satellite", "aerospace", "orbital", "trajectory", "spacecraft"],
                    "TELEM-LOG": ["telemetry", "logging", "monitoring", "diagnostics", "sensor data"],
                    "HARD-SPEC": ["hardware", "specification", "circuit", "processor", "memory", "architecture"],
                    "SYS-OPS": ["kernel", "driver", "embedded", "rtos", "firmware", "operating system"],
                    "OLYMPIAD": ["competition", "problem solving", "olympiad", "challenge", "mathematics"],
                    "ROS2-ROBOT": ["ros", "ros2", "node", "topic", "service", "gazebo", "robot operating"],
                    "PERSONALITY": ["polite", "helpful", "empathetic", "professional", "conversation"]
                }
        
        text_lower = text.lower()
        
        # Score each gate based on keyword matches
        scores = {}
        for gate, keywords in self._gate_keywords_cache.items():
            # Count keyword matches
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[gate] = score
        
        # Return gate with highest score, or MATH-CTRL as default
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        else:
            logger.debug("No keyword matches found, defaulting to MATH-CTRL")
            return "MATH-CTRL"
    
    def format_hierarchical_taxonomy(self, gate_name):
        """Format taxonomy showing full hierarchy"""
        if gate_name not in self.taxonomies:
            return "No taxonomy available"
        
        taxonomy = self.taxonomies[gate_name]
        lines = []
        
        for branch in taxonomy.get('branches', []):
            lines.append(f"\n## Branch: {branch['branch_name']} (ID: {branch['branch_id']})")
            lines.append(f"   Max Depth: {branch['max_depth']} layers")
            
            # Format hierarchical layers
            self._format_layers(branch['hierarchy'], lines, indent=1)
        
        return '\n'.join(lines[:100])  # Limit for token budget
    
    def _format_layers(self, layers, lines, indent=0):
        """Recursively format layers"""
        for layer in layers[:5]:  # Limit per level
            prefix = "  " * indent + "├─"
            lines.append(f"{prefix} {layer['name']} (ID: {layer['id']}, Depth: {layer['depth']})")
            
            if 'sublayers' in layer and indent < 3:  # Limit depth shown
                self._format_layers(layer['sublayers'], lines, indent + 1)
    
    def call_api(self, prompt, api):
        """Call API with error handling"""
        try:
            headers = {"Content-Type": "application/json"}
            if api['key']:
                headers["Authorization"] = f"Bearer {api['key']}"
            
            # Puter-specific headers
            if api.get('is_puter'):
                headers['Origin'] = 'https://puter.com'
                headers['Referer'] = 'https://puter.com/'
                if api.get('cookie'):
                    headers['Cookie'] = api['cookie']
            
            payload = {
                "model": api['model'],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 800
            }
            
            response = requests.post(
                api['url'],
                headers=headers,
                json=payload,
                timeout=api['timeout']
            )
            
            self.stats['api_calls'][api['name']] += 1
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'], api['name']
            
            return None, None
            
        except Exception as e:
            logger.debug(f"{api['name']} error: {e}")
            return None, None
    
    def ai_classify_batch(self, chunks_batch):
        """Classify a batch of chunks in a single API call"""
        if not chunks_batch:
            return []
        
        # Group by gate for better taxonomy context
        by_gate = {}
        for idx, chunk in enumerate(chunks_batch):
            text = chunk.get('content', '')
            gate_name = self.classify_gate(text)
            if gate_name not in by_gate:
                by_gate[gate_name] = []
            by_gate[gate_name].append((idx, chunk, text))
        
        results = [None] * len(chunks_batch)
        
        # Process each gate group
        for gate_name, gate_chunks in by_gate.items():
            taxonomy_str = self.format_hierarchical_taxonomy(gate_name)
            gate_data = self.taxonomies.get(gate_name, {})
            gate_id = gate_data.get('gate_id', 0)
            
            # Build batch prompt
            batch_texts = []
            for idx, chunk, text in gate_chunks[:10]:  # Max 10 per batch to avoid token limits
                batch_texts.append(f"[CHUNK_{idx}]\n{text[:800]}\n")
            
            prompt = f"""You are an expert in {gate_name}. Classify these {len(batch_texts)} text chunks using the hierarchical taxonomy.

HIERARCHICAL TAXONOMY:
{taxonomy_str[:1000]}

TEXTS TO CLASSIFY:
{"".join(batch_texts)}

Output JSON array (exact format, NO explanation):
[
  {{
    "chunk_id": 0,
    "gate": {{"name": "{gate_name}", "id": {gate_id}}},
    "room": {{"branch_name": "...", "branch_id": 0}},
    "place": {{"path": ["Layer1"], "path_ids": [10], "depth": 1}},
    "difficulty": {{"level": "Foundation", "level_id": 1}},
    "concepts": [],
    "prerequisites": [],
    "confidence": 0.85
  }}
]

RULES: Variable depth (1-7), match taxonomy exactly, confidence <0.7 = skip"""
            
            # Try APIs in rotation
            for i in range(len(WORKING_APIS)):
                api = WORKING_APIS[self.api_index % len(WORKING_APIS)]
                self.api_index += 1
                
                if not api['key']:
                    continue
                
                result, api_used = self.call_api(prompt, api)
                
                if result:
                    try:
                        # Smart parser for different model formats (Claude=array, GPT=object)
                        parsed = smart_parse_response(result, api)
                        if not parsed:
                            continue
                        
                        # Map results back to original indices
                        for item in parsed:
                            chunk_id = item.get('chunk_id', -1)
                            if 0 <= chunk_id < len(gate_chunks):
                                orig_idx, orig_chunk, _ = gate_chunks[chunk_id]
                                if self._validate_classification(item, gate_name):
                                    item['api_used'] = api_used
                                    results[orig_idx] = (orig_chunk, item)
                        break
                        
                    except Exception as e:
                        logger.debug(f"Batch parse error with {api_used}: {e}")
                        continue
        
        return [r for r in results if r is not None]
    
    def _validate_classification(self, classification, gate_name):
        """Validate hierarchical classification"""
        try:
            # Check required fields
            if not all(k in classification for k in ['gate', 'room', 'place', 'difficulty']):
                return False
            
            # Check place structure
            place = classification['place']
            if not isinstance(place['path'], list) or not isinstance(place['path_ids'], list):
                return False
            
            if len(place['path']) != len(place['path_ids']) or len(place['path']) != place['depth']:
                return False
            
            if place['depth'] < 1 or place['depth'] > 7:
                return False
            
            # Check difficulty
            if classification['difficulty']['level'] not in self.difficulty_map:
                return False
            
            return True
            
        except:
            return False
    
    def classify_chunk(self, chunk):
        """Legacy method - now using batch processing instead"""
        # Kept for compatibility but not used in new batch processing
        pass

def main():
    logger.info("=" * 80)
    logger.info("STAGE 4: HIERARCHICAL BATCH CLASSIFICATION (PARALLEL APIs)")
    logger.info("=" * 80)
    
    classifier = HierarchicalClassifier()
    
    if not classifier.taxonomies:
        logger.error("No taxonomies loaded! Run parse_hierarchical_taxonomies.py first.")
        return
    
    # Get input files and sort by size (process small files first)
    input_files = list(INPUT_DIR.glob("*.jsonl"))
    input_files_with_size = [(f, f.stat().st_size) for f in input_files]
    input_files_with_size.sort(key=lambda x: x[1])
    
    logger.info(f"Found {len(input_files)} files to process")
    logger.info(f"Using {len(WORKING_APIS)} APIs in PARALLEL (batch mode)")
    logger.info(f"Batch size: 10 chunks per API call")
    logger.info("=" * 80)
    
    # Initialize output file with resume capability
    output_file = OUTPUT_DIR / "chunks_hierarchical.jsonl"
    output_file.parent.mkdir(exist_ok=True)
    
    # RESUME CAPABILITY: Load already-processed chunk IDs
    processed_ids = set()
    if output_file.exists():
        # Backup existing file
        backup_file = OUTPUT_DIR / f"chunks_hierarchical_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        shutil.copy2(output_file, backup_file)
        logger.info(f"📦 Backed up existing output to: {backup_file.name}")
        
        # Load processed chunk IDs
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    chunk = json.loads(line)
                    # Use source + ID as unique identifier
                    chunk_id = f"{chunk.get('source', '')}_{chunk.get('id', '')}"
                    processed_ids.add(chunk_id)
                except:
                    pass
        logger.info(f"♻️  Resume mode: {len(processed_ids):,} chunks already processed (will skip)")
    else:
        logger.info("🆕 Starting fresh - no existing output found")
    
    # Process each file
    total_files = len(input_files_with_size)
    
    for file_idx, (input_file, file_size) in enumerate(input_files_with_size, 1):
        file_size_mb = file_size / (1024 * 1024)
        logger.info(f"\n[FILE {file_idx}/{total_files}] Processing: {input_file.name} ({file_size_mb:.1f} MB)")
        
        # Load chunks from this file
        file_chunks = []
        try:
            with open(input_file, 'r') as f:
                for line in f:
                    try:
                        chunk = json.loads(line)
                        # Check if already processed
                        chunk_id = f"{chunk.get('source', '')}_{chunk.get('id', '')}"
                        if chunk_id not in processed_ids:
                            file_chunks.append(chunk)
                    except:
                        pass
        except Exception as e:
            logger.error(f"Error reading {input_file.name}: {e}")
            continue
        
        total_in_file = len(file_chunks)
        logger.info(f"  Loaded {total_in_file:,} chunks (after skipping already-processed)")
        
        if not file_chunks:
            logger.info(f"  ✅ Skipping - all chunks already processed or no valid chunks")
            continue
        
        # Create batches of 10 chunks each
        BATCH_SIZE = 10
        batches = [file_chunks[i:i+BATCH_SIZE] for i in range(0, len(file_chunks), BATCH_SIZE)]
        logger.info(f"  Created {len(batches):,} batches ({BATCH_SIZE} chunks/batch)")
        
        # Distribute batches across APIs (round-robin)
        api_batches = {i: [] for i in range(len(WORKING_APIS))}
        for batch_idx, batch in enumerate(batches):
            api_idx = batch_idx % len(WORKING_APIS)
            api_batches[api_idx].append(batch)
        
        # Show distribution
        for api_idx, api in enumerate(WORKING_APIS):
            num_batches = len(api_batches[api_idx])
            num_chunks = sum(len(b) for b in api_batches[api_idx])
            logger.info(f"    {api['name']}: {num_batches} batches ({num_chunks} chunks)")
        
        # Process batches in parallel - each API handles its own batches
        all_results = []
        lock = threading.Lock()
        
        def process_api_batches(api_idx, api, batches_to_process):
            """Each API processes its assigned batches"""
            api_results = []
            for batch in tqdm(batches_to_process, 
                            desc=f"    {api['name'][:20]}", 
                            leave=False,
                            position=api_idx):
                # Process this batch
                batch_results = classifier.ai_classify_batch(batch)
                
                # Build output for each classified chunk
                for orig_chunk, classification in batch_results:
                    classifier.stats['total_processed'] += 1
                    
                    if not classification or classification.get('confidence', 0) < CONFIDENCE_THRESHOLD:
                        classifier.stats['failed'] += 1
                        continue
                    
                    # Build complete output
                    output = {
                        **orig_chunk,
                        "gate": classification["gate"],
                        "room": classification["room"],
                        "place": classification["place"],
                        "difficulty": classification["difficulty"],
                        "concepts": classification.get("concepts", []),
                        "prerequisites": classification.get("prerequisites", []),
                        "confidence": classification["confidence"],
                        "classified_by": classification.get("api_used", "unknown"),
                        "classified_at": datetime.now().isoformat(),
                        "status": "approved" if classification["confidence"] > 0.8 else "flagged"
                    }
                    
                    # Add personality analysis
                    text = orig_chunk.get('content', '')
                    output['personality'] = classifier.analyze_personality(text)
                    
                    # Update stats
                    classifier.stats['classified'] += 1
                    gate_name = classification['gate']['name']
                    classifier.stats['by_gate'][gate_name] = classifier.stats['by_gate'].get(gate_name, 0) + 1
                    depth = classification['place']['depth']
                    classifier.stats['by_depth'][depth] = classifier.stats['by_depth'].get(depth, 0) + 1
                    
                    api_results.append(output)
            
            with lock:
                all_results.extend(api_results)
        
        # Launch parallel threads - one per API
        threads = []
        for api_idx, api in enumerate(WORKING_APIS):
            if not api_batches[api_idx]:
                continue
            thread = threading.Thread(
                target=process_api_batches,
                args=(api_idx, api, api_batches[api_idx])
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all APIs to complete
        for thread in threads:
            thread.join()
        
        # Save results for this file
        logger.info(f"  Saving {len(all_results):,} classified chunks...")
        with open(output_file, 'a') as f:
            for chunk in all_results:
                f.write(json.dumps(chunk) + '\n')
        
        # Clear memory
        del file_chunks
        del batches
        del all_results
        
        logger.info(f"  ✅ Completed {input_file.name} - {file_idx}/{total_files} files done")
    
    # Final Stats
    stats = {
        "stage": 4,
        "version": "3.0_batch_parallel",
        "batch_size": BATCH_SIZE,
        "total_processed": classifier.stats['total_processed'],
        "classified": classifier.stats['classified'],
        "failed": classifier.stats['failed'],
        "success_rate": (classifier.stats['classified'] / classifier.stats['total_processed'] 
                        if classifier.stats['total_processed'] > 0 else 0),
        "by_gate": classifier.stats['by_gate'],
        "by_depth": classifier.stats['by_depth'],
        "api_calls": classifier.stats['api_calls'],
        "parallel_apis": len(WORKING_APIS),
        "files_processed": total_files,
        "completed_at": datetime.now().isoformat()
    }
    
    with open(OUTPUT_DIR / "stage4_hierarchical_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 4 COMPLETE (BATCH + PARALLEL)")
    logger.info(f"Files: {total_files}, Batch size: {BATCH_SIZE}")
    logger.info(f"Classified: {stats['classified']:,} / {stats['total_processed']:,} "
               f"({stats['success_rate']*100:.1f}%)")
    logger.info(f"By gate: {stats['by_gate']}")
    logger.info(f"API calls: {stats['api_calls']}")
    logger.info("✅ Ready for Stage 4.5: PostgreSQL Insert")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()

