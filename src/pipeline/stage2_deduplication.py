#!/usr/bin/env python3
"""
STAGE 2: Deduplication (250GB → 180GB)
Uses datasketch MinHash LSH for near-duplicate detection
Memory-efficient streaming with batching

Tools: datasketch, simhash
Input: raw/*.jsonl
Output: deduped/*.jsonl
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

try:
    from datasketch import MinHash, MinHashLSH
    DATASKETCH_AVAILABLE = True
except ImportError:
    DATASKETCH_AVAILABLE = False
    print("⚠️  datasketch not installed. Run: pip install datasketch --break-system-packages")

# Configuration - paths relative to script location
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
INPUT_DIR = PROJECT_DIR / "raw"
OUTPUT_DIR = PROJECT_DIR / "deduped"
OUTPUT_DIR.mkdir(exist_ok=True)

# MinHash parameters for 250GB scale
NUM_PERM = 128  # Hash permutations
THRESHOLD = 0.85  # 85% similarity = duplicate
BATCH_SIZE = 10000  # Process in batches to manage memory

# Files that need LSH (conversational datasets with near-duplicate risk)
LSH_FILES = [
    'ultrafeedback.jsonl',
    'openassistant_conversations.jsonl', 
    'anthropic_hh_rlhf.jsonl',
    'intel_orca_politeness.jsonl'
]

class DeduplicatorLSH:
    """Memory-efficient deduplication using LSH"""
    
    def __init__(self, use_lsh=True):
        self.use_lsh = use_lsh
        if use_lsh:
            self.lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
        else:
            self.lsh = None
        self.seen_hashes = set()  # Exact hash dedup
        self.stats = {
            'total_processed': 0,
            'exact_duplicates': 0,
            'near_duplicates': 0,
            'kept': 0
        }
    
    def create_minhash(self, text):
        """Generate MinHash for text"""
        m = MinHash(num_perm=NUM_PERM)
        # Tokenize by words
        words = text.lower().split()
        for word in words:
            m.update(word.encode('utf8'))
        return m
    
    def is_duplicate(self, chunk):
        """Check if chunk is duplicate (exact or near)"""
        # Generate content_hash if not present (for backward compatibility)
        if 'content_hash' not in chunk:
            content = chunk.get('content', '')
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            chunk['content_hash'] = content_hash  # Add for output
        else:
            content_hash = chunk['content_hash']
        
        # 1. Exact duplicate check (fast)
        if content_hash in self.seen_hashes:
            self.stats['exact_duplicates'] += 1
            return True
        
        # 2. Near-duplicate check (LSH) - only if enabled
        if self.use_lsh:
            minhash = self.create_minhash(chunk['content'])
            
            # Query LSH for similar documents
            result = self.lsh.query(minhash)
            if result:
                self.stats['near_duplicates'] += 1
                return True
            
            # Not a duplicate - add to LSH index
            self.lsh.insert(content_hash, minhash)
        
        # Not a duplicate - add to seen hashes
        self.seen_hashes.add(content_hash)
        
        return False
    
    def process_batch(self, chunks):
        """Process batch of chunks"""
        unique_chunks = []
        
        for chunk in chunks:
            self.stats['total_processed'] += 1
            
            if not self.is_duplicate(chunk):
                unique_chunks.append(chunk)
                self.stats['kept'] += 1
            
            # Progress update every 1000 chunks
            if self.stats['total_processed'] % 1000 == 0:
                self.print_progress()
        
        return unique_chunks
    
    def print_progress(self):
        """Print dedup progress"""
        total = self.stats['total_processed']
        kept = self.stats['kept']
        removed = self.stats['exact_duplicates'] + self.stats['near_duplicates']
        ratio = (removed / total * 100) if total > 0 else 0
        
        print(f"  Processed: {total:,} | Kept: {kept:,} | Removed: {removed:,} ({ratio:.1f}%)")

def process_file(input_file, deduplicator):
    """Process single JSONL file"""
    
    batch = []
    output_chunks = []
    
    with open(input_file, 'r') as f:
        for line in f:
            try:
                chunk = json.loads(line)
                batch.append(chunk)
                
                # Process in batches
                if len(batch) >= BATCH_SIZE:
                    unique = deduplicator.process_batch(batch)
                    output_chunks.extend(unique)
                    batch = []
                    
            except json.JSONDecodeError:
                continue
    
    # Process remaining
    if batch:
        unique = deduplicator.process_batch(batch)
        output_chunks.extend(unique)
    
    # Save deduplicated chunks
    output_file = OUTPUT_DIR / input_file.name
    with open(output_file, 'w') as f:
        for chunk in output_chunks:
            f.write(json.dumps(chunk) + '\n')
    
    print(f"  ✅ Saved {len(output_chunks):,} unique chunks to {output_file.name}")
    return len(output_chunks)

def main():
    if not DATASKETCH_AVAILABLE:
        print("\n❌ Cannot proceed without 'datasketch' library")
        print("Install with: pip install datasketch --break-system-packages")
        return
    
    print("=" * 80)
    print("STAGE 2: DEDUPLICATION (MinHash LSH)")
    print("=" * 80)
    print(f"Threshold: {THRESHOLD} | Permutations: {NUM_PERM} | Batch: {BATCH_SIZE}")
    print("=" * 80)
    
    # Process all JSONL files
    input_files = list(INPUT_DIR.glob("*.jsonl"))
    total_output = 0
    skipped = 0
    
    for input_file in input_files:
        output_file = OUTPUT_DIR / input_file.name
        
        # Skip if already processed
        if output_file.exists():
            print(f"\n⏭️  Skipping (already processed): {input_file.name}")
            skipped += 1
            continue
        
        # Determine if this file needs LSH
        use_lsh = input_file.name in LSH_FILES
        dedup_type = "LSH (near-duplicate)" if use_lsh else "Exact hash only"
        print(f"\n📄 Processing: {input_file.name} [{dedup_type}]")
        
        # Create fresh deduplicator for this file
        deduplicator = DeduplicatorLSH(use_lsh=use_lsh)
        count = process_file(input_file, deduplicator)
        total_output += count
    
    if skipped > 0:
        print(f"\n📊 Skipped {skipped} already-processed files")
    
    # Generate stats
    stats = {
        "stage": 2,
        "total_processed": deduplicator.stats['total_processed'],
        "exact_duplicates": deduplicator.stats['exact_duplicates'],
        "near_duplicates": deduplicator.stats['near_duplicates'],
        "kept": deduplicator.stats['kept'],
        "dedup_ratio": (deduplicator.stats['exact_duplicates'] + deduplicator.stats['near_duplicates']) / deduplicator.stats['total_processed'] if deduplicator.stats['total_processed'] > 0 else 0,
        "output_directory": str(OUTPUT_DIR),
        "completed_at": datetime.now().isoformat()
    }
    
    with open(OUTPUT_DIR / "stage2_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "=" * 80)
    print("STAGE 2 COMPLETE")
    print("=" * 80)
    print(f"Processed: {stats['total_processed']:,}")
    print(f"Kept: {stats['kept']:,}")
    print(f"Removed: {stats['exact_duplicates'] + stats['near_duplicates']:,} ({stats['dedup_ratio']*100:.1f}%)")
    print(f"Output: {OUTPUT_DIR}/")
    print("\n✅ Ready for Stage 3: Cleaning")
    print("=" * 80)

if __name__ == "__main__":
    main()
