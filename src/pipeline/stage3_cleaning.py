#!/usr/bin/env python3
"""
STAGE 3: Clean & Filter (180GB → 120GB)
Text cleaning + quality filtering

Tools: ftfy, clean-text, unidecode, textstat, spacy
Input: deduped/*.jsonl
Output: cleaned/*.jsonl
"""

import os
import json
import re
import gc
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

try:
    import ftfy
    from cleantext import clean
    from unidecode import unidecode
    import textstat
    CLEANING_AVAILABLE = True
except ImportError:
    CLEANING_AVAILABLE = False
    print("⚠️  Cleaning libraries not installed")
    print("Run: pip install ftfy clean-text unidecode textstat langdetect --break-system-packages")

# Optional: Language detection
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0  # Reproducible
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("⚠️  langdetect not installed (English filtering disabled)")

# Configuration - paths relative to script location
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
INPUT_DIR = PROJECT_DIR / "deduped"
OUTPUT_DIR = PROJECT_DIR / "cleaned"
OUTPUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 10000  # Increased for better throughput
MIN_LENGTH = 100  # Minimum chars
MAX_LENGTH = 50000  # Maximum chars
QUALITY_THRESHOLD = 0.4  # Heuristic score threshold

# High-value keywords (from 250gb.md)
KEYWORDS = [
    'ros2', 'ros', 'robot', 'navigation', 'slam', 'control', 'sensor',
    'algorithm', 'matrix', 'vector', 'calculus', 'physics', 'python',
    'function', 'class', 'import', 'package', 'node', 'topic'
]

class TextCleaner:
    """Production text cleaning pipeline - optimized for speed"""
    
    # Pre-compile all regex patterns for performance
    RE_WHITESPACE = re.compile(r'\s+')
    RE_HTML_TAG = re.compile(r'<[^>]+>')
    RE_HTML_ENTITY = re.compile(r'&[a-zA-Z]+;|&#\d+;')
    RE_HYPHEN_BREAK = re.compile(r'(\w)-\s*\n\s*(\w)')
    RE_PAGE_NUM = re.compile(r'^\d+\s*$', re.MULTILINE)
    RE_PAGE_HEADER = re.compile(r'^Page \d+.*$', re.MULTILINE)
    RE_PAGE_OF = re.compile(r'^\d+\s+of\s+\d+.*$', re.MULTILINE)
    RE_DATE_HEADER = re.compile(r'^[\d\-/]+\s*$', re.MULTILINE)
    RE_CAPS_HEADER = re.compile(r'^[A-Z\s]{20,}$', re.MULTILINE)
    RE_ORPHAN_REF = re.compile(r'\[\d+\]\s*\n')
    RE_TRIPLE_NEWLINE = re.compile(r'\n{3,}')
    RE_SPACE_PUNCT = re.compile(r'([a-z])\s+([.,;:\)])')
    RE_PAREN_SPACE = re.compile(r'([\(\[])\s+([a-zA-Z])')
    RE_DOLLAR_SPACE = re.compile(r'\$\s+|\s+\$')
    RE_TABLE_BORDER = re.compile(r'\|[\s\-]+\|')
    RE_SEPARATOR = re.compile(r'^\s*[\-=]{3,}\s*$', re.MULTILINE)
    RE_NEWLINE_SPACE = re.compile(r'\n\s+|\s+\n')
    
    def __init__(self):
        self.stats = {
            'total_processed': 0,
            'unicode_fixed': 0,
            'too_short': 0,
            'too_long': 0,
            'low_quality': 0,
            'non_english': 0,
            'kept': 0
        }
        # Reduced sampling for speed - only check 5% of chunks
        self.lang_check_counter = 0
        self.lang_sample_rate = 20  # Only check every 20th chunk (5%)
    
    def detect_content_type(self, text):
        """Detect if content is math/code/physics/telemetry - MUST PRESERVE these!"""
        # Count LaTeX/math indicators
        latex_count = text.count('$') + text.count('\\[') + text.count('\\(')
        latex_count += text.count('\\frac') + text.count('\\sum') + text.count('\\int')
        latex_count += text.count('\\partial') + text.count('\\nabla')
        
        # Count code indicators
        code_count = text.count('```') + text.count('def ') + text.count('class ')
        code_count += text.count('import ') + text.count('function ') + text.count('{')
        
        # Count physics/scientific indicators
        physics_count = 0
        # Greek letters in scientific text
        physics_count += text.count('alpha') + text.count('beta') + text.count('gamma')
        physics_count += text.count('theta') + text.count('omega') + text.count('delta')
        # Units of measurement
        physics_count += text.count('kg') + text.count('m/s') + text.count('N/m')
        physics_count += text.count('Hz') + text.count('rad/s') + text.count('deg/s')
        # Scientific notation patterns
        import re
        sci_notation = len(re.findall(r'\d+\.?\d*[eE][+-]?\d+', text))
        physics_count += sci_notation
        
        # Count telemetry indicators
        telemetry_count = 0
        # Timestamps (HH:MM:SS or similar)
        telemetry_count += len(re.findall(r'\d{2}:\d{2}:\d{2}', text))
        # Sensor data patterns (float numbers with units)
        telemetry_count += text.count('sensor') + text.count('timestamp')
        telemetry_count += text.count('voltage') + text.count('current') + text.count('temperature')
        # Data arrays/vectors [x, y, z]
        telemetry_count += text.count('[') + text.count(']') // 10  # Don't over-count
        
        # Determine content type
        is_math_heavy = latex_count > 5
        is_code_heavy = code_count > 3 or '```' in text
        is_physics = physics_count > 5 or sci_notation > 3
        is_telemetry = telemetry_count > 5
        
        return {
            'is_math': is_math_heavy,
            'is_code': is_code_heavy,
            'is_physics': is_physics,
            'is_telemetry': is_telemetry,
            'is_technical': is_math_heavy or is_code_heavy or is_physics or is_telemetry
        }
    
    def is_english(self, text, content_type=None):
        """Check if text is English (with sampling + technical content bypass)"""
        if not LANGDETECT_AVAILABLE:
            return True  # Skip check if not available
        
        # CRITICAL: Skip language check for technical content
        # Math/code/physics are universal and langdetect often wrongly flags them
        if content_type and content_type.get('is_technical'):
            return True  # Technical content is always "English" for our purposes
        
        # Sample-based checking: only verify every Nth chunk for regular text
        self.lang_check_counter += 1
        if self.lang_check_counter % self.lang_sample_rate != 0:
            return True  # Skip check for most chunks
        
        try:
            lang = detect(text[:500])  # Only check first 500 chars
            return lang == 'en'
        except:
            return True  # Assume English on error
    
    def fix_unicode(self, text):
        """Fix Unicode errors - handles all content types"""
        if not CLEANING_AVAILABLE:
            return text
        try:
            # ftfy handles: PDFs, LaTeX, HTML, code, etc.
            fixed = ftfy.fix_text(text)
            if fixed != text:
                self.stats['unicode_fixed'] += 1
            return fixed
        except Exception as e:
            # Fallback: return original text if fixing fails
            return text
    
    def remove_noise(self, text, content_type=None):
        """Remove URLs, emails, extra whitespace - PRESERVES math/code (optimized)"""
        if not CLEANING_AVAILABLE:
            # Fallback: basic cleanup
            import re
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        
        # If content is technical (math/code), use minimal cleaning
        if content_type and content_type.get('is_technical'):
            # Only remove URLs and emails, keep everything else
            try:
                text = clean(text,
                    fix_unicode=False,
                    to_ascii=False,
                    lower=False,
                    no_line_breaks=False,
                    no_urls=True,          # Remove URLs
                    no_emails=True,        # Remove emails
                    no_phone_numbers=False, # KEEP phone numbers (might be in code)
                    no_numbers=False,      # KEEP numbers (critical for math/code)
                    no_digits=False,
                    no_currency_symbols=False,
                    no_punct=False,        # KEEP all punctuation (code syntax!)
                    replace_with_url="",
                    replace_with_email="",
                )
            except Exception as e:
                # Fallback if cleantext fails
                import re
                text = re.sub(r'https?://\S+', '', text)  # Just remove URLs
                text = re.sub(r'\S+@\S+', '', text)       # Just remove emails
        else:
            # Normal cleaning for regular text
            try:
                text = clean(text,
                    fix_unicode=False,
                    to_ascii=False,
                    lower=False,
                    no_line_breaks=False,
                    no_urls=True,
                    no_emails=True,
                    no_phone_numbers=True,
                    no_numbers=False,      # KEEP numbers (important for technical content)
                    no_digits=False,
                    no_currency_symbols=False,
                    no_punct=False,        # KEEP punctuation
                    replace_with_punct="",
                    replace_with_url="",
                    replace_with_email="",
                    replace_with_phone_number="",
                )
            except Exception as e:
                # Fallback if cleantext fails
                import re
                text = re.sub(r'\s+', ' ', text)
                return text.strip()
        
        # Remove excessive whitespace (using pre-compiled pattern)
        text = self.RE_WHITESPACE.sub(' ', text)
        return text.strip()
    
    def remove_html_tags(self, text):
        """Remove HTML tags and entities - cleans messy HTML (optimized)"""
        # Remove HTML tags (pre-compiled)
        text = self.RE_HTML_TAG.sub(' ', text)
        
        # Remove HTML entities (pre-compiled)
        text = self.RE_HTML_ENTITY.sub(' ', text)
        
        # Clean up whitespace after removal (pre-compiled)
        text = self.RE_WHITESPACE.sub(' ', text)
        return text.strip()
    
    def clean_pdf_artifacts(self, text, content_type=None):
        """Clean PDF extraction artifacts - PRESERVES LaTeX/code (optimized)"""
        # If content has math/code, skip aggressive PDF cleaning
        if content_type and content_type.get('is_technical'):
            # Only do minimal cleaning for technical content
            # Fix broken hyphenation
            text = self.RE_HYPHEN_BREAK.sub(r'\1\2', text)
            # Fix excessive line breaks
            text = self.RE_TRIPLE_NEWLINE.sub('\n\n', text)
            # That's it - preserve everything else for math/code
            return text.strip()
        
        # Normal PDF cleaning for regular text
        # Fix broken hyphenation from PDFs (word- \n word -> word)
        text = self.RE_HYPHEN_BREAK.sub(r'\1\2', text)
        
        # Remove page numbers and headers/footers (pre-compiled patterns)
        text = self.RE_PAGE_NUM.sub('', text)
        text = self.RE_PAGE_HEADER.sub('', text)
        text = self.RE_PAGE_OF.sub('', text)
        
        # Remove common PDF header/footer patterns (pre-compiled)
        text = self.RE_DATE_HEADER.sub('', text)
        text = self.RE_CAPS_HEADER.sub('', text)
        
        # Clean reference markers [1], [2], etc.
        text = self.RE_ORPHAN_REF.sub(' ', text)
        
        # Fix excessive line breaks from PDFs
        text = self.RE_TRIPLE_NEWLINE.sub('\n\n', text)
        
        # Fix weird spacing patterns from PDFs (pre-compiled)
        text = self.RE_SPACE_PUNCT.sub(r'\1\2', text)
        text = self.RE_PAREN_SPACE.sub(r'\1\2', text)
        
        # Fix broken equations or LaTeX artifacts - CAREFUL with this
        # Only fix obvious spacing issues, don't destroy LaTeX
        text = self.RE_DOLLAR_SPACE.sub('$', text)
        
        # Remove table artifacts (pre-compiled patterns)
        text = self.RE_TABLE_BORDER.sub(' ', text)
        text = self.RE_SEPARATOR.sub('', text)
        
        # Fix multiple spaces and newline spacing (pre-compiled)
        text = self.RE_WHITESPACE.sub(' ', text)
        text = self.RE_NEWLINE_SPACE.sub('\n', text)
        
        return text.strip()
    
    def normalize_unicode(self, text):
        """Convert special Unicode to ASCII where needed"""
        # Keep most Unicode, only convert problematic chars
        return unidecode(text)
    
    def heuristic_quality_score(self, text, content_type=None):
        """Fast quality scoring with professionalism checks (optimized for speed)"""
        score = 0.5  # Base score
        
        # Length check
        length = len(text)
        if 500 <= length <= 5000:
            score += 0.1
        elif length < MIN_LENGTH:
            return 0.0  # Too short
        elif length > MAX_LENGTH:
            return 0.0  # Too long
        
        # Keyword density - domain relevance
        text_lower = text.lower()
        keyword_count = sum(1 for kw in KEYWORDS if kw in text_lower)
        if keyword_count >= 3:
            score += 0.2  # High domain relevance
        elif keyword_count >= 1:
            score += 0.1
        
        # Professionalism & Quality Indicators (for "Gemini-level" output)
        professional_score = 0
        
        # 1. Proper capitalization (sentences start with capitals)
        sentences = text.split('.')
        if len(sentences) > 2:
            capitalized = sum(1 for s in sentences if s.strip() and s.strip()[0].isupper())
            if capitalized / len(sentences) > 0.7:
                professional_score += 0.05
        
        # 2. Complete sentences (not just fragments)
        complete_sentences = text.count('.') + text.count('!') + text.count('?')
        if complete_sentences > 2:
            professional_score += 0.05
        
        # 3. Professional vocabulary indicators
        prof_words = ['however', 'therefore', 'furthermore', 'additionally', 'consequently',
                      'moreover', 'specifically', 'particularly', 'demonstrates', 'indicates']
        prof_count = sum(1 for word in prof_words if word in text_lower)
        if prof_count >= 2:
            professional_score += 0.05
        
        # 4. Avoid low-quality markers
        low_quality_markers = ['lol', 'omg', 'wtf', 'idk', 'tbh', '!!!', '...']
        low_quality_count = sum(1 for marker in low_quality_markers if marker in text_lower)
        if low_quality_count > 0:
            professional_score -= 0.1  # Penalty for informal language
        
        # 5. Code/technical content gets automatic quality boost
        if content_type and content_type.get('is_technical'):
            professional_score += 0.1  # Technical content is valuable
        
        score += professional_score
        
        return min(1.0, max(0.0, score))
    
    def clean_chunk(self, chunk):
        """Clean single chunk - PRESERVES math/code/physics/telemetry (production-grade)"""
        self.stats['total_processed'] += 1
        
        try:
            # Validate chunk structure
            if 'content' not in chunk:
                return None
            
            text = chunk['content']
            
            # Handle empty or None content
            if not text or not isinstance(text, str):
                self.stats['too_short'] += 1
                return None
            
            # STEP 0: Detect content type FIRST - this determines how we clean
            content_type = self.detect_content_type(text)
            
            # 1. Fix Unicode (handles PDFs, LaTeX, code, etc.)
            text = self.fix_unicode(text)
            
            # 2. Remove noise (URLs, emails, etc.) - RESPECTS content type
            text = self.remove_noise(text, content_type)
            
            # 3. Remove HTML tags and entities
            text = self.remove_html_tags(text)
            
            # 4. Clean PDF extraction artifacts - RESPECTS content type
            text = self.clean_pdf_artifacts(text, content_type)
            
            # 5. Language filter (English only - sampled) - BYPASSED for technical content
            if not self.is_english(text, content_type):
                self.stats['non_english'] += 1
                return None
            
            # 6. Normalize Unicode - SKIP for technical content
            # text = self.normalize_unicode(text)  # Disabled to preserve LaTeX/code/physics
            
            # 7. Length check
            if len(text) < MIN_LENGTH:
                self.stats['too_short'] += 1
                return None
            if len(text) > MAX_LENGTH:
                self.stats['too_long'] += 1
                return None
            
            # 8. Quality scoring with professionalism checks
            quality_score = self.heuristic_quality_score(text, content_type)
            if quality_score < QUALITY_THRESHOLD:
                self.stats['low_quality'] += 1
                return None
            
            # Update chunk with cleaned content + metadata
            chunk['content'] = text
            chunk['quality_score'] = quality_score
            chunk['cleaned_at'] = datetime.now().isoformat()
            # Save content type for later stages
            chunk['content_type'] = {k: v for k, v in content_type.items()}
            
            self.stats['kept'] += 1
            return chunk
            
        except Exception as e:
            # Robust error handling - log and skip problematic chunks
            return None
    
    def process_batch(self, chunks):
        """Process batch"""
        cleaned = []
        for chunk in chunks:
            result = self.clean_chunk(chunk)
            if result:
                cleaned.append(result)
        return cleaned

def process_file(input_file, cleaner):
    """Process single file - production-ready with streaming writes to prevent memory issues"""
    print(f"\n📄 {input_file.name} ({input_file.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # First pass: count total lines for progress bar
    print("  📊 Counting lines...")
    try:
        total_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8', errors='ignore'))
        print(f"  📊 Total chunks: {total_lines:,}")
    except Exception as e:
        print(f"  ⚠️  Error counting lines: {e}")
        total_lines = 0
    
    batch = []
    chunks_written = 0
    processed_count = 0
    error_count = 0
    
    # Open output file for streaming writes
    output_file = OUTPUT_DIR / input_file.name
    
    try:
        with open(output_file, 'w', encoding='utf-8') as out_f:
            with open(input_file, 'r', encoding='utf-8', errors='ignore') as in_f:
                for line in tqdm(in_f, desc="Cleaning", total=total_lines, unit=" chunks", 
                                mininterval=0.5, miniters=100):  # Update every 0.5s or 100 chunks
                    try:
                        chunk = json.loads(line)
                        batch.append(chunk)
                        
                        if len(batch) >= BATCH_SIZE:
                            # Process batch
                            cleaned = cleaner.process_batch(batch)
                            
                            # STREAM WRITE: Write immediately instead of accumulating
                            for chunk in cleaned:
                                out_f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                                chunks_written += 1
                            
                            batch = []
                            gc.collect()  # Force garbage collection to free memory
                            processed_count += BATCH_SIZE
                            
                            # Frequent status updates - every 5k chunks
                            if processed_count % 5000 == 0:
                                keep_rate = cleaner.stats['kept'] / cleaner.stats['total_processed'] * 100 if cleaner.stats['total_processed'] > 0 else 0
                                timestamp = datetime.now().strftime("%H:%M:%S")
                                print(f"\n  [{timestamp}] 📊 {processed_count:,}/{total_lines:,} chunks ({processed_count/total_lines*100:.1f}%) | Kept {cleaner.stats['kept']:,} ({keep_rate:.1f}%) | Errors: {error_count}")
                        
                    except json.JSONDecodeError as e:
                        error_count += 1
                        continue
                    except Exception as e:
                        error_count += 1
                        continue
            
            # Process remaining batch
            if batch:
                try:
                    cleaned = cleaner.process_batch(batch)
                    for chunk in cleaned:
                        out_f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                        chunks_written += 1
                except Exception as e:
                    print(f"\n  ⚠️  Error processing final batch: {e}")
    
    except Exception as e:
        print(f"  ❌ Error during processing: {e}")
        return 0
    
    keep_rate = chunks_written / total_lines * 100 if total_lines > 0 else 0
    print(f"\n  ✅ Kept {chunks_written:,} / {total_lines:,} chunks ({keep_rate:.1f}%)")
    if error_count > 0:
        print(f"  ⚠️  Encountered {error_count:,} errors (skipped)")
    return chunks_written

def main():
    if not CLEANING_AVAILABLE:
        print("\n❌ Cannot proceed without cleaning libraries")
        return
    
    print("=" * 80)
    print("STAGE 3: CLEAN & FILTER")
    print("=" * 80)
    
    cleaner = TextCleaner()
    
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
        
        count = process_file(input_file, cleaner)
        total_output += count
    
    if skipped > 0:
        print(f"\n📊 Skipped {skipped} already-processed files")
    
    # Stats
    stats = {
        "stage": 3,
        "total_processed": cleaner.stats['total_processed'],
        "kept": cleaner.stats['kept'],
        "removed": {
            "too_short": cleaner.stats['too_short'],
            "too_long": cleaner.stats['too_long'],
            "low_quality": cleaner.stats['low_quality']
        },
        "filter_ratio": (cleaner.stats['total_processed'] - cleaner.stats['kept']) / cleaner.stats['total_processed'] if cleaner.stats['total_processed'] > 0 else 0,
        "completed_at": datetime.now().isoformat()
    }
    
    with open(OUTPUT_DIR / "stage3_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "=" * 80)
    print("STAGE 3 COMPLETE")
    print("=" * 80)
    print(f"Processed: {stats['total_processed']:,}")
    print(f"Kept: {stats['kept']:,} ({stats['kept']/stats['total_processed']*100:.1f}%)")
    print(f"\n✅ Ready for Stage 4: AI Classification")
    print("=" * 80)

if __name__ == "__main__":
    main()
