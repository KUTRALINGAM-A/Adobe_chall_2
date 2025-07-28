# enhanced_persona_analyzer.py
import json
import os
import re
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict
import fitz
import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

# LLM Integration using llama-cpp-python
try:
    from llama_cpp import Llama
    LLM_AVAILABLE = True
except ImportError:
    print("Warning: llama-cpp-python not available. Using heuristic-only analysis.")
    LLM_AVAILABLE = False

class PersonaDrivenAnalyzer:
    def __init__(self,model_path="/app/model/TinyLlama/TinyLlama-1.1B-Chat-v0.6/ggml-model-q4_0.gguf"):
        self.llm = None
        self.llm_lock = threading.Lock()
        self.current_persona = None
        self.current_job = None
        
        if LLM_AVAILABLE and os.path.exists(model_path):
            try:
                print(f"Loading TinyLlama model from {model_path}...")
                self.llm = Llama(
                    model_path=model_path,
                    n_ctx=2048,
                    n_threads=6,
                    n_batch=512,
                    verbose=False,
                    use_mlock=True,
                    n_gpu_layers=0
                )
                print("✅ TinyLlama model loaded successfully!")
            except Exception as e:
                print(f"❌ Failed to load TinyLlama: {e}")
                self.llm = None
        
        self.ensure_nltk_data()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Enhanced heading patterns
        self.heading_patterns = [
            re.compile(r'^\d+\.\s+[A-Z].*', re.IGNORECASE),
            re.compile(r'^\d+\.\d+\s+[A-Z].*', re.IGNORECASE),
            re.compile(r'^\d+\.\d+\.\d+\s+[A-Z].*', re.IGNORECASE),
            re.compile(r'^[IVX]+\.\s+[A-Z].*', re.IGNORECASE),
            re.compile(r'^(Chapter|Section|Part|Unit|Module)\s+\d+', re.IGNORECASE),
            re.compile(r'^(Chapter|Section|Part|Unit|Module)\s+[A-Z]', re.IGNORECASE),
            re.compile(r'^[A-Z][A-Z\s]{3,50}$'),
            re.compile(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+){1,8}:?\s*$'),
            re.compile(r'^[A-Z]\.\s+[A-Z].*', re.IGNORECASE),
            re.compile(r'^[a-z]\)\s+[A-Z].*', re.IGNORECASE),
            re.compile(r'^(Introduction|Conclusion|Summary|Overview|Background|Methodology|Results|Discussion|References|Bibliography|Appendix|Abstract|Executive Summary|Key Findings|Recommendations)', re.IGNORECASE),
            re.compile(r'^(What|How|Why|When|Where|Who)\s+.*\?$', re.IGNORECASE),
            re.compile(r'^(Objective|Goals|Purpose|Scope|Requirements|Specifications|Features|Benefits|Challenges|Solutions|Implementation|Architecture|Design|Framework)', re.IGNORECASE),
        ]

    def ensure_nltk_data(self):
        """Download required NLTK data only if needed"""
        try:
            sent_tokenize("test")
            stopwords.words('english')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)

    def set_persona_context(self, persona: str, job_to_be_done: str):
        """Set the persona context for the analyzer"""
        self.current_persona = persona
        self.current_job = job_to_be_done
        
        self.persona_keywords = self._extract_persona_keywords(persona)
        self.job_keywords = self._extract_job_keywords(job_to_be_done)
        
        print(f"🎭 Persona context set: {persona}")
        print(f"🎯 Job context set: {job_to_be_done}")
        print(f"📝 Persona keywords: {list(self.persona_keywords)[:10]}")
        print(f"🔍 Job keywords: {list(self.job_keywords)[:10]}")

    def _extract_persona_keywords(self, persona: str) -> set:
        """Extract relevant keywords from persona description"""
        role_keywords = {
            'developer': ['code', 'programming', 'software', 'api', 'framework', 'library', 'function', 'method', 'class', 'variable'],
            'manager': ['strategy', 'planning', 'team', 'project', 'resource', 'budget', 'timeline', 'deliverable', 'stakeholder'],
            'analyst': ['data', 'analysis', 'metrics', 'report', 'trend', 'pattern', 'insight', 'performance', 'measurement'],
            'designer': ['design', 'ui', 'ux', 'interface', 'user', 'experience', 'visual', 'layout', 'aesthetic', 'prototype'],
            'researcher': ['research', 'study', 'methodology', 'experiment', 'hypothesis', 'data', 'analysis', 'findings', 'conclusion'],
            'student': ['learn', 'study', 'understand', 'concept', 'theory', 'practice', 'example', 'tutorial', 'guide', 'explanation'],
            'engineer': ['technical', 'specification', 'requirement', 'design', 'implementation', 'testing', 'optimization', 'system', 'architecture'],
            'consultant': ['advice', 'recommendation', 'solution', 'strategy', 'implementation', 'best practice', 'guideline', 'framework']
        }
        
        keywords = set()
        persona_lower = persona.lower()
        
        for role, role_keywords_list in role_keywords.items():
            if role in persona_lower:
                keywords.update(role_keywords_list)
        
        persona_words = re.findall(r'\b[a-zA-Z]{3,}\b', persona.lower())
        keywords.update(word for word in persona_words if word not in self.stop_words)
        
        return keywords

    def _extract_job_keywords(self, job_to_be_done: str) -> set:
        """Extract relevant keywords from job description"""
        job_words = re.findall(r'\b[a-zA-Z]{3,}\b', job_to_be_done.lower())
        return set(word for word in job_words if word not in self.stop_words)

    def query_llm_with_persona(self, prompt: str, max_tokens: int = 100, use_persona: bool = True) -> str:
        """Enhanced LLM query with persona context"""
        if not self.llm:
            return ""
        
        with self.llm_lock:
            try:
                if use_persona and self.current_persona:
                    persona_prompt = f"<|system|>\nYou are a {self.current_persona}. Think and respond from this perspective.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
                else:
                    persona_prompt = f"<|user|>\n{prompt}</s>\n<|assistant|>\n"
                
                response = self.llm(
                    persona_prompt,
                    max_tokens=max_tokens,
                    temperature=0.2,
                    top_p=0.7,
                    top_k=40,
                    repeat_penalty=1.1,
                    stop=["</s>", "<|user|>", "<|assistant|>", "<|system|>"],
                    echo=False
                )
                return response['choices'][0]['text'].strip()
            except Exception as e:
                print(f"LLM query error: {e}")
                return ""

    def extract_text_from_pdf_comprehensive(self, pdf_path: str) -> Dict[int, str]:
        """More comprehensive PDF text extraction"""
        page_texts = {}
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                max_pages = min(100, total_pages)
                
                for page_num in range(max_pages):
                    page = pdf.pages[page_num]
                    text = page.extract_text()
                    if text and len(text.strip()) > 10:
                        page_texts[page_num + 1] = text[:15000]
                        
                print(f"📄 Extracted text from {len(page_texts)} pages out of {total_pages}")
                        
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            try:
                doc = fitz.open(pdf_path)
                for page_num in range(min(100, len(doc))):
                    page = doc[page_num]
                    text = page.get_text()
                    if text and len(text.strip()) > 10:
                        page_texts[page_num + 1] = text[:15000]
                doc.close()
            except Exception as e2:
                print(f"Fallback extraction also failed: {e2}")
                
        return page_texts

    def clean_and_correct_title(self, title: str) -> str:
        """Clean and correct section titles to make them more readable"""
        if not title:
            return title
            
        # Remove excessive whitespace
        title = re.sub(r'\s+', ' ', title.strip())
        
        # Remove numbering patterns at the beginning
        title = re.sub(r'^\d+(\.\d+)*\s*[-.]?\s*', '', title)
        title = re.sub(r'^[A-Z]\.\s*', '', title)
        title = re.sub(r'^[ivx]+\.\s*', '', title, flags=re.IGNORECASE)
        
        # Remove trailing colons if not meaningful
        if title.endswith(':') and len(title) > 20:
            title = title[:-1]
            
        # Fix common OCR/extraction errors
        title = title.replace('  ', ' ')
        title = title.replace(' .', '.')
        title = title.replace(' ,', ',')
        
        # Capitalize properly if all caps
        if title.isupper() and len(title) > 10:
            words = title.split()
            corrected_words = []
            for word in words:
                if word in ['API', 'UI', 'UX', 'AI', 'ML', 'IT', 'HR', 'ROI', 'KPI']:
                    corrected_words.append(word)
                elif len(word) <= 3 and word.isupper():
                    corrected_words.append(word)
                else:
                    corrected_words.append(word.title())
            title = ' '.join(corrected_words)
        
        # Fix title case issues
        elif not title[0].isupper() and len(title) > 3:
            title = title[0].upper() + title[1:]
        
        # Remove artifacts from PDF extraction
        title = re.sub(r'^\W+', '', title)
        title = re.sub(r'\W+$', '', title)
        
        # Limit title length
        if len(title) > 100:
            title = title[:97] + '...'
        
        return title.strip()

    def is_likely_heading_persona_aware(self, line: str, prev_line: str = "", next_line: str = "", page_context: str = "") -> bool:
        """Enhanced heading detection with persona awareness and better title correction"""
        line = line.strip()
        
        # Basic filters
        if len(line) < 3:
            return False
        if len(line) > 200:
            return False
            
        # Skip obvious non-headings
        if line.endswith('.') and len(line) > 50 and not line.startswith(('1.', '2.', '3.', '4.', '5.')):
            return False
        if line.count('.') > 3 and not re.match(r'^\d+(\.\d+)*\s+', line):
            return False
        if re.search(r'\d{4}', line) and ('page' in line.lower() or 'copyright' in line.lower()):
            return False
        
        # Skip headers/footers patterns
        if re.search(r'(page \d+|chapter \d+)$', line.lower()):
            return False
        if len(line) > 100 and line.count(' ') > 15:
            return False
            
        # Check regex patterns first
        for pattern in self.heading_patterns:
            if pattern.match(line):
                return True
        
        # Enhanced heading scoring
        heading_score = 0
        
        # Clean the line for better analysis
        clean_line = re.sub(r'^\d+(\.\d+)*\s*', '', line)
        clean_line = re.sub(r'^[A-Z]\.\s*', '', clean_line)
        
        words = clean_line.split()
        if len(words) > 0:
            # Title case detection
            capitalized_words = sum(1 for word in words if word[0].isupper() and len(word) > 1)
            cap_ratio = capitalized_words / len(words)
            
            if cap_ratio >= 0.7:
                heading_score += 3
            elif cap_ratio >= 0.5:
                heading_score += 2
                
            # Length indicators
            if 2 <= len(words) <= 8:
                heading_score += 2
            elif len(words) <= 12:
                heading_score += 1
                
            # First word capitalized
            if clean_line and clean_line[0].isupper():
                heading_score += 1
                
        # Position indicators
        if prev_line.strip() == "":
            heading_score += 1
        if next_line.strip() == "":
            heading_score += 1
            
        # Persona-specific keyword matching
        line_lower = line.lower()
        persona_matches = sum(1 for keyword in self.persona_keywords if keyword in line_lower)
        job_matches = sum(1 for keyword in self.job_keywords if keyword in line_lower)
        
        if persona_matches > 0:
            heading_score += min(persona_matches * 2, 3)
        if job_matches > 0:
            heading_score += min(job_matches * 2, 3)
        
        # Professional/academic headings
        professional_patterns = [
            r'\b(introduction|overview|summary|conclusion|background)\b',
            r'\b(methodology|approach|method|technique)\b',
            r'\b(results|findings|analysis|discussion)\b',
            r'\b(implementation|design|architecture|framework)\b',
            r'\b(requirements|specifications|features|benefits)\b',
            r'\b(challenges|solutions|recommendations)\b',
            r'\b(objective|goals|purpose|scope)\b'
        ]
        
        for pattern in professional_patterns:
            if re.search(pattern, line_lower):
                heading_score += 2
                break
            
        # Formatting indicators
        if line.isupper() and 5 <= len(line) <= 40:
            heading_score += 2
        if line.endswith(':') and len(line) <= 50:
            heading_score += 2
        if re.match(r'^\d+[\.\)]', line):
            heading_score += 3
        if re.match(r'^[A-Z][\.\)]', line):
            heading_score += 2
            
        return heading_score >= 4

    def identify_sections_comprehensive(self, page_texts: Dict[int, str]) -> List[Dict[str, Any]]:
        """Comprehensive section identification with persona awareness"""
        sections = []
        
        for page_num, text in page_texts.items():
            lines = text.split('\n')
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                    
                # Get context
                prev_line = lines[i-1].strip() if i > 0 else ""
                next_line = lines[i+1].strip() if i < len(lines)-1 else ""
                page_context = text[:500]
                
                # Check if this line is a heading
                if self.is_likely_heading_persona_aware(line, prev_line, next_line, page_context):
                    # Clean and correct the title
                    clean_title = self.clean_and_correct_title(line)
                    if len(clean_title) < 3:
                        continue
                        
                    content = self._extract_section_content_comprehensive(lines, i)
                    
                    sections.append({
                        'title': clean_title,
                        'original_title': line,
                        'page_number': page_num,
                        'content': content,
                        'detection_method': 'persona_aware_heuristic'
                    })
                    
                    print(f"  📋 Found heading on page {page_num}: {clean_title[:60]}...")

        print(f"🔍 Total headings found: {len(sections)}")
        return sections

    def _extract_section_content_comprehensive(self, lines: List[str], heading_index: int) -> str:
        """Extract more comprehensive section content"""
        content_lines = []
        
        for i in range(heading_index + 1, min(heading_index + 30, len(lines))):
            line = lines[i].strip()
            
            if line and self.is_likely_heading_persona_aware(line):
                break
                
            if line:
                content_lines.append(line)
                
            if len(' '.join(content_lines)) > 1000:
                break
                
        return ' '.join(content_lines)[:1000]

    def calculate_persona_aware_relevance(self, section: Dict[str, Any]) -> float:
        """Enhanced relevance calculation with strong persona focus"""
        section_text = f"{section['title']} {section['content']}".lower()
        
        # Persona keyword matching
        persona_score = 0
        for keyword in self.persona_keywords:
            if keyword in section_text:
                title_weight = 3 if keyword in section['title'].lower() else 1
                persona_score += title_weight
        
        # Job keyword matching
        job_score = 0
        for keyword in self.job_keywords:
            if keyword in section_text:
                title_weight = 2 if keyword in section['title'].lower() else 1
                job_score += title_weight
        
        # Normalize scores
        max_persona_score = max(len(self.persona_keywords) * 2, 1)
        max_job_score = max(len(self.job_keywords) * 2, 1)
        
        normalized_persona = min(persona_score / max_persona_score, 1.0)
        normalized_job = min(job_score / max_job_score, 1.0)
        
        # Base relevance calculation
        base_score = (normalized_persona * 0.6) + (normalized_job * 0.4)
        base_score = max(0.15, base_score)
        
        # LLM enhancement for promising sections
        if self.llm and base_score > 0.3:
            llm_boost = self._persona_aware_llm_relevance(section)
            base_score = (base_score * 0.6) + (llm_boost * 0.4)
        
        return min(1.0, base_score)

    def _persona_aware_llm_relevance(self, section: Dict[str, Any]) -> float:
        """LLM relevance assessment with persona context"""
        content_preview = section['content'][:200]
        
        prompt = f"As a {self.current_persona} working on '{self.current_job}', rate the relevance (0-10) of this section:\nTitle: {section['title']}\nContent: {content_preview}\nRelevance score:"
        
        response = self.query_llm_with_persona(prompt, max_tokens=10, use_persona=True)
        
        try:
            numbers = re.findall(r'\d+', response)
            if numbers:
                score = float(numbers[0]) / 10.0
                return max(0.0, min(1.0, score))
        except:
            pass
        return 0.5

    def create_persona_aware_summary(self, section: Dict[str, Any]) -> str:
        """Create summary tailored to the persona's perspective"""
        title = section['title']
        content = section['content'][:300]
        
        # Heuristic summary
        try:
            sentences = sent_tokenize(content) if content else []
        except:
            sentences = content.split('.') if content else []
        
        if len(sentences) > 1:
            summary_content = '. '.join(sentences[:2])
        else:
            summary_content = content[:200]
        
        # Enhanced LLM summary with persona
        if self.llm and section.get('relevance_score', 0) > 0.4:
            llm_summary = self._persona_aware_llm_summary(title, content)
            if llm_summary and len(llm_summary.strip()) > 20:
                return f"{title}: {llm_summary.strip()}"
        
        return f"{title}: {summary_content}"[:300]

    def _persona_aware_llm_summary(self, title: str, content: str) -> str:
        """Generate persona-aware summary using LLM"""
        prompt = f"As a {self.current_persona} needing to {self.current_job}, summarize this section in 1-2 sentences focusing on what's most relevant to me:\nTitle: {title}\nContent: {content[:250]}\nSummary:"
        
        return self.query_llm_with_persona(prompt, max_tokens=60, use_persona=True)

    def process_documents_comprehensive(self, input_json_path: str, pdf_folder: str) -> Dict[str, Any]:
        """Enhanced main processing with comprehensive document coverage"""
        
        # Load configuration
        with open(input_json_path, 'r') as f:
            config = json.load(f)

        documents = config.get('documents', [])
        persona = config.get('persona', 'General User')
        job_to_be_done = config.get('job_to_be_done', 'Extract information')

        if isinstance(persona, dict):
            persona = persona.get('role', 'General User')
        if isinstance(job_to_be_done, dict):
            job_to_be_done = job_to_be_done.get('task', 'Extract information')

        # Set persona context
        self.set_persona_context(persona, job_to_be_done)

        print(f"🚀 Comprehensive processing with persona-driven analysis")
        print(f"👤 Persona: {persona}")
        print(f"🎯 Job: {job_to_be_done}")

        all_sections = []
        processed_docs = 0
        total_docs = len(documents)

        # Process each document
        for doc_info in documents:
            doc_name = doc_info if isinstance(doc_info, str) else doc_info.get('filename', '')
            pdf_path = os.path.join(pdf_folder, doc_name)

            if not os.path.exists(pdf_path):
                print(f"❌ File not found: {pdf_path}")
                continue

            print(f"📄 Processing ({processed_docs + 1}/{total_docs}): {doc_name}")
            
            page_texts = self.extract_text_from_pdf_comprehensive(pdf_path)
            if not page_texts:
                print(f"❌ No text extracted from {doc_name}")
                continue
                
            sections = self.identify_sections_comprehensive(page_texts)
            
            # Calculate relevance scores with persona awareness
            for section in sections:
                section['document'] = doc_name
                section['relevance_score'] = self.calculate_persona_aware_relevance(section)

            all_sections.extend(sections)
            processed_docs += 1
            print(f"✅ Extracted {len(sections)} sections from {doc_name}")

        print(f"📊 Total sections found across {processed_docs} documents: {len(all_sections)}")

        # Sort by relevance first
        all_sections.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Ensure balanced selection from all PDFs (exactly 10 sections)
        selected_sections = []
        doc_section_count = defaultdict(int)
        used_docs = set()
        
        # Step 1: Get at least one section from each document
        for section in all_sections:
            doc_name = section['document']
            if (doc_name not in used_docs and 
                section['relevance_score'] > 0.25 and
                len(selected_sections) < 10):
                selected_sections.append(section)
                doc_section_count[doc_name] += 1
                used_docs.add(doc_name)
                
        # Step 2: Fill remaining slots with highest relevance sections
        max_per_doc = max(2, 10 // len(used_docs)) if used_docs else 2
        
        for section in all_sections:
            if (len(selected_sections) >= 10):
                break
                
            doc_name = section['document']
            if (section not in selected_sections and 
                doc_section_count[doc_name] < max_per_doc and
                section['relevance_score'] > 0.2):
                selected_sections.append(section)
                doc_section_count[doc_name] += 1
        
        # Step 3: If still not enough, fill with any remaining good sections
        for section in all_sections:
            if len(selected_sections) >= 10:
                break
            if (section not in selected_sections and 
                section['relevance_score'] > 0.15):
                selected_sections.append(section)
                doc_section_count[section['document']] += 1
                
        # Ensure exactly 10 sections
        selected_sections = selected_sections[:10]
        
        # Recalculate doc counts for final selection
        final_doc_count = defaultdict(int)
        for section in selected_sections:
            final_doc_count[section['document']] += 1

        print(f"🎯 Selected exactly 10 sections from {len(final_doc_count)} documents")
        print(f"📋 Final distribution: {dict(final_doc_count)}")

        # Create persona-aware summaries
        refined_sections = []
        for section in selected_sections:
            summary = self.create_persona_aware_summary(section)
            refined_sections.append({
                'document': section['document'],
                'refined_text': summary,
                'page_number': section['page_number'],
                'relevance_score': section['relevance_score']
            })

        # Prepare comprehensive output
        output = {
            'metadata': {
                'input_documents': [d if isinstance(d, str) else d.get('filename', '') for d in documents],
                'processed_documents': processed_docs,
                'persona': persona,
                'job_to_be_done': job_to_be_done,
                'processing_timestamp': datetime.now().isoformat(),
                'analysis_method': 'Persona-driven balanced selection',
                'total_sections_found': len(all_sections),
                'final_sections_selected': len(selected_sections),
                'sections_per_document': dict(final_doc_count)
            },
            'extracted_sections': [
                {
                    'document': section['document'],
                    'section_title': section['title'],
                    'importance_rank': i + 1,
                    'page_number': section['page_number'],
                    'relevance_score': round(section['relevance_score'], 3)
                }
                for i, section in enumerate(selected_sections)
            ],
            'subsection_analysis': refined_sections
        }

        return output

def main():
    """Enhanced main execution"""
    analyzer = PersonaDrivenAnalyzer()
    
    input_json = os.getenv('INPUT_JSON', '/app/challenge1b_input.json')
    pdf_folder = os.getenv('PDF_FOLDER', '/app/pdf')
    output_path = os.getenv('OUTPUT_PATH', '/app/result.json')

    print("🎭 Starting PERSONA-DRIVEN comprehensive document analysis...")
    start_time = datetime.now()
    
    result = analyzer.process_documents_comprehensive(input_json, pdf_folder)
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✅ Analysis complete in {processing_time:.2f} seconds!")
    print(f"📊 Selected exactly 10 sections from {result['metadata']['processed_documents']} documents")
    print(f"📁 Distribution: {result['metadata']['sections_per_document']}")
    print(f"💾 Results saved to: {output_path}")

if __name__ == "__main__":
    main()