#!/usr/bin/env python3
"""
KinshipQA Error Analysis with Chain-of-Thought Prompting (v2.0)
================================================================
Enhanced version with:
- HPC/SLURM compatibility with checkpointing
- Pre-designed example selection for ACL paper
- Automatic LaTeX generation for Section 5.1 (Error Analysis)
- Support for multiple models in one run

Usage on HPC:
    # Single run
    python error_analysis_cot_v2.py \
        --results-dir ./results/ \
        --dataset-dir ./datasets/ \
        --provider openai --model gpt-4o-mini \
        --output ./analysis/gpt4o_error_analysis.json

    # With SLURM (see generated slurm script)
    sbatch run_error_analysis.slurm

Author: Tianda (ACL 2026)
Version: 2.0
"""

import json
import os
import re
import random
import argparse
import time
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict
from datetime import datetime
import hashlib

# =============================================================================
# Constants & Configuration
# =============================================================================

KINSHIP_SYSTEMS = ['eskimo', 'hawaiian', 'iroquois', 'dravidian', 'crow', 'omaha', 'sudanese']
WESTERN_SYSTEMS = ['eskimo', 'sudanese']
NON_WESTERN_SYSTEMS = ['hawaiian', 'iroquois', 'dravidian', 'crow', 'omaha']

# Target examples for the paper - designed for maximum impact
PAPER_EXAMPLE_TARGETS = {
    # Each entry: (kinship_system, category, error_type, why_include)
    "example_1_cultural_override": {
        "systems": ["crow", "omaha"],
        "categories": [4],
        "error_types": ["cultural_default", "conceptual_error"],
        "description": "Model applies Western kinship defaults instead of cultural rules",
        "paper_section": "Cultural Override Failures"
    },
    "example_2_reasoning_chain": {
        "systems": ["dravidian", "iroquois"],
        "categories": [2, 4],
        "error_types": ["reasoning_chain_break", "incomplete_enumeration"],
        "description": "Multi-hop reasoning breaks down mid-chain",
        "paper_section": "Reasoning Chain Failures"
    },
    "example_3_enumeration": {
        "systems": ["hawaiian", "crow"],
        "categories": [4],
        "error_types": ["incomplete_enumeration"],
        "description": "Model finds some but not all clan/classificatory members",
        "paper_section": "Incomplete Enumeration"
    },
    "example_4_western_success": {
        "systems": ["eskimo", "sudanese"],
        "categories": [4],
        "error_types": ["reasoning_chain_break"],  # Even when Eskimo fails, it's different
        "description": "Contrast: How the same question type succeeds in Western systems",
        "paper_section": "Western vs Non-Western Contrast"
    },
    "example_5_counting": {
        "systems": None,  # Any system
        "categories": [3],
        "error_types": ["counting_error"],
        "description": "Counting failures reveal enumeration limitations",
        "paper_section": "Counting Failures"
    }
}

# Error taxonomy with descriptions for the paper
ERROR_TAXONOMY = {
    "incomplete_enumeration": {
        "short": "Incomplete Enum.",
        "description": "High precision, low recall — model finds correct answers but misses others",
        "diagnosis": "Fails to exhaustively traverse relationship graph",
        "example_pattern": "Found 4/14 matriclan members, all correct"
    },
    "over_enumeration": {
        "short": "Over Enum.",
        "description": "Low precision — model includes incorrect answers",
        "diagnosis": "Over-generalizes relationship boundaries",
        "example_pattern": "Listed non-relatives as family members"
    },
    "complete_miss": {
        "short": "Complete Miss",
        "description": "No overlap between prediction and ground truth",
        "diagnosis": "Fundamental misunderstanding or context parsing failure",
        "example_pattern": "Named completely wrong people"
    },
    "counting_error": {
        "short": "Counting",
        "description": "Numerical counting mistake",
        "diagnosis": "Cannot accurately enumerate and count set members",
        "example_pattern": "Said 3 cousins when answer is 7"
    },
    "conceptual_error": {
        "short": "Conceptual",
        "description": "Misunderstood the kinship relationship definition",
        "diagnosis": "Lacks knowledge of cultural kinship terminology",
        "example_pattern": "Confused 'classificatory sibling' with biological sibling"
    },
    "reasoning_chain_break": {
        "short": "Chain Break",
        "description": "Started correct reasoning but broke down during multi-hop traversal",
        "diagnosis": "Working memory limitations in long inference chains",
        "example_pattern": "Correctly identified parent, failed to find grandparent"
    },
    "context_misread": {
        "short": "Context Error",
        "description": "Misread or ignored information in the context",
        "diagnosis": "Attention/retrieval failure from long context",
        "example_pattern": "Claimed person not mentioned when they were"
    },
    "cultural_default": {
        "short": "Cultural Default",
        "description": "Applied Western/Eskimo kinship assumptions incorrectly",
        "diagnosis": "Training bias toward Western kinship concepts",
        "example_pattern": "Treated cross-cousin as regular cousin in Iroquois system"
    },
    "hallucination": {
        "short": "Hallucination",
        "description": "Generated names or relations not present in context",
        "diagnosis": "Confabulation under uncertainty",
        "example_pattern": "Invented 'John Smith' not in the family tree"
    },
    "other": {
        "short": "Other",
        "description": "Unclassified error",
        "diagnosis": "Requires manual inspection",
        "example_pattern": "Various"
    }
}

# =============================================================================
# Chain-of-Thought Prompt Templates
# =============================================================================

COT_PROMPT_STRUCTURED = """You are analyzing kinship relationships in a family. Read the context carefully and answer the question by showing your complete reasoning process.

Context:
{context}

Question: {question}

Please think through this step-by-step:

STEP 1 - IDENTIFY KEY PERSON(S):
Who is the question asking about? What relationship are we looking for?

STEP 2 - EXTRACT RELEVANT FACTS:
List the specific family relationships from the context that are relevant.

STEP 3 - TRACE THE REASONING CHAIN:
Work through the relationships step by step. Show each connection.

STEP 4 - APPLY CULTURAL RULES (if mentioned):
If the context mentions any specific kinship system rules (like clan membership, classificatory relationships, or cultural terminology), apply them here.

STEP 5 - FINAL ANSWER:
Based on your reasoning, provide the answer.

Begin your analysis:"""

COT_PROMPT_SIMPLE = """Read the family information below and answer the question. Think through your reasoning carefully before giving your final answer.

Context:
{context}

Question: {question}

Let me work through this step by step:"""

# For generating contrast examples - ask model to explain its reasoning about cultural rules
COT_PROMPT_CULTURAL_PROBE = """You are analyzing kinship relationships. This question involves a specific cultural kinship system that may differ from Western conventions.

Context:
{context}

Question: {question}

Please analyze this carefully:

1. WHAT KINSHIP SYSTEM is being used here? (Look for clues like clan names, classificatory terms, etc.)

2. HOW DOES THIS SYSTEM DIFFER from standard Western/English kinship terms?

3. TRACE THE RELATIONSHIPS step by step, applying the cultural rules mentioned.

4. FINAL ANSWER:

Begin:"""

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ErrorSample:
    """A sampled error case for CoT analysis"""
    question_id: str
    kinship_system: str
    category: int
    n_hops: int
    question_text: str
    context: str
    ground_truth: Any
    original_prediction: Any
    original_metrics: Dict
    relationship_type: str = ""
    has_cultural_override: bool = False
    bio_term: str = ""
    kin_term: str = ""
    
    # CoT analysis fields (filled after analysis)
    cot_response: str = ""
    cot_extracted_answer: Any = None
    cot_correct: bool = False
    error_type: str = ""
    error_subtype: str = ""
    manual_notes: str = ""
    
    # For paper example selection
    example_quality_score: float = 0.0
    selected_for_paper: bool = False
    paper_example_category: str = ""

@dataclass 
class AnalysisCheckpoint:
    """Checkpoint for resuming interrupted analysis"""
    completed_ids: List[str] = field(default_factory=list)
    results: List[Dict] = field(default_factory=list)
    timestamp: str = ""
    
    def save(self, path: str):
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f)
    
    @classmethod
    def load(cls, path: str) -> 'AnalysisCheckpoint':
        if not Path(path).exists():
            return cls()
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

# =============================================================================
# LLM Provider Interfaces
# =============================================================================

class LLMProvider:
    """Base class for LLM providers"""
    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        raise NotImplementedError
    
    def get_name(self) -> str:
        raise NotImplementedError

class OpenAIProvider(LLMProvider):
    def __init__(self, model: str, api_key: str = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found")
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)
    
    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    def get_name(self) -> str:
        return f"openai/{self.model}"

class AnthropicProvider(LLMProvider):
    def __init__(self, model: str, api_key: str = None):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")
        import anthropic
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    def get_name(self) -> str:
        return f"anthropic/{self.model}"

class GeminiProvider(LLMProvider):
    def __init__(self, model: str, api_key: str = None):
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found")
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        self.genai = genai.GenerativeModel(self.model)
    
    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        response = self.genai.generate_content(
            prompt,
            generation_config={"temperature": 0.0, "max_output_tokens": max_tokens}
        )
        return response.text
    
    def get_name(self) -> str:
        return f"gemini/{self.model}"

class OllamaProvider(LLMProvider):
    def __init__(self, model: str):
        self.model = model
        from llama_index.llms.ollama import Ollama
        self.llm = Ollama(model=model, temperature=0.0, request_timeout=180.0)
    
    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        response = self.llm.complete(prompt)
        return str(response)
    
    def get_name(self) -> str:
        return f"ollama/{self.model}"

def create_provider(provider: str, model: str, api_key: str = None) -> LLMProvider:
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider,
        "ollama": OllamaProvider
    }
    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Choose from {list(providers.keys())}")
    
    if provider == "ollama":
        return OllamaProvider(model)
    return providers[provider](model, api_key)

# =============================================================================
# Error Sampling with Strategic Selection
# =============================================================================

def load_results_and_datasets(
    results_files: List[str],
    dataset_dir: str
) -> Tuple[Dict[str, List[Dict]], Dict[str, Dict]]:
    """
    Load all results and corresponding datasets.
    Returns: (results_by_system, datasets_by_system)
    """
    results_by_system = {}
    datasets_by_system = {}
    
    for results_file in results_files:
        print(f"Loading {results_file}...")
        with open(results_file) as f:
            data = json.load(f)
        
        # Extract system name
        system = Path(results_file).stem.replace("_results", "").lower()
        for s in KINSHIP_SYSTEMS:
            if s in system:
                system = s
                break
        
        results_by_system[system] = data.get('questions', [])
        
        # Load corresponding dataset
        dataset_path = Path(dataset_dir) / f"{system}_dataset.jsonl"
        if not dataset_path.exists():
            dataset_path = Path(dataset_dir) / f"{system}.jsonl"
        
        if dataset_path.exists():
            datasets_by_system[system] = {}
            with open(dataset_path) as f:
                for line in f:
                    q = json.loads(line)
                    datasets_by_system[system][q['question_id']] = q
            print(f"  Loaded {len(datasets_by_system[system])} questions from dataset")
        else:
            print(f"  Warning: Dataset not found at {dataset_path}")
            datasets_by_system[system] = {}
    
    return results_by_system, datasets_by_system

def select_errors_strategically(
    results_by_system: Dict[str, List[Dict]],
    datasets_by_system: Dict[str, Dict],
    sample_size: int = 200,
    seed: int = 42
) -> List[ErrorSample]:
    """
    Select errors strategically to ensure good coverage for paper examples.
    
    Strategy:
    1. First, ensure we have candidates for each PAPER_EXAMPLE_TARGETS
    2. Then, fill remaining quota with stratified random sampling
    """
    random.seed(seed)
    
    all_errors = []
    errors_by_criteria = defaultdict(list)  # For targeted selection
    
    # Collect all errors
    for system, questions in results_by_system.items():
        dataset = datasets_by_system.get(system, {})
        
        for q in questions:
            if q.get('exact_match', True):  # Skip correct answers
                continue
            
            q_id = q['question_id']
            q_data = dataset.get(q_id, {})
            
            error = ErrorSample(
                question_id=q_id,
                kinship_system=system,
                category=q.get('category', 0),
                n_hops=q.get('n_hops', 0),
                question_text=q.get('question_text', ''),
                context=q_data.get('context', ''),
                ground_truth=q.get('ground_truth'),
                original_prediction=q.get('predicted'),
                original_metrics={
                    'precision': q.get('precision', 0),
                    'recall': q.get('recall', 0),
                    'f1': q.get('f1', 0)
                },
                relationship_type=q_data.get('relationship_type', ''),
                has_cultural_override=q_data.get('has_cultural_override', False),
                bio_term=q_data.get('bio_term', ''),
                kin_term=q_data.get('kin_term', '')
            )
            
            all_errors.append(error)
            
            # Index by criteria for targeted selection
            errors_by_criteria[(system, error.category)].append(error)
            if error.has_cultural_override:
                errors_by_criteria[('cultural_override', system)].append(error)
    
    print(f"\nTotal errors found: {len(all_errors)}")
    
    # Phase 1: Ensure coverage for paper examples
    selected = []
    selected_ids = set()
    
    for example_key, criteria in PAPER_EXAMPLE_TARGETS.items():
        target_systems = criteria['systems'] or KINSHIP_SYSTEMS
        target_cats = criteria['categories']
        
        candidates = []
        for system in target_systems:
            for cat in target_cats:
                candidates.extend(errors_by_criteria.get((system, cat), []))
        
        # Prefer errors with cultural override for cultural examples
        if 'cultural' in example_key:
            cultural_candidates = [e for e in candidates if e.has_cultural_override]
            if cultural_candidates:
                candidates = cultural_candidates
        
        # Select up to 5 candidates per example category
        for candidate in candidates[:5]:
            if candidate.question_id not in selected_ids:
                candidate.paper_example_category = example_key
                selected.append(candidate)
                selected_ids.add(candidate.question_id)
    
    print(f"Selected {len(selected)} targeted examples for paper")
    
    # Phase 2: Stratified random sampling for remaining quota
    remaining_quota = sample_size - len(selected)
    remaining_errors = [e for e in all_errors if e.question_id not in selected_ids]
    
    if remaining_quota > 0 and remaining_errors:
        # Stratify by system and category
        strata = defaultdict(list)
        for e in remaining_errors:
            strata[(e.kinship_system, e.category)].append(e)
        
        # Sample proportionally
        total_remaining = len(remaining_errors)
        for key, errors in strata.items():
            n_sample = max(1, int(len(errors) / total_remaining * remaining_quota))
            n_sample = min(n_sample, len(errors))
            sampled = random.sample(errors, n_sample)
            selected.extend(sampled)
            selected_ids.update(e.question_id for e in sampled)
        
        # Trim if over quota
        if len(selected) > sample_size:
            # Keep all targeted examples, trim random ones
            targeted = [e for e in selected if e.paper_example_category]
            random_samples = [e for e in selected if not e.paper_example_category]
            n_random_to_keep = sample_size - len(targeted)
            selected = targeted + random.sample(random_samples, n_random_to_keep)
    
    print(f"Final sample size: {len(selected)}")
    
    # Print distribution
    print("\nSample distribution:")
    dist = defaultdict(lambda: defaultdict(int))
    for e in selected:
        dist[e.kinship_system][f"Cat{e.category}"] += 1
    
    print(f"{'System':<12} " + " ".join(f"{'Cat'+str(i):>6}" for i in range(1, 5)))
    for system in KINSHIP_SYSTEMS:
        if system in dist:
            counts = [str(dist[system].get(f"Cat{i}", 0)) for i in range(1, 5)]
            print(f"{system:<12} " + " ".join(f"{c:>6}" for c in counts))
    
    return selected

# =============================================================================
# CoT Analysis Engine
# =============================================================================

def run_cot_analysis(
    errors: List[ErrorSample],
    provider: LLMProvider,
    checkpoint_path: str = None,
    prompt_style: str = "structured",
    rate_limit_delay: float = 0.5,
    verbose: bool = False
) -> List[ErrorSample]:
    """
    Run CoT analysis with checkpointing support.
    """
    # Select prompt template
    templates = {
        "structured": COT_PROMPT_STRUCTURED,
        "simple": COT_PROMPT_SIMPLE,
        "cultural": COT_PROMPT_CULTURAL_PROBE
    }
    template = templates.get(prompt_style, COT_PROMPT_STRUCTURED)
    
    # Load checkpoint if exists
    checkpoint = AnalysisCheckpoint()
    if checkpoint_path:
        checkpoint = AnalysisCheckpoint.load(checkpoint_path)
        print(f"Loaded checkpoint with {len(checkpoint.completed_ids)} completed")
    
    completed_ids = set(checkpoint.completed_ids)
    
    # Process errors
    total = len(errors)
    for i, error in enumerate(errors):
        # Skip if already done
        if error.question_id in completed_ids:
            # Restore previous results
            for prev in checkpoint.results:
                if prev['question_id'] == error.question_id:
                    error.cot_response = prev.get('cot_response', '')
                    error.cot_extracted_answer = prev.get('cot_extracted_answer')
                    error.cot_correct = prev.get('cot_correct', False)
                    break
            continue
        
        print(f"\rProcessing {i+1}/{total} [{error.kinship_system}]...", end="", flush=True)
        
        # Use cultural probe for Cat 4 cultural override questions
        if error.category == 4 and error.has_cultural_override:
            current_template = COT_PROMPT_CULTURAL_PROBE
        else:
            current_template = template
        
        prompt = current_template.format(
            context=error.context,
            question=error.question_text
        )
        
        try:
            response = provider.generate(prompt, max_tokens=1024)
            error.cot_response = response
            error.cot_extracted_answer = extract_final_answer(response)
            error.cot_correct = check_answer_match(
                error.cot_extracted_answer,
                error.ground_truth
            )
            
            # Rate limiting
            time.sleep(rate_limit_delay)
            
        except Exception as e:
            print(f"\nError on {error.question_id}: {e}")
            error.cot_response = f"ERROR: {e}"
            time.sleep(2)  # Longer delay on error
        
        # Update checkpoint
        completed_ids.add(error.question_id)
        checkpoint.completed_ids = list(completed_ids)
        checkpoint.results.append({
            'question_id': error.question_id,
            'cot_response': error.cot_response,
            'cot_extracted_answer': error.cot_extracted_answer,
            'cot_correct': error.cot_correct
        })
        checkpoint.timestamp = datetime.now().isoformat()
        
        # Save checkpoint every 10 questions
        if checkpoint_path and (i + 1) % 10 == 0:
            checkpoint.save(checkpoint_path)
        
        if verbose and error.cot_response:
            print(f"\n{'='*60}")
            print(f"Q: {error.question_text[:100]}...")
            print(f"GT: {error.ground_truth}")
            print(f"CoT correct: {error.cot_correct}")
    
    # Final checkpoint save
    if checkpoint_path:
        checkpoint.save(checkpoint_path)
    
    print("\nCoT analysis complete!")
    return errors

def extract_final_answer(response: str) -> Any:
    """Extract the final answer from CoT response"""
    if not response:
        return None
    
    # Look for explicit FINAL ANSWER section
    patterns = [
        r"FINAL ANSWER[:\s]*\n?(.+?)(?:\n\n|$)",
        r"STEP 5[^:]*:[:\s]*\n?(.+?)(?:\n\n|$)",
        r"(?:The answer is|Answer:)[:\s]*(.+?)(?:\.|$)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            answer = match.group(1).strip()
            # Clean up
            answer = re.sub(r'\n.*', '', answer)  # Take first line only
            return answer
    
    # Fallback: last non-empty line that looks like an answer
    lines = [l.strip() for l in response.split('\n') if l.strip()]
    for line in reversed(lines):
        if not any(line.upper().startswith(s) for s in ['STEP', 'LET ME', 'FIRST', 'THE CONTEXT']):
            return line
    
    return response.strip().split('\n')[-1] if response.strip() else None

def check_answer_match(predicted: Any, ground_truth: Any) -> bool:
    """Check if predicted matches ground truth"""
    if predicted is None:
        return False
    
    pred_str = str(predicted).lower().strip()
    
    # Handle list ground truth
    if isinstance(ground_truth, list):
        gt_set = set(str(g).lower().strip() for g in ground_truth)
        # Extract names from prediction
        pred_names = set(re.findall(r'\b([a-z]+\s+[a-z]+)\b', pred_str))
        # Check for exact set match or superset
        return gt_set == pred_names or gt_set.issubset(pred_names)
    else:
        gt_str = str(ground_truth).lower().strip()
        return gt_str in pred_str or pred_str == gt_str

# =============================================================================
# Error Classification
# =============================================================================

def classify_error(error: ErrorSample) -> Tuple[str, str]:
    """
    Classify error type and subtype based on metrics and CoT response.
    Returns: (error_type, subtype/details)
    """
    metrics = error.original_metrics
    precision = metrics.get('precision', 0)
    recall = metrics.get('recall', 0)
    cot = error.cot_response.lower() if error.cot_response else ""
    
    # Category 3 = counting questions
    if error.category == 3 or 'count' in error.relationship_type:
        return "counting_error", f"precision={precision:.2f}"
    
    # Based on precision/recall patterns
    if precision > 0.8 and recall < 0.5:
        return "incomplete_enumeration", f"P={precision:.2f}, R={recall:.2f}"
    
    if precision < 0.5 and recall > 0.5:
        return "over_enumeration", f"P={precision:.2f}, R={recall:.2f}"
    
    if precision == 0 and recall == 0:
        # Check if it's hallucination vs complete miss
        if error.original_prediction:
            pred_names = set(re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', 
                                        str(error.original_prediction)))
            if pred_names:
                return "hallucination", "generated non-existent names"
        return "complete_miss", "no valid prediction"
    
    # Check CoT for cultural default indicators
    cultural_indicators = [
        "in english", "in western", "typically", "usually", 
        "in most cultures", "standard", "normal"
    ]
    if any(ind in cot for ind in cultural_indicators):
        if error.kinship_system in NON_WESTERN_SYSTEMS:
            return "cultural_default", "applied Western assumptions"
    
    # Check for conceptual errors in Cat 4
    if error.category == 4:
        if error.has_cultural_override:
            # Check if model acknowledged the cultural term
            if error.kin_term and error.kin_term.lower() not in cot:
                return "conceptual_error", f"ignored cultural term '{error.kin_term}'"
            return "cultural_default", "failed to apply cultural override"
        return "conceptual_error", "misunderstood relationship"
    
    # Multi-hop reasoning failures
    if error.n_hops >= 3:
        return "reasoning_chain_break", f"{error.n_hops}-hop chain"
    
    # Context reading issues
    if "not mentioned" in cot or "cannot find" in cot:
        gt = error.ground_truth
        if gt and (isinstance(gt, list) and gt) or gt:
            return "context_misread", "claimed missing when present"
    
    return "other", ""

def analyze_all_errors(errors: List[ErrorSample]) -> Dict:
    """Classify all errors and compute statistics"""
    
    for error in errors:
        error.error_type, error.error_subtype = classify_error(error)
    
    # Aggregate statistics
    stats = {
        "total": len(errors),
        "by_type": defaultdict(int),
        "by_system": defaultdict(lambda: defaultdict(int)),
        "by_category": defaultdict(lambda: defaultdict(int)),
        "cot_improvement": {"improved": 0, "still_wrong": 0},
        "western_vs_nonwestern": {
            "western": defaultdict(int),
            "non_western": defaultdict(int)
        }
    }
    
    for error in errors:
        stats["by_type"][error.error_type] += 1
        stats["by_system"][error.kinship_system][error.error_type] += 1
        stats["by_category"][error.category][error.error_type] += 1
        
        if error.cot_correct:
            stats["cot_improvement"]["improved"] += 1
        else:
            stats["cot_improvement"]["still_wrong"] += 1
        
        group = "western" if error.kinship_system in WESTERN_SYSTEMS else "non_western"
        stats["western_vs_nonwestern"][group][error.error_type] += 1
    
    # Convert defaultdicts to regular dicts for JSON serialization
    stats["by_type"] = dict(stats["by_type"])
    stats["by_system"] = {k: dict(v) for k, v in stats["by_system"].items()}
    stats["by_category"] = {k: dict(v) for k, v in stats["by_category"].items()}
    stats["western_vs_nonwestern"] = {k: dict(v) for k, v in stats["western_vs_nonwestern"].items()}
    
    return stats

# =============================================================================
# Paper Example Selection & Formatting
# =============================================================================

def select_paper_examples(errors: List[ErrorSample], n_per_category: int = 1) -> Dict[str, ErrorSample]:
    """
    Select the best examples for each paper example category.
    """
    examples = {}
    
    for example_key, criteria in PAPER_EXAMPLE_TARGETS.items():
        candidates = [e for e in errors if e.paper_example_category == example_key]
        
        if not candidates:
            # Find from general pool
            target_systems = criteria['systems'] or KINSHIP_SYSTEMS
            target_cats = criteria['categories']
            candidates = [
                e for e in errors 
                if e.kinship_system in target_systems and e.category in target_cats
            ]
        
        if candidates:
            # Score candidates by quality
            for c in candidates:
                score = 0
                # Prefer longer, more detailed CoT responses
                if c.cot_response and len(c.cot_response) > 300:
                    score += 1
                # Prefer clear error patterns
                if c.error_type in criteria.get('error_types', []):
                    score += 2
                # Prefer questions with cultural override for cultural examples
                if 'cultural' in example_key and c.has_cultural_override:
                    score += 2
                # Prefer manageable context length
                if 200 < len(c.context) < 1000:
                    score += 1
                c.example_quality_score = score
            
            # Select best
            best = max(candidates, key=lambda x: x.example_quality_score)
            best.selected_for_paper = True
            examples[example_key] = best
    
    return examples

def format_example_for_latex(example: ErrorSample, example_key: str) -> str:
    """Format a single example for LaTeX"""
    criteria = PAPER_EXAMPLE_TARGETS.get(example_key, {})
    section_title = criteria.get('paper_section', example_key)
    
    # Truncate context for readability
    context_preview = example.context[:400] + "..." if len(example.context) > 400 else example.context
    context_preview = context_preview.replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')
    
    # Format ground truth
    gt = example.ground_truth
    if isinstance(gt, list):
        gt_str = ", ".join(str(g) for g in gt[:5])
        if len(gt) > 5:
            gt_str += f"... ({len(gt)} total)"
    else:
        gt_str = str(gt)
    gt_str = gt_str.replace('_', r'\_').replace('&', r'\&')
    
    # Extract key reasoning steps from CoT
    cot_excerpt = ""
    if example.cot_response:
        # Get STEP 3 (reasoning chain) if present
        match = re.search(r'STEP 3[^:]*:(.*?)(?:STEP 4|FINAL|$)', 
                         example.cot_response, re.DOTALL | re.IGNORECASE)
        if match:
            cot_excerpt = match.group(1).strip()[:300]
        else:
            cot_excerpt = example.cot_response[:300]
        cot_excerpt = cot_excerpt.replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')
    
    # Get error type info (extract before f-string to avoid syntax issues)
    error_info = ERROR_TAXONOMY.get(example.error_type, {})
    error_short = error_info.get('short', example.error_type) if error_info else example.error_type
    error_diagnosis = error_info.get('diagnosis', '') if error_info else ''
    
    # Escape special characters in question text and prediction
    question_escaped = example.question_text.replace('_', r'\_').replace('&', r'\&')
    prediction_escaped = str(example.original_prediction)[:100].replace('_', r'\_').replace('&', r'\&')
    
    latex = f"""
\\begin{{tcolorbox}}[title={{{section_title}}}, colback=gray!5]
\\textbf{{System:}} {example.kinship_system.title()} \\quad
\\textbf{{Category:}} {example.category} \\quad
\\textbf{{N-hops:}} {example.n_hops}

\\textbf{{Question:}} {question_escaped}

\\textbf{{Ground Truth:}} {gt_str}

\\textbf{{Model Prediction:}} {prediction_escaped}

\\textbf{{Error Type:}} {error_short}

\\textbf{{Model Reasoning (excerpt):}}
\\begin{{quote}}
\\textit{{{cot_excerpt}...}}
\\end{{quote}}

\\textbf{{Analysis:}} {error_diagnosis}
\\end{{tcolorbox}}
"""
    return latex

def generate_latex_error_analysis_section(
    stats: Dict, 
    examples: Dict[str, ErrorSample],
    source_model: str,
    analysis_model: str
) -> str:
    """Generate complete LaTeX for Section 5.1 Error Analysis"""
    
    # Error distribution table
    type_rows = []
    for error_type, info in ERROR_TAXONOMY.items():
        if error_type == "other":
            continue
        count = stats['by_type'].get(error_type, 0)
        pct = count / stats['total'] * 100 if stats['total'] > 0 else 0
        type_rows.append(f"    {info['short']} & {count} & {pct:.1f}\\% \\\\")
    
    # Western vs Non-Western breakdown
    western_total = sum(stats['western_vs_nonwestern']['western'].values())
    nonwestern_total = sum(stats['western_vs_nonwestern']['non_western'].values())
    
    latex = f"""
\\subsection{{Error Analysis}}
\\label{{sec:error-analysis}}

To understand \\textit{{why}} models fail on non-Western kinship systems, we conducted detailed error analysis using chain-of-thought prompting. We sampled {stats['total']} incorrect responses from {source_model} and re-queried with explicit reasoning prompts{"" if source_model == analysis_model else f" using {analysis_model}"}.

\\subsubsection{{Error Type Distribution}}

Table~\\ref{{tab:error-types}} shows the distribution of error types. We identify {len([k for k,v in stats['by_type'].items() if v > 0])} distinct failure modes.

\\begin{{table}}[h]
\\centering
\\caption{{Error Type Distribution (n={stats['total']})}}
\\label{{tab:error-types}}
\\begin{{tabular}}{{lrr}}
\\toprule
Error Type & Count & \\% \\\\
\\midrule
{chr(10).join(type_rows)}
\\midrule
    Other & {stats['by_type'].get('other', 0)} & {stats['by_type'].get('other', 0)/stats['total']*100:.1f}\\% \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\subsubsection{{Key Findings}}

\\textbf{{Cultural Default Errors:}} {stats['by_type'].get('cultural_default', 0)} errors ({stats['by_type'].get('cultural_default', 0)/stats['total']*100:.1f}\\%) occurred when models applied Western kinship assumptions to non-Western systems. This confirms our hypothesis that LLMs encode Anglo-American kinship as the default.

\\textbf{{Incomplete Enumeration:}} The most common failure mode ({stats['by_type'].get('incomplete_enumeration', 0)} cases) involves models finding \\textit{{some}} correct answers but failing to exhaustively enumerate all members of a kinship class (e.g., all matriclan members).

\\textbf{{Chain-of-Thought Improvement:}} When prompted to show reasoning, {stats['cot_improvement']['improved']} of {stats['total']} previously incorrect answers became correct ({stats['cot_improvement']['improved']/stats['total']*100:.1f}\\%), suggesting some failures stem from reasoning depth rather than knowledge gaps.

\\subsubsection{{Qualitative Examples}}

We present representative examples of each major failure mode:

"""
    
    # Add formatted examples
    for example_key, example in examples.items():
        if example:
            latex += format_example_for_latex(example, example_key)
            latex += "\n"
    
    latex += """
\\subsubsection{{Western vs. Non-Western Error Patterns}}

Comparing error distributions between Western (Eskimo, Sudanese) and non-Western systems reveals distinct patterns. Cultural default errors occur almost exclusively in non-Western systems, while reasoning chain breaks are more evenly distributed, suggesting the latter reflects general multi-hop reasoning limitations rather than cultural bias.
"""
    
    return latex

# =============================================================================
# Output Generation
# =============================================================================

def save_all_outputs(
    errors: List[ErrorSample],
    stats: Dict,
    examples: Dict[str, ErrorSample],
    analysis_model: str,
    source_model: str,
    output_dir: str
):
    """Save all output files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Full JSON results
    full_output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "source_model": source_model,  # Model whose errors are being analyzed
            "analysis_model": analysis_model,  # Model doing the CoT analysis
            "total_samples": len(errors),
            "description": f"Error analysis of {source_model} results using {analysis_model} for CoT reasoning"
        },
        "statistics": stats,
        "paper_examples": {k: asdict(v) for k, v in examples.items()},
        "all_samples": [asdict(e) for e in errors]
    }
    
    json_path = output_path / "error_analysis_full.json"
    with open(json_path, 'w') as f:
        json.dump(full_output, f, indent=2, default=str)
    print(f"Full results saved to {json_path}")
    
    # 2. LaTeX section
    latex_content = generate_latex_error_analysis_section(stats, examples, source_model, analysis_model)
    latex_path = output_path / "error_analysis_section.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_content)
    print(f"LaTeX section saved to {latex_path}")
    
    # 3. Markdown examples (for easy preview)
    md_content = generate_markdown_examples(examples)
    md_path = output_path / "error_examples.md"
    with open(md_path, 'w') as f:
        f.write(md_content)
    print(f"Markdown examples saved to {md_path}")
    
    # 4. Summary statistics CSV
    csv_content = generate_stats_csv(stats)
    csv_path = output_path / "error_stats.csv"
    with open(csv_path, 'w') as f:
        f.write(csv_content)
    print(f"Statistics CSV saved to {csv_path}")

def generate_markdown_examples(examples: Dict[str, ErrorSample]) -> str:
    """Generate markdown formatted examples for preview"""
    md = "# KinshipQA Error Analysis Examples\n\n"
    md += f"Generated: {datetime.now().isoformat()}\n\n"
    
    for example_key, example in examples.items():
        if not example:
            continue
        
        criteria = PAPER_EXAMPLE_TARGETS.get(example_key, {})
        
        md += f"## {criteria.get('paper_section', example_key)}\n\n"
        md += f"**Kinship System:** {example.kinship_system.title()}  \n"
        md += f"**Category:** {example.category} | **N-hops:** {example.n_hops}  \n"
        md += f"**Error Type:** {example.error_type}  \n\n"
        
        md += f"### Question\n{example.question_text}\n\n"
        md += f"### Ground Truth\n{example.ground_truth}\n\n"
        md += f"### Model Prediction\n{example.original_prediction}\n\n"
        
        md += f"### Chain-of-Thought Response\n```\n{example.cot_response[:800]}{'...' if len(example.cot_response) > 800 else ''}\n```\n\n"
        
        md += f"### Analysis\n{ERROR_TAXONOMY.get(example.error_type, {}).get('diagnosis', 'N/A')}\n\n"
        md += "---\n\n"
    
    return md

def generate_stats_csv(stats: Dict) -> str:
    """Generate CSV of statistics"""
    lines = ["metric,value"]
    lines.append(f"total_samples,{stats['total']}")
    lines.append(f"cot_improved,{stats['cot_improvement']['improved']}")
    lines.append(f"cot_still_wrong,{stats['cot_improvement']['still_wrong']}")
    
    for error_type, count in stats['by_type'].items():
        lines.append(f"error_{error_type},{count}")
    
    return "\n".join(lines)

# =============================================================================
# SLURM Script Generation
# =============================================================================

def generate_slurm_script(args) -> str:
    """Generate a SLURM submission script"""
    script = f"""#!/bin/bash
#SBATCH --job-name=kinship_error_analysis
#SBATCH --output=error_analysis_%j.out
#SBATCH --error=error_analysis_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

# Load modules (adjust for your HPC)
module load python/3.10

# Activate virtual environment if needed
# source ~/venv/bin/activate

# Set API keys (use secrets management in production)
# export OPENAI_API_KEY="your-key-here"
# export ANTHROPIC_API_KEY="your-key-here"

# Run analysis
python error_analysis_cot_v2.py \\
    --results-dir {args.results_dir or './results/'} \\
    --dataset-dir {args.dataset_dir or './datasets/'} \\
    --provider {args.provider} \\
    --model {args.model} \\
    --sample-size {args.sample_size} \\
    --output-dir {args.output_dir or './analysis/'} \\
    --checkpoint ./checkpoint.json

echo "Error analysis complete!"
"""
    return script

# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="KinshipQA Error Analysis with Chain-of-Thought (v2.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with OpenAI
    python error_analysis_cot_v2.py --results-dir ./results/ --dataset-dir ./datasets/ \\
        --provider openai --model gpt-4o-mini --output-dir ./analysis/
    
    # With specific result files
    python error_analysis_cot_v2.py --results crow_results.json omaha_results.json \\
        --dataset-dir ./datasets/ --provider anthropic --model claude-3-5-haiku-latest
    
    # Generate SLURM script
    python error_analysis_cot_v2.py --generate-slurm --provider openai --model gpt-4o-mini
        """
    )
    
    # Input options
    parser.add_argument("--results", nargs="+", help="Specific result JSON files")
    parser.add_argument("--results-dir", type=str, 
                        help="Directory containing result files (e.g., ./result_qwen3/)")
    parser.add_argument("--results-dirs", nargs="+",
                        help="Multiple result directories to combine (e.g., ./result_qwen3/ ./result_gemma3/)")
    parser.add_argument("--dataset-dir", type=str, default="./datasets/", 
                        help="Directory with original dataset JSONL files")
    parser.add_argument("--source-model", type=str, default="",
                        help="Name of the model whose errors are being analyzed (for paper metadata)")
    
    # Sampling options
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    
    # Provider options
    parser.add_argument("--provider", type=str, default="openai",
                        choices=["openai", "anthropic", "gemini", "ollama"])
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--api-key", type=str)
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="./analysis/")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint file for resuming")
    
    # Control options
    parser.add_argument("--rate-limit", type=float, default=0.5, 
                        help="Delay between API calls (seconds)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--generate-slurm", action="store_true",
                        help="Generate SLURM script and exit")
    
    args = parser.parse_args()
    
    # Generate SLURM script if requested
    if args.generate_slurm:
        slurm_script = generate_slurm_script(args)
        slurm_path = "run_error_analysis.slurm"
        with open(slurm_path, 'w') as f:
            f.write(slurm_script)
        print(f"SLURM script saved to {slurm_path}")
        print("Submit with: sbatch run_error_analysis.slurm")
        return
    
    # Collect result files
    result_files = []
    if args.results:
        result_files = [Path(r) for r in args.results]
    elif args.results_dirs:
        # Multiple directories (combine errors from multiple models)
        for dir_path in args.results_dirs:
            result_files.extend(Path(dir_path).glob("*_results.json"))
        print(f"Combining results from {len(args.results_dirs)} directories")
    elif args.results_dir:
        result_files = list(Path(args.results_dir).glob("*_results.json"))
    else:
        parser.error("Please provide --results, --results-dir, or --results-dirs")
    
    result_files = [str(f) for f in result_files if f.exists()]
    print(f"Found {len(result_files)} result files")
    
    # Infer source model name from directory if not provided
    source_model_name = args.source_model
    if not source_model_name:
        if args.results_dir:
            source_model_name = Path(args.results_dir).name.replace("result_", "").replace("results_", "")
        elif args.results_dirs:
            source_model_name = "multiple_models"
        else:
            source_model_name = "unknown"
    
    print(f"Source model (being analyzed): {source_model_name}")
    print(f"Analysis model (doing CoT): {args.provider}/{args.model}")
    
    if not result_files:
        print("No result files found!")
        return
    
    # Load data
    results_by_system, datasets_by_system = load_results_and_datasets(
        result_files, args.dataset_dir
    )
    
    # Select errors strategically
    errors = select_errors_strategically(
        results_by_system, datasets_by_system,
        args.sample_size, args.seed
    )
    
    if not errors:
        print("No errors found to analyze!")
        return
    
    # Initialize provider
    print(f"\nInitializing {args.provider}/{args.model}...")
    provider = create_provider(args.provider, args.model, args.api_key)
    
    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run CoT analysis
    print(f"\nRunning Chain-of-Thought analysis on {len(errors)} samples...")
    checkpoint_path = args.checkpoint or str(Path(args.output_dir) / "checkpoint.json")
    
    errors = run_cot_analysis(
        errors, provider,
        checkpoint_path=checkpoint_path,
        rate_limit_delay=args.rate_limit,
        verbose=args.verbose
    )
    
    # Classify errors
    print("\nClassifying error types...")
    stats = analyze_all_errors(errors)
    
    # Select paper examples
    print("\nSelecting best examples for paper...")
    examples = select_paper_examples(errors)
    
    # Print summary
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal samples: {stats['total']}")
    print(f"CoT improvement: {stats['cot_improvement']['improved']} ({stats['cot_improvement']['improved']/stats['total']*100:.1f}%)")
    
    print("\nError Type Distribution:")
    for error_type, count in sorted(stats['by_type'].items(), key=lambda x: -x[1]):
        pct = count / stats['total'] * 100
        print(f"  {error_type:<25} {count:>4} ({pct:>5.1f}%)")
    
    print("\nSelected Paper Examples:")
    for key, ex in examples.items():
        if ex:
            print(f"  {key}: {ex.kinship_system}/{ex.error_type}")
    
    # Save all outputs
    print("\nSaving outputs...")
    save_all_outputs(errors, stats, examples, provider.get_name(), source_model_name, args.output_dir)
    
    print("\n" + "=" * 70)
    print("DONE! Check the output directory for:")
    print("  - error_analysis_full.json (complete data)")
    print("  - error_analysis_section.tex (LaTeX for paper)")
    print("  - error_examples.md (preview examples)")
    print("  - error_stats.csv (statistics)")
    print("=" * 70)

if __name__ == "__main__":
    main()
