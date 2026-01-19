#!/usr/bin/env python3
"""
KinshipQA LLM Tester v5.1
=========================
Evaluates LLM performance on kinship reasoning questions.
Updated to work with pipeline v6.1 dataset format.

v5.1 Changes:
- Fixed None response handling for Gemini (safety filters, empty responses)
- Improved error handling and retry logic
- Better logging for debugging

Supports: Ollama, OpenAI GPT, Google Gemini, Anthropic Claude

Installation:
    pip install google-genai openai anthropic llama-index-llms-ollama

Usage:
    python llm_tester_v5.py --all --dataset-dir ./datasets/ --provider gemini --model gemini-2.5-flash

Author: Tianda (ACL 2026)
Version: 5.1
"""

import json
import os
import re
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple, Set
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime


# =============================================================================
# Constants
# =============================================================================

KINSHIP_SYSTEMS = ['eskimo', 'hawaiian', 'iroquois', 'dravidian', 'crow', 'omaha', 'sudanese']
WESTERN_SYSTEMS = ['eskimo', 'sudanese']
NON_WESTERN_SYSTEMS = ['hawaiian', 'iroquois', 'dravidian', 'crow', 'omaha']

RATE_LIMITS = {
    "ollama": 1000,
    "openai": 60,
    "gemini": 15,
    "anthropic": 50
}


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TestConfig:
    """Configuration for LLM testing"""
    dataset_path: str = ""
    output_path: str = "results.json"
    provider: str = "ollama"
    model: str = None
    api_key: Optional[str] = None
    limit: Optional[int] = None
    categories: Optional[List[int]] = None
    verbose: bool = False
    debug: bool = False
    temperature: float = 0.0
    max_tokens: int = 512
    retry_count: int = 3
    retry_delay: float = 2.0
    
    def __post_init__(self):
        if self.model is None:
            defaults = {
                "ollama": "llama3.1:8b",
                "openai": "gpt-4o-mini",
                "gemini": "gemini-2.5-flash",
                "anthropic": "claude-3-5-haiku-latest"
            }
            self.model = defaults.get(self.provider, "llama3.1:8b")


# =============================================================================
# LLM Provider Interfaces
# =============================================================================

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict:
        pass
    
    def get_rate_limit_delay(self) -> float:
        return 0.0


class OllamaProvider(LLMProvider):
    """Ollama provider for local models"""
    
    def __init__(self, model: str, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self.llm = None
        self._initialize()
    
    def _initialize(self):
        try:
            from llama_index.llms.ollama import Ollama
            self.llm = Ollama(
                model=self.model,
                temperature=self.temperature,
                request_timeout=120.0
            )
        except ImportError:
            raise ImportError("Please install: pip install llama-index-llms-ollama")
    
    def generate(self, prompt: str) -> str:
        response = self.llm.complete(prompt)
        result = str(response) if response else ""
        return result
    
    def get_model_info(self) -> Dict:
        return {"provider": "ollama", "model": self.model}


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider"""
    
    def __init__(self, model: str, api_key: str = None, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        self._initialize()
    
    def _initialize(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("Please install: pip install openai")
    
    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=512
        )
        content = response.choices[0].message.content
        return content if content else ""
    
    def get_model_info(self) -> Dict:
        return {"provider": "openai", "model": self.model}
    
    def get_rate_limit_delay(self) -> float:
        return 1.0


class GeminiProvider(LLMProvider):
    """Google Gemini provider using new google-genai SDK"""
    
    def __init__(self, model: str, api_key: str = None, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.client = None
        self._initialize()
    
    def _initialize(self):
        try:
            from google import genai
            from google.genai import types
            self.genai = genai
            self.types = types
            self.client = genai.Client(api_key=self.api_key)
        except ImportError:
            raise ImportError("Please install: pip install google-genai")
    
    def generate(self, prompt: str) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=self.types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=512
                )
            )
            
            # Handle various response scenarios
            if response is None:
                return "[ERROR: No response from Gemini]"
            
            # Check if response was blocked by safety filters
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                block_reason = getattr(response.prompt_feedback, 'block_reason', None)
                if block_reason:
                    return f"[BLOCKED: {block_reason}]"
            
            # Try to get text from response
            if hasattr(response, 'text') and response.text:
                return response.text
            
            # Try candidates
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        parts_text = []
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                parts_text.append(part.text)
                        if parts_text:
                            return " ".join(parts_text)
                
                # Check finish reason
                finish_reason = getattr(candidate, 'finish_reason', None)
                if finish_reason and str(finish_reason) != "STOP":
                    return f"[BLOCKED: finish_reason={finish_reason}]"
            
            return "[ERROR: Empty response from Gemini]"
            
        except Exception as e:
            error_str = str(e)
            if "SAFETY" in error_str.upper() or "BLOCKED" in error_str.upper():
                return f"[BLOCKED: {error_str[:100]}]"
            raise  # Re-raise other errors for retry logic
    
    def get_model_info(self) -> Dict:
        return {"provider": "gemini", "model": self.model}
    
    def get_rate_limit_delay(self) -> float:
        return 4.0  # Gemini free tier is 15 RPM


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider"""
    
    def __init__(self, model: str, api_key: str = None, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = None
        self._initialize()
    
    def _initialize(self):
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("Please install: pip install anthropic")
    
    def generate(self, prompt: str) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        if message.content and len(message.content) > 0:
            return message.content[0].text
        return ""
    
    def get_model_info(self) -> Dict:
        return {"provider": "anthropic", "model": self.model}
    
    def get_rate_limit_delay(self) -> float:
        return 1.2


def create_provider(config: TestConfig) -> LLMProvider:
    """Factory function to create LLM provider"""
    if config.provider == "ollama":
        return OllamaProvider(config.model, config.temperature)
    elif config.provider == "openai":
        return OpenAIProvider(config.model, config.api_key, config.temperature)
    elif config.provider == "gemini":
        return GeminiProvider(config.model, config.api_key, config.temperature)
    elif config.provider == "anthropic":
        return AnthropicProvider(config.model, config.api_key, config.temperature)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")


# =============================================================================
# Answer Extraction & Comparison
# =============================================================================

def normalize_name(name: str) -> str:
    """Normalize a name for comparison"""
    if not name:
        return ""
    name = re.sub(r'\s+', ' ', name.strip().lower())
    name = re.sub(r'[^\w\s]', '', name)
    return name


def extract_names_from_response(response: str) -> List[str]:
    """Extract person names from LLM response"""
    if not response:
        return []
    
    response = response.strip()
    
    # Skip error/blocked responses
    if response.startswith("[ERROR") or response.startswith("[BLOCKED"):
        return []
    
    # Strategy 1: Find all "First Last" name patterns
    name_pattern = r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b'
    matches = re.findall(name_pattern, response)
    if matches:
        seen = set()
        unique_names = []
        for name in matches:
            if name.lower() not in seen:
                seen.add(name.lower())
                unique_names.append(name)
        return unique_names
    
    # Strategy 2: Single capitalized word
    single_name = re.findall(r'\b([A-Z][a-z]{2,})\b', response)
    if single_name:
        return single_name[:1]
    
    return [response.strip()]


def extract_number_from_response(response: str) -> Optional[int]:
    """Extract a number from LLM response"""
    if not response:
        return None
    
    if response.startswith("[ERROR") or response.startswith("[BLOCKED"):
        return None
    
    numbers = re.findall(r'\b(\d+)\b', response)
    if numbers:
        return int(numbers[0])
    
    word_to_num = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'ten': 10, 'eleven': 11, 'twelve': 12
    }
    
    response_lower = response.lower()
    for word, num in word_to_num.items():
        if word in response_lower:
            return num
    
    return None


def extract_relationship_from_response(response: str) -> Optional[str]:
    """Extract biological relationship term from response"""
    if not response:
        return None
    
    if response.startswith("[ERROR") or response.startswith("[BLOCKED"):
        return None
    
    response_lower = response.lower()
    
    relationships = [
        "father's brother's child", "father's sister's child", 
        "mother's brother's child", "mother's sister's child",
        "paternal grandfather", "paternal grandmother", 
        "maternal grandfather", "maternal grandmother",
        "paternal uncle", "maternal uncle", "paternal aunt", "maternal aunt",
        "father's brother", "father's sister", "mother's brother", "mother's sister",
        "cross-cousin", "parallel cousin", 
        "classificatory sibling", "classificatory father", "classificatory mother",
        "uncle", "aunt", "cousin", "nephew", "niece", 
        "grandparent", "grandchild", "grandfather", "grandmother",
        "fbc", "fzc", "mbc", "mzc", "fb", "fz", "mb", "mz"
    ]
    
    for rel in relationships:
        if rel in response_lower:
            return rel
    
    return response.strip().lower()


def compare_relationship(predicted: str, ground_truth: str) -> bool:
    """Compare relationship terms flexibly"""
    if not predicted or not ground_truth:
        return False
    
    pred = predicted.lower().strip()
    gt = ground_truth.lower().strip()
    
    if pred == gt:
        return True
    if gt in pred:
        return True
    if pred in gt:
        return True
    
    gt_clean = re.sub(r'\s*\([^)]*\)', '', gt).strip()
    pred_clean = re.sub(r'\s*\([^)]*\)', '', pred).strip()
    
    if pred_clean == gt_clean:
        return True
    if gt_clean in pred_clean:
        return True
    if pred_clean in gt_clean:
        return True
    
    return False


def compare_answers(predicted: Any, ground_truth: Any, question_type: str = "name") -> Dict:
    """Compare predicted answer with ground truth"""
    
    # Handle None/error responses
    if predicted is None or (isinstance(predicted, str) and 
                             (predicted.startswith("[ERROR") or predicted.startswith("[BLOCKED"))):
        return {
            "exact_match": False,
            "partial_match": False,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "predicted": [predicted] if predicted else [],
            "ground_truth": [ground_truth] if not isinstance(ground_truth, list) else ground_truth,
            "error": True
        }
    
    # Normalize ground truth to list
    if isinstance(ground_truth, list):
        gt_list = [normalize_name(str(g)) for g in ground_truth]
    else:
        gt_list = [normalize_name(str(ground_truth))]
    
    gt_set = set(gt_list)
    
    if question_type == "count":
        pred_num = extract_number_from_response(str(predicted))
        gt_num = ground_truth if isinstance(ground_truth, int) else int(ground_truth)
        exact = pred_num == gt_num
        return {
            "exact_match": exact,
            "partial_match": exact,
            "precision": 1.0 if exact else 0.0,
            "recall": 1.0 if exact else 0.0,
            "f1": 1.0 if exact else 0.0,
            "predicted": [pred_num],
            "ground_truth": [gt_num]
        }
    
    elif question_type == "relationship":
        pred_rel = extract_relationship_from_response(str(predicted))
        gt_rel = str(ground_truth)
        exact = compare_relationship(pred_rel, gt_rel)
        return {
            "exact_match": exact,
            "partial_match": exact,
            "precision": 1.0 if exact else 0.0,
            "recall": 1.0 if exact else 0.0,
            "f1": 1.0 if exact else 0.0,
            "predicted": [pred_rel],
            "ground_truth": [ground_truth]
        }
    
    else:  # name-based questions
        pred_names = extract_names_from_response(str(predicted))
        pred_list = [normalize_name(n) for n in pred_names]
        pred_set = set(pred_list)
        
        if not pred_set and not gt_set:
            precision = recall = f1 = 1.0
        elif not pred_set:
            precision = recall = f1 = 0.0
        elif not gt_set:
            precision = recall = f1 = 0.0
        else:
            correct = pred_set.intersection(gt_set)
            precision = len(correct) / len(pred_set)
            recall = len(correct) / len(gt_set)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        exact_match = pred_set == gt_set
        partial_match = len(pred_set.intersection(gt_set)) > 0
        
        return {
            "exact_match": exact_match,
            "partial_match": partial_match,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predicted": pred_names,
            "ground_truth": list(ground_truth) if isinstance(ground_truth, list) else [ground_truth]
        }


def determine_question_type(question: Dict) -> str:
    """Determine the type of question for answer extraction"""
    q_text = question.get('question_text', '').lower()
    
    if 'how many' in q_text:
        return "count"
    elif 'biological relationship' in q_text or 'what is their' in q_text:
        return "relationship"
    else:
        return "name"


# =============================================================================
# Prompt Generation
# =============================================================================

def create_prompt(question: Dict) -> str:
    """Create a prompt for the LLM"""
    context = question.get('context', '')
    q_text = question.get('question_text', '')
    
    prompt = f"""Answer the following question based on the given context. 
Be concise and provide only the answer without explanation.

Context: {context}

Question: {q_text}

Answer:"""
    
    return prompt


# =============================================================================
# Testing Logic
# =============================================================================

@dataclass
class QuestionResult:
    """Result for a single question"""
    question_id: str
    category: int
    n_hops: int
    kinship_system: str
    question_text: str
    ground_truth: Any
    predicted: Any
    exact_match: bool
    partial_match: bool
    precision: float
    recall: float
    f1: float
    response_time: float
    has_cultural_override: bool = False
    bio_term: str = ""
    kin_term: str = ""
    raw_response: str = ""
    is_error: bool = False


@dataclass
class TestResults:
    """Aggregated test results"""
    kinship_system: str
    model_info: Dict
    total_questions: int = 0
    exact_matches: int = 0
    partial_matches: int = 0
    total_precision: float = 0.0
    total_recall: float = 0.0
    total_f1: float = 0.0
    total_time: float = 0.0
    error_count: int = 0
    
    by_category: Dict = field(default_factory=lambda: defaultdict(lambda: {
        "total": 0, "exact": 0, "partial": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0
    }))
    
    by_hops: Dict = field(default_factory=lambda: defaultdict(lambda: {
        "total": 0, "exact": 0, "partial": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0
    }))
    
    by_override: Dict = field(default_factory=lambda: {
        True: {"total": 0, "exact": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
        False: {"total": 0, "exact": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    })
    
    question_results: List[QuestionResult] = field(default_factory=list)
    
    def add_result(self, result: QuestionResult):
        self.question_results.append(result)
        self.total_questions += 1
        
        if result.is_error:
            self.error_count += 1
        
        if result.exact_match:
            self.exact_matches += 1
        if result.partial_match:
            self.partial_matches += 1
        
        self.total_precision += result.precision
        self.total_recall += result.recall
        self.total_f1 += result.f1
        self.total_time += result.response_time
        
        cat = result.category
        self.by_category[cat]["total"] += 1
        if result.exact_match:
            self.by_category[cat]["exact"] += 1
        if result.partial_match:
            self.by_category[cat]["partial"] += 1
        self.by_category[cat]["precision"] += result.precision
        self.by_category[cat]["recall"] += result.recall
        self.by_category[cat]["f1"] += result.f1
        
        hops = result.n_hops
        self.by_hops[hops]["total"] += 1
        if result.exact_match:
            self.by_hops[hops]["exact"] += 1
        if result.partial_match:
            self.by_hops[hops]["partial"] += 1
        self.by_hops[hops]["precision"] += result.precision
        self.by_hops[hops]["recall"] += result.recall
        self.by_hops[hops]["f1"] += result.f1
        
        override = result.has_cultural_override
        self.by_override[override]["total"] += 1
        if result.exact_match:
            self.by_override[override]["exact"] += 1
        self.by_override[override]["precision"] += result.precision
        self.by_override[override]["recall"] += result.recall
        self.by_override[override]["f1"] += result.f1
    
    def get_summary(self) -> Dict:
        n = self.total_questions or 1
        
        by_cat_summary = {}
        for cat, stats in self.by_category.items():
            cat_n = stats["total"] or 1
            by_cat_summary[f"cat_{cat}"] = {
                "total": stats["total"],
                "exact_match": stats["exact"] / cat_n,
                "partial_match": stats["partial"] / cat_n,
                "precision": stats["precision"] / cat_n,
                "recall": stats["recall"] / cat_n,
                "f1": stats["f1"] / cat_n
            }
        
        by_hops_summary = {}
        for hops, stats in self.by_hops.items():
            hop_n = stats["total"] or 1
            by_hops_summary[f"hop_{hops}"] = {
                "total": stats["total"],
                "exact_match": stats["exact"] / hop_n,
                "partial_match": stats["partial"] / hop_n,
                "precision": stats["precision"] / hop_n,
                "recall": stats["recall"] / hop_n,
                "f1": stats["f1"] / hop_n
            }
        
        by_override_summary = {}
        for override, stats in self.by_override.items():
            ov_n = stats["total"] or 1
            if stats["total"] > 0:
                key = "with_override" if override else "no_override"
                by_override_summary[key] = {
                    "total": stats["total"],
                    "exact_match": stats["exact"] / ov_n,
                    "precision": stats["precision"] / ov_n,
                    "recall": stats["recall"] / ov_n,
                    "f1": stats["f1"] / ov_n
                }
        
        return {
            "kinship_system": self.kinship_system,
            "model": self.model_info,
            "total_questions": self.total_questions,
            "error_count": self.error_count,
            "accuracy": {
                "exact_match": self.exact_matches / n,
                "partial_match": self.partial_matches / n,
                "precision": self.total_precision / n,
                "recall": self.total_recall / n,
                "f1": self.total_f1 / n
            },
            "by_category": by_cat_summary,
            "by_hops": by_hops_summary,
            "by_cultural_override": by_override_summary,
            "timing": {
                "total_seconds": self.total_time,
                "avg_per_question": self.total_time / n
            }
        }


def run_test(config: TestConfig, provider: LLMProvider) -> TestResults:
    """Run the test on a dataset"""
    
    # Load dataset
    questions = []
    with open(config.dataset_path) as f:
        for line in f:
            questions.append(json.loads(line))
    
    # Filter by categories if specified
    if config.categories:
        questions = [q for q in questions if q.get('category') in config.categories]
    
    # Limit if specified
    if config.limit:
        questions = questions[:config.limit]
    
    # Extract system name
    system_name = Path(config.dataset_path).stem.replace("_dataset", "").replace("_natural", "")
    
    results = TestResults(
        kinship_system=system_name,
        model_info=provider.get_model_info()
    )
    
    rate_limit_delay = provider.get_rate_limit_delay()
    
    for i, question in enumerate(questions):
        if config.verbose:
            print(f"\n[{i+1}/{len(questions)}] {question.get('question_id', 'unknown')}")
        
        prompt = create_prompt(question)
        question_type = determine_question_type(question)
        
        # Get response with retry logic
        start_time = time.time()
        response = None
        is_error = False
        
        for attempt in range(config.retry_count):
            try:
                response = provider.generate(prompt)
                
                # Check if response is valid (not None, not empty)
                if response is None:
                    response = "[ERROR: Provider returned None]"
                    is_error = True
                elif response.startswith("[ERROR") or response.startswith("[BLOCKED"):
                    is_error = True
                
                break  # Success, exit retry loop
                
            except Exception as e:
                if attempt < config.retry_count - 1:
                    print(f"\n  Retry {attempt + 1}/{config.retry_count} after error: {e}")
                    time.sleep(config.retry_delay * (attempt + 1))
                else:
                    response = f"[ERROR: {str(e)[:100]}]"
                    is_error = True
        
        response_time = time.time() - start_time
        
        # Ensure response is a string
        if response is None:
            response = "[ERROR: No response]"
            is_error = True
        
        # Compare answers
        comparison = compare_answers(response, question.get('ground_truth'), question_type)
        
        # Create result
        result = QuestionResult(
            question_id=question.get('question_id', ''),
            category=question.get('category', 0),
            n_hops=question.get('n_hops', 0),
            kinship_system=system_name,
            question_text=question.get('question_text', ''),
            ground_truth=question.get('ground_truth'),
            predicted=comparison['predicted'],
            exact_match=comparison['exact_match'],
            partial_match=comparison['partial_match'],
            precision=comparison['precision'],
            recall=comparison['recall'],
            f1=comparison['f1'],
            response_time=response_time,
            has_cultural_override=question.get('has_cultural_override', False),
            bio_term=question.get('bio_term', ''),
            kin_term=question.get('kin_term', ''),
            raw_response=str(response)[:500] if response else "",  # Safe truncation
            is_error=is_error or comparison.get('error', False)
        )
        
        results.add_result(result)
        
        # Rate limiting
        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)
        
        if config.verbose:
            status = "✓" if comparison['exact_match'] else ("⚠" if is_error else "✗")
            print(f"  {status} GT: {question.get('ground_truth')} | Pred: {comparison['predicted']}")
        
        if config.debug and not comparison['exact_match']:
            print(f"  DEBUG: Raw response: {response[:200] if response else 'None'}...")
    
    return results


# =============================================================================
# Reporting
# =============================================================================

def print_results(results: TestResults):
    """Print formatted results"""
    summary = results.get_summary()
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {summary['kinship_system'].upper()}")
    print(f"Model: {summary['model']}")
    print("=" * 70)
    
    acc = summary['accuracy']
    print(f"\n📊 Overall Accuracy:")
    print(f"  Exact Match:   {acc['exact_match']*100:.1f}%")
    print(f"  Partial Match: {acc['partial_match']*100:.1f}%")
    print(f"  Precision:     {acc['precision']*100:.1f}%")
    print(f"  Recall:        {acc['recall']*100:.1f}%")
    print(f"  F1 Score:      {acc['f1']*100:.1f}%")
    
    if summary.get('error_count', 0) > 0:
        print(f"  ⚠️  Errors:     {summary['error_count']} ({summary['error_count']/summary['total_questions']*100:.1f}%)")
    
    print(f"\n📊 By Category:")
    print(f"  {'Category':<12} {'Total':>6} {'Exact%':>8} {'F1%':>8}")
    print(f"  {'-'*36}")
    for cat_name, stats in sorted(summary['by_category'].items()):
        print(f"  {cat_name:<12} {stats['total']:>6} {stats['exact_match']*100:>7.1f}% {stats['f1']*100:>7.1f}%")
    
    print(f"\n📊 By N-Hops (Complexity):")
    print(f"  {'Hops':<12} {'Total':>6} {'Exact%':>8} {'F1%':>8}")
    print(f"  {'-'*36}")
    for hop_name, stats in sorted(summary['by_hops'].items()):
        print(f"  {hop_name:<12} {stats['total']:>6} {stats['exact_match']*100:>7.1f}% {stats['f1']*100:>7.1f}%")
    
    if summary['by_cultural_override']:
        print(f"\n📊 By Cultural Override:")
        print(f"  {'Type':<16} {'Total':>6} {'Exact%':>8} {'F1%':>8}")
        print(f"  {'-'*40}")
        for override_type, stats in summary['by_cultural_override'].items():
            print(f"  {override_type:<16} {stats['total']:>6} {stats['exact_match']*100:>7.1f}% {stats['f1']*100:>7.1f}%")
    
    timing = summary['timing']
    print(f"\n⏱️  Timing:")
    print(f"  Total: {timing['total_seconds']:.1f}s")
    print(f"  Avg per question: {timing['avg_per_question']:.2f}s")


def save_results(results: TestResults, output_path: str):
    """Save results to JSON file"""
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    summary = results.get_summary()
    
    output = {
        "summary": summary,
        "questions": [asdict(r) for r in results.question_results]
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_path}")


def print_cross_system_summary(all_results: Dict[str, TestResults]):
    """Print summary across all kinship systems"""
    print("\n" + "=" * 80)
    print("CROSS-SYSTEM SUMMARY")
    print("=" * 80)
    
    print(f"\n{'System':<12} {'Total':>6} {'Exact%':>8} {'F1%':>8} {'Cat1%':>8} {'Cat2%':>8} {'Cat3%':>8} {'Cat4%':>8}")
    print("-" * 80)
    
    for system in KINSHIP_SYSTEMS:
        if system not in all_results:
            continue
        
        summary = all_results[system].get_summary()
        acc = summary['accuracy']
        cats = summary['by_category']
        
        cat1 = cats.get('cat_1', {}).get('exact_match', 0) * 100
        cat2 = cats.get('cat_2', {}).get('exact_match', 0) * 100
        cat3 = cats.get('cat_3', {}).get('exact_match', 0) * 100
        cat4 = cats.get('cat_4', {}).get('exact_match', 0) * 100
        
        print(f"{system.capitalize():<12} {summary['total_questions']:>6} {acc['exact_match']*100:>7.1f}% {acc['f1']*100:>7.1f}% {cat1:>7.1f}% {cat2:>7.1f}% {cat3:>7.1f}% {cat4:>7.1f}%")
    
    print("\n" + "-" * 80)
    
    western_exact = []
    non_western_exact = []
    
    for system, results in all_results.items():
        summary = results.get_summary()
        if system in WESTERN_SYSTEMS:
            western_exact.append(summary['accuracy']['exact_match'])
        else:
            non_western_exact.append(summary['accuracy']['exact_match'])
    
    if western_exact:
        print(f"Western Systems (Eskimo, Sudanese):     {sum(western_exact)/len(western_exact)*100:.1f}% avg exact match")
    if non_western_exact:
        print(f"Non-Western Systems:                    {sum(non_western_exact)/len(non_western_exact)*100:.1f}% avg exact match")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="KinshipQA LLM Tester v5.1",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("dataset", nargs="?", help="Path to dataset JSONL file")
    parser.add_argument("--all", action="store_true", help="Test all 7 kinship systems")
    parser.add_argument("--dataset-dir", type=str, default=".", help="Directory containing datasets")
    
    parser.add_argument("--provider", type=str, default="ollama",
                        choices=["ollama", "openai", "gemini", "anthropic"])
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--api-key", type=str, help="API key")
    
    parser.add_argument("--output", type=str, default="results.json")
    parser.add_argument("--output-dir", type=str, default="results")
    
    parser.add_argument("--limit", type=int, help="Limit questions per dataset")
    parser.add_argument("--categories", type=str, help="Test specific categories (e.g., '1,2,3,4')")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()
    
    categories = None
    if args.categories:
        categories = [int(c.strip()) for c in args.categories.split(",")]
    
    config = TestConfig(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        limit=args.limit,
        categories=categories,
        verbose=args.verbose,
        debug=args.debug
    )
    
    print(f"Initializing {config.provider} provider with model {config.model}...")
    provider = create_provider(config)
    
    if args.all:
        dataset_dir = Path(args.dataset_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = {}
        
        for system in KINSHIP_SYSTEMS:
            dataset_path = dataset_dir / f"{system}_dataset.jsonl"
            if not dataset_path.exists():
                dataset_path = dataset_dir / f"{system}.jsonl"
            if not dataset_path.exists():
                print(f"Warning: Dataset not found for {system}, skipping...")
                continue
            
            config.dataset_path = str(dataset_path)
            config.output_path = str(output_dir / f"{system}_results.json")
            
            print(f"\n{'#' * 60}")
            print(f"# Testing: {system.upper()}")
            print(f"{'#' * 60}")
            
            results = run_test(config, provider)
            all_results[system] = results
            
            print_results(results)
            save_results(results, config.output_path)
        
        print_cross_system_summary(all_results)
        
        combined_output = output_dir / "combined_results.json"
        combined = {
            system: results.get_summary() 
            for system, results in all_results.items()
        }
        with open(combined_output, 'w') as f:
            json.dump(combined, f, indent=2)
        print(f"\nCombined results saved to {combined_output}")
    
    else:
        if not args.dataset:
            parser.error("Please provide a dataset path or use --all")
        
        config.dataset_path = args.dataset
        config.output_path = args.output
        
        results = run_test(config, provider)
        print_results(results)
        save_results(results, config.output_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
