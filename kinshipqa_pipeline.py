#!/usr/bin/env python3


import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD


# =============================================================================
# NAMESPACES
# =============================================================================
FAMILY = Namespace("http://example.org/family/")
KIN = Namespace("http://example.org/kinship/")
PERSON = Namespace("http://example.org/person/")


# =============================================================================
# PATH-TO-TERM MAPPING TABLE
# =============================================================================
# This is the core contribution: mapping biological paths to kinship terms
# across different cultural systems.

@dataclass
class PathMapping:
    """Represents how a biological path maps to terms in a kinship system."""
    bio_term: str                    # Universal biological term
    kin_term: Optional[str] = None   # Cultural term (if different)
    kin_predicate: Optional[str] = None  # RDF predicate in kin: namespace
    has_override: bool = False       # Whether this path has cultural override
    
    
# Biological paths (universal) - tuple of predicate names
# Format: (predicate1, predicate2, ...) representing traversal path
BIOLOGICAL_PATHS: Dict[Tuple[str, ...], Dict] = {
    # 1-hop paths
    ("hasFather",): {"term": "father", "hops": 1},
    ("hasMother",): {"term": "mother", "hops": 1},
    ("hasSibling",): {"term": "sibling", "hops": 1},
    ("hasBrother",): {"term": "brother", "hops": 1},
    ("hasSister",): {"term": "sister", "hops": 1},
    ("hasSpouse",): {"term": "spouse", "hops": 1},
    ("hasChild",): {"term": "child", "hops": 1},
    ("hasSon",): {"term": "son", "hops": 1},
    ("hasDaughter",): {"term": "daughter", "hops": 1},
    
    # 2-hop paths - grandparents
    ("hasFather", "hasFather"): {"term": "paternal grandfather", "hops": 2},
    ("hasFather", "hasMother"): {"term": "paternal grandmother", "hops": 2},
    ("hasMother", "hasFather"): {"term": "maternal grandfather", "hops": 2},
    ("hasMother", "hasMother"): {"term": "maternal grandmother", "hops": 2},
    
    # 2-hop paths - aunts/uncles (KEY for kinship distinction)
    ("hasFather", "hasBrother"): {"term": "paternal uncle (FB)", "hops": 2},
    ("hasFather", "hasSister"): {"term": "paternal aunt (FZ)", "hops": 2},
    ("hasMother", "hasBrother"): {"term": "maternal uncle (MB)", "hops": 2},
    ("hasMother", "hasSister"): {"term": "maternal aunt (MZ)", "hops": 2},
    
    # 3-hop paths - cousins (KEY for cross/parallel distinction)
    ("hasFather", "hasBrother", "hasChild"): {"term": "father's brother's child (FBC)", "hops": 3},
    ("hasFather", "hasSister", "hasChild"): {"term": "father's sister's child (FZC)", "hops": 3},
    ("hasMother", "hasBrother", "hasChild"): {"term": "mother's brother's child (MBC)", "hops": 3},
    ("hasMother", "hasSister", "hasChild"): {"term": "mother's sister's child (MZC)", "hops": 3},
    
    # 3-hop paths - great-grandparents
    ("hasFather", "hasFather", "hasFather"): {"term": "paternal great-grandfather", "hops": 3},
    ("hasFather", "hasFather", "hasMother"): {"term": "paternal great-grandmother", "hops": 3},
    ("hasMother", "hasMother", "hasMother"): {"term": "maternal great-grandmother", "hops": 3},
    ("hasMother", "hasMother", "hasFather"): {"term": "maternal great-grandfather", "hops": 3},
    
    # 4-hop paths - second cousins, cousins' children
    ("hasFather", "hasBrother", "hasChild", "hasChild"): {"term": "paternal uncle's grandchild", "hops": 4},
    ("hasMother", "hasSister", "hasChild", "hasChild"): {"term": "maternal aunt's grandchild", "hops": 4},
    ("hasFather", "hasFather", "hasBrother", "hasChild"): {"term": "father's paternal uncle's child", "hops": 4},
    ("hasMother", "hasMother", "hasSister", "hasChild"): {"term": "mother's maternal aunt's child", "hops": 4},
}


# Kinship system overrides - where cultural classification differs from biological
KINSHIP_OVERRIDES: Dict[str, Dict[Tuple[str, ...], Dict]] = {
    
    "hawaiian": {
        # Generational lumping: all parents' siblings → classificatory parent
        ("hasFather", "hasBrother"): {
            "bio_term": "paternal uncle (FB)",
            "kin_term": "classificatory father",
            "kin_predicate": "kin:hasClassificatoryParent",
        },
        ("hasFather", "hasSister"): {
            "bio_term": "paternal aunt (FZ)",
            "kin_term": "classificatory mother",
            "kin_predicate": "kin:hasClassificatoryParent",
        },
        ("hasMother", "hasBrother"): {
            "bio_term": "maternal uncle (MB)",
            "kin_term": "classificatory father",
            "kin_predicate": "kin:hasClassificatoryParent",
        },
        ("hasMother", "hasSister"): {
            "bio_term": "maternal aunt (MZ)",
            "kin_term": "classificatory mother",
            "kin_predicate": "kin:hasClassificatoryParent",
        },
        # All cousins → classificatory siblings
        ("hasFather", "hasBrother", "hasChild"): {
            "bio_term": "father's brother's child (FBC)",
            "kin_term": "classificatory sibling",
            "kin_predicate": "kin:hasClassificatorySibling",
        },
        ("hasFather", "hasSister", "hasChild"): {
            "bio_term": "father's sister's child (FZC)",
            "kin_term": "classificatory sibling",
            "kin_predicate": "kin:hasClassificatorySibling",
        },
        ("hasMother", "hasBrother", "hasChild"): {
            "bio_term": "mother's brother's child (MBC)",
            "kin_term": "classificatory sibling",
            "kin_predicate": "kin:hasClassificatorySibling",
        },
        ("hasMother", "hasSister", "hasChild"): {
            "bio_term": "mother's sister's child (MZC)",
            "kin_term": "classificatory sibling",
            "kin_predicate": "kin:hasClassificatorySibling",
        },
    },
    
    "iroquois": {
        # Parallel relatives merged (same-sex parent's sibling)
        ("hasFather", "hasBrother"): {
            "bio_term": "paternal uncle (FB)",
            "kin_term": "classificatory father",
            "kin_predicate": "kin:hasClassificatoryParent",
        },
        ("hasMother", "hasSister"): {
            "bio_term": "maternal aunt (MZ)",
            "kin_term": "classificatory mother",
            "kin_predicate": "kin:hasClassificatoryParent",
        },
        # Cross relatives distinguished (opposite-sex parent's sibling)
        ("hasFather", "hasSister"): {
            "bio_term": "paternal aunt (FZ)",
            "kin_term": "cross-aunt",
            "kin_predicate": "kin:hasCrossAunt",
        },
        ("hasMother", "hasBrother"): {
            "bio_term": "maternal uncle (MB)",
            "kin_term": "cross-uncle",
            "kin_predicate": "kin:hasCrossUncle",
        },
        # Parallel cousins = classificatory siblings
        ("hasFather", "hasBrother", "hasChild"): {
            "bio_term": "father's brother's child (FBC)",
            "kin_term": "parallel cousin (classificatory sibling)",
            "kin_predicate": "kin:hasParallelCousin",
        },
        ("hasMother", "hasSister", "hasChild"): {
            "bio_term": "mother's sister's child (MZC)",
            "kin_term": "parallel cousin (classificatory sibling)",
            "kin_predicate": "kin:hasParallelCousin",
        },
        # Cross cousins distinguished
        ("hasFather", "hasSister", "hasChild"): {
            "bio_term": "father's sister's child (FZC)",
            "kin_term": "cross-cousin",
            "kin_predicate": "kin:hasCrossCousin",
        },
        ("hasMother", "hasBrother", "hasChild"): {
            "bio_term": "mother's brother's child (MBC)",
            "kin_term": "cross-cousin",
            "kin_predicate": "kin:hasCrossCousin",
        },
    },
    
    "dravidian": {
        # Inherits Iroquois parallel/cross distinction
        ("hasFather", "hasBrother"): {
            "bio_term": "paternal uncle (FB)",
            "kin_term": "classificatory father (Periyappa/Chithappa)",
            "kin_predicate": "kin:hasClassificatoryParent",
        },
        ("hasMother", "hasSister"): {
            "bio_term": "maternal aunt (MZ)",
            "kin_term": "classificatory mother (Chitti/Periyamma)",
            "kin_predicate": "kin:hasClassificatoryParent",
        },
        # Cross relatives with Tamil terms
        ("hasFather", "hasSister"): {
            "bio_term": "paternal aunt (FZ)",
            "kin_term": "Athai (potential mother-in-law)",
            "kin_predicate": "kin:hasAthai",
        },
        ("hasMother", "hasBrother"): {
            "bio_term": "maternal uncle (MB)",
            "kin_term": "Mama (potential father-in-law)",
            "kin_predicate": "kin:hasMama",
        },
        # Parallel cousins
        ("hasFather", "hasBrother", "hasChild"): {
            "bio_term": "father's brother's child (FBC)",
            "kin_term": "parallel cousin (classificatory sibling)",
            "kin_predicate": "kin:hasParallelCousin",
        },
        ("hasMother", "hasSister", "hasChild"): {
            "bio_term": "mother's sister's child (MZC)",
            "kin_term": "parallel cousin (classificatory sibling)",
            "kin_predicate": "kin:hasParallelCousin",
        },
        # Cross cousins - prescribed marriage partners!
        ("hasFather", "hasSister", "hasChild"): {
            "bio_term": "father's sister's child (FZC)",
            "kin_term": "cross-cousin (potential spouse)",
            "kin_predicate": "kin:hasPotentialSpouse",
        },
        ("hasMother", "hasBrother", "hasChild"): {
            "bio_term": "mother's brother's child (MBC)",
            "kin_term": "cross-cousin (potential spouse)",
            "kin_predicate": "kin:hasPotentialSpouse",
        },
    },
    
    "crow": {
        # Matrilineal - parallel relatives merged
        ("hasFather", "hasBrother"): {
            "bio_term": "paternal uncle (FB)",
            "kin_term": "classificatory father",
            "kin_predicate": "kin:hasClassificatoryParent",
        },
        ("hasMother", "hasSister"): {
            "bio_term": "maternal aunt (MZ)",
            "kin_term": "classificatory mother",
            "kin_predicate": "kin:hasClassificatoryParent",
        },
        # Generational skewing: FZ line collapsed
        ("hasFather", "hasSister"): {
            "bio_term": "paternal aunt (FZ)",
            "kin_term": "female father (skewed generation)",
            "kin_predicate": "kin:hasSkewedGeneration",
        },
        ("hasFather", "hasSister", "hasChild"): {
            "bio_term": "father's sister's child (FZC)",
            "kin_term": "father/female father (skewed - same as FZ)",
            "kin_predicate": "kin:hasSkewedGeneration",
        },
        # Cross relations
        ("hasMother", "hasBrother"): {
            "bio_term": "maternal uncle (MB)",
            "kin_term": "cross-uncle",
            "kin_predicate": "kin:hasCrossUncle",
        },
        ("hasMother", "hasBrother", "hasChild"): {
            "bio_term": "mother's brother's child (MBC)",
            "kin_term": "cross-cousin",
            "kin_predicate": "kin:hasCrossCousin",
        },
    },
    
    "omaha": {
        # Patrilineal - parallel relatives merged  
        ("hasFather", "hasBrother"): {
            "bio_term": "paternal uncle (FB)",
            "kin_term": "classificatory father",
            "kin_predicate": "kin:hasClassificatoryParent",
        },
        ("hasMother", "hasSister"): {
            "bio_term": "maternal aunt (MZ)",
            "kin_term": "classificatory mother",
            "kin_predicate": "kin:hasClassificatoryParent",
        },
        # Generational skewing: MB line collapsed
        ("hasMother", "hasBrother"): {
            "bio_term": "maternal uncle (MB)",
            "kin_term": "male mother (skewed generation)",
            "kin_predicate": "kin:hasSkewedGeneration",
        },
        ("hasMother", "hasBrother", "hasChild"): {
            "bio_term": "mother's brother's child (MBC)",
            "kin_term": "mother/male mother (skewed - same as MB)",
            "kin_predicate": "kin:hasSkewedGeneration",
        },
        # Cross relations
        ("hasFather", "hasSister"): {
            "bio_term": "paternal aunt (FZ)",
            "kin_term": "cross-aunt",
            "kin_predicate": "kin:hasCrossAunt",
        },
        ("hasFather", "hasSister", "hasChild"): {
            "bio_term": "father's sister's child (FZC)",
            "kin_term": "cross-cousin",
            "kin_predicate": "kin:hasCrossCousin",
        },
    },
    
    # Eskimo and Sudanese have NO overrides - they use biological terms
    "eskimo": {},
    "sudanese": {},
}


def resolve_path_to_terms(path: Tuple[str, ...], kinship_system: str) -> Optional[Dict]:
    """
    Given a biological path and kinship system, return both terms.
    
    Returns:
        {
            "bio_term": str,           # Universal biological name
            "kin_term": str,           # Kinship-system-specific name
            "has_override": bool,      # Whether cultural term differs
            "kin_predicate": str|None, # RDF predicate if override exists
            "hops": int,               # Number of hops in path
        }
    """
    # Get biological term
    bio_info = BIOLOGICAL_PATHS.get(path)
    if bio_info is None:
        return None  # Unknown path
    
    bio_term = bio_info["term"]
    hops = bio_info["hops"]
    
    # Check for kinship system override
    overrides = KINSHIP_OVERRIDES.get(kinship_system, {})
    
    if path in overrides:
        override = overrides[path]
        return {
            "bio_term": bio_term,
            "kin_term": override["kin_term"],
            "has_override": True,
            "kin_predicate": override.get("kin_predicate"),
            "hops": hops,
        }
    else:
        return {
            "bio_term": bio_term,
            "kin_term": bio_term,  # Same as biological
            "has_override": False,
            "kin_predicate": None,
            "hops": hops,
        }


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Question:
    """Represents a generated question with all metadata."""
    question_id: str
    question_text: str
    category: int
    n_hops: int
    kinship_system: str
    anchor_person: str
    anchor_name: str
    target_persons: List[str]
    target_names: List[str]
    ground_truth: Any  # Can be list, int, or str depending on question type
    context: str
    path: Optional[Tuple[str, ...]] = None
    bio_term: Optional[str] = None
    kin_term: Optional[str] = None
    has_cultural_override: bool = False
    proof_graph_size: int = 0
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        if d['path']:
            d['path'] = list(d['path'])  # Convert tuple to list for JSON
        return d


@dataclass 
class GenerationStats:
    """Track generation statistics."""
    total_generated: int = 0
    by_category: Dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0, 3: 0, 4: 0})
    by_hops: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    by_path: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    rejected: int = 0


# =============================================================================
# GRAPH UTILITIES
# =============================================================================

class KinshipGraph:
    """Wrapper around RDFLib graph with kinship-specific utilities."""
    
    def __init__(self, ttl_path: str, kinship_system: str):
        self.graph = Graph()
        self.graph.parse(ttl_path, format="turtle")
        self.kinship_system = kinship_system
        
        # Bind namespaces
        self.graph.bind("family", FAMILY)
        self.graph.bind("kin", KIN)
        self.graph.bind("person", PERSON)
        
        # Cache persons
        self._persons: List[URIRef] = []
        self._person_names: Dict[URIRef, str] = {}
        self._person_genders: Dict[URIRef, str] = {}
        self._load_persons()
        
    def _load_persons(self):
        """Load all persons and their attributes."""
        query = """
        SELECT ?person ?name ?gender WHERE {
            ?person a family:Person .
            ?person rdfs:label ?name .
            OPTIONAL {
                { ?person a family:Male . BIND("male" AS ?gender) }
                UNION
                { ?person a family:Female . BIND("female" AS ?gender) }
            }
        }
        """
        for row in self.graph.query(query):
            person_uri = row.person
            self._persons.append(person_uri)
            self._person_names[person_uri] = str(row.name)
            self._person_genders[person_uri] = str(row.gender) if row.gender else "unknown"
    
    @property
    def persons(self) -> List[URIRef]:
        return self._persons
    
    def get_name(self, person: URIRef) -> str:
        return self._person_names.get(person, str(person))
    
    def get_gender(self, person: URIRef) -> str:
        return self._person_genders.get(person, "unknown")
    
    def get_person_id(self, person: URIRef) -> str:
        """Extract person ID from URI."""
        return str(person).split("/")[-1]
    
    def traverse_path(self, start: URIRef, path: Tuple[str, ...]) -> Set[URIRef]:
        """
        Traverse a path from start node, returning all reachable endpoints.
        
        Args:
            start: Starting person URI
            path: Tuple of predicate names to follow
            
        Returns:
            Set of person URIs reachable via this path
        """
        current_set = {start}
        
        for predicate_name in path:
            next_set = set()
            predicate = FAMILY[predicate_name]
            
            for node in current_set:
                for _, _, obj in self.graph.triples((node, predicate, None)):
                    if isinstance(obj, URIRef) and obj in self._persons:
                        next_set.add(obj)
            
            current_set = next_set
            if not current_set:
                break
                
        return current_set
    
    def find_persons_at_distance(self, start: URIRef, n_hops: int, 
                                  predicates: List[str] = None) -> Dict[URIRef, List[Tuple[str, ...]]]:
        """
        Find all persons exactly n hops away from start.
        
        Returns:
            Dict mapping person URI to list of paths that reach them
        """
        if predicates is None:
            predicates = ["hasFather", "hasMother", "hasSibling", "hasBrother", 
                         "hasSister", "hasChild", "hasSon", "hasDaughter"]
        
        # BFS with path tracking
        results: Dict[URIRef, List[Tuple[str, ...]]] = defaultdict(list)
        
        # For efficiency, enumerate known paths of this length
        target_paths = [p for p, info in BIOLOGICAL_PATHS.items() 
                        if info["hops"] == n_hops]
        
        for path in target_paths:
            endpoints = self.traverse_path(start, path)
            for endpoint in endpoints:
                if endpoint != start:  # Exclude self
                    results[endpoint].append(path)
        
        return dict(results)
    
    def get_relationship_context(self, anchor: URIRef, targets: Set[URIRef], 
                                  path: Tuple[str, ...]) -> Tuple[str, int]:
        """
        Generate natural language context for reasoning.
        Returns (context_string, proof_graph_size).
        
        Uses PRIMITIVE relationships only - LLM must reason through the chain.
        """
        # Collect all persons involved in the reasoning path
        involved_persons = {anchor}
        context_triples = []
        
        # Trace the path and collect intermediate nodes
        current_set = {anchor}
        for i, pred_name in enumerate(path):
            next_set = set()
            predicate = FAMILY[pred_name]
            
            for node in current_set:
                for _, _, obj in self.graph.triples((node, predicate, None)):
                    if isinstance(obj, URIRef) and obj in self._persons:
                        next_set.add(obj)
                        involved_persons.add(obj)
                        # Record this triple
                        context_triples.append((node, pred_name, obj))
            
            current_set = next_set
        
        # Add targets
        involved_persons.update(targets)
        
        # Generate natural language context
        sentences = []
        seen_facts = set()
        
        for subj, pred, obj in context_triples:
            fact_key = (subj, pred, obj)
            if fact_key in seen_facts:
                continue
            seen_facts.add(fact_key)
            
            subj_name = self.get_name(subj)
            obj_name = self.get_name(obj)
            
            # Convert predicate to natural language
            pred_nl = self._predicate_to_nl(pred)
            sentence = f"{subj_name}'s {pred_nl} is {obj_name}."
            sentences.append(sentence)
        
        context = " ".join(sentences)
        proof_graph_size = len(involved_persons) + len(context_triples)
        
        return context, proof_graph_size
    
    def _predicate_to_nl(self, pred_name: str) -> str:
        """Convert predicate name to natural language."""
        mapping = {
            "hasFather": "father",
            "hasMother": "mother",
            "hasParent": "parent",
            "hasSibling": "sibling",
            "hasBrother": "brother",
            "hasSister": "sister",
            "hasChild": "child",
            "hasSon": "son",
            "hasDaughter": "daughter",
            "hasSpouse": "spouse",
            "hasHusband": "husband",
            "hasWife": "wife",
            "hasGrandfather": "grandfather",
            "hasGrandmother": "grandmother",
            "hasGrandparent": "grandparent",
            "hasUncle": "uncle",
            "hasAunt": "aunt",
            "hasCousin": "cousin",
            "hasNephew": "nephew",
            "hasNiece": "niece",
        }
        return mapping.get(pred_name, pred_name.replace("has", "").lower())
    
    def execute_sparql(self, query: str) -> List[Any]:
        """Execute a SPARQL query and return results."""
        try:
            results = list(self.graph.query(query))
            return results
        except Exception as e:
            print(f"SPARQL error: {e}")
            return []


# =============================================================================
# QUESTION GENERATORS
# =============================================================================

class QuestionGenerator:
    """Main question generator using path-based approach."""
    
    def __init__(self, kg: KinshipGraph, kinship_system: str, seed: int = 42):
        self.kg = kg
        self.kinship_system = kinship_system
        self.rng = random.Random(seed)
        self.stats = GenerationStats()
        self.question_counter = 0
        
    def _next_id(self) -> str:
        self.question_counter += 1
        return f"{self.kinship_system}_{self.question_counter:04d}"
    
    # -------------------------------------------------------------------------
    # Category 1: Simple Fact Retrieval (1-hop, template-based)
    # -------------------------------------------------------------------------
    
    def generate_cat1_questions(self, count: int = 50) -> List[Question]:
        """Generate Category 1 questions - simple 1-hop fact retrieval."""
        templates = [
            {"pred": "hasFather", "q": "Who is {name}'s father?", "type": "person"},
            {"pred": "hasMother", "q": "Who is {name}'s mother?", "type": "person"},
            {"pred": "hasSpouse", "q": "Who is {name}'s spouse?", "type": "person"},
            {"pred": "hasSibling", "q": "Who are {name}'s siblings?", "type": "list"},
            {"pred": "hasChild", "q": "Who are {name}'s children?", "type": "list"},
        ]
        
        questions = []
        seen_questions = set()  # Track question texts to avoid duplicates
        attempts = 0
        max_attempts = count * 10
        
        while len(questions) < count and attempts < max_attempts:
            attempts += 1
            
            # Random person and template
            anchor = self.rng.choice(self.kg.persons)
            template = self.rng.choice(templates)
            
            # Check if relationship exists
            predicate = FAMILY[template["pred"]]
            targets = list(self.kg.graph.objects(anchor, predicate))
            targets = [t for t in targets if isinstance(t, URIRef) and t in self.kg.persons]
            
            if not targets:
                continue
            
            anchor_name = self.kg.get_name(anchor)
            target_names = sorted([self.kg.get_name(t) for t in targets])  # Sort for consistency
            
            # Generate question text
            q_text = template["q"].format(name=anchor_name)
            
            # Check for duplicate question text
            if q_text in seen_questions:
                continue
            seen_questions.add(q_text)
            
            # Simple context - just the direct fact
            if template["type"] == "person":
                context = f"{anchor_name}'s {self._predicate_to_nl(template['pred'])} is {target_names[0]}."
                ground_truth = target_names[0]
            else:
                context = f"{anchor_name} has the following {self._predicate_to_nl(template['pred'])}s: {', '.join(target_names)}."
                ground_truth = target_names
            
            q = Question(
                question_id=self._next_id(),
                question_text=q_text,
                category=1,
                n_hops=1,
                kinship_system=self.kinship_system,
                anchor_person=self.kg.get_person_id(anchor),
                anchor_name=anchor_name,
                target_persons=[self.kg.get_person_id(t) for t in targets],
                target_names=target_names,
                ground_truth=ground_truth,
                context=context,
                path=(template["pred"],),
                bio_term=self._predicate_to_nl(template["pred"]),
                proof_graph_size=2,
            )
            
            questions.append(q)
            self.stats.by_category[1] += 1
            self.stats.by_hops[1] += 1
            
        self.stats.total_generated += len(questions)
        return questions
    
    def _predicate_to_nl(self, pred_name: str) -> str:
        return self.kg._predicate_to_nl(pred_name)
    
    # -------------------------------------------------------------------------
    # Category 2: Multi-hop Reasoning (path-based, biological terms)
    # -------------------------------------------------------------------------
    
    def generate_cat2_questions(self, count: int = 150, 
                                 hops_distribution: Dict[int, int] = None) -> List[Question]:
        """Generate Category 2 questions - multi-hop with biological terms."""
        if hops_distribution is None:
            hops_distribution = {2: 50, 3: 50, 4: 50}
        
        questions = []
        
        for n_hops, target_count in hops_distribution.items():
            hop_questions = self._generate_cat2_for_hops(n_hops, target_count)
            questions.extend(hop_questions)
        
        return questions
    
    def _generate_cat2_for_hops(self, n_hops: int, count: int) -> List[Question]:
        """Generate Cat2 questions for specific hop count."""
        # Get all paths of this length
        valid_paths = [p for p, info in BIOLOGICAL_PATHS.items() 
                       if info["hops"] == n_hops]
        
        if not valid_paths:
            return []
        
        questions = []
        seen_questions = set()  # Track question texts to avoid duplicates
        attempts = 0
        max_attempts = count * 20
        
        while len(questions) < count and attempts < max_attempts:
            attempts += 1
            
            # Random anchor and path
            anchor = self.rng.choice(self.kg.persons)
            path = self.rng.choice(valid_paths)
            
            # Traverse path - get ALL targets
            targets = self.kg.traverse_path(anchor, path)
            targets = {t for t in targets if t != anchor}
            
            if not targets:
                self.stats.rejected += 1
                continue
            
            # Get terms
            terms = resolve_path_to_terms(path, self.kinship_system)
            if not terms:
                continue
            
            anchor_name = self.kg.get_name(anchor)
            target_names = sorted([self.kg.get_name(t) for t in targets])  # Sort for consistency
            
            # Generate context (primitive relationships only) - include ALL targets
            context, pg_size = self.kg.get_relationship_context(anchor, targets, path)
            
            # Question using biological term - ALWAYS use plural-friendly phrasing
            bio_term = terms["bio_term"]
            # Use "Who is/are" based on actual count, but always include ALL answers
            if len(targets) == 1:
                q_text = f"Who is {anchor_name}'s {bio_term}?"
            else:
                # For terms already containing descriptive text, adjust phrasing
                if bio_term.endswith(')'):
                    # e.g., "paternal uncle (FB)" -> "paternal uncles (FB)"
                    base = bio_term.rsplit(' (', 1)[0]
                    suffix = ' (' + bio_term.rsplit(' (', 1)[1] if ' (' in bio_term else ''
                    q_text = f"Who are {anchor_name}'s {base}s{suffix}?"
                else:
                    q_text = f"Who are {anchor_name}'s {bio_term}s?"
            
            # Check for duplicate question text
            if q_text in seen_questions:
                self.stats.rejected += 1
                continue
            seen_questions.add(q_text)
            
            # Ground truth is ALWAYS a list of ALL valid answers
            ground_truth = target_names if len(targets) > 1 else target_names[0]
            
            q = Question(
                question_id=self._next_id(),
                question_text=q_text,
                category=2,
                n_hops=n_hops,
                kinship_system=self.kinship_system,
                anchor_person=self.kg.get_person_id(anchor),
                anchor_name=anchor_name,
                target_persons=[self.kg.get_person_id(t) for t in targets],
                target_names=target_names,
                ground_truth=ground_truth,
                context=context,
                path=path,
                bio_term=bio_term,
                kin_term=terms["kin_term"],
                has_cultural_override=terms["has_override"],
                proof_graph_size=pg_size,
            )
            
            questions.append(q)
            self.stats.by_category[2] += 1
            self.stats.by_hops[n_hops] += 1
            self.stats.by_path[str(path)] += 1
        
        self.stats.total_generated += len(questions)
        return questions
    
    # -------------------------------------------------------------------------
    # Category 3: Comparative & Constraints (template-based)
    # -------------------------------------------------------------------------
    
    def generate_cat3_questions(self, count: int = 100) -> List[Question]:
        """Generate Category 3 questions - counting, filtering, comparison."""
        templates = [
            {"type": "count_children", "q": "How many children does {name} have?"},
            {"type": "count_siblings", "q": "How many siblings does {name} have?"},
            {"type": "count_grandchildren", "q": "How many grandchildren does {name} have?"},
            {"type": "filter_male_siblings", "q": "Who are {name}'s brothers?"},
            {"type": "filter_female_siblings", "q": "Who are {name}'s sisters?"},
            {"type": "filter_sons", "q": "Who are {name}'s sons?"},
            {"type": "filter_daughters", "q": "Who are {name}'s daughters?"},
        ]
        
        questions = []
        seen_questions = set()  # Track question texts to avoid duplicates
        attempts = 0
        max_attempts = count * 15
        
        while len(questions) < count and attempts < max_attempts:
            attempts += 1
            
            anchor = self.rng.choice(self.kg.persons)
            template = self.rng.choice(templates)
            anchor_name = self.kg.get_name(anchor)
            
            # Generate question text first to check for duplicates
            q_text = template["q"].format(name=anchor_name)
            if q_text in seen_questions:
                continue
            
            q_data = self._generate_cat3_question(anchor, anchor_name, template)
            
            if q_data:
                seen_questions.add(q_text)
                questions.append(q_data)
                self.stats.by_category[3] += 1
                self.stats.by_hops[q_data.n_hops] += 1
        
        self.stats.total_generated += len(questions)
        return questions
    
    def _generate_cat3_question(self, anchor: URIRef, anchor_name: str, 
                                 template: Dict) -> Optional[Question]:
        """Generate a single Cat3 question."""
        t_type = template["type"]
        
        if t_type == "count_children":
            children = list(self.kg.graph.objects(anchor, FAMILY.hasChild))
            children = [c for c in children if isinstance(c, URIRef)]
            if not children:
                return None
            
            child_names = sorted([self.kg.get_name(c) for c in children])
            context = f"{anchor_name} has children: {', '.join(child_names)}."
            ground_truth = len(children)
            n_hops = 1
            target_names = child_names
            
        elif t_type == "count_siblings":
            siblings = list(self.kg.graph.objects(anchor, FAMILY.hasSibling))
            siblings = [s for s in siblings if isinstance(s, URIRef)]
            if not siblings:
                return None
            
            sib_names = sorted([self.kg.get_name(s) for s in siblings])
            context = f"{anchor_name} has siblings: {', '.join(sib_names)}."
            ground_truth = len(siblings)
            n_hops = 1
            target_names = sib_names
            
        elif t_type == "count_grandchildren":
            # 2-hop query
            children = list(self.kg.graph.objects(anchor, FAMILY.hasChild))
            grandchildren = []
            for child in children:
                gc = list(self.kg.graph.objects(child, FAMILY.hasChild))
                grandchildren.extend([g for g in gc if isinstance(g, URIRef)])
            
            if not grandchildren:
                return None
            
            child_names = sorted([self.kg.get_name(c) for c in children if isinstance(c, URIRef)])
            gc_names = sorted([self.kg.get_name(g) for g in grandchildren])
            context = f"{anchor_name}'s children are: {', '.join(child_names)}. "
            context += f"Their children include: {', '.join(gc_names)}."
            ground_truth = len(grandchildren)
            n_hops = 2
            target_names = gc_names
            
        elif t_type.startswith("filter_"):
            # Gender filtering
            if "siblings" in t_type or "brother" in t_type or "sister" in t_type:
                pred = FAMILY.hasSibling
                n_hops = 1
            else:
                pred = FAMILY.hasChild
                n_hops = 1
            
            relatives = list(self.kg.graph.objects(anchor, pred))
            relatives = [r for r in relatives if isinstance(r, URIRef)]
            
            if "male" in t_type or "brother" in t_type or "sons" in t_type:
                filtered = [r for r in relatives if self.kg.get_gender(r) == "male"]
            else:
                filtered = [r for r in relatives if self.kg.get_gender(r) == "female"]
            
            if not filtered:
                return None
            
            all_names = sorted([self.kg.get_name(r) for r in relatives])
            filtered_names = sorted([self.kg.get_name(r) for r in filtered])
            
            rel_type = "siblings" if "sibling" in t_type or "brother" in t_type or "sister" in t_type else "children"
            context = f"{anchor_name}'s {rel_type} are: {', '.join(all_names)}."
            ground_truth = filtered_names
            target_names = filtered_names
            
        else:
            return None
        
        return Question(
            question_id=self._next_id(),
            question_text=template["q"].format(name=anchor_name),
            category=3,
            n_hops=n_hops,
            kinship_system=self.kinship_system,
            anchor_person=self.kg.get_person_id(anchor),
            anchor_name=anchor_name,
            target_persons=[],  # Varies by question type
            target_names=target_names,
            ground_truth=ground_truth,
            context=context,
            proof_graph_size=len(target_names) + 2,
        )
    
    # -------------------------------------------------------------------------
    # Category 4: Cultural Disambiguation (path-based, requires override)
    # -------------------------------------------------------------------------
    
    def generate_cat4_questions(self, count: int = 200,
                                 hops_distribution: Dict[int, int] = None) -> List[Question]:
        """Generate Category 4 questions - cultural reasoning with overrides."""
        if hops_distribution is None:
            hops_distribution = {2: 70, 3: 70, 4: 60}
        
        # Find paths that have cultural overrides for this system
        override_paths = list(KINSHIP_OVERRIDES.get(self.kinship_system, {}).keys())
        
        if not override_paths:
            print(f"Warning: No cultural overrides for {self.kinship_system}. "
                  f"Cat4 will use disambiguation questions only.")
            return self._generate_cat4_disambiguation_only(count, hops_distribution)
        
        questions = []
        
        for n_hops, target_count in hops_distribution.items():
            # Filter paths by hop count
            hop_paths = [p for p in override_paths 
                         if BIOLOGICAL_PATHS.get(p, {}).get("hops") == n_hops]
            
            if not hop_paths:
                continue
            
            hop_questions = self._generate_cat4_for_hops(hop_paths, n_hops, target_count)
            questions.extend(hop_questions)
        
        return questions
    
    def _generate_cat4_for_hops(self, paths: List[Tuple[str, ...]], 
                                 n_hops: int, count: int) -> List[Question]:
        """Generate Cat4 questions for specific paths and hop count."""
        questions = []
        seen_questions = set()  # Track question texts to avoid duplicates
        attempts = 0
        max_attempts = count * 30
        
        while len(questions) < count and attempts < max_attempts:
            attempts += 1
            
            anchor = self.rng.choice(self.kg.persons)
            path = self.rng.choice(paths)
            
            # Traverse path - get ALL targets
            targets = self.kg.traverse_path(anchor, path)
            targets = {t for t in targets if t != anchor}
            
            if not targets:
                self.stats.rejected += 1
                continue
            
            # Get terms (should have override)
            terms = resolve_path_to_terms(path, self.kinship_system)
            if not terms or not terms["has_override"]:
                continue
            
            anchor_name = self.kg.get_name(anchor)
            target_names = sorted([self.kg.get_name(t) for t in targets])  # Sort for consistency
            
            # Generate context - include ALL targets
            context, pg_size = self.kg.get_relationship_context(anchor, targets, path)
            
            # Randomly choose question type
            q_type = self.rng.choice(["cultural_term", "disambiguation", "comparison"])
            
            kin_term = terms["kin_term"]
            bio_term = terms["bio_term"]
            
            if q_type == "cultural_term":
                # Ask using cultural term - use plural if multiple targets
                if len(targets) == 1:
                    q_text = (f"In the {self.kinship_system.capitalize()} kinship system, "
                             f"who is {anchor_name}'s {kin_term}?")
                else:
                    q_text = (f"In the {self.kinship_system.capitalize()} kinship system, "
                             f"who are {anchor_name}'s {kin_term}s?")
                ground_truth = target_names if len(targets) > 1 else target_names[0]
                
            elif q_type == "disambiguation":
                # Ask to identify biological relationship from cultural term
                # Use ALL target names in the question to avoid ambiguity
                if len(targets) == 1:
                    target_name = target_names[0]
                    q_text = (f"In the {self.kinship_system.capitalize()} system, "
                             f"{target_name} is called {anchor_name}'s '{kin_term}'. "
                             f"What is their actual biological relationship?")
                else:
                    # For multiple targets, list them all
                    targets_str = ', '.join(target_names)
                    q_text = (f"In the {self.kinship_system.capitalize()} system, "
                             f"{targets_str} are called {anchor_name}'s '{kin_term}s'. "
                             f"What is their actual biological relationship?")
                ground_truth = bio_term
                
            else:  # comparison
                # Ask to list cultural classification - always gets all targets
                q_text = (f"List all of {anchor_name}'s {kin_term}s according to the "
                         f"{self.kinship_system.capitalize()} system.")
                ground_truth = target_names if len(targets) > 1 else target_names[0]
            
            # Check for duplicate question text
            if q_text in seen_questions:
                self.stats.rejected += 1
                continue
            seen_questions.add(q_text)
            
            q = Question(
                question_id=self._next_id(),
                question_text=q_text,
                category=4,
                n_hops=n_hops,
                kinship_system=self.kinship_system,
                anchor_person=self.kg.get_person_id(anchor),
                anchor_name=anchor_name,
                target_persons=[self.kg.get_person_id(t) for t in targets],
                target_names=target_names,
                ground_truth=ground_truth,
                context=context,
                path=path,
                bio_term=terms["bio_term"],
                kin_term=terms["kin_term"],
                has_cultural_override=True,
                proof_graph_size=pg_size,
            )
            
            questions.append(q)
            self.stats.by_category[4] += 1
            self.stats.by_hops[n_hops] += 1
            self.stats.by_path[str(path)] += 1
        
        self.stats.total_generated += len(questions)
        return questions
    
    def _generate_cat4_disambiguation_only(self, count: int, 
                                           hops_distribution: Dict[int, int]) -> List[Question]:
        """Fallback for systems without cultural overrides (Eskimo, Sudanese)."""
        # For Eskimo/Sudanese, generate questions about the LACK of distinction
        questions = []
        seen_questions = set()  # Track question texts to avoid duplicates
        
        # Use standard biological paths and ask comparative questions
        for n_hops, target_count in hops_distribution.items():
            valid_paths = [p for p, info in BIOLOGICAL_PATHS.items() 
                           if info["hops"] == n_hops]
            
            attempts = 0
            generated = 0
            
            while generated < target_count and attempts < target_count * 20:
                attempts += 1
                
                anchor = self.rng.choice(self.kg.persons)
                path = self.rng.choice(valid_paths)
                
                targets = self.kg.traverse_path(anchor, path)
                targets = {t for t in targets if t != anchor}
                
                if not targets:
                    continue
                
                terms = resolve_path_to_terms(path, self.kinship_system)
                if not terms:
                    continue
                
                anchor_name = self.kg.get_name(anchor)
                target_names = sorted([self.kg.get_name(t) for t in targets])
                context, pg_size = self.kg.get_relationship_context(anchor, targets, path)
                
                # For non-override systems, ask about the specific biological term
                bio_term = terms["bio_term"]
                if len(targets) == 1:
                    q_text = (f"In the {self.kinship_system.capitalize()} (descriptive) system, "
                             f"who is specifically {anchor_name}'s {bio_term}?")
                else:
                    # Handle plural for terms with parenthetical info
                    if bio_term.endswith(')'):
                        base = bio_term.rsplit(' (', 1)[0]
                        suffix = ' (' + bio_term.rsplit(' (', 1)[1] if ' (' in bio_term else ''
                        q_text = (f"In the {self.kinship_system.capitalize()} (descriptive) system, "
                                 f"who are specifically {anchor_name}'s {base}s{suffix}?")
                    else:
                        q_text = (f"In the {self.kinship_system.capitalize()} (descriptive) system, "
                                 f"who are specifically {anchor_name}'s {bio_term}s?")
                
                # Check for duplicate question text
                if q_text in seen_questions:
                    continue
                seen_questions.add(q_text)
                
                q = Question(
                    question_id=self._next_id(),
                    question_text=q_text,
                    category=4,
                    n_hops=n_hops,
                    kinship_system=self.kinship_system,
                    anchor_person=self.kg.get_person_id(anchor),
                    anchor_name=anchor_name,
                    target_persons=[self.kg.get_person_id(t) for t in targets],
                    target_names=target_names,
                    ground_truth=target_names if len(targets) > 1 else target_names[0],
                    context=context,
                    path=path,
                    bio_term=bio_term,
                    has_cultural_override=False,
                    proof_graph_size=pg_size,
                )
                
                questions.append(q)
                self.stats.by_category[4] += 1
                self.stats.by_hops[n_hops] += 1
                generated += 1
        
        self.stats.total_generated += len(questions)
        return questions


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class KinshipQAPipeline:
    """Main pipeline for generating KinshipQA datasets."""
    
    DEFAULT_CONFIG = {
        "cat1": {"count": 50},  # Sanity check only
        "cat2": {
            "count": 150,
            "hops_distribution": {2: 50, 3: 50, 4: 50}
        },
        "cat3": {"count": 100},
        "cat4": {
            "count": 200,
            "hops_distribution": {2: 70, 3: 70, 4: 60}
        }
    }
    
    def __init__(self, ttl_path: str, kinship_system: str, 
                 config: Dict = None, seed: int = 42):
        self.ttl_path = ttl_path
        self.kinship_system = kinship_system
        self.config = config or self.DEFAULT_CONFIG
        self.seed = seed
        
        print(f"Loading {kinship_system} ontology from {ttl_path}...")
        self.kg = KinshipGraph(ttl_path, kinship_system)
        print(f"  Loaded {len(self.kg.persons)} persons")
        
        self.generator = QuestionGenerator(self.kg, kinship_system, seed)
    
    def generate_dataset(self) -> List[Question]:
        """Generate complete dataset."""
        all_questions = []
        
        # Category 1
        print(f"\nGenerating Category 1 (1-hop facts)...")
        cat1 = self.generator.generate_cat1_questions(self.config["cat1"]["count"])
        print(f"  Generated {len(cat1)} Cat1 questions")
        all_questions.extend(cat1)
        
        # Category 2
        print(f"\nGenerating Category 2 (multi-hop biological)...")
        cat2 = self.generator.generate_cat2_questions(
            self.config["cat2"]["count"],
            self.config["cat2"]["hops_distribution"]
        )
        print(f"  Generated {len(cat2)} Cat2 questions")
        all_questions.extend(cat2)
        
        # Category 3
        print(f"\nGenerating Category 3 (constraints)...")
        cat3 = self.generator.generate_cat3_questions(self.config["cat3"]["count"])
        print(f"  Generated {len(cat3)} Cat3 questions")
        all_questions.extend(cat3)
        
        # Category 4
        print(f"\nGenerating Category 4 (cultural disambiguation)...")
        cat4 = self.generator.generate_cat4_questions(
            self.config["cat4"]["count"],
            self.config["cat4"]["hops_distribution"]
        )
        print(f"  Generated {len(cat4)} Cat4 questions")
        all_questions.extend(cat4)
        
        # Print statistics
        self._print_stats()
        
        return all_questions
    
    def _print_stats(self):
        """Print generation statistics."""
        stats = self.generator.stats
        
        print(f"\n{'='*60}")
        print(f"GENERATION STATISTICS: {self.kinship_system.upper()}")
        print(f"{'='*60}")
        print(f"Total questions: {stats.total_generated}")
        print(f"Rejected attempts: {stats.rejected}")
        
        print(f"\nBy Category:")
        for cat, count in sorted(stats.by_category.items()):
            print(f"  Cat {cat}: {count}")
        
        print(f"\nBy N-Hops:")
        for hops, count in sorted(stats.by_hops.items()):
            print(f"  {hops}-hop: {count}")
        
        print(f"\nTop Paths Used:")
        sorted_paths = sorted(stats.by_path.items(), key=lambda x: -x[1])[:10]
        for path, count in sorted_paths:
            print(f"  {path}: {count}")
    
    def save_dataset(self, output_path: str, questions: List[Question]):
        """Save dataset to JSONL file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for q in questions:
                f.write(json.dumps(q.to_dict()) + '\n')
        
        print(f"\nSaved {len(questions)} questions to {output_path}")
        
        # Also save summary
        summary_path = output_path.with_suffix('.summary.json')
        summary = {
            "kinship_system": self.kinship_system,
            "total_questions": len(questions),
            "by_category": dict(self.generator.stats.by_category),
            "by_hops": dict(self.generator.stats.by_hops),
            "config": self.config,
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Saved summary to {summary_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="KinshipQA Dataset Generator v6.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate single system
  python kinshipqa_pipeline_v6.py --ttl hawaiian.ttl --system hawaiian --output hawaiian_dataset.jsonl
  
  # Generate all systems
  python kinshipqa_pipeline_v6.py --all --ttl-dir ./ontologies --output-dir ./datasets
  
  # Custom configuration
  python kinshipqa_pipeline_v6.py --ttl iroquois.ttl --system iroquois --cat2-count 200 --cat4-count 300
        """
    )
    
    # Input options
    parser.add_argument("--ttl", type=str, help="Path to TTL file")
    parser.add_argument("--system", type=str, 
                        choices=["eskimo", "hawaiian", "iroquois", "dravidian", 
                                "crow", "omaha", "sudanese"],
                        help="Kinship system name")
    parser.add_argument("--all", action="store_true", 
                        help="Generate for all kinship systems")
    parser.add_argument("--ttl-dir", type=str, default=".",
                        help="Directory containing TTL files (for --all)")
    
    # Output options
    parser.add_argument("--output", type=str, help="Output JSONL file path")
    parser.add_argument("--output-dir", type=str, default="./datasets",
                        help="Output directory (for --all)")
    
    # Generation options
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cat1-count", type=int, default=50)
    parser.add_argument("--cat2-count", type=int, default=150)
    parser.add_argument("--cat3-count", type=int, default=100)
    parser.add_argument("--cat4-count", type=int, default=200)
    
    args = parser.parse_args()
    
    # Build config
    config = {
        "cat1": {"count": args.cat1_count},
        "cat2": {
            "count": args.cat2_count,
            "hops_distribution": {2: args.cat2_count//3, 3: args.cat2_count//3, 
                                  4: args.cat2_count - 2*(args.cat2_count//3)}
        },
        "cat3": {"count": args.cat3_count},
        "cat4": {
            "count": args.cat4_count,
            "hops_distribution": {2: args.cat4_count//3, 3: args.cat4_count//3,
                                  4: args.cat4_count - 2*(args.cat4_count//3)}
        }
    }
    
    if args.all:
        # Generate for all systems
        systems = ["eskimo", "hawaiian", "iroquois", "dravidian", 
                   "crow", "omaha", "sudanese"]
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for system in systems:
            ttl_path = Path(args.ttl_dir) / f"{system}.ttl"
            if not ttl_path.exists():
                print(f"Warning: {ttl_path} not found, skipping {system}")
                continue
            
            output_path = output_dir / f"{system}_dataset.jsonl"
            
            print(f"\n{'#'*60}")
            print(f"# GENERATING: {system.upper()}")
            print(f"{'#'*60}")
            
            pipeline = KinshipQAPipeline(str(ttl_path), system, config, args.seed)
            questions = pipeline.generate_dataset()
            pipeline.save_dataset(str(output_path), questions)
    
    else:
        # Generate single system
        if not args.ttl or not args.system:
            parser.error("--ttl and --system required unless using --all")
        
        output_path = args.output or f"{args.system}_dataset.jsonl"
        
        pipeline = KinshipQAPipeline(args.ttl, args.system, config, args.seed)
        questions = pipeline.generate_dataset()
        pipeline.save_dataset(output_path, questions)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
