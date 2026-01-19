# -*- coding: utf-8 -*-
"""
Kinship Tree Generator with Ontology Export v2.0
=================================================

Combined module for generating family trees and exporting to RDF/OWL ontologies.
Supports all 7 kinship systems with their specific cultural properties.

Kinship Systems Supported:
- Eskimo (lineal) - Western bilateral system
- Hawaiian (generational) - All same-generation relatives merged
- Iroquois (bifurcate merging) - Cross/parallel distinction
- Crow (matrilineal) - Matrilineal with generational skewing
- Omaha (patrilineal) - Patrilineal with generational skewing
- Sudanese (descriptive) - Unique term for each relative
- Dravidian (classificatory) - Cross-cousin marriage preference

Usage:
    # Generate single ontology
    python kinship_tree_generator.py --system dravidian --output dravidian.ttl
    
    # Generate all 7 ontologies
    python kinship_tree_generator.py --all --output-dir ./ontologies/
    
    # With custom parameters
    python kinship_tree_generator.py --system crow --start-year 1900 --end-year 1960 --seed 42

"""

import random
import argparse
from collections import deque
from pathlib import Path
from datetime import datetime

from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD


# =============================================================================
# Person Class
# =============================================================================

class Person:
    def __init__(self, person_id, sex, first_name, surname, year_of_birth, mother_id=-1, father_id=-1):
        self.id = person_id
        self.sex = sex  # 'M' or 'F'
        self.first_name = first_name
        self.surname = surname
        self.year_of_birth = year_of_birth
        self.mother_id = mother_id
        self.father_id = father_id
        self.children_ids = []
        self.spouse_id = None

    def age_in_year(self, year):
        return year - self.year_of_birth

    def is_married(self):
        return self.spouse_id is not None

    def __repr__(self):
        return f"{self.first_name} {self.surname} (ID:{self.id}, {self.sex}, born {self.year_of_birth})"


# =============================================================================
# Population Simulator
# =============================================================================

class PopulationSimulator:
    """
    Simulates a population following kinship-system-specific marriage rules.
    """
    
    # Name pools for generating unique names
    MALE_NAMES = [
        'Kevin', 'Keith', 'Garry', 'Larry', 'Jerry', 'James', 'William', 'Michael',
        'David', 'Robert', 'Richard', 'Thomas', 'Charles', 'Daniel', 'Matthew',
        'Anthony', 'Mark', 'Donald', 'Steven', 'Paul', 'Andrew', 'Joshua', 'Kenneth',
        'George', 'Edward', 'Brian', 'Ronald', 'Timothy', 'Jason', 'Jeffrey',
        'Ryan', 'Jacob', 'Gary', 'Nicholas', 'Eric', 'Jonathan', 'Stephen', 'Larry',
        'Justin', 'Scott', 'Brandon', 'Benjamin', 'Samuel', 'Raymond', 'Gregory',
        'Frank', 'Alexander', 'Patrick', 'Jack', 'Dennis', 'Henry', 'Peter'
    ]
    
    FEMALE_NAMES = [
        'Olivia', 'Susan', 'Christine', 'Jenny', 'Sarah', 'Emma', 'Sophia', 'Isabella',
        'Emily', 'Abigail', 'Mia', 'Elizabeth', 'Charlotte', 'Amelia', 'Harper',
        'Evelyn', 'Jessica', 'Jennifer', 'Amanda', 'Ashley', 'Stephanie', 'Nicole',
        'Melissa', 'Rebecca', 'Laura', 'Michelle', 'Kimberly', 'Lisa', 'Angela',
        'Helen', 'Samantha', 'Katherine', 'Nancy', 'Betty', 'Margaret', 'Sandra',
        'Dorothy', 'Ruth', 'Sharon', 'Deborah', 'Barbara', 'Patricia', 'Carol',
        'Nancy', 'Linda', 'Karen', 'Donna', 'Cynthia', 'Janet', 'Diane', 'Catherine'
    ]
    
    def __init__(self, start_year=1900, end_year=1999, kinship_system='eskimo', random_seed=42):
        """
        Initialize population simulator with kinship system.
        
        Args:
            start_year: Starting year of simulation
            end_year: Ending year of simulation
            kinship_system: One of ['eskimo', 'hawaiian', 'iroquois', 'crow', 'omaha', 'sudanese', 'dravidian']
            random_seed: Random seed for reproducibility
        """
        self.start_year = start_year
        self.end_year = end_year
        self.current_year = start_year
        
        # Set random seed
        random.seed(random_seed)
        
        # Validate kinship system
        valid_systems = ['eskimo', 'hawaiian', 'iroquois', 'crow', 'omaha', 'sudanese', 'dravidian']
        if kinship_system.lower() not in valid_systems:
            raise ValueError(f"kinship_system must be one of {valid_systems}")
        
        self.kinship_system = kinship_system.lower()

        self.next_person_id = 1
        self.people = {}  # id -> Person
        self.marriages = []  # list of tuples: (man_id, woman_id, year_of_marriage)
        
        # Track used names to avoid collisions
        self.used_name_combos = set()
        
        # For tracking clans (needed for Crow/Omaha)
        self.matriclans = {}  # person_id -> clan_id (matrilineal)
        self.patricians = {}  # person_id -> clan_id (patrilineal)
        self.next_clan_id = 1
        
        # Shuffle name pools for variety
        self.available_male_names = self.MALE_NAMES.copy()
        self.available_female_names = self.FEMALE_NAMES.copy()
        random.shuffle(self.available_male_names)
        random.shuffle(self.available_female_names)
        self.male_name_index = 0
        self.female_name_index = 0

        # Initialize the population
        self._initialize_population()

    def _get_unique_name(self, sex, surname):
        """
        Get a unique first name that doesn't collide with existing name+surname combos.
        
        Args:
            sex: 'M' or 'F'
            surname: The surname to pair with
            
        Returns:
            A unique first name
        """
        if sex == 'M':
            names = self.available_male_names
            start_index = self.male_name_index
        else:
            names = self.available_female_names
            start_index = self.female_name_index
        
        # Try each name in the pool
        for i in range(len(names)):
            idx = (start_index + i) % len(names)
            candidate = names[idx]
            combo = (candidate, surname)
            
            if combo not in self.used_name_combos:
                self.used_name_combos.add(combo)
                if sex == 'M':
                    self.male_name_index = (idx + 1) % len(names)
                else:
                    self.female_name_index = (idx + 1) % len(names)
                return candidate
        
        # If all names used, add a numeric suffix
        base_name = random.choice(names)
        suffix = 2
        while True:
            candidate = f"{base_name}_{suffix}"
            combo = (candidate, surname)
            if combo not in self.used_name_combos:
                self.used_name_combos.add(combo)
                return candidate
            suffix += 1

    def _add_person(self, sex, first_name, surname, year_of_birth, mother_id=-1, father_id=-1):
        """Add a person to the population."""
        # Track used name combo
        self.used_name_combos.add((first_name, surname))
        
        p = Person(self.next_person_id, sex, first_name, surname, year_of_birth, mother_id, father_id)
        self.people[self.next_person_id] = p
        
        # Assign clans for Crow/Omaha systems
        if self.kinship_system == 'crow':
            # Matriclan: inherit from mother
            if mother_id != -1 and mother_id in self.matriclans:
                self.matriclans[p.id] = self.matriclans[mother_id]
            else:
                # Founding member gets new clan
                self.matriclans[p.id] = self.next_clan_id
                self.next_clan_id += 1
                
        elif self.kinship_system == 'omaha':
            # Patrician: inherit from father
            if father_id != -1 and father_id in self.patricians:
                self.patricians[p.id] = self.patricians[father_id]
            else:
                # Founding member gets new clan
                self.patricians[p.id] = self.next_clan_id
                self.next_clan_id += 1
        
        self.next_person_id += 1
        return p

    def _initialize_population(self):
        """Initialize with 5 founding couples."""
        initial_couples = [
            ("John", "Mary", "Smith"),
            ("Peter", "Jane", "Jones"),
            ("Paul", "Anne", "Brown"),
            ("Stephen", "Eve", "Johnson"),
            ("Mike", "Olga", "Williams")
        ]

        for (m_name, f_name, surname) in initial_couples:
            man = self._add_person('M', m_name, surname, 1880)
            woman = self._add_person('F', f_name, surname, 1880)
            # Marry them immediately at start_year
            man.spouse_id = woman.id
            woman.spouse_id = man.id
            self.marriages.append((man.id, woman.id, self.start_year))

    def run_simulation(self):
        """Run the population simulation."""
        for year in range(self.start_year, self.end_year + 1):
            self.current_year = year
            
            # Have children with 50% probability
            self._have_children(year)

            # Marry all eligible singles according to kinship rules
            self._marry_singles(year)

    # =========================================================================
    # Ancestry and Relationship Methods
    # =========================================================================
    
    def _get_ancestors(self, person_id):
        """Get all ancestors of a person with their generational distance."""
        ancestors = {}
        queue = deque()
        queue.append((person_id, 0))

        while queue:
            current_id, distance = queue.popleft()

            if current_id not in ancestors:
                ancestors[current_id] = distance

            if current_id not in self.people:
                continue
                
            current_person = self.people[current_id]
            mother_id = current_person.mother_id
            father_id = current_person.father_id

            if mother_id != -1 and mother_id not in ancestors:
                queue.append((mother_id, distance + 1))
            if father_id != -1 and father_id not in ancestors:
                queue.append((father_id, distance + 1))

        return ancestors

    def _find_mrca(self, person_a_id, person_b_id):
        """Find the Most Recent Common Ancestor (MRCA) of two individuals."""
        ancestors_a = self._get_ancestors(person_a_id)
        ancestors_b = self._get_ancestors(person_b_id)

        common_ancestors = set(ancestors_a.keys()).intersection(ancestors_b.keys())

        if not common_ancestors:
            return (None, None, None)

        min_sum = float('inf')
        mrca_id = None
        distance_a = None
        distance_b = None

        for ancestor_id in common_ancestors:
            da = ancestors_a[ancestor_id]
            db = ancestors_b[ancestor_id]
            total = da + db
            if total < min_sum:
                min_sum = total
                mrca_id = ancestor_id
                distance_a = da
                distance_b = db

        return (mrca_id, distance_a, distance_b)

    def _are_siblings(self, person_a_id, person_b_id):
        """Check if two people are siblings (share at least one parent)."""
        if person_a_id == person_b_id:
            return False
            
        person_a = self.people[person_a_id]
        person_b = self.people[person_b_id]
        
        if person_a.mother_id == -1 or person_a.father_id == -1:
            return False
        if person_b.mother_id == -1 or person_b.father_id == -1:
            return False
            
        return (person_a.mother_id == person_b.mother_id or 
                person_a.father_id == person_b.father_id)

    def _are_parent_child(self, person_a_id, person_b_id):
        """Check if two people are parent and child."""
        person_a = self.people[person_a_id]
        person_b = self.people[person_b_id]
        
        return (person_a.mother_id == person_b_id or 
                person_a.father_id == person_b_id or
                person_b.mother_id == person_a_id or 
                person_b.father_id == person_a_id)

    def _get_cousin_type(self, person_a_id, person_b_id):
        """
        Determine cousin relationship type.
        Returns: ('parallel', degree) or ('cross', degree) or (None, None)
        """
        mrca_id, dist_a, dist_b = self._find_mrca(person_a_id, person_b_id)
        
        if mrca_id is None:
            return (None, None)
        
        if dist_a != dist_b or dist_a < 2:
            return (None, None)
        
        degree = dist_a - 1
        
        person_a = self.people[person_a_id]
        person_b = self.people[person_b_id]
        
        # Find which child of MRCA leads to person A
        current = person_a
        path_a_parent = None
        for _ in range(dist_a - 1):
            if current.mother_id == -1 and current.father_id == -1:
                return (None, None)
            if current.mother_id != -1:
                parent = self.people.get(current.mother_id)
                if parent and (parent.id == mrca_id or parent.mother_id == mrca_id or parent.father_id == mrca_id):
                    path_a_parent = current.mother_id
                    break
                current = parent if parent else current
            if current.father_id != -1:
                parent = self.people.get(current.father_id)
                if parent and (parent.id == mrca_id or parent.mother_id == mrca_id or parent.father_id == mrca_id):
                    path_a_parent = current.father_id
                    break
                current = parent if parent else current
        else:
            if person_a.mother_id != -1:
                m = self.people[person_a.mother_id]
                if m.mother_id == mrca_id or m.father_id == mrca_id:
                    path_a_parent = person_a.mother_id
                elif person_a.father_id != -1:
                    path_a_parent = person_a.father_id
                else:
                    return (None, None)
            elif person_a.father_id != -1:
                path_a_parent = person_a.father_id
            else:
                return (None, None)
        
        # Find which child of MRCA leads to person B
        current = person_b
        path_b_parent = None
        for _ in range(dist_b - 1):
            if current.mother_id == -1 and current.father_id == -1:
                return (None, None)
            if current.mother_id != -1:
                parent = self.people.get(current.mother_id)
                if parent and (parent.id == mrca_id or parent.mother_id == mrca_id or parent.father_id == mrca_id):
                    path_b_parent = current.mother_id
                    break
                current = parent if parent else current
            if current.father_id != -1:
                parent = self.people.get(current.father_id)
                if parent and (parent.id == mrca_id or parent.mother_id == mrca_id or parent.father_id == mrca_id):
                    path_b_parent = current.father_id
                    break
                current = parent if parent else current
        else:
            if person_b.mother_id != -1:
                m = self.people[person_b.mother_id]
                if m.mother_id == mrca_id or m.father_id == mrca_id:
                    path_b_parent = person_b.mother_id
                elif person_b.father_id != -1:
                    path_b_parent = person_b.father_id
                else:
                    return (None, None)
            elif person_b.father_id != -1:
                path_b_parent = person_b.father_id
            else:
                return (None, None)
        
        if path_a_parent is None or path_b_parent is None:
            return (None, None)
            
        parent_a = self.people.get(path_a_parent)
        parent_b = self.people.get(path_b_parent)
        
        if not parent_a or not parent_b:
            return (None, None)
        
        if parent_a.sex == parent_b.sex:
            return ('parallel', degree)
        else:
            return ('cross', degree)

    def _get_matriclan(self, person_id):
        """Get the matrilineal clan of a person (for Crow system)."""
        return self.matriclans.get(person_id)
    
    def _get_patriclan(self, person_id):
        """Get the patrilineal clan of a person (for Omaha system)."""
        return self.patricians.get(person_id)

    # =========================================================================
    # Kinship System Marriage Rules
    # =========================================================================

    def _can_marry_eskimo(self, person_a_id, person_b_id):
        """Eskimo: Prohibit nuclear family and first cousins."""
        if self._are_siblings(person_a_id, person_b_id):
            return False
        if self._are_parent_child(person_a_id, person_b_id):
            return False
        
        mrca_id, dist_a, dist_b = self._find_mrca(person_a_id, person_b_id)
        if mrca_id and dist_a == 2 and dist_b == 2:
            return False
        
        return True

    def _can_marry_hawaiian(self, person_a_id, person_b_id):
        """Hawaiian: All cousins classified as siblings - must marry outside."""
        if self._are_siblings(person_a_id, person_b_id):
            return False
        if self._are_parent_child(person_a_id, person_b_id):
            return False
        
        mrca_id, dist_a, dist_b = self._find_mrca(person_a_id, person_b_id)
        if mrca_id and dist_a >= 2 and dist_b >= 2:
            return False
        
        return True

    def _can_marry_iroquois(self, person_a_id, person_b_id):
        """Iroquois: Parallel cousins prohibited, cross cousins allowed."""
        if self._are_siblings(person_a_id, person_b_id):
            return False
        if self._are_parent_child(person_a_id, person_b_id):
            return False
        
        cousin_type, degree = self._get_cousin_type(person_a_id, person_b_id)
        if cousin_type == 'parallel':
            return False
        
        return True

    def _can_marry_crow(self, person_a_id, person_b_id):
        """Crow (matrilineal): Cannot marry own or father's matriclan."""
        if self._are_siblings(person_a_id, person_b_id):
            return False
        if self._are_parent_child(person_a_id, person_b_id):
            return False
        
        person_a = self.people[person_a_id]
        person_b = self.people[person_b_id]
        
        clan_a = self._get_matriclan(person_a_id)
        clan_b = self._get_matriclan(person_b_id)
        
        if clan_a is None or clan_b is None:
            return True
        
        if clan_a == clan_b:
            return False
        
        if person_a.father_id != -1:
            father_clan = self._get_matriclan(person_a.father_id)
            if father_clan and father_clan == clan_b:
                return False
        
        if person_b.father_id != -1:
            father_clan = self._get_matriclan(person_b.father_id)
            if father_clan and father_clan == clan_a:
                return False
        
        return True

    def _can_marry_omaha(self, person_a_id, person_b_id):
        """Omaha (patrilineal): Cannot marry own or mother's patriclan."""
        if self._are_siblings(person_a_id, person_b_id):
            return False
        if self._are_parent_child(person_a_id, person_b_id):
            return False
        
        person_a = self.people[person_a_id]
        person_b = self.people[person_b_id]
        
        clan_a = self._get_patriclan(person_a_id)
        clan_b = self._get_patriclan(person_b_id)
        
        if clan_a is None or clan_b is None:
            return True
        
        if clan_a == clan_b:
            return False
        
        if person_a.mother_id != -1:
            mother_clan = self._get_patriclan(person_a.mother_id)
            if mother_clan and mother_clan == clan_b:
                return False
        
        if person_b.mother_id != -1:
            mother_clan = self._get_patriclan(person_b.mother_id)
            if mother_clan and mother_clan == clan_a:
                return False
        
        return True

    def _can_marry_sudanese(self, person_a_id, person_b_id):
        """Sudanese: Prohibit close relatives (1st cousins and closer)."""
        if self._are_siblings(person_a_id, person_b_id):
            return False
        if self._are_parent_child(person_a_id, person_b_id):
            return False
        
        mrca_id, dist_a, dist_b = self._find_mrca(person_a_id, person_b_id)
        if mrca_id and dist_a <= 2 and dist_b <= 2:
            return False
        
        return True

    def _can_marry_dravidian(self, person_a_id, person_b_id):
        """Dravidian: Cross-cousin marriage prescribed, parallel cousins prohibited."""
        if self._are_siblings(person_a_id, person_b_id):
            return False
        if self._are_parent_child(person_a_id, person_b_id):
            return False
        
        cousin_type, degree = self._get_cousin_type(person_a_id, person_b_id)
        
        if cousin_type == 'parallel':
            return False
        
        return True

    def _can_marry(self, person_a_id, person_b_id):
        """Check if two people can marry according to the kinship system."""
        dispatch = {
            'eskimo': self._can_marry_eskimo,
            'hawaiian': self._can_marry_hawaiian,
            'iroquois': self._can_marry_iroquois,
            'crow': self._can_marry_crow,
            'omaha': self._can_marry_omaha,
            'sudanese': self._can_marry_sudanese,
            'dravidian': self._can_marry_dravidian
        }
        return dispatch.get(self.kinship_system, self._can_marry_eskimo)(person_a_id, person_b_id)

    # =========================================================================
    # Simulation Methods
    # =========================================================================

    def _marry_singles(self, year):
        """Match eligible singles according to kinship rules."""
        single_men = []
        single_women = []

        for p in self.people.values():
            if not p.is_married() and p.age_in_year(year) >= 18:
                if p.sex == 'M':
                    single_men.append(p)
                else:
                    single_women.append(p)

        single_men.sort(key=lambda x: x.age_in_year(year))
        single_women.sort(key=lambda x: x.age_in_year(year))

        # For Dravidian system, prioritize cross-cousin marriages
        if self.kinship_system == 'dravidian':
            self._marry_cross_cousins_first(single_men, single_women, year)

        used_women = set()
        for man in single_men:
            if man.is_married():
                continue
                
            for woman in single_women:
                if woman.id in used_women or woman.is_married():
                    continue
                
                age_diff = abs(man.age_in_year(year) - woman.age_in_year(year))
                if age_diff > 10:
                    continue
                
                if self._can_marry(man.id, woman.id):
                    man.spouse_id = woman.id
                    woman.spouse_id = man.id
                    self.marriages.append((man.id, woman.id, year))
                    used_women.add(woman.id)
                    break

    def _marry_cross_cousins_first(self, single_men, single_women, year):
        """For Dravidian system: marry cross-cousins first (prescribed)."""
        for man in single_men:
            if man.is_married():
                continue
            for woman in single_women:
                if woman.is_married():
                    continue
                
                cousin_type, degree = self._get_cousin_type(man.id, woman.id)
                if cousin_type == 'cross' and degree == 1:
                    age_diff = abs(man.age_in_year(year) - woman.age_in_year(year))
                    if age_diff <= 10:
                        man.spouse_id = woman.id
                        woman.spouse_id = man.id
                        self.marriages.append((man.id, woman.id, year))
                        break

    def _have_children(self, year):
        """Create children for married couples."""
        processed_couples = set()

        for p in list(self.people.values()):
            if p.is_married() and p.sex == 'M':
                wife = self.people[p.spouse_id]
                mother, father = (wife, p) if wife.sex == 'F' else (p, wife)
                
                if (mother.id, father.id) in processed_couples or (father.id, mother.id) in processed_couples:
                    continue
                processed_couples.add((mother.id, father.id))

                if mother.age_in_year(year) < 40 and len(mother.children_ids) < 3 and random.random() < 0.5:
                    self._create_child(mother, father, year)

    def _create_child(self, mother, father, year):
        """Create a child for a couple with a unique name."""
        sex = 'M' if random.random() < 0.5 else 'F'
        child_surname = father.surname
        
        # Get a unique name - this is the fix for the name collision bug
        child_first_name = self._get_unique_name(sex, child_surname)
        
        child = self._add_person(sex, child_first_name, child_surname, year, mother.id, father.id)

        mother.children_ids.append(child.id)
        father.children_ids.append(child.id)

    # =========================================================================
    # Reporting Methods
    # =========================================================================

    def print_summary(self):
        """Print summary of the population."""
        print(f"\n{'='*60}")
        print(f"KINSHIP SYSTEM: {self.kinship_system.upper()}")
        print(f"{'='*60}")
        
        print(f"\n=== Population Statistics ===")
        print(f"Total people: {len(self.people)}")
        print(f"Total marriages: {len(self.marriages)}")
        
        married_count = sum(1 for p in self.people.values() if p.is_married())
        print(f"Married individuals: {married_count}")
        print(f"Single adults: {sum(1 for p in self.people.values() if not p.is_married() and p.age_in_year(self.end_year) >= 18)}")
        
        # Check for name uniqueness
        name_counts = {}
        for p in self.people.values():
            full_name = f"{p.first_name} {p.surname}"
            name_counts[full_name] = name_counts.get(full_name, 0) + 1
        
        duplicates = {k: v for k, v in name_counts.items() if v > 1}
        if duplicates:
            print(f"\n⚠ Warning: Found {len(duplicates)} duplicate names")
        else:
            print(f"\n✓ All names are unique")

    def print_kinship_analysis(self):
        """Analyze and print kinship patterns in marriages."""
        print(f"\n=== Kinship Analysis ===")
        
        cousin_marriages = {'parallel': 0, 'cross': 0, 'other': 0}
        
        for man_id, woman_id, year in self.marriages:
            cousin_type, degree = self._get_cousin_type(man_id, woman_id)
            if cousin_type == 'parallel':
                cousin_marriages['parallel'] += 1
            elif cousin_type == 'cross':
                cousin_marriages['cross'] += 1
            else:
                cousin_marriages['other'] += 1
        
        print(f"Parallel cousin marriages: {cousin_marriages['parallel']}")
        print(f"Cross cousin marriages: {cousin_marriages['cross']}")
        print(f"Non-cousin marriages: {cousin_marriages['other']}")
        
        if self.kinship_system in ['crow', 'omaha']:
            clan_key = 'matriclans' if self.kinship_system == 'crow' else 'patricians'
            clans = self.matriclans if self.kinship_system == 'crow' else self.patricians
            print(f"\nTotal {clan_key}: {len(set(clans.values()))}")


# =============================================================================
# Kinship Ontology Exporter
# =============================================================================

class KinshipOntologyExporter:
    """
    Export family tree data to RDF/OWL ontology with kinship-specific relationships.
    """
    
    def __init__(self, simulator):
        """Initialize exporter with a PopulationSimulator instance."""
        self.sim = simulator
        self.graph = Graph()
        
        # Define namespaces
        self.FAMILY = Namespace("http://example.org/family/")
        self.KIN = Namespace("http://example.org/kinship/")
        self.PERSON = Namespace("http://example.org/person/")
        
        # Bind namespaces
        self.graph.bind("family", self.FAMILY)
        self.graph.bind("kin", self.KIN)
        self.graph.bind("person", self.PERSON)
        self.graph.bind("owl", OWL)
        self.graph.bind("rdf", RDF)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("xsd", XSD)
    
    def create_ontology(self):
        """Create the complete ontology with classes, properties, and individuals."""
        self._create_ontology_header()
        self._define_classes()
        self._define_properties()
        self._add_persons()
        self._add_basic_relationships()
        self._add_kinship_relationships()
        self._add_inferred_relationships()
        return self.graph
    
    def _create_ontology_header(self):
        """Create ontology metadata."""
        ontology_uri = URIRef(f"http://example.org/ontology/{self.sim.kinship_system}_kinship")
        
        self.graph.add((ontology_uri, RDF.type, OWL.Ontology))
        self.graph.add((ontology_uri, RDFS.label, 
                       Literal(f"{self.sim.kinship_system.capitalize()} Kinship Ontology")))
        self.graph.add((ontology_uri, RDFS.comment, 
                       Literal(f"Family tree ontology following {self.sim.kinship_system} kinship system rules. Generated on {datetime.now().isoformat()}")))
    
    def _define_classes(self):
        """Define ontology classes."""
        classes = [
            (self.FAMILY.Person, "Person"),
            (self.FAMILY.Male, "Male"),
            (self.FAMILY.Female, "Female"),
            (self.FAMILY.Marriage, "Marriage"),
            (self.FAMILY.Clan, "Clan")
        ]
        
        for cls, label in classes:
            self.graph.add((cls, RDF.type, OWL.Class))
            self.graph.add((cls, RDFS.label, Literal(label)))
    
    def _define_properties(self):
        """Define ontology properties including system-specific ones."""
        system = self.sim.kinship_system
        
        # Basic properties
        basic_props = [
            (self.FAMILY.firstName, "first name", XSD.string),
            (self.FAMILY.surname, "surname", XSD.string),
            (self.FAMILY.yearOfBirth, "year of birth", XSD.integer),
        ]
        
        for prop, label, range_type in basic_props:
            self.graph.add((prop, RDF.type, OWL.DatatypeProperty))
            self.graph.add((prop, RDFS.label, Literal(label)))
            self.graph.add((prop, RDFS.range, range_type))
        
        # Object properties for relationships
        rel_props = [
            (self.FAMILY.hasFather, "has father"),
            (self.FAMILY.hasMother, "has mother"),
            (self.FAMILY.hasParent, "has parent"),
            (self.FAMILY.hasChild, "has child"),
            (self.FAMILY.hasSon, "has son"),
            (self.FAMILY.hasDaughter, "has daughter"),
            (self.FAMILY.hasSibling, "has sibling"),
            (self.FAMILY.hasBrother, "has brother"),
            (self.FAMILY.hasSister, "has sister"),
            (self.FAMILY.hasSpouse, "has spouse"),
            (self.FAMILY.hasHusband, "has husband"),
            (self.FAMILY.hasWife, "has wife"),
            (self.FAMILY.hasCousin, "has cousin"),
            (self.FAMILY.hasUncle, "has uncle"),
            (self.FAMILY.hasAunt, "has aunt"),
            (self.FAMILY.hasNephew, "has nephew"),
            (self.FAMILY.hasNiece, "has niece"),
            (self.FAMILY.hasGrandparent, "has grandparent"),
            (self.FAMILY.hasGrandfather, "has grandfather"),
            (self.FAMILY.hasGrandmother, "has grandmother"),
            (self.FAMILY.hasGrandchild, "has grandchild"),
            # New: Maternal/Paternal distinctions (for Eskimo/Sudanese)
            (self.FAMILY.hasMaternalGrandfather, "has maternal grandfather"),
            (self.FAMILY.hasMaternalGrandmother, "has maternal grandmother"),
            (self.FAMILY.hasPaternalGrandfather, "has paternal grandfather"),
            (self.FAMILY.hasPaternalGrandmother, "has paternal grandmother"),
            (self.FAMILY.hasMaternalUncle, "has maternal uncle (mother's brother)"),
            (self.FAMILY.hasMaternalAunt, "has maternal aunt (mother's sister)"),
            (self.FAMILY.hasPaternalUncle, "has paternal uncle (father's brother)"),
            (self.FAMILY.hasPaternalAunt, "has paternal aunt (father's sister)"),
        ]
        
        for prop, label in rel_props:
            self.graph.add((prop, RDF.type, OWL.ObjectProperty))
            self.graph.add((prop, RDFS.label, Literal(label)))
            self.graph.add((prop, RDFS.domain, self.FAMILY.Person))
            self.graph.add((prop, RDFS.range, self.FAMILY.Person))
        
        # System-specific properties
        if system in ['iroquois', 'dravidian']:
            props = [
                (self.KIN.hasClassificatoryParent, "has classificatory parent"),
                (self.KIN.hasClassificatorySibling, "has classificatory sibling"),
                (self.KIN.hasCrossCousin, "has cross-cousin"),
                (self.KIN.hasParallelCousin, "has parallel cousin"),
                (self.KIN.hasCrossAunt, "has cross-aunt (father's sister)"),
                (self.KIN.hasCrossUncle, "has cross-uncle (mother's brother)"),
            ]
            for prop, label in props:
                self.graph.add((prop, RDF.type, OWL.ObjectProperty))
                self.graph.add((prop, RDFS.label, Literal(label)))
        
        if system == 'dravidian':
            props = [
                (self.KIN.hasPotentialSpouse, "has potential spouse (cross-cousin)"),
                (self.KIN.hasAffinalRelative, "has affinal relative"),
                (self.KIN.hasMama, "has Mama (mother's brother)"),
                (self.KIN.hasAthai, "has Athai (father's sister)"),
            ]
            for prop, label in props:
                self.graph.add((prop, RDF.type, OWL.ObjectProperty))
                self.graph.add((prop, RDFS.label, Literal(label)))
        
        if system == 'hawaiian':
            props = [
                (self.KIN.hasClassificatorySibling, "has classificatory sibling"),
                (self.KIN.hasClassificatoryParent, "has classificatory parent"),
            ]
            for prop, label in props:
                self.graph.add((prop, RDF.type, OWL.ObjectProperty))
                self.graph.add((prop, RDFS.label, Literal(label)))
        
        if system == 'crow':
            props = [
                (self.KIN.sameMatriclan, "same matriclan"),
                (self.KIN.fathersMatriclan, "father's matriclan"),
                (self.KIN.hasSkewedGeneration, "has skewed generation relationship"),
            ]
            for prop, label in props:
                self.graph.add((prop, RDF.type, OWL.ObjectProperty))
                self.graph.add((prop, RDFS.label, Literal(label)))
        
        if system == 'omaha':
            props = [
                (self.KIN.samePatriclan, "same patriclan"),
                (self.KIN.mothersPatriclan, "mother's patriclan"),
                (self.KIN.hasSkewedGeneration, "has skewed generation relationship"),
            ]
            for prop, label in props:
                self.graph.add((prop, RDF.type, OWL.ObjectProperty))
                self.graph.add((prop, RDFS.label, Literal(label)))
    
    def _add_persons(self):
        """Add all persons as individuals with unique labels."""
        for person_id, person in self.sim.people.items():
            person_uri = self.PERSON[f"person_{person_id}"]
            
            # Add type
            self.graph.add((person_uri, RDF.type, self.FAMILY.Person))
            if person.sex == 'M':
                self.graph.add((person_uri, RDF.type, self.FAMILY.Male))
            else:
                self.graph.add((person_uri, RDF.type, self.FAMILY.Female))
            
            # Add properties
            self.graph.add((person_uri, self.FAMILY.firstName, 
                          Literal(person.first_name, datatype=XSD.string)))
            self.graph.add((person_uri, self.FAMILY.surname, 
                          Literal(person.surname, datatype=XSD.string)))
            self.graph.add((person_uri, self.FAMILY.yearOfBirth, 
                          Literal(person.year_of_birth, datatype=XSD.integer)))
            
            # Label includes birth year for disambiguation
            self.graph.add((person_uri, RDFS.label, 
                          Literal(f"{person.first_name} {person.surname}")))
            
            # Add clan membership for Crow/Omaha
            if self.sim.kinship_system == 'crow':
                clan_id = self.sim._get_matriclan(person_id)
                if clan_id:
                    clan_uri = self.FAMILY[f"matriclan_{clan_id}"]
                    self.graph.add((clan_uri, RDF.type, self.FAMILY.Clan))
                    self.graph.add((person_uri, self.FAMILY.belongsToClan, clan_uri))
            elif self.sim.kinship_system == 'omaha':
                clan_id = self.sim._get_patriclan(person_id)
                if clan_id:
                    clan_uri = self.FAMILY[f"patriclan_{clan_id}"]
                    self.graph.add((clan_uri, RDF.type, self.FAMILY.Clan))
                    self.graph.add((person_uri, self.FAMILY.belongsToClan, clan_uri))
    
    def _add_basic_relationships(self):
        """Add basic family relationships."""
        for person_id, person in self.sim.people.items():
            person_uri = self.PERSON[f"person_{person_id}"]
            
            # Parent relationships
            if person.mother_id != -1:
                mother_uri = self.PERSON[f"person_{person.mother_id}"]
                self.graph.add((person_uri, self.FAMILY.hasMother, mother_uri))
                self.graph.add((person_uri, self.FAMILY.hasParent, mother_uri))
                self.graph.add((mother_uri, self.FAMILY.hasChild, person_uri))
                if person.sex == 'M':
                    self.graph.add((mother_uri, self.FAMILY.hasSon, person_uri))
                else:
                    self.graph.add((mother_uri, self.FAMILY.hasDaughter, person_uri))
            
            if person.father_id != -1:
                father_uri = self.PERSON[f"person_{person.father_id}"]
                self.graph.add((person_uri, self.FAMILY.hasFather, father_uri))
                self.graph.add((person_uri, self.FAMILY.hasParent, father_uri))
                self.graph.add((father_uri, self.FAMILY.hasChild, person_uri))
                if person.sex == 'M':
                    self.graph.add((father_uri, self.FAMILY.hasSon, person_uri))
                else:
                    self.graph.add((father_uri, self.FAMILY.hasDaughter, person_uri))
            
            # Spouse relationships
            if person.spouse_id is not None:
                spouse_uri = self.PERSON[f"person_{person.spouse_id}"]
                self.graph.add((person_uri, self.FAMILY.hasSpouse, spouse_uri))
                if person.sex == 'M':
                    self.graph.add((person_uri, self.FAMILY.hasWife, spouse_uri))
                else:
                    self.graph.add((person_uri, self.FAMILY.hasHusband, spouse_uri))
            
            # Sibling relationships
            for other_id, other_person in self.sim.people.items():
                if other_id != person_id and self.sim._are_siblings(person_id, other_id):
                    other_uri = self.PERSON[f"person_{other_id}"]
                    self.graph.add((person_uri, self.FAMILY.hasSibling, other_uri))
                    if other_person.sex == 'M':
                        self.graph.add((person_uri, self.FAMILY.hasBrother, other_uri))
                    else:
                        self.graph.add((person_uri, self.FAMILY.hasSister, other_uri))
    
    def _add_kinship_relationships(self):
        """Add kinship-system-specific relationships."""
        system = self.sim.kinship_system
        
        for person_id, person in self.sim.people.items():
            person_uri = self.PERSON[f"person_{person_id}"]
            
            # === Cousin relationships ===
            for other_id, other_person in self.sim.people.items():
                if other_id == person_id:
                    continue
                
                cousin_type, degree = self.sim._get_cousin_type(person_id, other_id)
                other_uri = self.PERSON[f"person_{other_id}"]
                
                if cousin_type and degree:
                    # General cousin relationship
                    self.graph.add((person_uri, self.FAMILY.hasCousin, other_uri))
                    
                    # System-specific cousin relationships
                    if system in ['iroquois', 'dravidian']:
                        if cousin_type == 'parallel':
                            self.graph.add((person_uri, self.KIN.hasParallelCousin, other_uri))
                            self.graph.add((person_uri, self.KIN.hasClassificatorySibling, other_uri))
                        elif cousin_type == 'cross':
                            self.graph.add((person_uri, self.KIN.hasCrossCousin, other_uri))
                            if system == 'dravidian' and degree == 1:
                                self.graph.add((person_uri, self.KIN.hasPotentialSpouse, other_uri))
                    
                    elif system == 'hawaiian':
                        self.graph.add((person_uri, self.KIN.hasClassificatorySibling, other_uri))
            
            # === Aunt/Uncle relationships with maternal/paternal distinction ===
            if person.mother_id != -1:
                mother = self.sim.people[person.mother_id]
                for other_id, other_person in self.sim.people.items():
                    if self.sim._are_siblings(mother.id, other_id):
                        other_uri = self.PERSON[f"person_{other_id}"]
                        if other_person.sex == 'M':
                            self.graph.add((person_uri, self.FAMILY.hasUncle, other_uri))
                            # Maternal uncle (mother's brother)
                            self.graph.add((person_uri, self.FAMILY.hasMaternalUncle, other_uri))
                            if system in ['iroquois', 'dravidian']:
                                self.graph.add((person_uri, self.KIN.hasCrossUncle, other_uri))
                            if system == 'dravidian':
                                self.graph.add((person_uri, self.KIN.hasMama, other_uri))
                        else:
                            self.graph.add((person_uri, self.FAMILY.hasAunt, other_uri))
                            # Maternal aunt (mother's sister)
                            self.graph.add((person_uri, self.FAMILY.hasMaternalAunt, other_uri))
                            if system in ['iroquois', 'dravidian', 'hawaiian']:
                                self.graph.add((person_uri, self.KIN.hasClassificatoryParent, other_uri))
            
            if person.father_id != -1:
                father = self.sim.people[person.father_id]
                for other_id, other_person in self.sim.people.items():
                    if self.sim._are_siblings(father.id, other_id):
                        other_uri = self.PERSON[f"person_{other_id}"]
                        if other_person.sex == 'M':
                            self.graph.add((person_uri, self.FAMILY.hasUncle, other_uri))
                            # Paternal uncle (father's brother)
                            self.graph.add((person_uri, self.FAMILY.hasPaternalUncle, other_uri))
                            if system in ['iroquois', 'dravidian', 'hawaiian']:
                                self.graph.add((person_uri, self.KIN.hasClassificatoryParent, other_uri))
                        else:
                            self.graph.add((person_uri, self.FAMILY.hasAunt, other_uri))
                            # Paternal aunt (father's sister)
                            self.graph.add((person_uri, self.FAMILY.hasPaternalAunt, other_uri))
                            if system in ['iroquois', 'dravidian']:
                                self.graph.add((person_uri, self.KIN.hasCrossAunt, other_uri))
                            if system == 'dravidian':
                                self.graph.add((person_uri, self.KIN.hasAthai, other_uri))
            
            # === Clan-based relationships ===
            if system == 'crow':
                own_clan = self.sim._get_matriclan(person_id)
                if person.father_id != -1:
                    fathers_clan = self.sim._get_matriclan(person.father_id)
                    
                    for other_id, other_person in self.sim.people.items():
                        if other_id == person_id:
                            continue
                        other_clan = self.sim._get_matriclan(other_id)
                        other_uri = self.PERSON[f"person_{other_id}"]
                        
                        if other_clan == own_clan:
                            self.graph.add((person_uri, self.KIN.sameMatriclan, other_uri))
                        elif other_clan == fathers_clan:
                            self.graph.add((person_uri, self.KIN.fathersMatriclan, other_uri))
            
            elif system == 'omaha':
                own_clan = self.sim._get_patriclan(person_id)
                if person.mother_id != -1:
                    mothers_clan = self.sim._get_patriclan(person.mother_id)
                    
                    for other_id, other_person in self.sim.people.items():
                        if other_id == person_id:
                            continue
                        other_clan = self.sim._get_patriclan(other_id)
                        other_uri = self.PERSON[f"person_{other_id}"]
                        
                        if other_clan == own_clan:
                            self.graph.add((person_uri, self.KIN.samePatriclan, other_uri))
                        elif other_clan == mothers_clan:
                            self.graph.add((person_uri, self.KIN.mothersPatriclan, other_uri))
    
    def _add_inferred_relationships(self):
        """Add inferred relationships with maternal/paternal distinctions."""
        for person_id, person in self.sim.people.items():
            person_uri = self.PERSON[f"person_{person_id}"]
            
            # Maternal grandparents
            if person.mother_id != -1:
                mother = self.sim.people[person.mother_id]
                if mother.mother_id != -1:
                    grandma_uri = self.PERSON[f"person_{mother.mother_id}"]
                    self.graph.add((person_uri, self.FAMILY.hasGrandmother, grandma_uri))
                    self.graph.add((person_uri, self.FAMILY.hasMaternalGrandmother, grandma_uri))
                    self.graph.add((person_uri, self.FAMILY.hasGrandparent, grandma_uri))
                    self.graph.add((grandma_uri, self.FAMILY.hasGrandchild, person_uri))
                if mother.father_id != -1:
                    grandpa_uri = self.PERSON[f"person_{mother.father_id}"]
                    self.graph.add((person_uri, self.FAMILY.hasGrandfather, grandpa_uri))
                    self.graph.add((person_uri, self.FAMILY.hasMaternalGrandfather, grandpa_uri))
                    self.graph.add((person_uri, self.FAMILY.hasGrandparent, grandpa_uri))
                    self.graph.add((grandpa_uri, self.FAMILY.hasGrandchild, person_uri))
            
            # Paternal grandparents
            if person.father_id != -1:
                father = self.sim.people[person.father_id]
                if father.mother_id != -1:
                    grandma_uri = self.PERSON[f"person_{father.mother_id}"]
                    self.graph.add((person_uri, self.FAMILY.hasGrandmother, grandma_uri))
                    self.graph.add((person_uri, self.FAMILY.hasPaternalGrandmother, grandma_uri))
                    self.graph.add((person_uri, self.FAMILY.hasGrandparent, grandma_uri))
                    self.graph.add((grandma_uri, self.FAMILY.hasGrandchild, person_uri))
                if father.father_id != -1:
                    grandpa_uri = self.PERSON[f"person_{father.father_id}"]
                    self.graph.add((person_uri, self.FAMILY.hasGrandfather, grandpa_uri))
                    self.graph.add((person_uri, self.FAMILY.hasPaternalGrandfather, grandpa_uri))
                    self.graph.add((person_uri, self.FAMILY.hasGrandparent, grandpa_uri))
                    self.graph.add((grandpa_uri, self.FAMILY.hasGrandchild, person_uri))
    
    def export_to_file(self, filename):
        """Export the ontology to a Turtle (.ttl) file."""
        if len(self.graph) == 0:
            self.create_ontology()
        
        self.graph.serialize(destination=filename, format='turtle')
        return filename
    
    def get_statistics(self):
        """Get statistics about the ontology."""
        return {
            'total_triples': len(self.graph),
            'total_persons': len(self.sim.people),
            'total_marriages': len(self.sim.marriages),
            'kinship_system': self.sim.kinship_system
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_ontology(system, output_path, start_year=1900, end_year=1950, seed=42, verbose=True):
    """
    Generate a kinship ontology for a specific system.
    
    Args:
        system: Kinship system name
        output_path: Output file path
        start_year: Simulation start year
        end_year: Simulation end year
        seed: Random seed
        verbose: Print progress
    
    Returns:
        Statistics dictionary
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Generating: {system.upper()} kinship system")
        print(f"{'='*60}")
    
    # Run simulation
    sim = PopulationSimulator(
        start_year=start_year,
        end_year=end_year,
        kinship_system=system,
        random_seed=seed
    )
    sim.run_simulation()
    
    if verbose:
        sim.print_summary()
    
    # Export ontology
    exporter = KinshipOntologyExporter(sim)
    exporter.create_ontology()
    exporter.export_to_file(output_path)
    
    stats = exporter.get_statistics()
    
    if verbose:
        print(f"\n✓ Exported to: {output_path}")
        print(f"  Total triples: {stats['total_triples']}")
        print(f"  Total persons: {stats['total_persons']}")
        print(f"  Total marriages: {stats['total_marriages']}")
    
    return stats


def generate_all_ontologies(output_dir, start_year=1900, end_year=1950, seed=42, verbose=True):
    """
    Generate ontologies for all 7 kinship systems.
    
    Args:
        output_dir: Output directory
        start_year: Simulation start year
        end_year: Simulation end year
        seed: Random seed
        verbose: Print progress
    
    Returns:
        Dictionary of system -> statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    systems = ['eskimo', 'hawaiian', 'iroquois', 'crow', 'omaha', 'sudanese', 'dravidian']
    all_stats = {}
    
    print("\n" + "="*70)
    print("KINSHIP ONTOLOGY GENERATION - ALL SYSTEMS")
    print("="*70)
    print(f"Output directory: {output_path}")
    print(f"Simulation period: {start_year} - {end_year}")
    print(f"Random seed: {seed}")
    
    for system in systems:
        ttl_path = output_path / f"{system}.ttl"
        stats = generate_ontology(
            system=system,
            output_path=str(ttl_path),
            start_year=start_year,
            end_year=end_year,
            seed=seed,
            verbose=verbose
        )
        all_stats[system] = stats
    
    # Print summary
    print("\n" + "="*70)
    print("GENERATION COMPLETE - SUMMARY")
    print("="*70)
    print(f"\n{'System':<12} {'Triples':>10} {'Persons':>10} {'Marriages':>10}")
    print("-"*45)
    for system, stats in all_stats.items():
        print(f"{system:<12} {stats['total_triples']:>10} {stats['total_persons']:>10} {stats['total_marriages']:>10}")
    
    print(f"\n✓ All {len(systems)} ontologies generated in: {output_path}")
    
    return all_stats


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate kinship family tree ontologies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate single ontology
  python kinship_tree_generator.py --system dravidian --output dravidian.ttl

  # Generate all 7 ontologies
  python kinship_tree_generator.py --all --output-dir ./ontologies/

  # Custom simulation parameters
  python kinship_tree_generator.py --all --output-dir ./data/ --start-year 1900 --end-year 1970 --seed 123
        """
    )
    
    parser.add_argument('--system', type=str, 
                       choices=['eskimo', 'hawaiian', 'iroquois', 'crow', 'omaha', 'sudanese', 'dravidian'],
                       help='Kinship system to generate')
    parser.add_argument('--output', type=str, help='Output file path (for single system)')
    parser.add_argument('--all', action='store_true', help='Generate all 7 kinship systems')
    parser.add_argument('--output-dir', type=str, default='./ontologies',
                       help='Output directory (for --all)')
    parser.add_argument('--start-year', type=int, default=1900, help='Simulation start year')
    parser.add_argument('--end-year', type=int, default=1950, help='Simulation end year')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    if args.all:
        generate_all_ontologies(
            output_dir=args.output_dir,
            start_year=args.start_year,
            end_year=args.end_year,
            seed=args.seed,
            verbose=not args.quiet
        )
    elif args.system:
        if not args.output:
            args.output = f"{args.system}.ttl"
        generate_ontology(
            system=args.system,
            output_path=args.output,
            start_year=args.start_year,
            end_year=args.end_year,
            seed=args.seed,
            verbose=not args.quiet
        )
    else:
        parser.print_help()
        print("\n⚠ Please specify --system or --all")
