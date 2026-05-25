#!/usr/bin/env python3
"""
KinshipQA Result Analyzer
=========================
Automatically reads all result folders and generates tables for the paper.

Tables generated:
- Table 4: Main results (Western vs Non-Western, by Category)
- Table 5: Performance by kinship system
- Table 6: Cultural override effect
- Table 7: Performance by n-hops complexity

Usage:
    python analyze_results.py --results-dir ./
    python analyze_results.py --results-dir ./ --output-dir ./tables/
    python analyze_results.py --results-dir ./ --format csv
    python analyze_results.py --results-dir ./ --format latex

Author: Tianda (ACL 2026)
"""

import json
import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import csv

# =============================================================================
# Constants
# =============================================================================

KINSHIP_SYSTEMS = ['eskimo', 'hawaiian', 'iroquois', 'dravidian', 'crow', 'omaha', 'sudanese']
WESTERN_SYSTEMS = ['eskimo', 'sudanese']
NON_WESTERN_SYSTEMS = ['hawaiian', 'iroquois', 'dravidian', 'crow', 'omaha']

SYSTEM_TYPES = {
    'eskimo': 'Descriptive',
    'sudanese': 'Descriptive', 
    'hawaiian': 'Generational',
    'iroquois': 'Bifurcate',
    'dravidian': 'Bifurcate',
    'crow': 'Mat. Skewing',
    'omaha': 'Pat. Skewing'
}

# =============================================================================
# Data Loading
# =============================================================================

def find_result_folders(base_dir: str) -> Dict[str, Path]:
    """Find all result folders matching pattern results_* or result_*"""
    base_path = Path(base_dir)
    result_folders = {}
    
    # Look for folders matching patterns
    for pattern in ['results_*', 'result_*']:
        for folder in base_path.glob(pattern):
            if folder.is_dir():
                # Extract model name from folder name
                model_name = folder.name.replace('results_', '').replace('result_', '')
                result_folders[model_name] = folder
    
    # Also check for combined_results*.json files directly in base_dir
    for json_file in base_path.glob('combined_results*.json'):
        # Extract model name from filename
        name = json_file.stem.replace('combined_results_', '').replace('combined_results', '')
        if name:
            result_folders[f"combined_{name}"] = json_file
        else:
            result_folders["combined"] = json_file
    
    return result_folders


def load_results_from_folder(folder_path: Path) -> Optional[Dict]:
    """Load results from a folder containing individual system results or combined JSON"""
    
    # If it's a JSON file directly
    if folder_path.is_file() and folder_path.suffix == '.json':
        with open(folder_path) as f:
            return json.load(f)
    
    # Check for combined_results.json first
    combined_path = folder_path / 'combined_results.json'
    if combined_path.exists():
        with open(combined_path) as f:
            return json.load(f)
    
    # Otherwise, load individual system files
    results = {}
    for system in KINSHIP_SYSTEMS:
        # Try different naming patterns
        for pattern in [f'{system}_results.json', f'{system}.json']:
            file_path = folder_path / pattern
            if file_path.exists():
                with open(file_path) as f:
                    data = json.load(f)
                    # Handle both formats: direct system data or wrapped in 'summary'
                    if 'summary' in data:
                        results[system] = data['summary']
                    else:
                        results[system] = data
                break
    
    return results if results else None


def load_all_results(base_dir: str) -> Dict[str, Dict]:
    """Load all results from all model folders"""
    all_results = {}
    folders = find_result_folders(base_dir)
    
    print(f"Found {len(folders)} result sources:")
    for name, path in folders.items():
        print(f"  - {name}: {path}")
        results = load_results_from_folder(path)
        if results:
            all_results[name] = results
            print(f"    Loaded {len(results)} systems")
        else:
            print(f"    WARNING: No results found")
    
    return all_results


# =============================================================================
# Metric Extraction Helpers
# =============================================================================

def get_accuracy(data: Dict, metric: str = 'exact_match') -> float:
    """Extract accuracy from result data"""
    if 'accuracy' in data:
        return data['accuracy'].get(metric, 0)
    return data.get(metric, 0)


def get_category_accuracy(data: Dict, cat: int) -> float:
    """Extract category-specific accuracy"""
    by_cat = data.get('by_category', {})
    # Handle different key formats
    for key in [f'cat_{cat}', f'cat{cat}', str(cat)]:
        if key in by_cat:
            return by_cat[key].get('exact_match', 0)
    return 0


def get_hop_accuracy(data: Dict, hop: int) -> float:
    """Extract hop-specific accuracy"""
    by_hops = data.get('by_hops', {})
    # Handle different key formats
    for key in [f'hop_{hop}', f'{hop}_hop', f'{hop}-hop', str(hop)]:
        if key in by_hops:
            return by_hops[key].get('exact_match', 0)
    return 0


def get_override_accuracy(data: Dict, with_override: bool) -> Optional[float]:
    """Extract cultural override accuracy"""
    by_override = data.get('by_cultural_override', {})
    
    if with_override:
        for key in ['with_override', 'w_override', 'override']:
            if key in by_override:
                return by_override[key].get('exact_match', 0)
    else:
        for key in ['without_override', 'no_override', 'w/o_override']:
            if key in by_override:
                return by_override[key].get('exact_match', 0)
    
    return None


# =============================================================================
# Table Generation
# =============================================================================

def generate_table4(all_results: Dict[str, Dict]) -> Dict:
    """
    Generate Table 4: Main Results
    
    Columns:
    - Model
    - Western (avg of Eskimo, Sudanese)
    - Non-Western (avg of other 5)
    - Δ Gap
    - Cat.1 (all systems)
    - Cat.2 (all systems)
    - Cat.3 (all systems)
    - Cat.4 (NON-WESTERN ONLY)
    """
    table_data = []
    
    for model_name, results in all_results.items():
        row = {'model': model_name}
        
        # Western accuracy
        western_acc = []
        for sys in WESTERN_SYSTEMS:
            if sys in results:
                western_acc.append(get_accuracy(results[sys]))
        row['western'] = sum(western_acc) / len(western_acc) * 100 if western_acc else 0
        
        # Non-Western accuracy
        nonwestern_acc = []
        for sys in NON_WESTERN_SYSTEMS:
            if sys in results:
                nonwestern_acc.append(get_accuracy(results[sys]))
        row['non_western'] = sum(nonwestern_acc) / len(nonwestern_acc) * 100 if nonwestern_acc else 0
        
        # Gap
        row['gap'] = row['western'] - row['non_western']
        
        # Category accuracies (1-3: all systems, 4: non-western only)
        for cat in [1, 2, 3]:
            cat_acc = []
            for sys in KINSHIP_SYSTEMS:
                if sys in results:
                    cat_acc.append(get_category_accuracy(results[sys], cat))
            row[f'cat{cat}'] = sum(cat_acc) / len(cat_acc) * 100 if cat_acc else 0
        
        # Cat 4: NON-WESTERN ONLY
        cat4_acc = []
        for sys in NON_WESTERN_SYSTEMS:
            if sys in results:
                cat4_acc.append(get_category_accuracy(results[sys], 4))
        row['cat4'] = sum(cat4_acc) / len(cat4_acc) * 100 if cat4_acc else 0
        
        table_data.append(row)
    
    return {
        'title': 'Table 4: Main Results (Exact Match %)',
        'description': 'Western=Eskimo+Sudanese, Non-Western=Hawaiian+Iroquois+Dravidian+Crow+Omaha. Cat.4 is Non-Western only.',
        'columns': ['Model', 'Western', 'Non-West.', 'Δ Gap', 'Cat.1', 'Cat.2', 'Cat.3', 'Cat.4'],
        'data': table_data
    }


def generate_table5(all_results: Dict[str, Dict]) -> Dict:
    """
    Generate Table 5: Performance by Kinship System
    
    Columns:
    - System
    - Type
    - Overall (mean ± std across models)
    - Cat.4 (mean ± std across models)
    """
    table_data = []
    
    for system in KINSHIP_SYSTEMS:
        row = {
            'system': system.capitalize(),
            'type': SYSTEM_TYPES[system]
        }
        
        # Collect overall and cat4 across all models
        overall_scores = []
        cat4_scores = []
        
        for model_name, results in all_results.items():
            if system in results:
                overall_scores.append(get_accuracy(results[system]) * 100)
                cat4_scores.append(get_category_accuracy(results[system], 4) * 100)
        
        # Calculate mean and std
        if overall_scores:
            mean_overall = sum(overall_scores) / len(overall_scores)
            std_overall = (sum((x - mean_overall) ** 2 for x in overall_scores) / len(overall_scores)) ** 0.5
            row['overall_mean'] = mean_overall
            row['overall_std'] = std_overall
        else:
            row['overall_mean'] = 0
            row['overall_std'] = 0
        
        if cat4_scores:
            mean_cat4 = sum(cat4_scores) / len(cat4_scores)
            std_cat4 = (sum((x - mean_cat4) ** 2 for x in cat4_scores) / len(cat4_scores)) ** 0.5
            row['cat4_mean'] = mean_cat4
            row['cat4_std'] = std_cat4
        else:
            row['cat4_mean'] = 0
            row['cat4_std'] = 0
        
        table_data.append(row)
    
    return {
        'title': 'Table 5: Performance by Kinship System',
        'description': 'Mean ± std across all models',
        'columns': ['System', 'Type', 'Overall', 'Cat.4'],
        'data': table_data
    }


def generate_table6(all_results: Dict[str, Dict]) -> Dict:
    """
    Generate Table 6: Cultural Override Effect
    
    Columns:
    - System
    - w/ Override
    - w/o Override
    - Gap
    """
    table_data = []
    
    for system in NON_WESTERN_SYSTEMS:
        row = {'system': system.capitalize()}
        
        with_override_scores = []
        without_override_scores = []
        
        for model_name, results in all_results.items():
            if system in results:
                w_override = get_override_accuracy(results[system], True)
                wo_override = get_override_accuracy(results[system], False)
                
                if w_override is not None:
                    with_override_scores.append(w_override * 100)
                if wo_override is not None:
                    without_override_scores.append(wo_override * 100)
        
        # Calculate mean and std for w/ override
        if with_override_scores:
            mean_w = sum(with_override_scores) / len(with_override_scores)
            std_w = (sum((x - mean_w) ** 2 for x in with_override_scores) / len(with_override_scores)) ** 0.5
            row['with_override_mean'] = mean_w
            row['with_override_std'] = std_w
        else:
            row['with_override_mean'] = 0
            row['with_override_std'] = 0
        
        # Calculate mean and std for w/o override
        if without_override_scores:
            mean_wo = sum(without_override_scores) / len(without_override_scores)
            std_wo = (sum((x - mean_wo) ** 2 for x in without_override_scores) / len(without_override_scores)) ** 0.5
            row['without_override_mean'] = mean_wo
            row['without_override_std'] = std_wo
        else:
            row['without_override_mean'] = 0
            row['without_override_std'] = 0
        
        # Gap
        row['gap'] = row['without_override_mean'] - row['with_override_mean']
        
        table_data.append(row)
    
    # Add average row
    avg_row = {'system': 'Average'}
    avg_w = sum(r['with_override_mean'] for r in table_data) / len(table_data)
    avg_wo = sum(r['without_override_mean'] for r in table_data) / len(table_data)
    std_w = (sum((r['with_override_mean'] - avg_w) ** 2 for r in table_data) / len(table_data)) ** 0.5
    std_wo = (sum((r['without_override_mean'] - avg_wo) ** 2 for r in table_data) / len(table_data)) ** 0.5
    avg_row['with_override_mean'] = avg_w
    avg_row['with_override_std'] = std_w
    avg_row['without_override_mean'] = avg_wo
    avg_row['without_override_std'] = std_wo
    avg_row['gap'] = avg_wo - avg_w
    table_data.append(avg_row)
    
    return {
        'title': 'Table 6: Cultural Override Effect (Category 4)',
        'description': 'Performance on questions with vs without cultural override. Mean ± std across models.',
        'columns': ['System', 'w/ Override', 'w/o Override', 'Gap'],
        'data': table_data
    }


def generate_table7(all_results: Dict[str, Dict]) -> Dict:
    """
    Generate Table 7: Performance by N-Hops
    
    Columns:
    - System Type (Western/Non-Western)
    - 1-hop
    - 2-hop
    - 3-hop
    - 4-hop
    """
    table_data = []
    
    for sys_type, systems in [('Western', WESTERN_SYSTEMS), ('Non-Western', NON_WESTERN_SYSTEMS)]:
        row = {'system_type': sys_type}
        
        for hop in [1, 2, 3, 4]:
            hop_scores = []
            for model_name, results in all_results.items():
                for system in systems:
                    if system in results:
                        hop_scores.append(get_hop_accuracy(results[system], hop) * 100)
            
            if hop_scores:
                row[f'hop{hop}'] = sum(hop_scores) / len(hop_scores)
            else:
                row[f'hop{hop}'] = 0
        
        table_data.append(row)
    
    # Add gap row
    if len(table_data) == 2:
        gap_row = {'system_type': 'Δ Gap'}
        for hop in [1, 2, 3, 4]:
            gap_row[f'hop{hop}'] = table_data[0][f'hop{hop}'] - table_data[1][f'hop{hop}']
        table_data.append(gap_row)
    
    return {
        'title': 'Table 7: Performance by N-Hops Complexity',
        'description': 'Averaged across models. Gap = Western - Non-Western.',
        'columns': ['System Type', '1-hop', '2-hop', '3-hop', '4-hop'],
        'data': table_data
    }


# =============================================================================
# Output Formatting
# =============================================================================

def print_table_console(table: Dict):
    """Print table to console in readable format"""
    print("\n" + "=" * 80)
    print(table['title'])
    print(table['description'])
    print("=" * 80)
    
    data = table['data']
    if not data:
        print("No data available")
        return
    
    # Determine column widths based on content
    if 'model' in data[0]:
        # Table 4 format
        print(f"\n{'Model':<20} {'Western':>8} {'Non-West':>9} {'Δ Gap':>7} {'Cat.1':>7} {'Cat.2':>7} {'Cat.3':>7} {'Cat.4':>7}")
        print("-" * 80)
        for row in data:
            print(f"{row['model']:<20} {row['western']:>7.1f}% {row['non_western']:>8.1f}% {row['gap']:>+6.1f}% {row['cat1']:>6.1f}% {row['cat2']:>6.1f}% {row['cat3']:>6.1f}% {row['cat4']:>6.1f}%")
    
    elif 'system' in data[0] and 'type' in data[0]:
        # Table 5 format
        print(f"\n{'System':<12} {'Type':<14} {'Overall':>16} {'Cat.4':>16}")
        print("-" * 60)
        for row in data:
            overall_str = f"{row['overall_mean']:.1f} ± {row['overall_std']:.1f}"
            cat4_str = f"{row['cat4_mean']:.1f} ± {row['cat4_std']:.1f}"
            print(f"{row['system']:<12} {row['type']:<14} {overall_str:>16} {cat4_str:>16}")
    
    elif 'system' in data[0] and 'with_override_mean' in data[0]:
        # Table 6 format
        print(f"\n{'System':<12} {'w/ Override':>16} {'w/o Override':>16} {'Gap':>8}")
        print("-" * 55)
        for row in data:
            w_str = f"{row['with_override_mean']:.1f} ± {row['with_override_std']:.1f}"
            wo_str = f"{row['without_override_mean']:.1f} ± {row['without_override_std']:.1f}"
            print(f"{row['system']:<12} {w_str:>16} {wo_str:>16} {row['gap']:>+7.1f}")
    
    elif 'system_type' in data[0]:
        # Table 7 format
        print(f"\n{'System Type':<14} {'1-hop':>8} {'2-hop':>8} {'3-hop':>8} {'4-hop':>8}")
        print("-" * 50)
        for row in data:
            if row['system_type'] == 'Δ Gap':
                print(f"{row['system_type']:<14} {row['hop1']:>+7.1f} {row['hop2']:>+7.1f} {row['hop3']:>+7.1f} {row['hop4']:>+7.1f}")
            else:
                print(f"{row['system_type']:<14} {row['hop1']:>7.1f}% {row['hop2']:>7.1f}% {row['hop3']:>7.1f}% {row['hop4']:>7.1f}%")


def save_table_csv(table: Dict, output_path: Path):
    """Save table to CSV format"""
    data = table['data']
    if not data:
        return
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Saved: {output_path}")


def save_table_json(table: Dict, output_path: Path):
    """Save table to JSON format"""
    with open(output_path, 'w') as f:
        json.dump(table, f, indent=2)
    
    print(f"Saved: {output_path}")


def generate_latex_table4(table: Dict) -> str:
    """Generate LaTeX for Table 4"""
    latex = """\\begin{table*}[t]
\\centering
\\caption{Main results on KinshipQA (Exact Match \\%). Western includes Eskimo and Sudanese systems; Non-Western includes Hawaiian, Iroquois, Dravidian, Crow, and Omaha. $\\Delta$ Gap = Western $-$ Non-Western. Cat.4 scores are computed on non-Western systems only.}
\\label{tab:main-results}
\\begin{tabular}{l|ccc|cccc}
\\toprule
 & \\multicolumn{3}{c|}{By Culture} & \\multicolumn{4}{c}{By Category} \\\\
Model & Western & Non-West. & $\\Delta$ Gap & Cat.1 & Cat.2 & Cat.3 & Cat.4 \\\\
\\midrule
"""
    for row in table['data']:
        latex += f"{row['model']} & {row['western']:.1f} & {row['non_western']:.1f} & {row['gap']:+.1f} & {row['cat1']:.1f} & {row['cat2']:.1f} & {row['cat3']:.1f} & {row['cat4']:.1f} \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\end{table*}
"""
    return latex


# =============================================================================
# Summary Statistics
# =============================================================================

def print_summary(all_results: Dict[str, Dict]):
    """Print summary statistics"""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    # Count models
    print(f"\nModels evaluated: {len(all_results)}")
    for model in all_results:
        print(f"  - {model}")
    
    # Overall accuracy across all models
    print("\n" + "-" * 40)
    print("Overall Accuracy (Exact Match)")
    print("-" * 40)
    
    all_western = []
    all_nonwestern = []
    
    for model_name, results in all_results.items():
        western = [get_accuracy(results[s]) for s in WESTERN_SYSTEMS if s in results]
        nonwestern = [get_accuracy(results[s]) for s in NON_WESTERN_SYSTEMS if s in results]
        
        if western:
            all_western.extend(western)
        if nonwestern:
            all_nonwestern.extend(nonwestern)
    
    if all_western:
        mean_w = sum(all_western) / len(all_western) * 100
        print(f"Western avg:     {mean_w:.1f}%")
    if all_nonwestern:
        mean_nw = sum(all_nonwestern) / len(all_nonwestern) * 100
        print(f"Non-Western avg: {mean_nw:.1f}%")
    if all_western and all_nonwestern:
        print(f"Gap:             {mean_w - mean_nw:+.1f}%")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze KinshipQA results and generate paper tables',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--results-dir', type=str, default='./',
                        help='Base directory containing result folders')
    parser.add_argument('--output-dir', type=str, default='./tables/',
                        help='Output directory for generated tables')
    parser.add_argument('--format', type=str, default='all',
                        choices=['console', 'csv', 'json', 'latex', 'all'],
                        help='Output format')
    
    args = parser.parse_args()
    
    # Load all results
    print("Loading results...")
    all_results = load_all_results(args.results_dir)
    
    if not all_results:
        print("ERROR: No results found!")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate tables
    print("\nGenerating tables...")
    
    table4 = generate_table4(all_results)
    table5 = generate_table5(all_results)
    table6 = generate_table6(all_results)
    table7 = generate_table7(all_results)
    
    tables = [
        ('table4', table4),
        ('table5', table5),
        ('table6', table6),
        ('table7', table7)
    ]
    
    # Output based on format
    if args.format in ['console', 'all']:
        for name, table in tables:
            print_table_console(table)
    
    if args.format in ['csv', 'all']:
        for name, table in tables:
            save_table_csv(table, output_dir / f'{name}.csv')
    
    if args.format in ['json', 'all']:
        for name, table in tables:
            save_table_json(table, output_dir / f'{name}.json')
    
    if args.format in ['latex', 'all']:
        # Generate LaTeX for Table 4 (main results)
        latex_content = generate_latex_table4(table4)
        latex_path = output_dir / 'table4.tex'
        with open(latex_path, 'w') as f:
            f.write(latex_content)
        print(f"Saved: {latex_path}")
    
    # Print summary
    print_summary(all_results)
    
    print("\n" + "=" * 80)
    print("DONE!")
    print(f"Tables saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
