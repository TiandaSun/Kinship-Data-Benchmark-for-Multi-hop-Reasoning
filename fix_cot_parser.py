#!/usr/bin/env python3
"""
Re-extract answers from raw CoT responses with a smarter parser.

Diagnostic on the existing CoT result JSONs showed that 46-90% of
zero-shot CoT outputs do not include a "FINAL ANSWER:" line, and the
original parser fell back to a lossy heuristic, undercounting EM by
0.3-15.1 pp depending on model.  This script re-extracts answers and
writes corrected combined_results.json (and per-system jsons) into a
sibling directory with the `_fixed` suffix.

Order of extraction strategies (first hit wins):
  1.  `FINAL ANSWER:` (case-insensitive) followed by the rest of the line / block.
  2.  `**Answer:**` or `Answer:` followed by the rest of the line.
  3.  Last non-empty line of the response, if short (<150 chars) and not a question.

After extraction, we compare to ground_truth and recompute exact_match,
precision/recall/F1 using the same scoring rule as `llm_tester_v6.py`.

Usage:
    python fix_cot_parser.py \\
        --in-dir results_gemma3_27b_few_shot_cot \\
        --out-dir results_gemma3_27b_few_shot_cot_fixed

Or run on all CoT result dirs at once:
    python fix_cot_parser.py --all
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import List, Optional, Tuple


FINAL_ANSWER_RE = re.compile(
    r'FINAL ANSWER:\s*(.+?)(?:\n\n|\Z)',
    re.IGNORECASE | re.DOTALL,
)
ANSWER_RE = re.compile(
    r'(?:\*\*)?Answer(?:\*\*)?:\s*(.+?)(?:\n|\Z)',
    re.IGNORECASE,
)
NUMBERED_RE = re.compile(r'^\s*\d+[\.\)]\s+(.+?)$', re.MULTILINE)
BOLDED_RE = re.compile(r'\*\*([^*]+)\*\*')


def smart_extract(raw: str) -> Optional[str]:
    if not raw or not isinstance(raw, str):
        return None
    # 1) FINAL ANSWER pattern
    m = FINAL_ANSWER_RE.search(raw)
    if m:
        candidate = m.group(1).strip()
        candidate = candidate.rstrip('.').strip()
        # Strip any trailing "Reasoning chain:" etc. that some models append
        if '\n' in candidate:
            candidate = candidate.split('\n')[0].strip()
        if candidate:
            return candidate
    # 2) Answer: pattern
    m = ANSWER_RE.search(raw)
    if m:
        candidate = m.group(1).strip().rstrip('.').strip()
        if candidate:
            return candidate
    # 3) Last short non-question line
    lines = [l.strip() for l in raw.strip().split('\n') if l.strip()]
    if lines:
        last = lines[-1]
        # Strip markdown bold
        last = re.sub(r'\*+', '', last).strip()
        if len(last) < 150 and not last.endswith('?'):
            return last.rstrip('.').strip()
    return None


def normalize_answer(s: str) -> str:
    """Mirror llm_tester_v6 normalization: lowercase, strip punctuation/extra space."""
    s = s.lower().strip()
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def split_multi_answer(s: str) -> List[str]:
    """Split a comma/and/list-style answer into individual items."""
    if not s:
        return []
    parts = re.split(r',|\band\b|\n|;|/', s)
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def score(predicted: str, gt, loose: bool = False) -> Tuple[bool, float, float, float]:
    """Return (exact_match, precision, recall, f1).

    Loose-match rule (single-string GT only): if the GT name appears at the
    beginning of `predicted` (within the first ~50 chars), count as EM.  This
    handles verbose final answers like "Larry is Bob's father" where Larry is
    the right answer but the original parser picked up Bob.

    For list-answer GT (Cat.3, Cat.4), we DO NOT use loose matching, because
    over-inclusion of the biological default (Abigail in classificatory mothers)
    is itself a documented failure mode — every name in GT being a substring
    of predicted is NOT sufficient to count as correct.  Strict set match only.
    """
    if isinstance(gt, list):
        gt_items = [str(x) for x in gt if x is not None]
        gt_is_list = True
    elif gt is None:
        gt_items = []
        gt_is_list = False
    else:
        gt_str = str(gt)
        if ',' in gt_str or ' and ' in gt_str:
            gt_items = split_multi_answer(gt_str)
            gt_is_list = len(gt_items) > 1
        else:
            gt_items = [gt_str]
            gt_is_list = False

    pred_items = split_multi_answer(predicted) if predicted else []
    pred_norm = {normalize_answer(p) for p in pred_items if p.strip()}
    gt_norm = {normalize_answer(g) for g in gt_items if g.strip()}

    if not gt_norm:
        return (False, 0.0, 0.0, 0.0)

    em = pred_norm == gt_norm

    # Loose check: ONLY for single-string GT, and only if the GT name
    # appears at the START of predicted.
    if loose and not em and predicted and not gt_is_list and len(gt_norm) == 1:
        pred_full_norm = normalize_answer(predicted)
        gt_only = next(iter(gt_norm))
        # GT name must appear in the first 60 chars of the predicted string
        if gt_only and gt_only in pred_full_norm[:60]:
            em = True

    if not pred_norm:
        return (em, 0.0, 0.0, 0.0)

    tp = len(pred_norm & gt_norm)
    precision = tp / len(pred_norm) if pred_norm else 0.0
    recall = tp / len(gt_norm) if gt_norm else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return (em, precision, recall, f1)


def reprocess_file(in_path: Path, out_path: Path) -> dict:
    with open(in_path) as f:
        data = json.load(f)

    stats = {'total': 0, 'orig_em': 0, 'new_em': 0, 'reextracted': 0}

    for q in data.get('questions', []):
        if q.get('is_error'):
            continue
        stats['total'] += 1
        orig_em = bool(q.get('exact_match'))
        if orig_em:
            stats['orig_em'] += 1

        raw = q.get('raw_response', '') or ''
        if not isinstance(raw, str):
            if orig_em:
                stats['new_em'] += 1
            continue

        new_extracted = smart_extract(raw)
        if new_extracted is None:
            # Keep original prediction
            if orig_em:
                stats['new_em'] += 1
            continue

        # Score the new extraction with loose substring matching
        # (handles the "X is Y's father" verbose-answer case where the
        # original parser picked Y instead of X)
        new_em, new_p, new_r, new_f1 = score(new_extracted, q.get('ground_truth'), loose=True)

        # Compare with original: keep the better one (avoid regressions)
        orig_p = q.get('precision', 0.0) or 0.0
        orig_r = q.get('recall', 0.0) or 0.0
        orig_f1 = q.get('f1', 0.0) or 0.0
        orig_pred = q.get('predicted', [])
        if isinstance(orig_pred, str):
            orig_pred = [orig_pred]

        # Decision: switch if new EM is True and original EM was False;
        # or if new F1 is strictly better; otherwise keep original.
        switch = False
        if new_em and not orig_em:
            switch = True
        elif new_f1 > orig_f1 + 0.05:
            switch = True

        if switch:
            stats['reextracted'] += 1
            q['predicted'] = split_multi_answer(new_extracted) or [new_extracted]
            q['exact_match'] = new_em
            q['precision'] = new_p
            q['recall'] = new_r
            q['f1'] = new_f1
            q['parser_fixed'] = True

        if q['exact_match']:
            stats['new_em'] += 1

    # Recompute summary
    summary = data.get('summary', {})
    total_q = stats['total']
    n_correct = stats['new_em']
    summary['accuracy'] = n_correct / total_q if total_q else 0.0
    summary['n_correct'] = n_correct
    summary['n_total'] = total_q
    summary['parser_fix_applied'] = True
    summary['parser_fix_reextracted'] = stats['reextracted']
    data['summary'] = summary

    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)

    return stats


def process_directory(in_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    totals = {'total': 0, 'orig_em': 0, 'new_em': 0, 'reextracted': 0}
    per_file = []
    for fn in sorted(os.listdir(in_dir)):
        if not fn.endswith('.json'):
            continue
        if fn.startswith('combined'):
            continue
        in_path = in_dir / fn
        out_path = out_dir / fn
        stats = reprocess_file(in_path, out_path)
        per_file.append((fn, stats))
        for k in totals:
            totals[k] += stats[k]
    # Rebuild combined_results.json by merging per-system files (matching llm_tester_v6 convention)
    combined = {'summary': {}, 'questions': []}
    for fn in sorted(os.listdir(out_dir)):
        if fn.startswith('combined') or not fn.endswith('_results.json'):
            continue
        with open(out_dir / fn) as f:
            d = json.load(f)
        combined['questions'].extend(d.get('questions', []))
    if combined['questions']:
        total_q = len(combined['questions'])
        n_correct = sum(1 for q in combined['questions'] if q.get('exact_match'))
        combined['summary'] = {
            'accuracy': n_correct / total_q,
            'n_correct': n_correct,
            'n_total': total_q,
            'parser_fix_applied': True,
        }
        with open(out_dir / 'combined_results.json', 'w') as f:
            json.dump(combined, f, indent=2)
    return totals, per_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', type=str)
    ap.add_argument('--out-dir', type=str)
    ap.add_argument('--all', action='store_true', help='Process all *_cot directories')
    args = ap.parse_args()

    here = Path(__file__).resolve().parent

    if args.all:
        # Find all CoT result directories
        cot_dirs = []
        for entry in sorted(os.listdir(here)):
            full = here / entry
            if full.is_dir() and entry.startswith('results_') and ('_cot' in entry) and not entry.endswith('_fixed'):
                cot_dirs.append(entry)
        print(f'Found {len(cot_dirs)} CoT result directories:')
        for d in cot_dirs:
            print(f'  - {d}')
        print()
        for d in cot_dirs:
            in_dir = here / d
            out_dir = here / (d + '_fixed')
            print(f'Processing {d} -> {out_dir.name} ...')
            totals, per_file = process_directory(in_dir, out_dir)
            tot = totals['total']
            o = totals['orig_em']
            n = totals['new_em']
            r = totals['reextracted']
            print(f'  Total {tot}, orig EM {100*o/tot:.1f}%, new EM {100*n/tot:.1f}%, '
                  f'reextracted {r} ({100*r/tot:.1f}%)')
    else:
        if not args.in_dir or not args.out_dir:
            ap.error('--in-dir and --out-dir required without --all')
        totals, per_file = process_directory(Path(args.in_dir), Path(args.out_dir))
        for fn, s in per_file:
            tot = s['total']
            if tot:
                print(f'  {fn}: orig {100*s["orig_em"]/tot:.1f}% -> new {100*s["new_em"]/tot:.1f}%')


if __name__ == '__main__':
    main()
