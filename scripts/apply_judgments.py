#!/usr/bin/env python3
"""Apply auto-judgments to an evaluation dataset, producing an augmented version.

Usage:
    # Dry-run: preview changes without writing
    python scripts/apply_judgments.py \\
        --judgments data/eval_runs/auto_judgments.jsonl \\
        --dataset data/eval/rag_retrieval_v1.jsonl \\
        --dry-run

    # Apply and write new dataset
    python scripts/apply_judgments.py \\
        --judgments data/eval_runs/auto_judgments.jsonl \\
        --dataset data/eval/rag_retrieval_v1.jsonl \\
        --output data/eval/rag_retrieval_v2.jsonl \\
        --backup
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.eval.auto_judge import AutoJudgment, load_judgments  # noqa: E402
from src.eval.rag_dataset import (  # noqa: E402
    DEFAULT_RAG_RETRIEVAL_BENCHMARK,
    RagRetrievalCase,
    load_rag_retrieval_cases,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply LLM auto-judgments to an evaluation dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--judgments", "-j",
        default="data/eval_runs/auto_judgments.jsonl",
        help="Path to auto_judgments JSONL (from auto_judge_review.py)",
    )
    parser.add_argument(
        "--dataset", "-d",
        default=str(DEFAULT_RAG_RETRIEVAL_BENCHMARK),
        help="Path to the evaluation dataset JSONL to augment",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/eval/rag_retrieval_v2.jsonl",
        help="Path to write the augmented dataset JSONL",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create a timestamped backup of the original dataset before overwriting",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the original dataset (--dataset) instead of writing --output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing any files",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Only apply judgments with confidence >= this threshold (default: 0.0 = all)",
    )
    parser.add_argument(
        "--skip-needs-review",
        action="store_true",
        help="Skip judgments marked needs_review (apply only high-confidence ones)",
    )
    return parser


def load_original_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load raw JSONL lines as dicts (preserving original structure)."""
    lines: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            lines.append(json.loads(line))
    return lines


def apply_judgments_to_dataset(
    original_lines: list[dict[str, Any]],
    cases: list[RagRetrievalCase],
    judgments: list[AutoJudgment],
    *,
    min_confidence: float = 0.0,
    skip_needs_review: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply judgments to the dataset, returning new JSONL lines + stats.

    Rules:
    - grade=0, confidence>=min → add to judged_irrelevant_chunk_ids
    - grade>=1, matched_unit_id given, confidence>=min → add to unit's alternatives
    - grade>=1, matched_unit_id empty → treated as grade=0 (no unit to match)
    - If matched_unit_id doesn't exist in the case → create new unit
    - When skip_needs_review=True, judgments with needs_review are ignored
    """

    # Index cases by case_id
    case_map: dict[str, RagRetrievalCase] = {case.case_id: case for case in cases}

    # Build index: (case_id, chunk_id) → judgment
    judgment_map: dict[tuple[str, str], AutoJudgment] = {}
    for j in judgments:
        key = (j.case_id, j.chunk_id)
        if key in judgment_map:
            # Keep the higher-confidence judgment
            if j.confidence > judgment_map[key].confidence:
                judgment_map[key] = j
        else:
            judgment_map[key] = j

    # Group judgments by case_id for application
    by_case: dict[str, list[AutoJudgment]] = defaultdict(list)
    for j in judgment_map.values():
        by_case[j.case_id].append(j)

    # Build case_id -> line index map
    case_line_index: dict[str, int] = {}
    for idx, line in enumerate(original_lines):
        cid = str(line.get("case_id", ""))
        if cid:
            case_line_index[cid] = idx

    # Prepare output
    new_lines = deepcopy(original_lines)
    stats = {
        "total_judgments": len(judgments),
        "applied_irrelevant": 0,
        "applied_grade1": 0,
        "applied_grade2": 0,
        "created_units": 0,
        "skipped_low_confidence": 0,
        "skipped_needs_review": 0,
        "skipped_invalid": 0,
        "cases_modified": 0,
    }

    for case_id, case_judgments in sorted(by_case.items()):
        case = case_map.get(case_id)
        if case is None:
            continue
        line_idx = case_line_index.get(case_id)
        if line_idx is None:
            continue

        line = new_lines[line_idx]
        case_modified = False

        # Current evidence_units indexed by unit_id
        units_by_id: dict[str, dict[str, Any]] = {}
        for unit in line.get("evidence_units", []):
            uid = unit.get("unit_id", "")
            if uid:
                units_by_id[uid] = unit

        for j in case_judgments:
            # Confidence filter
            if j.confidence < min_confidence:
                stats["skipped_low_confidence"] += 1
                continue

            # Needs-review filter
            if skip_needs_review and j.needs_review:
                stats["skipped_needs_review"] += 1
                continue

            if j.grade == 0:
                # Add to judged_irrelevant_chunk_ids
                if j.chunk_id not in line.setdefault("judged_irrelevant_chunk_ids", []):
                    line["judged_irrelevant_chunk_ids"].append(j.chunk_id)
                    stats["applied_irrelevant"] += 1
                    case_modified = True

            elif j.grade in {1, 2}:
                unit_id = j.matched_unit_id.strip()
                if not unit_id:
                    # grade≥1 but no unit matched — treat as irrelevant
                    if j.chunk_id not in line.setdefault("judged_irrelevant_chunk_ids", []):
                        line["judged_irrelevant_chunk_ids"].append(j.chunk_id)
                        stats["applied_irrelevant"] += 1
                        case_modified = True
                    continue

                if unit_id not in units_by_id:
                    # Create new evidence unit
                    # Derive description from the judgment reasoning
                    desc = build_unit_description(j, case)
                    new_unit = {
                        "unit_id": unit_id,
                        "required": False,
                        "description": desc,
                        "alternatives": [],
                    }
                    line.setdefault("evidence_units", []).append(new_unit)
                    units_by_id[unit_id] = new_unit
                    stats["created_units"] += 1

                unit = units_by_id[unit_id]
                alternatives = unit.setdefault("alternatives", [])
                existing_ids = {alt.get("chunk_id") for alt in alternatives if isinstance(alt, dict)}
                if j.chunk_id not in existing_ids:
                    alternatives.append({
                        "chunk_id": j.chunk_id,
                        "grade": j.grade,
                    })
                    if j.grade == 2:
                        stats["applied_grade2"] += 1
                    else:
                        stats["applied_grade1"] += 1
                    case_modified = True
            else:
                stats["skipped_invalid"] += 1

        if case_modified:
            # Update annotation status
            if line.get("annotation_status") in {"draft", "", None}:
                line["annotation_status"] = "llm_augmented"
            stats["cases_modified"] += 1

    return new_lines, stats


def build_unit_description(judgment: AutoJudgment, case: RagRetrievalCase) -> str:
    """Generate a meaningful description for an auto-created evidence unit."""
    if judgment.reasoning:
        return f"[LLM auto-judged] {judgment.reasoning}"
    return f"[LLM auto-judged] Related evidence for: {case.question}"


def validate_output(lines: list[dict[str, Any]]) -> list[str]:
    """Basic validation of output lines against dataset constraints."""
    errors: list[str] = []
    for idx, line in enumerate(lines, start=1):
        # Check required fields
        for field in {"case_id", "split", "category", "question", "filters", "evidence_units"}:
            if field not in line:
                errors.append(f"line {idx}: missing required field '{field}'")

        # Check evidence_units structure
        units = line.get("evidence_units", [])
        if not isinstance(units, list) or not units:
            errors.append(f"line {idx}: evidence_units must be a non-empty list")
            continue

        unit_ids = []
        for ui, unit in enumerate(units):
            if not isinstance(unit, dict):
                errors.append(f"line {idx}.evidence_units[{ui}]: must be an object")
                continue
            uid = unit.get("unit_id", "")
            if not uid:
                errors.append(f"line {idx}.evidence_units[{ui}]: missing unit_id")
            unit_ids.append(uid)

            alts = unit.get("alternatives", [])
            if not isinstance(alts, list):
                errors.append(f"line {idx}.evidence_units[{ui}]: alternatives must be a list")
                continue
            for ai, alt in enumerate(alts):
                if not isinstance(alt, dict):
                    errors.append(f"line {idx}.evidence_units[{ui}].alternatives[{ai}]: must be an object")
                    continue
                if not alt.get("chunk_id"):
                    errors.append(f"line {idx}.evidence_units[{ui}].alternatives[{ai}]: missing chunk_id")

    return errors


def main() -> None:
    args = build_parser().parse_args()
    judgments_path = Path(args.judgments)
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)

    # Load judgments
    print(f"Loading judgments: {judgments_path}")
    try:
        judgments = load_judgments(judgments_path)
    except FileNotFoundError:
        print(f"ERROR: judgments file not found: {judgments_path}")
        print("Run scripts/auto_judge_review.py first.")
        sys.exit(1)
    print(f"  total judgments: {len(judgments)}")

    if not judgments:
        print("No judgments to apply. Exiting.")
        return

    # Load cases (for validation and unit lookup)
    print(f"Loading dataset: {dataset_path}")
    cases = load_rag_retrieval_cases(dataset_path)
    print(f"  cases: {len(cases)}")

    # Load original JSONL lines
    original_lines = load_original_jsonl(dataset_path)
    print(f"  lines: {len(original_lines)}")

    # Summary of judgments before applying
    grade_counts = {0: 0, 1: 0, 2: 0}
    needs_review_count = 0
    for j in judgments:
        grade_counts[j.grade] = grade_counts.get(j.grade, 0) + 1
        if j.needs_review:
            needs_review_count += 1
    print(f"\nJudgment breakdown:")
    print(f"  grade=0: {grade_counts[0]}")
    print(f"  grade=1: {grade_counts[1]}")
    print(f"  grade=2: {grade_counts[2]}")
    print(f"  needs_review: {needs_review_count}")

    if args.skip_needs_review:
        print(f"  -> Skipping {needs_review_count} needs_review judgments")
    if args.min_confidence > 0:
        print(f"  -> Minimum confidence threshold: {args.min_confidence}")

    # Apply
    print(f"\nApplying judgments...")
    new_lines, stats = apply_judgments_to_dataset(
        original_lines,
        cases,
        judgments,
        min_confidence=args.min_confidence,
        skip_needs_review=args.skip_needs_review,
    )

    print(f"\nApplication stats:")
    print(f"  applied as irrelevant (grade=0):  {stats['applied_irrelevant']:5d}")
    print(f"  applied as grade=1 (partial):     {stats['applied_grade1']:5d}")
    print(f"  applied as grade=2 (direct):      {stats['applied_grade2']:5d}")
    print(f"  new evidence_units created:        {stats['created_units']:5d}")
    print(f"  cases modified:                    {stats['cases_modified']:5d}")
    print(f"  skipped (low confidence):          {stats['skipped_low_confidence']:5d}")
    print(f"  skipped (needs_review):            {stats['skipped_needs_review']:5d}")
    print(f"  skipped (invalid):                 {stats['skipped_invalid']:5d}")

    # Validate output
    print(f"\nValidating output...")
    errors = validate_output(new_lines)
    if errors:
        for err in errors[:20]:
            print(f"  VALIDATION ERROR: {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors")
        print(f"\nERROR: {len(errors)} validation failures. Aborting.")
        sys.exit(1)
    print(f"  validation: OK")

    if args.dry_run:
        print(f"\n[Dry-run mode] No files written.")
        print(f"Would write {len(new_lines)} lines to {'dataset (in-place)' if args.in_place else output_path}")

        # Show a sample of what changed
        changed = 0
        for orig_line, new_line in zip(original_lines, new_lines):
            if orig_line != new_line:
                changed += 1
        print(f"  lines with changes: {changed}")
        return

    # Write output
    write_path = dataset_path if args.in_place else output_path

    if args.backup and write_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = write_path.with_suffix(f".backup_{timestamp}.jsonl")
        shutil.copy2(write_path, backup_path)
        print(f"\nBackup created: {backup_path}")

    write_path.parent.mkdir(parents=True, exist_ok=True)
    with write_path.open("w", encoding="utf-8") as fh:
        for line in new_lines:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"\nAugmented dataset written: {write_path}")
    print(f"  lines: {len(new_lines)}")

    # Show new annotation status distribution
    statuses: dict[str, int] = defaultdict(int)
    for line in new_lines:
        statuses[line.get("annotation_status", "draft")] += 1
    print(f"  annotation_statuses: {dict(sorted(statuses.items()))}")

    # Next steps
    print(f"\nNext: re-run evaluation with the augmented dataset:")
    print(f"  python -m src.eval.rag_runner \\")
    print(f"    --benchmark {write_path} \\")
    print(f"    --retrievers bm25 semantic rrf")


if __name__ == "__main__":
    main()