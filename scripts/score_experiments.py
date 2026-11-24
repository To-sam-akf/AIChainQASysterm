"""Independently judge saved experiment outputs without regenerating answers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from aika.eval.score_runner import run_scoring


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="existing generation run directory")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "data/eval_runs/scores")
    parser.add_argument("--resume", type=Path, help="existing scoring directory")
    parser.add_argument("--rules-only", action="store_true", default=None, help="no API calls; semantic items stay pending")
    parser.add_argument("--judge-model", help="override JUDGE_MODEL / LLM_MODEL")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-errors", action="store_true", help="retry only failed judgments, retaining prior calls and costs")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--calibration-from", type=Path, help="reuse eight unchanged answers selected in an earlier scoring run; never reuse approvals")
    parser.add_argument("--import-reviews", type=Path, help="attributed human review JSONL copied from the generated template")
    parser.add_argument("--pricing", type=Path, help="optional operator-supplied provider rates and attribution")
    args = parser.parse_args(argv)
    try:
        result = run_scoring(
            args.run_dir, output_dir=args.output_dir, resume=args.resume, rules_only=args.rules_only,
            model=args.judge_model, dry_run=args.dry_run, retry_errors=args.retry_errors, seed=args.seed,
            reviews_path=args.import_reviews, prices=json.loads(args.pricing.read_text()) if args.pricing else None,
            calibration_from=args.calibration_from,
        )
    except (ValueError, OSError, KeyError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    fields = ("run_id", "output_dir", "status", "assessed", "planned_assessments", "pending_answers", "judge_errors",
              "calibration_ready", "ready_for_product_conclusions", "dry_run")
    print(json.dumps({key: result[key] for key in fields if key in result}, ensure_ascii=False, indent=2))
    return 1 if result.get("judge_errors") or result.get("status") == "interrupted" else 0


if __name__ == "__main__":
    raise SystemExit(main())
