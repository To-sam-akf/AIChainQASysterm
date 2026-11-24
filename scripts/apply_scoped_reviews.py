"""Apply attributed partial reviews into a new, isolated score snapshot.

Set AIQASYS_FROZEN_ROOT to the extracted source archive of the input scoring
experiment. This command never writes into the original scoring directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
frozen = os.environ.get('AIQASYS_FROZEN_ROOT')
if not frozen:
    raise SystemExit('Set AIQASYS_FROZEN_ROOT to the source experiment frozen code directory')
sys.path.insert(0, frozen)
from aika.eval.answer_judge import apply_coverage_review, validate_judgment
from aika.eval.benchmark_v2 import load_benchmark_v2
from aika.eval.experiment_artifacts import digest
from aika.eval.faithfulness import is_pending, score_metrics
from aika.eval.score_runner import judging_code_hash

spec = importlib.util.spec_from_file_location('aiqasys_scoped_review', ROOT / 'aika/eval/scoped_review.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply(source: Path, review_file: Path, destination: Path):
    source, review_file, destination = source.resolve(), review_file.resolve(), destination.resolve()
    if destination == source or source in destination.parents or destination.exists():
        raise ValueError('Destination must be a new directory outside the original score directory')
    manifest = json.loads((source / 'manifest.json').read_text())
    if judging_code_hash() != manifest['code_hash'] or sha(source / 'source.zip') != manifest['code']['source_sha256']:
        raise ValueError('Frozen scoring code or source archive does not match the source experiment')
    generation = Path(manifest['generation_dir'])
    source_paths = [source / 'scores.jsonl', source / 'human_reviews.jsonl', source / 'review_packet.jsonl',
                    generation / 'results.jsonl', generation / 'benchmark.jsonl', source / 'manifest.json', source / 'source.zip']
    hashes = {str(path): sha(path) for path in source_paths}
    review_hash = sha(review_file)
    rows = [json.loads(line) for line in source_paths[0].read_text().splitlines() if line.strip()]
    by_id = {r['answer_id']: r for r in rows}
    packets = {r['answer_id']: r['input'] for r in map(json.loads, source_paths[2].read_text().splitlines())}
    generated = {(r['case_id'], r['variant']): r for r in map(json.loads, source_paths[3].read_text().splitlines())}
    cases = {c.case_id: c for c in load_benchmark_v2(source_paths[4], require_human_review=True)}
    reviews = [module.ScopedReview.model_validate_json(line) for line in review_file.read_text().splitlines() if line.strip()]
    if not reviews or len(by_id) != len(rows):
        raise ValueError('Review file must be nonempty and answer IDs must be unique')
    before_full = sum(bool(row.get('human_review')) for row in rows)
    for review in reviews:
        row = by_id.get(review.answer_id)
        if row is None:
            raise ValueError('Scoped review targets an unknown answer')
        key = (row['case_id'], row['variant'])
        by_id[review.answer_id] = module.apply_scoped_review(
            row, review, case=cases[key[0]], generation=generated[key], payload=packets[review.answer_id],
            digest=digest, apply_coverage_review=apply_coverage_review, validate_judgment=validate_judgment,
            score_metrics=score_metrics, is_pending=is_pending)
    result = [by_id[row['answer_id']] for row in rows]
    if sum(bool(row.get('human_review')) for row in result) != before_full:
        raise ValueError('A partial review unexpectedly changed the full-review count')
    if any(sha(Path(path)) != h for path, h in hashes.items()) or sha(review_file) != review_hash:
        raise ValueError('Source data changed while applying partial reviews')
    summary = {'status': 'partial_review_snapshot', 'source_scoring_experiment': str(source),
               'source_hashes': hashes, 'source_code_hash': manifest['code_hash'],
               'scoped_review_input_sha256': review_hash, 'scoped_review_records': len(reviews),
               'scoped_workflow_code_sha256': {'apply_scoped_reviews.py': sha(Path(__file__)),
                                               'scoped_review.py': sha(ROOT / 'aika/eval/scoped_review.py')},
               'answers_with_scoped_reviews': len({r.answer_id for r in reviews}),
               'full_human_reviews_preserved': before_full, 'pending_answers': sum(bool(r['pending']) for r in result),
               'ready_for_product_conclusions': False, 'original_score_unchanged': True,
               'new_model_calls': 0, 'notes': '局部审核不等于整份答案的人工审核；需重新评估校准、版本分歧及正式结论。'}
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='.scoped-review-', dir=destination.parent))
    try:
        (tmp / 'scores.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in result))
        (tmp / 'scoped_reviews.jsonl').write_text(''.join(json.dumps(r.model_dump(mode='json'), ensure_ascii=False) + '\n' for r in reviews))
        (tmp / 'manifest.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
        os.rename(tmp, destination)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--reviews', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(apply(args.source, args.reviews, args.output), ensure_ascii=False))
