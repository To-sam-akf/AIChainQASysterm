"""Validate feature inventory and the separate lightweight task lifecycle."""
import json
import sys
from pathlib import Path

TRANSITIONS = {'planned': {'in_progress', 'blocked'}, 'in_progress': {'done', 'blocked'},
               'blocked': {'planned', 'in_progress'}, 'done': set()}


def validate(data):
    errors = []
    features, tasks = data.get('features', []), data.get('tasks', [])
    all_items = features + tasks
    ids = [item.get('id') for item in all_items]
    if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        errors.append('IDs must be nonempty and globally unique')
    for label, items in [('feature', features), ('task', tasks)]:
        lookup = {item.get('id'): item for item in items}
        visiting, visited = set(), set()
        def visit(node):
            if node in visiting:
                errors.append(f'{label} dependency cycle: {node}')
                return
            if node in visited:
                return
            visiting.add(node)
            for dep in lookup[node].get('dependencies', []):
                if dep not in lookup:
                    errors.append(f'{label} {node}: unknown dependency {dep}')
                else:
                    visit(dep)
            visiting.remove(node)
            visited.add(node)
        for node in lookup:
            visit(node)
    if sum(t.get('status') == 'in_progress' for t in tasks) > 1:
        errors.append('WIP exceeds one')
    task_by_id = {t.get('id'): t for t in tasks}
    for task in tasks:
        ident, status = task.get('id'), task.get('status')
        history = task.get('history', [])
        if not history or history[0] != 'planned' or history[-1] != status:
            errors.append(f'{ident}: invalid history')
        if any(b not in TRANSITIONS.get(a, set()) for a, b in zip(history, history[1:])):
            errors.append(f'{ident}: illegal transition')
        if status not in TRANSITIONS:
            errors.append(f'{ident}: invalid status')
        if not str(task.get('acceptance_contract', '')).strip():
            errors.append(f'{ident}: missing acceptance contract')
        if status == 'blocked' and not str(task.get('blocker', '')).strip():
            errors.append(f'{ident}: missing blocker')
        if status == 'done' and (not str(task.get('evidence', '')).strip() or not str(task.get('verification_command', '')).strip()):
            errors.append(f'{ident}: missing acceptance evidence')
        if status in {'in_progress', 'done'}:
            for dep in task.get('dependencies', []):
                if dep in task_by_id and task_by_id[dep].get('status') != 'done':
                    errors.append(f'{ident}: unfinished dependency {dep}')
    return errors


def main():
    try:
        data = json.loads(Path(__file__).with_name('features.json').read_text(encoding='utf-8'))
        errors = validate(data)
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        errors = [f'invalid inventory: {exc}']
    for error in errors:
        print(error, file=sys.stderr)
    if not errors:
        print('Task gate: OK')
    return int(bool(errors))


if __name__ == '__main__':
    sys.exit(main())
