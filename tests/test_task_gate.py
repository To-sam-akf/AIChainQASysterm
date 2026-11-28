import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('task_gate', Path(__file__).resolve().parents[1] / 'docs/project-tracking/task_gate.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def valid():
    return {'features': [{'id': 'F1', 'dependencies': []}], 'tasks': [
        {'id': 'T1', 'dependencies': [], 'status': 'done', 'history': ['planned', 'in_progress', 'done'],
         'acceptance_contract': 'contract', 'evidence': 'passed', 'verification_command': 'pytest'},
        {'id': 'T2', 'dependencies': ['T1'], 'status': 'in_progress', 'history': ['planned', 'in_progress'],
         'acceptance_contract': 'contract'}]}


def test_valid_inventory():
    assert gate.validate(valid()) == []


def test_duplicate_unknown_and_cycle():
    d = valid(); d['features'].append({'id': 'F1', 'dependencies': ['missing']})
    assert any('unique' in e for e in gate.validate(d))
    assert any('unknown' in e for e in gate.validate(d))
    d = valid(); d['tasks'][0]['dependencies'] = ['T2']
    assert any('cycle' in e for e in gate.validate(d))


def test_wip_transition_contract_evidence_blocker_and_dependencies():
    d = valid(); d['tasks'][0].update(status='in_progress', history=['planned', 'in_progress'])
    assert any('WIP' in e for e in gate.validate(d))
    assert any('unfinished dependency' in e for e in gate.validate(d))
    d = valid(); d['tasks'][0]['history'] = ['planned', 'done']; d['tasks'][0]['evidence'] = ''
    assert any('transition' in e for e in gate.validate(d))
    assert any('evidence' in e for e in gate.validate(d))
    d = valid(); d['tasks'][1].update(status='blocked', history=['planned', 'blocked'], acceptance_contract='')
    errors = gate.validate(d)
    assert any('blocker' in e for e in errors)
    assert any('contract' in e for e in errors)
