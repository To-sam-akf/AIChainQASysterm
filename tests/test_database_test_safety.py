from pathlib import Path

import pytest

from conftest import validate_test_database
from aika.llm_client import load_dotenv


def test_dotenv_cannot_override_offline_database_settings(tmp_path, monkeypatch):
    env_file = tmp_path / '.env'
    env_file.write_text('TEST_DATABASE_URL=postgresql://localhost/aika_test_danger\nDATABASE_URL=postgresql://localhost/production\n')
    monkeypatch.setenv('TEST_DATABASE_URL', '')
    monkeypatch.setenv('DATABASE_URL', '')
    load_dotenv(env_file)
    import os
    assert os.environ['TEST_DATABASE_URL'] == os.environ['DATABASE_URL'] == ''


@pytest.mark.parametrize('url', ['', 'postgresql://remote/aika_test_x', 'postgresql://localhost/prod',
    'postgresql://localhost/aika_test_x?host=remote', 'postgresql://localhost/aika_test_'])
def test_unsafe_targets_rejected(url):
    with pytest.raises(ValueError):
        validate_test_database(url, '')


def test_ordinary_database_target_rejected():
    with pytest.raises(ValueError):
        validate_test_database('postgresql://a@localhost/aika_test_x', 'postgresql://b@localhost:5432/aika_test_x')


def test_dedicated_local_url_accepted_without_connecting():
    assert validate_test_database('postgresql://localhost/aika_test_example', 'postgresql://localhost/production')


@pytest.mark.parametrize('ordinary', ['', 'postgresql://127.0.0.1/aika_test_example',
    'postgresql://[::1]/aika_test_example', 'invalid', 'postgresql://localhost/production?host=other'])
def test_uncertain_or_same_database_rejected(ordinary):
    with pytest.raises(ValueError):
        validate_test_database('postgresql://localhost/aika_test_example', ordinary)


def test_dotenv_cannot_authorize_destructive_test(tmp_path, monkeypatch):
    import os
    import shutil
    import subprocess
    import sys

    shutil.copy(Path(__file__).with_name('conftest.py'), tmp_path / 'conftest.py')
    (tmp_path / '.env').write_text('TEST_DATABASE_URL=postgresql://localhost/aika_test_from_dotenv\n')
    (tmp_path / 'test_gate.py').write_text(
        'import os\nfrom pathlib import Path\n'
        'for line in Path(__file__).with_name(".env").read_text().splitlines():\n'
        '    key, value = line.split("=", 1); os.environ.setdefault(key, value)\n'
        'def test_never_reaches_body(destructive_postgres_url):\n'
        '    raise AssertionError("must not reach a database")\n'
    )
    env = os.environ.copy()
    env.pop('TEST_DATABASE_URL', None)
    env.pop('DATABASE_URL', None)
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', str(tmp_path), '--allow-destructive-postgres'],
                            env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Unsafe test database' in result.stdout
    assert 'must not reach a database' not in result.stdout
