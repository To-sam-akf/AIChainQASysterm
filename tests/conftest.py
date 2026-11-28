"""Safety gates for tests that can mutate a database."""
import os
from urllib.parse import urlsplit, unquote

import pytest


def pytest_addoption(parser):
    parser.addoption('--allow-destructive-postgres', action='store_true', default=False,
                     help='Enable destructive tests against an isolated local test database')


def pytest_configure(config):
    # Snapshot before application code can load .env.
    config._aika_test_database_url = os.environ.get('TEST_DATABASE_URL', '').strip()
    config._aika_database_url = os.environ.get('DATABASE_URL', '').strip()
    if not config.getoption('--allow-destructive-postgres'):
        os.environ['TEST_DATABASE_URL'] = ''
        os.environ['DATABASE_URL'] = ''
        os.environ['QA_DISABLE_POSTGRES'] = 'true'


def validate_test_database(url, ordinary_url):
    try:
        parsed = urlsplit(url)
        name = unquote(parsed.path.lstrip('/'))
        if (parsed.scheme not in {'postgres', 'postgresql'} or
                parsed.hostname not in {'localhost', '127.0.0.1', '::1'} or
                parsed.query or parsed.fragment or '/' in name or
                not name.startswith('aika_test_') or len(name) <= len('aika_test_')):
            raise ValueError
        ordinary = urlsplit(ordinary_url)
        ordinary_name = unquote(ordinary.path.lstrip('/'))
        if (ordinary.scheme not in {'postgres', 'postgresql'} or not ordinary.hostname or
                not ordinary_name or '/' in ordinary_name or ordinary.query or ordinary.fragment):
            raise ValueError
        # Hostnames can resolve to the same server, including localhost and loopback.
        # A different database name is the only isolation we can establish here.
        if ordinary_name == name:
            raise ValueError
    except ValueError:
        raise ValueError('Unsafe test database: require a dedicated local aika_test_* database and an explicit, different ordinary DATABASE_URL without URL options') from None
    return url


@pytest.fixture
def destructive_postgres_url(request):
    if not request.config.getoption('--allow-destructive-postgres'):
        pytest.skip('destructive PostgreSQL test requires --allow-destructive-postgres')
    try:
        return validate_test_database(request.config._aika_test_database_url, request.config._aika_database_url)
    except ValueError as exc:
        pytest.fail(str(exc), pytrace=False)
