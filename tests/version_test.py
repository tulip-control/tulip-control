"""Test the management of `tulip.__version__`.

When testing out of source, first run `setup.py`
to generate the module `tulip._version`.
"""
import importlib
import os
import os.path
import sys
import unittest.mock as mock

import git
import pkg_resources
from pkg_resources.extern import packaging
import pytest
import tulip
import tulip._version


def test_tulip_has_pep440_version():
    """Check that `tulip.__version__` complies to PEP440."""
    version = tulip.__version__
    assert version is not None, version
    version_ = tulip._version.version
    assert version == version_, (version, version_)
    assert_pep440(version)


@mock.patch('git.Repo')
def test_git_version(mock_repo):
    """Mock `git` repository for testing `setup.git_version`."""
    path = os.path.realpath(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)  # parent dir
    path = os.path.join(path, 'setup.py')
    module_spec = importlib.util.spec_from_file_location('setup', path)
    setup = importlib.util.module_from_spec(module_spec)
    sys.modules['setup'] = setup
    module_spec.loader.exec_module(setup)
    # mocking
    version = '0.1.2'
    instance = mock_repo.return_value
    instance.head.commit.hexsha = '0123'
    # dirty repo
    v = setup.git_version(version)
    assert_pep440(v)
    assert 'dev' in v, v
    assert 'dirty' in v, v
    # not dirty, not tagged
    instance.is_dirty.return_value = False
    instance.git.describe.side_effect = git.GitCommandError('0', 0)
    v = setup.git_version(version)
    assert_pep440(v)
    assert 'dev' in v, v
    assert 'dirty' not in v, v
    # tagged as version that matches `setup.py`
    instance.git.describe.side_effect = None
    instance.git.describe.return_value = 'v0.1.2'
    v = setup.git_version(version)
    assert_pep440(v)
    assert v == '0.1.2', v
    # tagged as wrong version
    instance.git.describe.return_value = 'v0.1.3'
    with pytest.raises(AssertionError):
        setup.git_version(version)
    # release: no repo
    mock_repo.side_effect = Exception('no repo found')
    with pytest.raises(Exception):
        setup.git_version(version)


def assert_pep440(version):
    """Raise `AssertionError` if `version` violates PEP440."""
    v = pkg_resources.parse_version(version)
    assert isinstance(v, packaging.version.Version), v


if __name__ == '__main__':
    test_git_version()
