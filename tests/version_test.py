"""Test the management of `tulip.__version__`.

When testing out of source, first run `setup.py`
to generate the module `tulip._version`.
"""
import importlib
import os
import os.path
import shutil
import subprocess
import sys
import tempfile

import packaging.version
import pytest
import tulip
import tulip._version


try:
    subprocess.check_output(['git', 'version'])
    no_git = False
except FileNotFoundError:
    no_git = True


def test_tulip_has_pep440_version():
    """Check that `tulip.__version__` complies to PEP440."""
    version = tulip.__version__
    assert version is not None, version
    version_ = tulip._version.version
    assert version == version_, (version, version_)
    assert_pep440(version)


def assert_pep440(version):
    """Raise `AssertionError` if `version` violates PEP440."""
    v = packaging.version.parse(version)
    assert isinstance(v, packaging.version.Version), v


@pytest.mark.skipif(no_git, reason='requires Git installation')
def test_git_version():
    # Create temporary Git repository with 1 file and 1 commit
    repo_dir = tempfile.mkdtemp()
    subprocess.check_call(['git', 'init'], cwd=repo_dir)
    subprocess.check_call(['git', 'config', 'user.name', 'Gregor Samsa'], cwd=repo_dir)
    subprocess.check_call(['git', 'config', 'user.email', 'gsamsa@example.com'], cwd=repo_dir)
    with open(os.path.join(repo_dir, 'data.txt'), 'wt') as fp:
        fp.write('text')
    subprocess.check_call(['git', 'add', 'data.txt'], cwd=repo_dir)
    subprocess.check_call(['git', 'commit', '-m', 'init repo'], cwd=repo_dir)

    # Import setup.py to call git_version()
    path = os.path.realpath(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)  # parent dir
    path = os.path.join(path, 'setup.py')
    module_spec = importlib.util.spec_from_file_location('setup', path)
    setup = importlib.util.module_from_spec(module_spec)
    sys.modules['setup'] = setup
    module_spec.loader.exec_module(setup)

    # With no tags, this should return version string with commit suffix
    pwd = os.getcwd()
    os.chdir(repo_dir)
    vstring = setup.git_version('1.2.3')
    os.chdir(pwd)

    expected_prefix = '1.2.3.dev0+'
    assert vstring.startswith(expected_prefix)

    # SHA-1 hash has length of 40
    assert len(vstring) == len(expected_prefix) + 40

    # Add tag to test case of no commit suffix
    subprocess.check_call(['git', 'tag', 'v1.2.3'], cwd=repo_dir)

    pwd = os.getcwd()
    os.chdir(repo_dir)
    vstring = setup.git_version('1.2.3')
    os.chdir(pwd)

    assert vstring == '1.2.3'

    # Add an untracked file, which should result in "dirty" suffix
    with open(os.path.join(repo_dir, 'untracked.txt'), 'wt') as fp:
        fp.write('more text')

    pwd = os.getcwd()
    os.chdir(repo_dir)
    vstring = setup.git_version('1.2.3')
    os.chdir(pwd)

    assert vstring.endswith('.dirty')
    assert len(vstring) == len(expected_prefix) + 40 + len('.dirty')

    # Delete temporary directory
    shutil.rmtree(repo_dir)


if __name__ == '__main__':
    test_git_version()
