"""Test the management of `tulip.__version__`.

When testing out of source, first run `setup.py`
to generate the module `tulip._version`.
"""
import importlib
import os
import os.path
import sys
import unittest.mock as mock

import packaging
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


def assert_pep440(version):
    """Raise `AssertionError` if `version` violates PEP440."""
    v = packaging.version.parse(version)
    assert isinstance(v, packaging.version.Version), v


if __name__ == '__main__':
    test_git_version()
