#!/usr/bin/env python
"""Installation script."""
import logging
import subprocess
import sys
import traceback as _tb

import setuptools
# inline:
# import git
# import polytope
# import tulip.spec.lexyacc


NAME = 'tulip'
VERSION_FILE = f'{NAME}/_version.py'
MAJOR = 1
MINOR = 4
MICRO = 1
VERSION = f'{MAJOR}.{MINOR}.{MICRO}'
VERSION_TEXT = (
    '# This file was generated from setup.py\n'
    "version = '{version}'\n")
logging.basicConfig(level=logging.WARNING)
_logger = logging.getLogger(__name__)


def git_version(
        version:
            str
        ) -> str:
    """Return version with local version identifier."""
    import git
    repo = git.Repo('.git')
    repo.git.status()
    sha = repo.head.commit.hexsha
    if repo.is_dirty():
        return f'{version}.dev0+{sha}.dirty'
    # commit is clean
    # is it release of `version` ?
    try:
        tag = repo.git.describe(
            match='v[0-9]*',
            exact_match=True,
            tags=True,
            dirty=True)
    except git.GitCommandError:
        return f'{version}.dev0+{sha}'
    assert tag == 'v' + version, (tag, version)
    return version


def run_setup() -> None:
    """Build parser, get version from `git`, install."""
    # Build PLY table, to be installed as tulip package data
    try:
        import tulip.spec.lexyacc
        tabmodule = tulip.spec.lexyacc.TABMODULE.split('.')[-1]
        outputdir = 'tulip/spec'
        parser = tulip.spec.lexyacc.Parser()
        parser.build(
            tabmodule,
            outputdir=outputdir,
            write_tables=True,
            debug=True,
            debuglog=_logger)
        import tulip.interfaces.ltl2ba
        tabmodule = tulip.interfaces.ltl2ba.TABMODULE.split('.')[-1]
        outputdir = 'tulip/interfaces'
        parser = tulip.interfaces.ltl2ba.Parser()
        parser.build(
            tabmodule,
            outputdir=outputdir,
            write_tables=True,
            debug=True,
            debuglog=_logger)
        plytable_build_failed = False
    except Exception as e:
        tb = ''.join(
            _tb.format_exception(e))
        print(
            f'Failed to build PLY tables, raising:\n{tb}')
        plytable_build_failed = True
    # version
    try:
        version = git_version(VERSION)
    except AssertionError:
        raise
    except Exception:
        print('No git info: Assume release.')
        version = VERSION
    s = VERSION_TEXT.format(version=version)
    with open(VERSION_FILE, 'w') as f:
        f.write(s)
    # setup
    setuptools.setup(
        name=NAME,
        version=version,
        package_dir={
            'tulip': 'tulip'})
    # ply failed ?
    if plytable_build_failed:
        print('!' * 65 +
              '    Failed to build PLY table.  ' +
              'Please run setup.py again.' +
              '!' * 65)


def install_cvxopt() -> None:
    """Install `cvxopt` version compatible with polytope requirements."""
    import polytope
    ver = polytope.__version__
    # Download all files for the current version of polytope
    subprocess.check_call([
        sys.executable, "-m",
        "pip", "download",
        "-d", "temp",
        "--no-deps",
        f"polytope=={ver}"])
    # Extract the tar archive
    subprocess.check_call([
        "tar", "xzf",
        f"temp/polytope-{ver}.tar.gz",
        "-C", "temp"])
    # Install cvxopt according to requirements file provided by polytope
    subprocess.check_call([
        sys.executable, "-m",
        "pip", "install",
        "-r", f"temp/polytope-{ver}/requirements/extras.txt"])


if __name__ == '__main__':
    run_setup()
