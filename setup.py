#!/usr/bin/env python
"""Installation script."""
import logging
import subprocess
import sys

from setuptools import setup
# inline:
# import git
# import polytope
# import tulip.spec.lexyacc


NAME = 'tulip'
PROJECT_URLS = {
    'Bug Tracker': 'https://github.com/tulip-control/tulip-control/issues',
    'Documentation': 'https://tulip-control.sourceforge.io/doc/',
    'API Documentation': 'https://tulip-control.sourceforge.io/api-doc/',
    'Source Code': 'https://github.com/tulip-control/tulip-control'}
VERSION_FILE = f'{NAME}/_version.py'
MAJOR = 1
MINOR = 4
MICRO = 0
VERSION = f'{MAJOR}.{MINOR}.{MICRO}'
VERSION_TEXT = (
    '# This file was generated from setup.py\n'
    "version = '{version}'\n")
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
classifiers = [
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3 :: Only',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Topic :: Scientific/Engineering']
package_data = {
    'tulip.spec': ['parsetab.py']}


def git_version(version):
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


def run_setup():
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
            debuglog=logger)
        import tulip.interfaces.ltl2ba
        tabmodule = tulip.interfaces.ltl2ba.TABMODULE.split('.')[-1]
        outputdir = 'tulip/interfaces'
        parser = tulip.interfaces.ltl2ba.Parser()
        parser.build(
            tabmodule,
            outputdir=outputdir,
            write_tables=True,
            debug=True,
            debuglog=logger)
        plytable_build_failed = False
    except Exception as e:
        print(f'Failed to build PLY tables: {e}')
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
    setup(
        name=NAME,
        version=version,
        description='Temporal Logic Planning (TuLiP) Toolbox',
        author='Caltech Control and Dynamical Systems',
        author_email='tulip@tulip-control.org',
        url='http://tulip-control.org',
        project_urls=PROJECT_URLS,
        license='BSD',
        classifiers=classifiers,
        python_requires='>=3.7',
        install_requires=[
            'graphviz >= 0.20',
            'networkx >= 2.0',
            'numpy >= 1.7',
            'omega >= 0.3.1, < 0.4.0',
            'ply >= 3.4, <= 3.10',
            'polytope >= 0.2.1',
            'scipy'],
        tests_require=[
            'matplotlib >= 2.0.0',
            'gr1py >= 0.2.0',
            'mock',
            'pytest',
            'setuptools >= 39.0.0'],
        packages=[
            'tulip',
            'tulip.transys',
            'tulip.transys.export',
            'tulip.abstract',
            'tulip.spec',
            'tulip.interfaces'],
        package_dir={
            'tulip': 'tulip'},
        package_data=package_data)
    # ply failed ?
    if plytable_build_failed:
        print('!' * 65 +
              '    Failed to build PLY table.  ' +
              'Please run setup.py again.' +
              '!' * 65)


def install_cvxopt():
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
