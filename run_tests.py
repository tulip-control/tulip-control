#!/usr/bin/env python
"""Driver script for testing TuLiP.  Try calling it with "-h" flag."""
from __future__ import print_function
import argparse
import importlib
import os
import pprint
import sys

import pytest


BASE = [
    'dumpsmach_test',
    'form_test',
    'gr1_test',
    'gridworld_test',
    'mathfunc_test',
    'mdp_test',
    'mvp_test',
    'omega_interface_test',
    'spec_test',
    'synth_test',
    'transform_test',
    'translation_test',
    'transys_automata_test',
    'transys_labeled_graphs_test',
    'transys_machines_test',
    'transys_mathset_test',
    'transys_ts_test']
HYBRID = [
    'abstract_test',
    'hybrid_test',
    'prop2part_test',
    'transys_simu_abstract_test']
TEST_FAMILIES_HELP = '''
name of a family of tests. If the `-f` or `--testfiles`
switch is not used, then at most one of the following
can be requested (not case sensitive):

* `base`: (default) tests that should pass on a
  basic TuLiP installation with required dependencies.

* `hybrid`: tests that should pass when using `polytope`.

* `full`: all tests. Every optional dependency or
  external tool is potentially used.

If the `-f` or `--testfiles` switch is used, then
TEST [TEST ...] are interpreted as described below.
'''
TEST_FILES_HELP = '''
space-separated list of test file names,
where the suffix "_test.py" is added to
each given name.  E.g.,

    run_tests.py gr1cint

causes the `gr1cint_test.py` file to be used
and no others.

If no arguments are given,
then default is to run all tests.

If TESTFILES... each have a prefix of "-",
then all tests *except* those listed will be run.
In this case, use the pseudoargument `--`.
For example:

    run_tests.py --fast -- -abstract -synth

Besides what is below, OPTIONS... are passed on to `pytest`.
'''


class ArgParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: {m}\n'.format(m=message))
        self.print_help()
        sys.exit(1)


def main():
    """Entry point."""
    args, unknown_args = _parse_args()
    if (not args.testfiles) and len(args.testfamily) > 1:
        print('At most one family of tests can be requested.')
        sys.exit(1)
    # `chdir` needed to avoid generating temporary files
    # in the `tulip` directory (`pytest` does so even when
    # the option `--rootdir` is given, and `testpaths` defined
    # in the configuration file)
    if args.outofsource:
        print('`chdir("./tests")`')
        os.chdir('./tests')
    try:
        importlib.import_module('tulip')
    except ImportError:
        raise ImportError(
            '`tulip` package not found installed')
    pytest_args = list()
    # skip tests marked as slow
    if args.fast:
        pytest_args.extend(['-m', 'not slow'])
    # measure code coverage
    if args.cover:
        pytest_args.extend([
            '--cov=tulip', '--cov-report=html'])
    # determine what test files will be run
    if args.outofsource:
        tests_dir = '.'
        config_file = 'pytest.ini'
    else:
        tests_dir = args.dir
        config_file = 'tests/pytest.ini'
    if args.testfiles:
        more_args, testfiles = _test_files(tests_dir, args.testfamily)
        pytest_args.extend(more_args)
    else:
        args.testfamily, = args.testfamily
        testfiles = _test_family(args.testfamily)
    testfile_paths = [os.path.join(tests_dir, f) for f in testfiles]
    pytest_args.extend(testfile_paths)
    # other options
    pytest_args.extend(unknown_args)
    pytest_args.extend([
        '--verbosity=3',
        '--continue-on-collection-errors',
        '-c', config_file,
        ])
    print('calling pytest with arguments:\n{args}'.format(
        args=pprint.pformat(pytest_args)))
    ret = pytest.main(pytest_args)
    # return the exit value of `pytest`,
    # to inform CI runs whether all tests passed
    sys.exit(int(ret))


def _test_files(tests_dir, basenames):
    """Collect test files."""
    testfiles, excludefiles, more_args = _collect_testfiles_and_excludefiles(
        tests_dir, basenames)
    if testfiles and excludefiles:
        print(
            'You can specify files to exclude or include, but not both.\n'
            'Try calling this script with the "-h" flag.')
        exit(1)
    if excludefiles:
        more_args.extend(
            '--ignore-glob={omit}'.format(
                omit=os.path.join(tests_dir, omit))
            for omit in excludefiles)
    return more_args, testfiles


def _collect_testfiles_and_excludefiles(tests_dir, basenames):
    """Return Python files to include and to omit in tests."""
    available = _collect_testdir_files(tests_dir)
    more_args = list()
    testfiles = list()
    excludefiles = list()
    for basename in basenames:
        r = _add_files_matching_basename(
            basename, testfiles, excludefiles, available)
        if r is None:
            filename = _map_basename_to_filename(
                basename, testfiles, excludefiles, more_args, tests_dir)
    return testfiles, excludefiles, more_args


def _collect_testdir_files(tests_dir):
    """Return files contained in directory `tests_dir`."""
    available = list()
    for _, _, filenames in os.walk(tests_dir):
        available.extend(filenames)
        return available


def _add_files_matching_basename(
        basename, testfiles, excludefiles, available):
    """Add to lists of files the matching Python files."""
    match = _filter_filenames(basename, available)
    if len(match) > 1:
        raise ValueError(
            'ambiguous base name: `{b}`, matches: {m}'.format(
                b=basename, m=match))
    if not match:
        return
    filename = match[0]
    if not filename.endswith('_test.py'):
        return
    assert len(match) == 1 and filename.endswith('_test.py'), match
    if basename[0] == '-':
        excludefiles.append(filename)
    else:
        testfiles.append(filename)
    assert filename is not None, filename
    return filename


def _filter_filenames(basename, available):
    """Return Python files in `available` that start with `basename`.

    If `basename` starts with a hyphen `-`, then the hyphen is
    ignored when testing for matching.
    """
    desired_start = basename[1:] if basename.startswith('-') else basename
    match = [f for f in available
             if f.startswith(desired_start) and f.endswith('.py')]
    return match


def _map_basename_to_filename(
        basename, testfiles, excludefiles, more_args, tests_dir):
    filename = '{base}_test.py'.format(base=basename)
    path = os.path.join(tests_dir, filename)
    if os.path.exists(path):
        testfiles.append(filename)
        return
    neg = basename.startswith('-')
    if not neg:
        more_args.append(basename)
        return
    base = basename[1:]
    filename = '{base}_test.py'.format(base=base)
    path = os.path.join(tests_dir, filename)
    if os.path.exists(path):
        excludefiles.append(filename)
    else:
        more_args.append(basename)


def _test_family(testfamily):
    """Return `list` of test files that comprise `testfamily`."""
    if testfamily.lower() == 'base':
        testfiles = list(BASE)
    elif testfamily.lower() == 'hybrid':
        testfiles = BASE + HYBRID
    elif testfamily.lower() == 'full':
        testfiles = list()
    else:
        print('Unrecognized test family: "{f}"'.format(
            f=testfamily))
        sys.exit(1)
    testfiles = [name + '.py' for name in testfiles]
    return testfiles


def _parse_args():
    """Return known and unknown command-line arguments."""
    parser = ArgParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        'testfamily', metavar='TEST', nargs='*', default=['base'],
        help=TEST_FAMILIES_HELP)
    parser.add_argument(
        '-f,--testfiles', dest='testfiles', action='store_true',
        help=TEST_FILES_HELP)
    parser.add_argument(
        '--fast', action='store_true',
        help='exclude tests that are marked as slow')
    parser.add_argument(
        '--cover', action='store_true',
        help='generate a coverage report')
    parser.add_argument(
        '--outofsource', action='store_true',
        help='change directory to `tests/` before invoking `pytest`')
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


if __name__ == "__main__":
    main()
