#!/usr/bin/env python
"""
Driver script for testing TuLiP.  Try calling it with "-h" flag.
"""
from __future__ import print_function
import imp
import sys
import os
from os import walk
import argparse

import nose
from nose.plugins.base import Plugin


class ShowFunctions(Plugin):
    name = 'showfunctions'
    def describeTest(self, test):
        return str(test)


class ArgParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(1)

testfamilies_help = (
"""
name of a family of tests. If the `-f` or `--testfiles`
switch is not used, then at most one of the following
can be requested (not case sensitive):

* "base": (default) tests that should pass on a
  basic TuLiP installation with required dependencies.

* "hybrid": tests that should pass given dependencies
  implied by the "hybrid" extras: polytope.

* "full": all tests. Every optional dependency or
  external tool is potentially used.

If the `-f` or `--testfiles` switch is used, then
TEST [TEST ...] are interpreted as described below."""
)

testfiles_help = (
"""
space-separated list of test file names,
where the suffix "_test.py" is added to
each given name.  E.g.,

    run_tests.py gr1cint

causes the gr1cint_test.py file to be used
and no others.

If no arguments are given,
then default is to run all tests.

If TESTFILES... each have a prefix of "-",
then all tests *except* those listed will be run.
Use the pseudoargument -- in this case. E.g.,

    run_tests.py --fast -- -abstract -synth

Besides what is below, OPTIONS... are passed on to nose."""
)


def main():
    parser = ArgParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('testfamily', metavar='TEST', nargs='*', default=['base'],
                        help=testfamilies_help)
    parser.add_argument('-f,--testfiles', dest='testfiles', action='store_true',
                        help=testfiles_help)
    parser.add_argument('--fast', action='store_true',
                        help='exclude tests that are marked as slow')
    parser.add_argument('--cover', action='store_true',
                        help='generate a coverage report')
    parser.add_argument('--outofsource', action='store_true',
                        help='import tulip from outside the current directory')
    parser.add_argument(
        '-w', '--where', dest='dir', default='tests',
        help='search for tests in directory WHERE\n'
             '(this is exactly the "-w" or "--where" option of nose)')

    args, unknown_args = parser.parse_known_args()

    if (not args.testfiles) and len(args.testfamily) > 1:
        print('At most one family of tests can be requested.')
        sys.exit(1)

    if args.testfiles:
        basenames = args.testfamily
    else:
        basenames = None
        args.testfamily = args.testfamily[0]
    skip_slow = args.fast
    measure_coverage = args.cover
    require_nonlocaldir_tulip = args.outofsource
    tests_dir = args.dir

    if require_nonlocaldir_tulip:
        # Scrub local directory from search path for modules
        try:
            while True:
                sys.path.remove('')
        except ValueError:
            pass
        try:
            while True:
                sys.path.remove(os.path.abspath(os.curdir))
        except ValueError:
            pass
    try:
        modtuple = imp.find_module("tulip", sys.path)
        imp.load_module("tulip", *modtuple)
    except ImportError:
        if require_nonlocaldir_tulip:
            raise ImportError('tulip package not found, '
                              'besides in the local directory')
        else:
            raise()

    argv = ["nosetests"]
    if skip_slow:
        argv.append("--attr=!slow")

    if measure_coverage:
        argv.extend(["--with-coverage", "--cover-html", "--cover-package=tulip"])

    available = []
    for dirpath, dirnames, filenames in walk(tests_dir):
        available.extend(filenames)
        break

    testfiles = []
    excludefiles = []

    if args.testfiles:
        for basename in basenames:
            if basename[0] == '-':
                matchstart = lambda f, bname: f.startswith(bname[1:])
            else:
                matchstart = lambda f, bname: f.startswith(bname)
            match = [f for f in available
                     if matchstart(f, basename) and f.endswith('.py')]

            if len(match) > 1:
                raise Exception('ambiguous base name: %s, matches: %s' %
                                (basename, match))
            elif len(match) == 1 and match[0].endswith('_test.py'):
                if basename[0] == '-':
                    excludefiles.append(match[0])
                else:
                    testfiles.append(match[0])
                continue

            if os.path.exists(os.path.join(tests_dir, basename + '_test.py')):
                testfiles.append(basename + '_test.py')
            elif basename[0] == '-':
                if os.path.exists(os.path.join(tests_dir, basename[1:] + "_test.py")):
                    excludefiles.append(basename[1:] + "_test.py")
                else:
                    argv.append(basename)
            else:
                argv.append(basename)

        if testfiles and excludefiles:
            print("You can specify files to exclude or include, but not both.")
            print("Try calling it with \"-h\" flag.")
            exit(1)

        if excludefiles:
            argv.append("--exclude=" + "|".join(excludefiles))

    else:
        base = [
            'dumpsmach_test',
            'form_test',
            'gr1_test',
            'omega_interface_test',
            'spec_test',
            'synth_test',
            'transform_test',
            'translation_test',
            'transys_automata_test',
            'transys_labeled_graphs_test',
            'transys_machines_test',
            'transys_mathset_test',
            'transys_ts_test',
            'gridworld_test']
        hybrid = [
            'abstract_test',
            'hybrid_test',
            'prop2part_test']
        if args.testfamily.lower() == 'base':
            testfiles = base
        elif args.testfamily.lower() == 'hybrid':
            testfiles = base + hybrid
        elif args.testfamily.lower() == 'full':
            pass
        else:
            print('Unrecognized test family: "'+args.testfamily+'"')
            sys.exit(1)

    argv.extend(testfiles)
    argv += ["--where=" + tests_dir]
    argv.extend(unknown_args)

    print('calling nose')
    argv.extend(["--verbosity=3",
                 "--exe",
                 "--with-showfunctions"])
    nose.main(argv=argv,
              addplugins=[ShowFunctions()])


if __name__ == "__main__":
    main()
