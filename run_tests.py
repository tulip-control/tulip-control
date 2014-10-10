#!/usr/bin/env python
"""
Driver script for testing TuLiP.  Try calling it with "-h" flag.
"""

import imp
import sys
import os.path
from os import walk
import argparse

import nose

class ArgParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(1)

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

if __name__ == "__main__":
    parser = ArgParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('testfiles', nargs='*', default=None,
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

    basenames = args.testfiles
    skip_slow = args.fast
    measure_coverage = args.cover
    require_nonlocaldir_tulip = args.outofsource
    tests_dir = args.dir

    if require_nonlocaldir_tulip:
        # Scrub local directory from search path for modules
        import os
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
    for basename in basenames:
        match = [f for f in available
                 if f.startswith(basename) and f.endswith('.py')]

        if len(match) > 1:
            raise Exception('ambiguous base name: %s, matches: %s' %
                            (basename, match))
        elif match[0].endswith('_test.py'):
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

    argv.extend(testfiles)
    argv += ["--where=" + tests_dir]
    argv.extend(unknown_args)

    print('calling nose')
    nose.main(argv=argv + ["--verbosity=3", "--exe"])
