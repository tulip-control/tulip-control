#!/usr/bin/env python
"""
Driver script for testing nu-TuLiP.  Try calling it with "-h" flag.

SCL; 5 Sep 2013.
"""

import sys
import os.path
import nose

if __name__ == "__main__":
    if ("-h" in sys.argv) or ("--help" in sys.argv):
        print """Usage: run_tests.py [--cover] [--fast] [OPTIONS...] [[-]TESTFILES...]

    TESTFILES... is space-separated list of test file names, where the
    suffix "_test.py" is added to each given name.  E.g.,

      run_tests.py automaton

    causes the automaton_test.py file to be used and no others.  If no
    arguments are given, then default is to run all tests.  To exclude
    tests that are marked as slow, use the flag "--fast".
    If TESTFILES... each have a prefix of "-", then all tests *except*
    those listed will be run.  OPTIONS... are passed on to nose.
    """
        exit(1)

    if len(sys.argv) == 1:
        nose.main()

    if "--fast" in sys.argv:
        skip_slow = True
        sys.argv.remove("--fast")
    else:
        skip_slow = False

    if "--cover" in sys.argv:
        measure_coverage = True
        sys.argv.remove("--cover")
    else:
        measure_coverage = False

    argv = [sys.argv[0]]
    if skip_slow:
        argv.append("--attr=!slow")
    if measure_coverage:
        argv.extend(["--with-coverage", "--cover-html", "--cover-package=tulip"])
    testfiles = []
    excludefiles = []
    for basename in sys.argv[1:]:  # Only add extant file names
        try:
            with open(os.path.join("tests", basename+"_test.py"), "r") as f:
                testfiles.append(basename+"_test.py")
        except IOError:
            if basename[0] == "-":
                try:
                    with open(os.path.join("tests", basename[1:]+"_test.py"), "r") as f:
                        excludefiles.append(basename[1:]+"_test.py")
                except IOError:
                    argv.append(basename)
            else:
                argv.append(basename)
    if len(testfiles) > 0 and len(excludefiles) > 0:
        print "You can specify files to exclude or include, but not both."
        print "Try calling it with \"-h\" flag."
        exit(1)
    if len(excludefiles) > 0:
        argv.append("--exclude="+"|".join(excludefiles))
    argv.extend(testfiles)
    nose.main(argv=argv)
