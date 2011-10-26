#!/usr/bin/env python

from distutils.core import setup


###########################################
# Dependency or optional-checking functions
###########################################
# (see notes below.)

def check_graphlibs():
    """Check for presence of graph packages: python-graph."""
    try:
        import pygraph
    except ImportError:
        print 'python-graph not found. If you\'re interested, see http://code.google.com/p/python-graph/'
        print 'Some methods for the Automaton class will not be available.'

    # Dud return value to conform to typical behavior of check_... functions.
    return True

def check_mpt():
    import subprocess

    # Check for Matlab;
    # N.B., the user might have Matlab installed despite it not being
    # visible on the system path!  If set to use MPT, this would cause
    # TuLiP, to enter an interactive processing mode, wherein the user
    # must manually invoke runDiscretizeMatlab.m
    cmd = subprocess.Popen(['which', 'matlab'],
                           stdout=subprocess.PIPE, close_fds=True)
    matlab_visible = False
    for line in cmd.stdout:
        if 'matlab' in line:
            matlab_visible = True
            break
    if not matlab_visible:
        print 'WARNING: Matlab not found, thus we cannot check for MPT.'
        return False

    cmd = subprocess.Popen(['matlab',
                            '-nodesktop',
                            '-nosplash',
                            '-nojvm',
                            '-r',
                            'if (exist(\'mpt_init\', \'file\')==2) && (exist(\'polytope\', \'file\')==2), disp \'MPT FOUND; GO TIME\'; else, disp \'NO MPT; PANIC\'; end; pause(1); exit'],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in cmd.stdout:
        if 'MPT FOUND; GO TIME' in line:
            return True
    
    return False

def check_yices():
    import subprocess
    cmd = subprocess.Popen(['which', 'yices'],
                           stdout=subprocess.PIPE, close_fds=True)
    for line in cmd.stdout:
        if 'yices' in line:
            return True
    return False

def check_glpk():
    try:
        import cvxopt.glpk
    except ImportError:
        return False
    return True

def check_gephi():
    import subprocess
    cmd = subprocess.Popen(['which', 'gephi'], stdout=subprocess.PIPE)
    for line in cmd.stdout:
        if 'gephi' in line:
            return True
    return False


# Handle "check" argument to check for dependencies;
# occurs by default if "install" is given,
# unless both "install" and "nocheck" are given (but typical
# users do not need "nocheck").
#
# The behavior is such that if "check" is given as a command-line
# argument, but "install" is not, then setup (as provided by
# Distutils) is not invoked.

# You *must* have these to run TuLiP.  Each item in other_depends must
# be treated specially; thus other_depends is a dictionary with
#
#   keys   : names of dependency;

#   values : list of callable and string, which is printed on failure
#           (i.e. package not found); we interpret the return value
#           True to be success, and False failure.
other_depends = {'yices' : [check_yices, 'ERROR: Yices not found.']}

# These are nice to have but not necessary. Each item is of the form
#
#   keys   : name of optional package;
#   values : list of callable and two strings, first string printed on
#           success, second printed on failure (i.e. package not
#           found); we interpret the return value True to be success,
#           and False failure.
optionals = {'glpk' : [check_glpk, 'GLPK found.', 'GLPK seems to be missing\nand thus apparently not used by your installation of CVXOPT.\nIf you\'re interested, see http://www.gnu.org/s/glpk/'],
             'gephi' : [check_gephi, 'Gephi found.', 'Gephi seems to be missing. If you\'re interested in graph visualization, see http://gephi.org/'],
             'graphlibs' : [check_graphlibs, '', ''],
             'MPT' : [check_mpt, 'MPT found.', 'MPT not found (and not required). If you\'re curious, see http://control.ee.ethz.ch/~mpt/']}

import sys
perform_setup = True
check_deps = False
if 'install' in sys.argv[1:] and 'nocheck' not in sys.argv[1:]:
    check_deps = True
elif 'check' in sys.argv[1:]:
    perform_setup = False
    check_deps = True

# Pull "check" and "nocheck" from argument list, if present, to play
# nicely with Distutils setup.
try:
    sys.argv.remove('check')
except ValueError:
    pass
try:
    sys.argv.remove('nocheck')
except ValueError:
    pass

if check_deps:
    print "Checking for dependencies..."

    # Python package dependencies
    try:
        import numpy
    except:
        print 'ERROR: NumPy not found.'
        raise
    try:
        import scipy
    except:
        print 'ERROR: SciPy not found.'
        raise
    try:
        import cvxopt
    except:
        print 'ERROR: CVXOPT not found.'
        raise
    try:
        import matplotlib
    except:
        print 'ERROR: matplotlib not found.'
        raise

    # Other dependencies
    for (dep_key, dep_val) in other_depends.items():
        if not dep_val[0]():
            print dep_val[1]
            raise Exception('Failed dependency: '+dep_key)

    # Optional stuff
    for (opt_key, opt_val) in optionals.items():
        print 'Probing for optional '+opt_key+'...'
        if opt_val[0]():
            print opt_val[1]
        else:
            print opt_val[2]


if perform_setup:
    setup(name = 'tulip',
          version = '0.3a',
          description = 'Temporal Logic Planning (TuLiP) Toolbox',
          author = 'Caltech Control and Dynamical Systems',
          author_email = 'murray@cds.caltech.edu',
          url = 'http://tulip-control.sourceforge.net',
          requires = ['numpy', 'scipy', 'cvxopt', 'matplotlib'],
          packages = ['tulip'],
          package_dir = {'tulip' : 'tulip'},
          package_data={'tulip': ['matlab/*.m', 'jtlv_grgame.jar', 'polytope/*.py']},
          scripts = ['tools/aut2dot','tools/aut2gexf','tools/trim_aut']
          )
