#!/usr/bin/env python

from setuptools import setup

###########################################
# Dependency or optional-checking functions
###########################################
# (see notes below.)

def check_gr1c():
    import subprocess
    try:
        subprocess.call(["gr1c", "-V"], stdout=subprocess.PIPE)
    except OSError:
        return False
    return True

def check_mpl():
    try:
        import matplotlib
    except ImportError:
        return False
    return True

def check_pydot():
    try:
        import pydot
        from distutils.version import StrictVersion
        if StrictVersion(pydot.__version__) < StrictVersion('1.0.28'):
            print('Pydot version >= 1.0.28 required.' +
                'found: ' +pydot.__version__)
    except ImportError:
        return False
    return True

# Handle "dry-check" argument to check for dependencies without
# installing the tulip package; checking occurs by default if
# "install" is given, unless both "install" and "nocheck" are given
# (but typical users do not need "nocheck").

# You *must* have these to run TuLiP.  Each item in other_depends must
# be treated specially; thus other_depends is a dictionary with
#
#   keys   : names of dependency;

#   values : list of callable and string, which is printed on failure
#           (i.e. package not found); we interpret the return value
#           True to be success, and False failure.
other_depends = {}

gr1c_msg = 'gr1c not found.\n' +\
    'If you\'re interested in a GR(1) synthesis tool besides JTLV,\n' +\
    'see http://scottman.net/2012/gr1c'
mpl_msg = 'matplotlib not found.\n' +\
    'For many graphics drawing features in TuLiP, you must install\n' +\
    'matplotlib (http://matplotlib.org/).'
pydot_msg = 'pydot not found.\n' +\
    'Several graph image file creation and dot (http://www.graphviz.org/)\n' +\
    'export routines will be unavailable unless you install\n' +\
    'pydot (http://code.google.com/p/pydot/).'

# These are nice to have but not necessary. Each item is of the form
#
#   keys   : name of optional package;
#   values : list of callable and two strings, first string printed on
#           success, second printed on failure (i.e. package not
#           found); we interpret the return value True to be success,
#           and False failure.
optionals = {'gr1c' : [check_gr1c, 'gr1c found.', gr1c_msg],
             'matplotlib' : [check_mpl, 'matplotlib found.', mpl_msg],
             'pydot' : [check_pydot, 'pydot found.', pydot_msg]}

import sys
perform_setup = True
check_deps = False
if 'install' in sys.argv[1:] and 'nocheck' not in sys.argv[1:]:
    check_deps = True
elif 'dry-check' in sys.argv[1:]:
    perform_setup = False
    check_deps = True

# Pull "dry-check" and "nocheck" from argument list, if present, to play
# nicely with Distutils setup.
try:
    sys.argv.remove('dry-check')
except ValueError:
    pass
try:
    sys.argv.remove('nocheck')
except ValueError:
    pass

if check_deps:
    if not perform_setup:
        print('Checking for required dependencies...')

        # Python package dependencies
        try:
            import numpy
        except:
            print('ERROR: NumPy not found.')
            raise
        try:
            import scipy
        except:
            print('ERROR: SciPy not found.')
            raise
        try:
            import ply
        except:
            print('ERROR: PLY not found.')
            raise
        try:
            import networkx
        except:
            print('ERROR: NetworkX not found.')
            raise

        # Other dependencies
        for (dep_key, dep_val) in other_depends.items():
            if not dep_val[0]():
                print(dep_val[1] )
                raise Exception('Failed dependency: '+dep_key)

    # Optional stuff
    for (opt_key, opt_val) in optionals.items():
        print('Probing for optional '+opt_key+'...')
        if opt_val[0]():
            print("\t"+opt_val[1] )
        else:
            print("\t"+opt_val[2] )


if perform_setup:
    # Build PLY table, to be installed as tulip package data
    try:
        import os
        import tulip.spec.plyparser
        tulip.spec.plyparser.rebuild_parsetab()
        os.rename("parsetab.py", "tulip/spec/parsetab.py")
        plytable_build_failed = False
    except:
        plytable_build_failed = True

    from tulip import __version__ as tulip_version
    setup(
        name = 'tulip',
        version = tulip_version,
        description = 'Temporal Logic Planning (TuLiP) Toolbox',
        author = 'Caltech Control and Dynamical Systems',
        author_email = 'tulip@tulip-control.org',
        url = 'http://tulip-control.org',
        license = 'BSD',
        requires = ['numpy', 'scipy', 'polytope', 'ply', 'networkx'],
        install_requires = [
            'numpy >= 1.7',
            'polytope >= 0.1.0',
            'ply >= 3.4',
            'networkx >= 1.6'
        ],
        packages = [
            'tulip', 'tulip.transys', 'tulip.transys.export',
            'tulip.abstract', 'tulip.spec',
            'tulip.interfaces'
        ],
        package_dir = {'tulip' : 'tulip'},
        package_data={
            'tulip.interfaces': ['jtlv_grgame.jar'],
            'tulip.transys.export' : ['d3.v3.min.js'],
            'tulip.spec' : ['parsetab.py']
        },
    )

    if plytable_build_failed:
        print("!"*65)
        print("    Failed to build PLY table.  Please run setup.py again.")
        print("!"*65)
