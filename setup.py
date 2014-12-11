#!/usr/bin/env python
import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

from setuptools import setup
import subprocess
import sys
import os

###########################################
# Dependency or optional-checking functions
###########################################
# (see notes below.)

GR1C_MIN_VERSION = (0,9,0)
def check_gr1c():
    try:
        v_str = subprocess.check_output(["gr1c", "-V"])
    except OSError:
        return False
    try:
        v_str = v_str.split()[1]
        major, minor, micro = v_str.split(".")
        major = int(major)
        minor = int(minor)
        micro = int(micro)
        if not (major > GR1C_MIN_VERSION[0]
                or (major == GR1C_MIN_VERSION[0]
                    and (minor > GR1C_MIN_VERSION[1]
                         or (minor == GR1C_MIN_VERSION[1]
                             and micro >= GR1C_MIN_VERSION[2])))):
            return False
    except:
        return False
    return True

def check_java():
    try:
        subprocess.check_output(['java', '-help'], stderr=subprocess.STDOUT)
    except OSError as e:
        if e.errno == os.errno.ENOENT:
            return False
        else:
            raise
    return True

def check_glpk():
    try:
        import cvxopt.glpk
    except ImportError:
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

java_msg = (
    'java not found.\n'
    "The jtlv synthesis tool included in the tulip distribution\n"
    'will not be able to run. Unless the tool gr1c is installed,\n'
    'it will not be possible to solve games.'
)

# You *must* have these to run TuLiP.  Each item in other_depends must
# be treated specially; thus other_depends is a dictionary with
#
#   keys   : names of dependency;

#   values : list of callable and string, which is printed on failure
#           (i.e. package not found); we interpret the return value
#           True to be success, and False failure.
other_depends = {'java': [check_java, 'Java  found.', java_msg]}

glpk_msg = 'GLPK seems to be missing\n' +\
    'and thus apparently not used by your installation of CVXOPT.\n' +\
    'If you\'re interested, see http://www.gnu.org/s/glpk/'
gr1c_msg = 'gr1c not found or of version prior to ' +\
    ".".join([str(vs) for vs in GR1C_MIN_VERSION]) +\
    '.\n' +\
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
optionals = {'glpk' : [check_glpk, 'GLPK found.', glpk_msg],
             'gr1c' : [check_gr1c, 'gr1c found.', gr1c_msg],
             'matplotlib' : [check_mpl, 'matplotlib found.', mpl_msg],
             'pydot' : [check_pydot, 'pydot found.', pydot_msg]}

def retrieve_git_info():
    """Return commit hash of HEAD, or "release", or None if failure.

    If the git command fails, then return None.

    If HEAD has tag with prefix "tulip-" or "vM" where M is an
    integer, then return 'release'.
    Tags with such names are regarded as version or release tags.

    Otherwise, return the commit hash as str.
    """
    # Is Git installed?
    try:
        subprocess.call(['git', '--version'],
                        stdout=subprocess.PIPE)
    except OSError:
        return None

    # Decide whether this is a release
    p = subprocess.Popen(
        ['git', 'describe', '--tags', '--candidates=0', 'HEAD'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    p.wait()
    if p.returncode == 0:
        tag = p.stdout.read()
        logger.debug('Most recent tag: ' + tag)
        if tag.startswith('tulip-'):
            return 'release'
        if len(tag) >= 2 and tag.startswith('v'):
            try:
                int(tag[1])
                return 'release'
            except ValueError:
                pass

    # Otherwise, return commit hash
    p = subprocess.Popen(
        ['git', 'log', '-1', '--format=%H'],
        stdout=subprocess.PIPE
    )
    p.wait()
    sha1 = p.stdout.read()
    logger.debug('SHA1: ' + sha1)
    return sha1


perform_setup = True
check_deps = False
if (
    ('install' in sys.argv[1:] or 'develop' in sys.argv[1:]) and
    'nocheck' not in sys.argv[1:]
):
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
        try:
            import cvxopt
        except:
            print('ERROR: CVXOPT not found.')
            raise

    # Other dependencies
    for (dep_key, dep_val) in other_depends.items():
        print('Probing for required dependency:' + dep_key + '...')
        if dep_val[0]():
            print('\t' + dep_val[1])
        else:
            print('\t' + dep_val[2])
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
        import tulip.spec.lexyacc

        tabmodule = 'parsetab'
        outputdir = 'tulip/spec'

        parser = tulip.spec.lexyacc.Parser()
        parser.rebuild_parsetab(tabmodule, outputdir=outputdir,
                                debuglog=logger)

        plytable_build_failed = False
    except Exception as e:
        logger.debug('Failed to build PLY tables: {e}'.format(e=e))
        plytable_build_failed = True

    # If .git directory is present, create commit_hash.txt accordingly
    # to indicate version information
    if os.path.exists('.git'):
        # Provide commit hash or empty file to indicate release
        sha1 = retrieve_git_info()
        if sha1 is None:
            sha1 = 'unknown-commit'
        elif sha1 is 'release':
            sha1 = ''
        else:
            logger.debug('dev sha1: ' + str(sha1) )
        commit_hash_header = "# DO NOT EDIT!  This file was automatically generated by setup.py of TuLiP"
        with open("tulip/commit_hash.txt", "w") as f:
            f.write(commit_hash_header+"\n")
            f.write(sha1+"\n")

    # Import tulip/version.py without importing tulip
    import imp
    version = imp.load_module("version",
                              *imp.find_module("version", ["tulip"]))
    tulip_version = version.version

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
            'networkx >= 1.6',
            'cvxopt'
        ],
        packages = [
            'tulip', 'tulip.transys', 'tulip.transys.export',
            'tulip.abstract', 'tulip.spec',
            'tulip.interfaces'
        ],
        package_dir = {'tulip' : 'tulip'},
        package_data={
            'tulip': ['commit_hash.txt'],
            'tulip.interfaces': ['jtlv_grgame.jar'],
            'tulip.transys.export' : ['d3.v3.min.js'],
            'tulip.spec' : ['parsetab.py']
        },
    )

    if plytable_build_failed:
        print("!"*65)
        print("    Failed to build PLY table.  Please run setup.py again.")
        print("!"*65)
