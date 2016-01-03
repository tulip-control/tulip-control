#!/usr/bin/env python
import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

from setuptools import setup
import subprocess
import sys
import os


classifiers = [
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Topic :: Scientific/Engineering']


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

gr1c_msg = 'gr1c not found or of version prior to ' +\
    ".".join([str(vs) for vs in GR1C_MIN_VERSION]) +\
    '.\n' +\
    'Unless you have some alternative synthesis tool installed,\n' +\
    'it will not be possible to realize GR(1) specifications.\n' +\
    'Consult installation instructions for gr1c at http://scottman.net/2012/gr1c\n' +\
    'or the TuLiP User\'s Guide about alternatives.'

# You *must* have these to run TuLiP.  Each item in other_depends must
# be treated specially; thus other_depends is a dictionary with
#
#   keys   : names of dependency;

#   values : list of callable and string, which is printed on failure
#           (i.e. package not found); we interpret the return value
#           True to be success, and False failure.
other_depends = {'gr1c' : [check_gr1c, 'gr1c found.', gr1c_msg]}

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
optionals = {'pydot' : [check_pydot, 'pydot found.', pydot_msg]}

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

package_data = {
    'tulip': ['commit_hash.txt'],
    'tulip.transys.export' : ['d3.v3.min.js'],
    'tulip.spec' : ['parsetab.py']
}

if check_deps:
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

    # Optional stuff for which the installation configuration will
    # change depending on the availability of each.
    if os.path.exists(os.path.join('tulip', 'interfaces', 'jtlv_grgame.jar')):
        print('Found optional JTLV-based solver.')
        package_data['tulip.interfaces'] = ['jtlv_grgame.jar']
    else:
        print('The jtlv synthesis tool was not found. '
              'Try extern/get-jtlv.sh to get it.\n'
              'It is an optional alternative to gr1c, '
              'the default GR(1) solver of TuLiP.')


if perform_setup:
    # Build PLY table, to be installed as tulip package data
    try:
        import tulip.spec.lexyacc
        tabmodule = tulip.spec.lexyacc.TABMODULE.split('.')[-1]
        outputdir = 'tulip/spec'
        parser = tulip.spec.lexyacc.Parser()
        parser.build(tabmodule, outputdir=outputdir,
                     write_tables=True,
                     debug=True, debuglog=logger)

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
        bugtrack_url='http://github.com/tulip-control/tulip-control/issues',
        license = 'BSD',
        classifiers=classifiers,
        install_requires = [
            'ply >= 3.4',
            'networkx >= 1.6',
            'numpy >= 1.7',
            'scipy',
        ],
        extras_require={
            'hybrid': ['cvxopt >= 1.1.7',
                       'polytope >= 0.1.1']},
        tests_require=[
            'nose',
            'matplotlib'],
        packages = [
            'tulip', 'tulip.transys', 'tulip.transys.export',
            'tulip.abstract', 'tulip.spec',
            'tulip.interfaces'
        ],
        package_dir = {'tulip' : 'tulip'},
        package_data=package_data
    )

    if plytable_build_failed:
        print("!"*65)
        print("    Failed to build PLY table.  Please run setup.py again.")
        print("!"*65)
