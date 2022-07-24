"""Collect and print environment information."""
import subprocess

try:
    import cvxopt
except ImportError:
    cvxopt = None
try:
    from cvxopt import glpk as cvxopt_glpk
except ImportError:
    cvxopt_glpk = None
try:
    import dd
except ImportError:
    dd = None
try:
    from dd import cudd as dd_cudd
except ImportError:
    dd_cudd = None
try:
    import scipy
except ImportError:
    scipy = None
try:
    import numpy
except ImportError:
    numpy = None
try:
    import polytope
except ImportError:
    polytope = None


from tulip.interfaces import gr1c as gr1c
from tulip.interfaces import gr1py as gr1py_int
from tulip.interfaces import omega as omega_int


__all__ = [
    'print_env']


def print_env():
    """Print to `stdout` relevant information about environment.

    Looks for solvers like `gr1c` and `glpk`, bindings with `cvxopt`,
    and reports this and other relevant information.
    """
    c = list()
    s = '---- ENVIRONMENT INFORMATION RELEVANT TO `tulip` ----\n'
    c.append(s)
    if gr1c.check_gr1c():
        s = 'Found `gr1c` in `$PATH`.\n'
    else:
        s = 'Did not find `gr1c` in `$PATH`.\n'
    c.append(s)
    s = _format_python_package_message(
        'gr1py', gr1py_int.gr1py,
        'https://pypi.python.org/pypi/gr1py')
    c.append(s)
    s = _format_python_package_message(
        'dd', dd,
        'https://pypi.python.org/pypi/dd')
    c.append(s)
    if dd_cudd is None:
        s = (
            'Could not import Cython module `dd.cudd`.\n'
            'Can be installed by compiling and linking the Cython bindings '
            'of `dd` to CUDD.\n')
    else:
        s = f'Found Cython module `dd.cudd` as:\n    {dd_cudd}\n.'
    c.append(s)
    s = _format_python_package_message(
        'omega', omega_int.omega,
        'https://pypi.python.org/pypi/omega')
    c.append(s)
    s = _format_python_package_message(
        'numpy', numpy,
        'https://pypi.python.org/pypi/numpy')
    c.append(s)
    s = _format_python_package_message(
        'scipy', scipy,
        'https://pypi.python.org/pypi/scipy')
    c.append(s)
    s = _format_python_package_message(
        'cvxopt', cvxopt,
        'https://pypi.python.org/pypi/cvxopt')
    c.append(s)
    if _check_glpsol():
        s = 'Found GLPK solver `glpsol` in `$PATH`.\n'
    else:
        s = 'Did not find `glpsol` in `$PATH`.\n'
    c.append(s)
    if cvxopt_glpk is None:
        s = (
            'Could not import module `cvxopt.glpk`.\n'
            'Can be installed by compiling and linking `cvxopt` to GLPK.\n')
    else:
        s = f'Found module `cvxopt.glpk` as:\n    {cvxopt_glpk}\n.'
    c.append(s)
    s = _format_python_package_message(
        'polytope', polytope, 'https://pypi.python.org/pypi/polytope')
    c.append(s)
    s = [
        'For details about what each package and solver does, '
        'see the documentation. In summary:',
        '',
        '---- DISCRETE SYNTHESIS ----',
        '`omega` is a Python solver for discrete synthesis,',
        '`gr1c` is a synthesizer written in C that uses the library CUDD.',
        'If `dd.cudd` has been built, then `omega` can use the C library '
        'CUDD for more efficient operation.',
        '',
        '---- CONTINUOUS SYNTHESIS ----',
        '`polytope` is a Python package that uses `scipy` and `numpy`.',
        'If `cvxopt` is installed, GLPK available, and '
        'cvxopt.glpk` compiled and linked successfully, ',
        'then `polytope` will use `cvxopt.glpk`, which is notably faster.',
        '',
        50 * '=']
    c.extend(s)
    s = '\n'.join(c)
    print(s)


def _format_python_package_message(name, package, url):
    """Return a `str` reporting information about package."""
    if package is None:
        return (
            f'Could not import Python package `{name}`.\n'
            f'Can be installed from:\n    <{url}>\n')
    return (
        f'Found Python package `{name}` as:\n'
        f'    {package}\n')


def _check_glpsol():
    """Return `True` if `glpsol` (of GLPK) in `$PATH`."""
    try:
        v = subprocess.check_output(['glpsol', '--version'])
    except OSError:
        return False
    return True
