"""Extra tests for interface to `omega.games.gr1`.

Compared to omega_interface_test.py, tests here can depend on
configurations of `omega` or `dd` that involve configurations beyond
the required dependencies of TuLiP. E.g., dd can be built with CUDD.
"""
import logging

from tulip.spec import form
from tulip.interfaces import omega as omega_int

from omega_interface_test import grspec_1


logging.getLogger('tulip').setLevel('ERROR')
logging.getLogger('astutils').setLevel('ERROR')
logging.getLogger('omega').setLevel('ERROR')
log = logging.getLogger('omega.games')
log.setLevel('WARNING')
log.addHandler(logging.StreamHandler())


def test_synthesis_cudd():
    sp = grspec_1()
    h = omega_int.synthesize_enumerated_streett(sp, use_cudd=True)
    assert h is not None
    n = len(h)
    assert n == 25, n


def test_is_circular_cudd():
    f = form.GRSpec()
    f.sys_vars['y'] = 'bool'
    f.env_prog = ['y']
    f.sys_prog = ['y']
    triv = omega_int.is_circular(f, use_cudd=True)
    assert triv, triv
