"""Tests for interface to `omega.games.gr1`."""
import logging

import networkx as nx

from tulip.spec import form
from tulip.interfaces import omega as omega_int
from tulip import synth


from nose import tools as nt


logging.getLogger('tulip').setLevel('ERROR')
logging.getLogger('astutils').setLevel('ERROR')
logging.getLogger('omega').setLevel('ERROR')
log = logging.getLogger('omega.games')
log.setLevel('WARNING')
log.addHandler(logging.StreamHandler())


def test_grspec_to_automaton():
    sp = grspec_0()
    a = omega_int._grspec_to_automaton(sp)
    dvars = dict(x=dict(type='bool', owner='env', dom=None),
                 y=dict(type='bool', owner='sys', dom=None))
    assert a.vars == dvars, (a.vars, dvars)
    r = a.action['env']
    assert r == list(), r
    r = a.action['sys']
    assert r == ['( ( X x ) -> ( X y ) )'], r
    r = a.win['<>[]']
    assert r == ['!(( ! x ))'], r
    r = a.win['[]<>']
    assert r == ['( ! y )'], r


def test_synthesis_bool():
    sp = grspec_0()
    h = omega_int.synthesize_enumerated_streett(sp)
    assert h is not None, 'no winning states'
    g = synth.strategy2mealy(h, sp)
    # fname = 'mealy.pdf'
    # g.save(fname)
    # pd = nx.drawing.nx_pydot.to_pydot(g)
    # pd.write_pdf(fname)
    assert g is not None
    assert len(g.inputs) == 1, g.inputs
    assert 'x' in g.inputs, g.inputs
    dom = g.inputs['x']
    dom_ = {False, True}
    assert dom == dom_, (dom, dom_)
    assert len(g.outputs) == 1, g.outputs
    assert 'y' in g.outputs, g.outputs
    dom = g.outputs['y']
    dom_ = {False, True}
    assert dom == dom_, (dom, dom_)
    assert len(g) == 5, [
        g.nodes(data=True), g.edges(data=True)]


def test_synthesis_fol():
    sp = grspec_1()
    h = omega_int.synthesize_enumerated_streett(sp)
    assert h is not None
    g = synth.strategy2mealy(h, sp)
    assert g is not None
    assert len(g.inputs) == 1, g.inputs
    assert 'x' in g.inputs, g.inputs
    dom = g.inputs['x']
    dom_ = set(xrange(5))
    assert dom == dom_, (dom, dom_)
    assert len(g.outputs) == 1, g.outputs
    assert 'y' in g.outputs, g.outputs
    dom = g.outputs['y']
    dom_ = set(xrange(5))
    assert dom == dom_, (dom, dom_)


def test_synthesis_strings():
    sp = grspec_4()
    h = omega_int.synthesize_enumerated_streett(sp)
    g = synth.strategy2mealy(h, sp)
    assert g is not None
    # outputs
    assert len(g.outputs) == 1, g.outputs
    assert 'y' in g.outputs, g.outputs
    dom = g.outputs['y']
    dom_ = {'a', 'b'}
    assert dom == dom_, (dom, dom_)
    # check simulation
    n = len(g)
    assert n == 4, n
    u = 'Sinit'
    u, r = g.reaction(u, dict(x=0))
    assert r == dict(y='a'), r
    u, r = g.reaction(u, dict(x=2))
    assert r == dict(y='b'), r
    u, r = g.reaction(u, dict(x=1))
    assert r == dict(y='b'), r
    u, r = g.reaction(u, dict(x=0))
    assert r == dict(y='a'), r


def test_synthesis_moore():
    sp = grspec_2()
    h = omega_int.synthesize_enumerated_streett(sp)
    g = synth.strategy2mealy(h, sp)
    # g.save('moore.pdf')
    assert g is not None
    n = len(g)
    assert n == 26, n


def test_synthesis_mealy_all_init():
    sp = grspec_3()
    h = omega_int.synthesize_enumerated_streett(sp)
    assert h is None, 'should be unrealizable'
    sp.env_init = ['y = x']
    h = omega_int.synthesize_enumerated_streett(sp)
    g = synth.strategy2mealy(h, sp)
    # g.save('moore.pdf')
    assert g is not None
    n = len(g)
    assert n == 6, n


def test_synthesis_unrealizable():
    sp = grspec_0()
    sp.sys_prog = ['False']
    h = omega_int.synthesize_enumerated_streett(sp)
    assert h is None, h


def test_is_circular_true():
    f = form.GRSpec()
    f.sys_vars['y'] = 'bool'
    f.env_prog = ['y']
    f.sys_prog = ['y']
    triv = omega_int.is_circular(f)
    assert triv, triv


def test_is_circular_false():
    f = form.GRSpec()
    f.env_vars['x'] = 'bool'
    f.env_prog = ['x']
    f.sys_prog = ['x']
    triv = omega_int.is_circular(f)
    assert not triv, triv


def grspec_0():
    sp = form.GRSpec()
    sp.moore = False
    sp.env_vars = dict(x='boolean')
    sp.sys_vars = dict(y='boolean')
    sp.sys_safety = ["x' -> y'"]
    sp.env_prog = ['!x']
    sp.sys_prog = ['!y']
    return sp


def grspec_1():
    sp = form.GRSpec()
    sp.moore = False
    sp.plus_one = False
    sp.env_vars = dict(x=(0, 4))
    sp.sys_vars = dict(y=(0, 4))
    sp.env_init = ['(0 <= y) & (y <= 4)']
    sp.sys_safety = ["y' = x'"]
    return sp


def grspec_2():
    sp = form.GRSpec()
    sp.moore = True
    sp.env_vars = dict(x=(0, 4))
    sp.sys_vars = dict(y=(0, 4))
    sp.env_init = ['(0 <= y) & (y <= 4)']
    sp.sys_safety = ["y' = x"]
    return sp

def grspec_3():
    sp = form.GRSpec()
    sp.moore = False
    sp.env_vars = dict(x=(0, 4))
    sp.sys_vars = dict(y=(0, 4))
    sp.env_init = ['(0 <= y) & (y <= 4)']
    sp.sys_safety = ["y = x"]
    return sp


def grspec_4():
    sp = form.GRSpec()
    sp.moore = False
    sp.env_init = ['(x = 0) & (y = "a")']
    sp.env_vars = dict(x=(0, 2))
    sp.sys_vars = dict(y=['a', 'b'])
    sp.sys_safety = [
        '(x = 0) -> (y = "a")',
        '(x > 0) -> (y = "b")']
    return sp


if __name__ == '__main__':
    test_synthesis_bool()
