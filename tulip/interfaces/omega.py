# Copyright by California Institute of Technology
# All rights reserved. See LICENSE file at:
# https://github.com/tulip-control/tulip-control
"""Interface to `omega` package.

`omega` constructs symbolic transducers,
represented as binary decision diagrams.
This module applies enumeration,
to return enumerated transducers.

U{https://pypi.python.org/pypi/omega}
"""
from __future__ import absolute_import
import logging
import time

try:
    import dd.bdd as _bdd
except ImportError:
    _bdd = None
try:
    from dd import cudd
except ImportError:
    cudd = None
try:
    import omega
    from omega.logic import bitvector as bv
    from omega.games import gr1
    from omega.symbolic import symbolic as sym
    from omega.games import enumeration as enum
except ImportError:
    omega = None
import networkx as nx


log = logging.getLogger(__name__)


def is_realizable(spec, use_cudd=False):
    """Return `True` if, and only if, realizable.

    See `synthesize_enumerated_streett` for more details.
    """
    aut = _grspec_to_automaton(spec)
    sym.fill_blanks(aut)
    bdd = _init_bdd(use_cudd)
    aut.bdd = bdd
    a = aut.build()
    t0 = time.time()
    z, _, _ = gr1.solve_streett_game(a)
    t1 = time.time()
    return gr1.is_realizable(z, a)


def synthesize_enumerated_streett(spec, use_cudd=False):
    """Return transducer enumerated as a graph.

    @type spec: `tulip.spec.form.GRSpec`
    @param use_cudd: efficient BDD computations with `dd.cudd`
    @rtype: `networkx.DiGraph`
    """
    aut = _grspec_to_automaton(spec)
    sym.fill_blanks(aut)
    bdd = _init_bdd(use_cudd)
    aut.bdd = bdd
    a = aut.build()
    assert a.action['sys'][0] != bdd.false
    t0 = time.time()
    z, yij, xijk = gr1.solve_streett_game(a)
    t1 = time.time()
    # unrealizable ?
    if not gr1.is_realizable(z, a):
        print('WARNING: unrealizable')
        return None
    t = gr1.make_streett_transducer(z, yij, xijk, a)
    t2 = time.time()
    (u,) = t.action['sys']
    assert u != bdd.false
    g = enum.action_to_steps(t, qinit=spec.qinit)
    h = _strategy_to_state_annotated(g, a)
    del u, yij, xijk
    t3 = time.time()
    log.info((
        'Winning set computed in {win} sec.\n'
        'Symbolic strategy computed in {sym} sec.\n'
        'Strategy enumerated in {enu} sec.').format(
            win=t1 - t0,
            sym=t2 - t1,
            enu=t3 - t2))
    return h


def is_circular(spec, use_cudd=False):
    """Return `True` if trivial winning set non-empty.

    @type spec: `tulip.spec.form.GRSpec`
    @param use_cudd: efficient BDD computations with `dd.cudd`
    @rtype: `bool`
    """
    aut = _grspec_to_automaton(spec)
    sym.fill_blanks(aut)
    bdd = _init_bdd(use_cudd)
    aut.bdd = bdd
    triv, t = gr1.trivial_winning_set(aut)
    return triv != t.bdd.false


def _init_bdd(use_cudd):
    if _bdd is None:
        raise ImportError(
            'Failed to import `dd.bdd`.\n'
            'Install package `dd`.')
    if not use_cudd:
        return _bdd.BDD()
    if cudd is None:
        raise ImportError(
            'Failed to import module `dd.cudd`.\n'
            'Compile the Cython bindings of `dd` to CUDD.')
    return cudd.BDD()


def _int_bounds(aut):
    """Create care set for enumeration.

    @type aut: `omega.symbolic.symbolic.Automaton`
    @return: node in a `dd.bdd.BDD`
    @rtype: `int`
    """
    int_types = {'int', 'saturating', 'modwrap'}
    bdd = aut.bdd
    u = bdd.true
    for var, d in aut.vars.iteritems():
        t = d['type']
        if t == 'bool':
            continue
        assert t in int_types, t
        dom = d['dom']
        p, q = dom
        e = "({p} <= {var}) & ({var} <= {q})".format(
            p=p, q=q, var=var)
        v = aut.add_expr(e)
        u = bdd.apply('and', u, v)
    return u


def _strategy_to_state_annotated(g, aut):
    """Move annotation to `dict` as value of `'state'` key.

    @type g: `nx.DiGraph`
    @type: aut: `omega.symbolic.symbolic.Automaton`
    @rtype: `nx.DiGraph`
    """
    h = nx.DiGraph()
    for u, d in g.nodes_iter(data=True):
        dvars = {k: d[k] for k in d if k in aut.vars}
        h.add_node(u, state=dvars)
    for u, v in g.edges_iter():
        h.add_edge(u, v)
    return h


def _grspec_to_automaton(g):
    """Return `symbolic.Automaton` from `GRSpec`.

    @type g: `tulip.spec.form.GRSpec`
    @rtype: `omega.symbolic.symbolic.Automaton`
    """
    if omega is None:
        raise ImportError(
            'Failed to import package `omega`.')
    a = sym.Automaton()
    d = dict(g.env_vars)
    d.update(g.sys_vars)
    for k, v in d.iteritems():
        if v in ('boolean', 'bool'):
            r = 'bool'
        elif isinstance(v, list):
            # string var -> integer var
            r = (0, len(v) - 1)
        elif isinstance(v, tuple):
            r = v
        else:
            raise ValueError(
                'unknown variable type: {v}'.format(v=v))
        d[k] = r
    g.str_to_int()
    # reverse mapping by `synth.strategy2mealy`
    a.vars = bv.make_table(d, env_vars=g.env_vars)
    f = g._bool_int.__getitem__
    a.init['env'] = map(f, g.env_init)
    a.init['sys'] = map(f, g.sys_init)
    a.action['env'] = map(f, g.env_safety)
    a.action['sys'] = map(f, g.sys_safety)
    a.win['<>[]'] = [
        '!({s})'.format(s=s)
        for s in map(f, g.env_prog)]
    a.win['[]<>'] = map(f, g.sys_prog)
    a.moore = g.moore
    a.plus_one = g.plus_one
    a.qinit = g.qinit
    return a
