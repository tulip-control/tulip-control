# Copyright by California Institute of Technology
# All rights reserved. See LICENSE file at:
# <https://github.com/tulip-control/tulip-control>
"""Interface to `omega` package.

`omega` constructs symbolic transducers,
represented as binary decision diagrams.
This module applies enumeration,
to return enumerated transducers.

<https://pypi.org/project/omega>
"""
import collections.abc as _abc
import logging
import time

try:
    import omega
    import omega.logic.bitvector as bv
    import omega.games.gr1 as gr1
    import omega.symbolic.temporal as trl
    import omega.games.enumeration as enum
except ImportError:
    omega = None
import networkx as nx

import tulip.spec as _spec


_log = logging.getLogger(__name__)


def is_realizable(
        spec:
            _spec.GRSpec
        ) -> bool:
    """Return `True` if, and only if, realizable.

    See `synthesize_enumerated_streett` for more details.
    """
    aut = _grspec_to_automaton(spec)
    t0 = time.time()
    z, _, _ = gr1.solve_streett_game(aut)
    t1 = time.time()
    return gr1.is_realizable(z, aut)


def synthesize_enumerated_streett(
        spec:
            _spec.GRSpec
        ) -> nx.DiGraph:
    """Return transducer enumerated as a graph."""
    aut = _grspec_to_automaton(spec)
    assert aut.action['sys'] != aut.false
    t0 = time.time()
    z, yij, xijk = gr1.solve_streett_game(aut)
    t1 = time.time()
    # unrealizable ?
    if not gr1.is_realizable(z, aut):
        print('WARNING: unrealizable')
        return None
    gr1.make_streett_transducer(z, yij, xijk, aut)
    t2 = time.time()
    g = enum.action_to_steps(
        aut, 'env', 'impl',
        qinit=aut.qinit)
    h = _strategy_to_state_annotated(g, aut)
    del z, yij, xijk
    t3 = time.time()
    win = t1 - t0
    sym = t2 - t1
    enu = t3 - t2
    _log.info(
        f'Winning set computed in {win} sec.\n'
        f'Symbolic strategy computed in {sym} sec.\n'
        f'Strategy enumerated in {enu} sec.')
    return h


def is_circular(
        spec:
            _spec.GRSpec
        ) -> bool:
    """Return `True` if trivial winning set non-empty."""
    aut = _grspec_to_automaton(spec)
    triv, t = gr1.trivial_winning_set(aut)
    return triv != t.bdd.false


def _int_bounds(aut: trl.Automaton):
    """Create care set for enumeration.

    @rtype:
        `dd.bdd.Function | dd.cudd.Function`
    """
    int_types = {'int', 'saturating', 'modwrap'}
    bdd = aut.bdd
    u = bdd.true
    for var, d in aut.vars.items():
        t = d['type']
        if t == 'bool':
            continue
        assert t in int_types, t
        dom = d['dom']
        p, q = dom
        e = f'({p} <= {var}) & ({var} <= {q})'
        v = aut.add_expr(e)
        u = bdd.apply('and', u, v)
    return u


def _strategy_to_state_annotated(
        g:
            nx.DiGraph,
        aut:
            trl.Automaton
        ) -> nx.DiGraph:
    """Move annotation to `dict` as value of `'state'` key."""
    h = nx.DiGraph()
    for u, d in g.nodes(data=True):
        dvars = {
            k: d[k]
            for k in d
            if k in aut.vars}
        h.add_node(u, state=dvars)
    for u, v in g.edges():
        h.add_edge(u, v)
    h.initial_nodes = set(g.initial_nodes)
    return h


def _grspec_to_automaton(
        g:
            _spec.GRSpec
        ) -> trl.Automaton:
    """Return `omega.symbolic.temporal.Automaton` from `GRSpec`."""
    if omega is None:
        raise ImportError(
            'Failed to import package `omega`.')
    a = trl.Automaton()
    d = dict(g.env_vars)
    d.update(g.sys_vars)
    for k, v in d.items():
        if v in ('boolean', 'bool'):
            r = 'bool'
        elif isinstance(v, list):
            # string var -> integer var
            r = (0, len(v) - 1)
        elif isinstance(v, tuple):
            r = v
        else:
            raise ValueError(
                f'unknown variable type: {v}')
        d[k] = r
    g.str_to_int()
    # reverse mapping by `synth.strategy2mealy`
    a.declare_variables(**d)
    a.varlist.update(
        env=list(g.env_vars.keys()),
        sys=list(g.sys_vars.keys()))
    f = g._bool_int.__getitem__
    a.init['env'] = _conj(g.env_init, f)
    a.init['sys'] = _conj(g.sys_init, f)
    a.action['env'] = _conj(g.env_safety, f)
    a.action['sys'] = _conj(g.sys_safety, f)
    if g.env_prog:
        w1 = [f'!({s})' for s in map(f, g.env_prog)]
    else:
        w1 = ['FALSE']
    if g.sys_prog:
        w2 = [f(sp) for sp in g.sys_prog]
    else:
        w2 = ['TRUE']
    a.win['<>[]'] = a.bdds_from(*w1)
    a.win['[]<>'] = a.bdds_from(*w2)
    # attributes for spec type
    a.moore = g.moore
    a.plus_one = g.plus_one
    a.qinit = g.qinit
    return a


def _conj(
        strings:
            _abc.Iterable,
        f:
            _abc.Callable
        ) -> str:
    if not strings:
        return 'TRUE'
    return ' & '.join(f(s) for s in strings)
