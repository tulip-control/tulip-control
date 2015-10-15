"""Interface to `omega` package.

`omega` constructs symbolic transducers,
represented as binary decision diagrams.
This module applies enumeration,
to return enumerated transducers.

U{https://pypi.python.org/pypi/omega}
"""
from __future__ import absolute_import
import networkx as nx
try:
    import omega
    from omega.logic import bitvector as bv
    from omega.games import gr1
    from omega.symbolic import symbolic as sym
    from omega.symbolic import enumeration as enum
except ImportError:
    omega = None


def synthesize_enumerated_streett(spec):
    """Return transducer enumerated as a graph."""
    aut = _grspec_to_automaton(spec)
    sym.fill_blanks(aut)
    a = aut.build()
    z, yij, xijk = gr1.solve_streett_game(a)
    try:
        t = gr1.make_streett_transducer(z, yij, xijk, a)
    except AssertionError as e:
        print(e)
        return None
    (u,) = t.action['sys']
    care = _int_bounds(t)
    g = enum.relation_to_graph(u, t, care_source=care,
                               care_target=care)
    h = _strategy_to_state_annotated(g, a)
    return h


def is_circular(spec):
    """Return `True` if trivial winning set non-empty."""
    aut = _grspec_to_automaton(spec)
    sym.fill_blanks(aut)
    triv, t = gr1.trivial_winning_set(aut)
    return triv != t.bdd.false


def _int_bounds(aut):
    """Create care set for enumeration."""
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
    """Move annotation to `dict` as value of `'state'` key."""
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
        if v == 'boolean':
            v = 'bool'
        d[k] = v
    a.vars = bv.make_table(d, env_vars=g.env_vars)
    a.init['env'] = list(g.env_init)
    a.init['sys'] = list(g.sys_init)
    a.action['env'] = list(g.env_safety)
    a.action['sys'] = list(g.sys_safety)
    a.win['<>[]'] = ['!({s})'.format(s=s) for s in g.env_prog]
    a.win['[]<>'] = list(g.sys_prog)
    return a
