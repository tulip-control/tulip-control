# Copyright (c) 2014, 2015 by California Institute of Technology
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the California Institute of Technology nor
#    the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CALTECH
# OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
#
"""Syntactic manipulation of trees."""
import collections.abc as _abc
import logging
import copy
import os
import warnings

import networkx as nx

import tulip.graphics as _graphics
import tulip.spec.ast as _ast
import tulip.spec.parser as parser


__all__ = [
    'Tree',
    'ast_to_labeled_graph',
    'check_for_undefined_identifiers',
    'sub_values',
    'sub_constants',
    'sub_bool_with_subtree',
    'pair_node_to_var',
    'infer_constants',
    'check_var_name_conflict',
    'collect_primed_vars']


_logger = logging.getLogger(__name__)
Node = _ast.NodeSpec


class Tree(
        nx.MultiDiGraph):
    """Abstract syntax tree as a graph data structure.

    Use this as a scaffold for syntactic manipulation.
    It makes traversals and graph rewriting easier,
    so it is preferred to working directly with the
    recursive AST classes.

    The attribute `self.root` is the tree's root `Node`.
    """

    def __init__(self):
        self.root = None
        super().__init__()

    def __repr__(self):
        return repr(self.root)

    def __str__(self):
        # need to override networkx.DiGraph.__str__
        return ('Abstract syntax tree as graph with edges:\n' +
                str([(str(u), str(v))
                    for u, v, _ in self.edges(keys=True)]))

    @property
    def variables(self) -> set[Node]:
        """Return the set of variables in `tree`.

        @rtype:
            `set` of `Var`
        """
        return {u for u in self if u.type == 'var'}

    @classmethod
    def from_recursive_ast(
            cls,
            u:
                Node
            ) -> 'Tree':
        tree = cls()
        tree.root = u
        tree._recurse(u)
        return tree

    def _recurse(
            self,
            u:
                Node
            ) -> Node:
        if hasattr(u, 'value'):
            # necessary this terminal is the root
            self.add_node(u)
        elif hasattr(u, 'operator'):
            for i, v in enumerate(u.operands):
                self.add_edge(u, v, key=i)
                self._recurse(v)
        else:
            raise Exception(f'unknown node type: {u}')
        return u

    def to_recursive_ast(
            self,
            u:
                Node |
                None=None
            ) -> Node:
        if u is None:
            u = self.root
        w = copy.copy(u)
        if not self.succ.get(u):
            assert hasattr(u, 'value')
        else:
            w.operands = [self.to_recursive_ast(v)
                          for _, v, _ in sorted(
                              self.edges(u, keys=True),
                              key=lambda x: x[2])]
            assert len(u.operands) == len(w.operands)
        return w

    def add_subtree(
            self,
            leaf:
                Node,
            tree:
                'Tree'
            ) -> None:
        """Add the `tree` at node `leaf`.

        @param tree:
            to be added, without copying AST nodes.
        """
        assert not self.succ.get(leaf)
        for u, v, k in tree.edges(keys=True):
            self.add_edge(u, v, key=k)
        # replace old leaf with subtree root
        ine = self.in_edges(leaf, keys=True)
        if ine:
            assert len(ine) == 1
            ((parent, _, k), ) = ine
            self.add_edge(parent, tree.root, key=k)
        else:
            self.root = tree.root
        self.remove_node(leaf)

    def _to_dot(
            self,
            detailed:
                bool=False
            ) -> _graphics._Digraph:
        """Create GraphViz dot string from given AST."""
        g = ast_to_labeled_graph(self, detailed)
        return _graphics.networkx_to_graphviz(g)

    def write(
            self,
            filename:
                str,
            detailed:
                bool=False
            ) -> None:
        """Layout AST and save result in PDF file."""
        fname, fext = os.path.splitext(filename)
        fext = fext[1:]  # drop .
        p = self._to_dot(detailed)
        p.graph_attr['ordering'] = 'out'
        p.render(
            filename=filename,
            format=fext)


def ast_to_labeled_graph(
        tree:
            Tree,
        detailed:
            bool
        ) -> nx.DiGraph:
    """Convert AST to digraph for graphics.

    @param tree:
        Abstract syntax tree
    """
    g = nx.DiGraph()
    for u in tree:
        if hasattr(u, 'operator'):
            label = u.operator
        elif hasattr(u, 'value'):
            label = u.value
        else:
            raise TypeError(
                'AST node must be Operator or Terminal, '
                f'got instead: {u}'
                f', of type: {type(u)}')
        # show both repr and AST node class in each vertex
        if detailed:
            label += f'\n{type(u).__name__}'
        g.add_node(id(u), label=label)
    for u, v, k in tree.edges(keys=True):
        g.add_edge(id(u), id(v), label=k)
    return g


def check_for_undefined_identifiers(
        tree:
            Tree,
        domains:
            dict):
    """Check that types in `tree` are incompatible with `domains`.

    Raise a `ValueError` if `tree` either:

      - contains a variable missing from `domains`
      - binary operator between variable and
        invalid value for that variable.

    @param domains:
        variable definitions:

        `{'varname': domain}`

        See `GRSpec` for more details of available domain types.
    """
    for u in tree:
        if u.type == 'var' and u.value not in domains:
            var = u.value
            tr = tree.to_recursive_ast()
            raise ValueError(
                f'Undefined variable "{var}" missing from '
                f'symbol table:\n\t{domains}\n'
                f'in subformula:\n\t{tr}')
        if u.type not in {'str', 'num'}:
            continue
        # is a Const or Num
        var, c = pair_node_to_var(tree, u)
        if c.type == 'str':
            dom = domains[var]
            if not isinstance(dom, list):
                raise Exception(
                    f'String constant "{c}" assigned to non-string '
                    f'variable "{var}" with domain:\n\t{dom}')
            if c.value not in domains[var.value]:
                raise ValueError(
                    f'String constant "{c}" is not in the domain '
                    f'of variable "{var}"')
        if c.type == 'num':
            dom = domains[var]
            if not isinstance(dom, tuple):
                raise Exception(
                    f'Number: {c}, assigned to non-integer {c}'
                    f'variable "{var}" with domain:\n\t{dom}')
            if not dom[0] <= c.value <= dom[1]:
                raise Exception(
                    f'Integer variable "{var}", is assigned the '
                    f'value: {c}, that is out of its domain:'
                    f'{dom[0]} ... {dom[1]}')


def sub_values(
        tree:
            Tree,
        var_values:
            dict
        ) -> None:
    """Substitute given values for variables.

    Modifies AST, with `Var` nodes replaced by
    `Num`, `Const`, or `Bool`.
    """
    old2new = dict()
    for u in tree.nodes():
        if u.type != 'var':
            continue
        val = var_values[u.value]
        # instantiate appropriate value type
        if isinstance(val, bool):
            v = _ast.nodes.Bool(val)
        elif isinstance(val, int):
            v = _ast.nodes.Num(val)
        elif isinstance(val, str):
            v = _ast.nodes.Str(val)
        else:
            raise TypeError(val)
        old2new[u] = v
    # replace variable by value
    nx.relabel_nodes(tree, old2new, copy=False)


def sub_constants(
        tree:
            Tree,
        var_str2int:
            dict[str, list]):
    """Replace string constants by integers.

    To be used for converting arbitrary finite domains
    to integer domains prior to calling gr1c.

    @param var_str2int:
        {'varname':['const_val0', ...], ...}
    """
    # _logger.info(f'substitute ints for constants in:\n\t{self}')
    old2new = dict()
    for u in tree.nodes():
        if u.type != 'str':
            continue
        var, op = pair_node_to_var(tree, u)
        # now: c, is the operator and: v, the variable
        str2int = var_str2int[str(var)]
        x = str2int.index(u.value)
        num = _ast.nodes.Num(str(x))
        # replace Const with Num
        old2new[u] = num
    nx.relabel_nodes(tree, old2new, copy=False)
    # _logger.info(f'result after substitution:\n\t{self}\n')


def sub_bool_with_subtree(
        tree:
            Tree,
        bool2subtree:
            dict[str, Tree]):
    """Replace selected Boolean variables with given AST.

    @param bool2subtree:
        map from each Boolean variable to some
        equivalent formula. A subset of Boolean varibles may be used.

        Note that the types of variables in `tree`
        are defined by `bool2subtree`.
    """
    for u in list(tree.nodes()):
        if u.type == 'var' and u.value in bool2subtree:
            # tree.write(f'{id(tree)}_before.png')
            tree.add_subtree(u, bool2subtree[u.value])
            # tree.write(f'id(tree)}_after.png')


def pair_node_to_var(
        tree:
            Tree,
        c:
            Node
        ) -> tuple[
            Node,
            Node]:
    """Find variable under `Binary` operator above given node.

    First move up from `nd`, stop at first `Binary` node.
    Then move down, until first `Var`.
    This assumes that only `Unary` operators appear between a
    `Binary` and its variable and constant operands.

    May be extended in the future, depending on what the
    tools support and is thus needed here.

    @type `c`:
        `Const` or
        `Num`
    @return:
        `(variable, constant)`
    """
    # find parent Binary operator
    while True:
        old = c
        c = next(iter(tree.predecessors(c)))
        if c.type == 'operator':
            if len(c.operands) == 2:
                break
    p, q = tree.successors(c)
    v = p if q == old else q
    # go down until terminal found
    # assuming correct syntax for gr1c
    while True:
        if not tree.succ.get(v):
            break
        v = next(iter(tree.successors(v)))
    # now: b, is the operator and: v, the variable
    return v, c


def infer_constants(
        formula:
            str,
        variables:
            dict |
            _abc.Iterable[str]
        ) -> str:
    """Enclose all non-variable names in quotes.

    @param formula:
        well-formed LTL formula
    @param variables:
        domains of variables (for `GRSpec`), or
        only their names.
        If the domains are given, then they are checked
        for ambiguities as for example a variable name
        duplicated as a possible value in the domain of
        a string variable (the same or another).

        If the names are given only, then a warning is raised,
        because ambiguities cannot be checked in that case,
        since they depend on what domains will be used.
    @return:
        `formula` with all string literals not in `variables`
        enclosed in double quotes
    """
    if isinstance(variables, dict):
        for var in variables:
            other_vars = dict(variables)
            other_vars.pop(var)
            _check_var_conflicts({var}, other_vars)
    else:
        _logger.error('infer constants does not know the variable domains.')
        warnings.warn(
            'infer_constants can give an incorrect result '
            'depending on the variable domains.\n'
            'If you give the variable domain definitions as dict, '
            'then infer_constants will check for ambiguities.')
    rec_ast = parser.parse(formula)
    tree = Tree.from_recursive_ast(rec_ast)
    old2new = dict()
    for u in tree:
        if u.type != 'var':
            continue
        if str(u) in variables:
            continue
        # Var (so NAME token) but not a variable
        # turn it into a string constant
        old2new[u] = _ast.nodes.Str(f'"{u}"')
    nx.relabel_nodes(tree, old2new, copy=False)
    return tree.to_recursive_ast().flatten()


def _check_var_conflicts(
        s:
            set[str],
        variables:
            dict):
    """Raise exception if set intersects existing variable name, or values.

    Values refers to arbitrary finite data types.

    @param s:
        set
    @param variables:
        definitions of variable types
    """
    # check conflicts with variable names
    vars_redefined = {x for x in s if x in variables}
    if vars_redefined:
        raise Exception(f'Variables redefined: {vars_redefined}')
    # check conflicts with values of arbitrary finite data types
    for var, domain in variables.items():
        # not arbitrary finite type ?
        if not isinstance(domain, list):
            continue
        # var has arbitrary finite type
        conflicting_values = {x for x in s if x in domain}
        if conflicting_values:
            raise Exception(
                f'Values redefined: {conflicting_values}')


def check_var_name_conflict(
        f:
            str,
        varname:
            str
        ) -> set[str]:
    t = parser.parse(f)
    g = Tree.from_recursive_ast(t)
    v = {x.value for x in g.variables}
    if varname in v:
        raise ValueError(f'var name "{varname}" already used')
    return v


def collect_primed_vars(
        t:
            Node
        ) -> set[str]:
    """Return variable identifiers that are within scope of operator `'`."""
    g = Tree.from_recursive_ast(t)
    # (node, context)
    Q = [(t, False)]
    primed = set()
    while Q:
        u, c = Q.pop()
        if u.type == 'var' and c:
            primed.add(u.value)
        try:
            c = (u.operator == 'X') or c
        except AttributeError:
            pass
        Q.extend((v, c) for v in g.successors(u))
    return primed


# defunct until further notice
def _flatten(
        tree:
            Tree,
        u:
            Node,
        to_lang:
            _abc.Callable,
        **kw
        ) -> str:
    """Recursively flatten `tree`."""
    s = tree.succ[u]
    if not s:
        return to_lang(u, **kw)
    elif len(s) == 2:
        l, r = s
        if 1 in s[l]:
            l, r = r, l
        l = _flatten(tree, l, to_lang, **kw)
        r = _flatten(tree, r, to_lang, **kw)
        return to_lang(u, l, r, **kw)
    else:
        c, = s
        if u.op == 'X':
            return to_lang(u, _flatten(tree, c, to_lang,
                           prime=True, **kw), **kw)
        else:
            return to_lang(u, _flatten(tree, c, to_lang, **kw), **kw)
