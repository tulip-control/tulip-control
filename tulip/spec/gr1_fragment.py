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
"""Test if given formula belongs to an LTL fragment that
is convertible to deterministic Buchi Automata
(readily expressible in GR(1) ).

reference
=========
1. Andreas Morgenstern and Klaus Schneider,
   A LTL Fragment for GR(1)-Synthesis,
   in Proceedings First International Workshop on
   Interactions, Games and Protocols (iWIGP),
   Electronic Proceedings in Theoretical Computer Science (EPTCS),
   50, pp. 33--45, 2011,
   http://doi.org/10.4204/EPTCS.50.3
"""
from __future__ import absolute_import

import logging
logger = logging.getLogger(__name__)

import networkx as nx
from tulip import transys as trs
from tulip.spec import lexyacc, GRSpec
from tulip.spec import transformation as tx
from tulip.spec import parser
from tulip.spec import ast as sast


def check(formula):
    """Parse formula string and create abstract syntax tree (AST).
    """
    ast = lexyacc.parse(formula)

    dfa = trs.automata.FiniteWordAutomaton(atomic_proposition_based=False,
                                           deterministic=True)

    dfa.alphabet |= {'!', 'W', 'U', 'G', 'F',
                     'U_left', 'U_right',
                     'W_left', 'W_right'}

    dfa.states.add_from({'gf', 'fg', 'g', 'f'})
    dfa.states.initial.add('gf')

    dfa.transitions.add('gf', 'fg', letter='!')
    dfa.transitions.add('fg', 'gf', letter='!')
    dfa.transitions.add('g', 'f', letter='!')
    dfa.transitions.add('f', 'g', letter='!')

    dfa.transitions.add('gf', 'gf', letter='W')
    dfa.transitions.add('gf', 'gf', letter='U_left')
    dfa.transitions.add('gf', 'gf', letter='G')

    dfa.transitions.add('fg', 'fg', letter='U')
    dfa.transitions.add('fg', 'fg', letter='F')
    dfa.transitions.add('fg', 'fg', letter='W_right')

    dfa.transitions.add('gf', 'f', letter='U_right')
    dfa.transitions.add('gf', 'f', letter='F')

    dfa.transitions.add('fg', 'g', letter='W_left')
    dfa.transitions.add('fg', 'g', letter='G')

    dfa.transitions.add('g', 'g', letter='W')
    dfa.transitions.add('g', 'g', letter='G')

    dfa.transitions.add('f', 'f', letter='U')
    dfa.transitions.add('f', 'f', letter='F')

    # plot tree automaton
    # dfa.save('dfa.pdf')

    # plot parse tree
    sast.dump_dot(ast, 'ast.dot')

    # sync product of AST with DFA,
    # to check acceptance
    Q = [(ast, 'gf')]
    while Q:
        s, q = Q.pop()
        logger.info('visiting: ' + str(s) + ', ' + str(q))

        if isinstance(s, sast.Unary):
            op = s.operator

            if op in {'!', 'G', 'F'}:
                t = dfa.transitions.find(q, letter=op)

                if not t:
                    raise Exception('not in fragment')

                qi, qj, w = t[0]

                Q.append((s.operand, qj))
            else:
                # ignore
                Q.append((s.operand, q))
        elif isinstance(s, sast.Binary):
            op = s.operator

            if op in {'W', 'U'}:
                t = dfa.transitions.find(q, letter=op)
                if t:
                    qi, qj, w = t[0]
                    Q.append((s.op_l, qj))
                    Q.append((s.op_r, qj))
                else:
                    t = dfa.transitions.find(q, letter=op + '_left')

                    if not t:
                        raise Exception('not in fragment')

                    qi, qj, w = t[0]

                    Q.append((s.op_l, qj))

                    t = dfa.transitions.find(q, letter=op + '_right')

                    if not t:
                        raise Exception('not in fragment')

                    qi, qj, w = t[0]

                    Q.append((s.op_r, qj))
            else:
                # ignore
                Q.append((s.op_l, q))
                Q.append((s.op_r, q))
        elif isinstance(s, sast.Var):
            print('reached var')

    return ast


def str_to_grspec(f):
    """Return `GRSpec` from LTL formula `f` as `str`.

    Formula `f` must be in the form:

      A -> G

    where each of A, G is a conjunction of terms: `B`, `[]C`, `[]<>B`.
    For more details on `B, C`, see [split_gr1].

    @type f: `str`
    @rtype: [GRSpec]
    """
    t = parser.parse(f)
    assert t.operator == '->'
    env, sys = t.operands
    d = {'assume': split_gr1(env),
         'assert': split_gr1(sys)}
    return GRSpec(env_init=d['assume']['init'],
                  env_safety=d['assume']['G'],
                  env_prog=d['assume']['GF'],
                  sys_init=d['assert']['init'],
                  sys_safety=d['assert']['G'],
                  sys_prog=d['assert']['GF'])


def split_gr1(f):
    """Return `dict` of GR(1) subformulae.

    The formula `f` is assumed to be a conjunction of expressions
    of the form:
      `B`, `[]C` or `[]<>B`
    where:
      - `C` can contain "next"
      - `B` cannot contain "next"

    @param f: temporal logic formula
    @type f: `str` or AST

    @return: conjunctions of formulae A, B as `str`, grouped by keys:
        `'init', 'G', 'GF'`
    @rtype: `dict` of `str`: `list` of `str`
    """
    # TODO: preprocess by applying syntactic identities: [][] = [] etc
    try:
        f + 's'
        t = parser.parse(f)
    except TypeError:
        t = f
    g = tx.Tree.from_recursive_ast(t)
    # collect boundary of conjunction operators
    Q = [g.root]
    b = list()  # use lists to preserve as much given syntactic order
    while Q:
        u = Q.pop()
        # terminal ?
        if not g.succ.get(u):
            b.append(u)
            continue
        # operator
        if u.operator == '&':
            # use `u.operands` instead of `g.successors`
            # to preserve original order
            Q.extend(u.operands)
        else:
            b.append(u)
    d = {'init': list(), 'G': list(), 'GF': list()}
    for u in b:
        # terminal ?
        if not g.succ.get(u):
            d['init'].append(u)
            continue
        # some operator
        if u.operator != 'G':
            d['init'].append(u)
            continue
        # G
        (v,) = u.operands
        # terminal in G ?
        if not g.succ.get(v):
            d['G'].append(v)
            continue
        # some operator in G
        if v.operator == 'F':
            (w,) = v.operands
            d['GF'].append(w)
        else:
            # not a GF
            d['G'].append(v)
    # assert only admissible temporal operators
    ops = {'G', 'F', 'U', 'V', 'R'}
    operators = {'G': ops}
    ops = set(ops)
    ops.add('X')
    operators.update(init=ops, GF=ops)
    for part, f in d.iteritems():
        ops = operators[part]
        for u in f:
            op = has_operator(u, g, ops)
            if op is None:
                continue
            raise AssertionError((
                'found inadmissible operator "{op}" '
                'in "{f}" formula').format(
                    op=op, f=u))
    # conjoin (except for progress)
    init = ' & '.join(u.flatten() for u in reversed(d['init']))
    d['init'] = [init]
    safe = ' & '.join(u.flatten() for u in reversed(d['G']))
    d['G'] = [safe]
    # flatten individual progress formulae
    d['GF'] = [u.flatten() for u in d['GF']]
    return d


def has_operator(u, g, operators):
    try:
        if u.operator in operators:
            return u.operator
    except AttributeError:
        pass
    for v in nx.descendants(g, u):
        # terminal
        if not g.succ.get(v):
            continue
        # operator
        # is it temporal except for 'X' ?
        if v.operator in operators:
            return v.operator
    return None


def stability_to_gr1(p, aux='aux'):
    """Convert C{<>[] p} to GR(1).

    Warning: This conversion is sound, but not complete.
    See p.2, U{[E10]
    <http://tulip-control.sourceforge.net/doc/bibliography.html#e10>}

    GR(1) form::

        !(aux) &&
        [](aux -> X aux) &&
        []<>(aux) &&

        [](aux -> p)

    @type p: str

    @param aux: name to use for auxiliary variable
    @type aux: str

    @rtype: L{GRSpec}
    """
    logging.warning(
        'Conversion of stability (<>[]p) to GR(1)' +
        'is sound, but NOT complete.'
    )

    a = aux
    a0 = a

    p = _paren(p)
    a = _paren(a)

    v = tx.check_var_name_conflict(p, a0)

    sys_vars = v | {a0}
    sys_init = {'!' + a}
    sys_safe = {a + ' -> ' + p,
                a + ' -> X ' + a}
    sys_prog = {a}

    return GRSpec(sys_vars=sys_vars, sys_init=sys_init,
                  sys_safety=sys_safe, sys_prog=sys_prog)


def response_to_gr1(p, q, aux='aux'):
    """Convert C{[](p -> <> q)} to GR(1).

    GR(1) form::

        []<>(aux) &&

        []( (p && !q) -> X ! aux) &&
        []( (! aux && !q) -> X ! aux)

    @type p: str

    @type q: str

    @param aux: name to use for auxiliary variable
    @type aux: str

    @rtype: L{GRSpec}
    """
    a = aux
    a0 = a

    p = _paren(p)
    q = _paren(q)
    a = _paren(a)

    s = p + ' -> <> ' + q
    v = tx.check_var_name_conflict(s, a0)

    sys_vars = v | {a0}
    # sys_init = {a}
    sys_safe = {
        '(' + p + ' && !' + q + ') -> X !' + a,
        '(!' + a + ' && !' + q + ') -> X !' + a
    }
    sys_prog = {a}

    return GRSpec(sys_vars=sys_vars,  # sys_init=sys_init,
                  sys_safety=sys_safe, sys_prog=sys_prog)


def eventually_to_gr1(p, aux='aux'):
    """Convert C{<> p} to GR(1).

    GR(1) form::

        !(aux) &&
        [](aux -> X aux) &&
        []<>(aux) &&

        []( (!p && !aux) -> X!(aux) )

    @type p: str

    @param aux: name to use for auxiliary variable
    @type aux: str

    @rtype: L{GRSpec}
    """
    a = aux
    a0 = a

    p = _paren(p)
    a = _paren(a)

    v = tx.check_var_name_conflict(p, a0)

    sys_vars = v | {a0}
    sys_init = {'!(' + a + ')'}
    sys_safe = {
        '(!' + p + ' && !' + a + ') -> X !' + a,
        a + ' -> X ' + a
    }
    sys_prog = {a}

    return GRSpec(sys_vars=sys_vars, sys_init=sys_init,
                  sys_safety=sys_safe, sys_prog=sys_prog)


def until_to_gr1(p, q, aux='aux'):
    """Convert C{p U q} to GR(1).

    GR(1) form::

        (!q -> !aux) &&
        [](q -> aux)
        [](aux -> X aux) &&
        []<>(aux) &&

        []( (!aux && X(!q) ) -> X!(aux) ) &&
        [](!aux -> p)

    @type p: str

    @param aux: name to use for auxiliary variable
    @type aux: str

    @rtype: L{GRSpec}
    """
    a = aux
    a0 = a

    p = _paren(p)
    q = _paren(q)
    a = _paren(a)

    s = p + ' && ' + q
    v = tx.check_var_name_conflict(s, a0)

    sys_vars = v | {a0}
    sys_init = {'!' + q + ' -> !' + a}
    sys_safe = {
        q + ' -> ' + a,
        '( (X !' + q + ') && !' + a + ') -> X !' + a,
        a + ' -> X ' + a,
        '(!' + a + ') -> ' + p
    }
    sys_prog = {a}

    return GRSpec(sys_vars=sys_vars, sys_init=sys_init,
                  sys_safety=sys_safe, sys_prog=sys_prog)


def _paren(x):
    return '({x})'.format(x=x)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    s = '(a U b) && []a && <>a && <>a && []<>(<>z)'
    parsed_formula = check(s)

    print('Parsing result: ' + str(parsed_formula))
