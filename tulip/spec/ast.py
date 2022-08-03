# Copyright (c) 2011-2015 by California Institute of Technology
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
"""Abstract syntax tree classes for LTL.

Syntax taken originally roughly from:
<http://spot.lip6.fr/wiki/LtlSyntax>
"""
import collections.abc as _abc
import logging
import typing as _ty


_logger = logging.getLogger(__name__)


OpMap = dict[str, str]
# prototype for flattening to a "canonical" string
OPMAP = {
    'False':
        'False',
    'True':
        'True',
    '!':
        '!',
    '|':
        '|',
    '&':
        '&',
    '->':
        '->',
    '<->':
        '<->',
    '^':
        '^',
    'ite':
        'ite',
    'X':
        'X',
    'G':
        'G',
    'F':
        'F',
    'U':
        'U',
    'W':
        'W',
    'V':
        'V',
    '<':
        '<',
    '<=':
        '<=',
    '=':
        '=',
    '>=':
        '>=',
    '>':
        '>',
    '!=':
        '!=',
    '+':
        '+',
    '-':
        '-',
    '*':
        '*',
    '/':
        '/',
    '<<>>':
        '<<>>'
        # arithmetic
    }
# this mapping is based on SPIN documentation:
#   <http://spinroot.com/spin/Man/ltl.html>
FULL_OPERATOR_NAMES = {
    'next':
        'X',
    'always':
        '[]',
    'eventually':
        '<>',
    'until':
        'U',
    'stronguntil':
        'U',
    'weakuntil':
        'W',
    'unless':
        'W',
        # see Baier - Katoen
    'release':
        'V',
    'implies':
        '->',
    'equivalent':
        '<->',
    'not':
        '!',
    'and':
        '&&',
    'or':
        '||',
    }


class NodeSpec(_ty.Protocol):
    """Base class for AST nodes."""

    opmap: (
        OpMap |
        None
        ) = None

    def __init__(self):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def flatten(self):
        raise NotImplementedError


class TerminalSpec(NodeSpec):
    """Terminal symbols of grammar.

    Include:

    - 0-ary function constants (numbers, strings)
    - 0-ary function variables (integer or string variable)
    - 0-ary connectives (Boolean constants)
    - 0-ary predicate constants
    - 0-ary predicate variables
    """

    def __init__(self, value):
        try:
            value + 'a'
        except TypeError:
            raise TypeError(
                f'value must be a string, got: {value}')
        self.type = 'terminal'
        self.value = value

    def __repr__(self):
        t = type(self).__name__
        v = repr(self.value)
        return f'{t}({v})'

    def __hash__(self):
        return id(self)

    def __str__(self, *arg, **kw):
        # *arg accommodates "depth" arg of Operator.__str__
        return self.value

    def __len__(self):
        """Return the number of operators and terminals.

        Note that this definition differs from the
        theoretical definition that a formula's length
        is the number of operators it contains.
        """
        return 1

    def __eq__(self, other):
        return (isinstance(other, type(self)) and
                self.value == other.value)

    def flatten(self, *arg, **kw):
        return self.value


class OperatorSpec(NodeSpec):
    """Takes a non-zero number of operands and returns a result.

    Cases:

      - function (arithmetic):
          maps (terms)^n to terms
      - predicate (relational operator):
          maps (terms)^n to atomic formulas
      - connective (logical operator):
          maps (wff)^n to wff
    """

    def __init__(self, operator, *operands):
        try:
            operator + 'a'
        except TypeError:
            raise TypeError(
                f'operator must be string, got: {operator}')
        self.type = 'operator'
        self.operator = operator
        self.operands = list(operands)

    def __repr__(self):
        t = type(self).__name__
        op = repr(self.operator)
        xyz = _comma(map(
            repr, self.operands))
        return f'{t}({op}, {xyz})'

    # depth allows limiting recursion to see a shallower view
    def __str__(self, depth=None):
        if depth is not None:
            depth = depth - 1
        if depth == 0:
            return '...'
        op = self.operator
        def str_of(x):
            return x.__str__(depth=depth)
        xyz = ' '.join(map(
            str_of, self.operands))
        return f'({op} {xyz})'

    def __len__(self):
        return 1 + sum(map(len, self.operands))

    def flatten(self, *arg, **kw):
        def flatten(x):
            return x.flatten(*arg, **kw)
        op = self.opmap[self.operator]
        args = _comma(map(
            flatten, self.operands))
        return f'( {op} {args} )'


class BinarySpec(OperatorSpec):
    def flatten(self, *arg, **kw):
        """Infix flattener for consistency with parser.

        Override it if you want prefix or postfix.
        """
        op = self.opmap[self.operator]
        def flatten(x):
            return x.flatten(*arg, **kw)
        first, second = map(flatten, self.operands)
        return f'( {first} {op} {second} )'


class NodesSpec(_ty.Protocol):
    """AST nodes for a generic grammar."""

    Node: NodeSpec
    Terminal: TerminalSpec
    Operator: OperatorSpec
    Unary: OperatorSpec
    Binary: BinarySpec


class VarSpec(TerminalSpec):
    """A 0-ary variable.

    Two cases:

      - 0-ary function variable (integer or string variable)
      - 0-ary propositional variable (atomic proposition)
    """

    def __init__(self, value):
        super().__init__(value)
        self.type = 'var'


class BoolSpec(TerminalSpec):
    """A 0-ary connective."""

    def __init__(self, value):
        try:
            value + 'a'
        except TypeError:
            raise TypeError(
                f'value must be string, got: {value}')
        if value.lower() not in {'true', 'false'}:
            raise TypeError(
                'value must be "true" or "false" '
                f'(case insensitive), got: {value}')
        self.value = value.title()
        self.type = 'bool'

    def flatten(self, *arg, **kw):
        return self.opmap[self.value]


class NumSpec(TerminalSpec):
    """A 0-ary function."""

    def __init__(self, value):
        super().__init__(value)
        self.type = 'num'


class StrSpec(TerminalSpec):
    """A 0-ary function."""
    # parser ensures that value has no quotes

    def __init__(self, value):
        super().__init__(value)
        self.type = 'str'


class ComparatorSpec(BinarySpec):
    """Binary relational operator (2-ary predicate)."""


class ArithmeticSpec(BinarySpec):
    """Binary function.

    Maps terms to terms.
    """


def make_nodes(
        opmap:
            OpMap |
            None=None
        ) -> NodesSpec:
    """Return class with attributes the AST node classes.

    The tree is defined recursively,
    not with a graph data structure.
    `Tree` is a graph data structure for that purpose.
    """
    if opmap is None:
        _opmap = OPMAP
    else:
        _opmap = opmap
    class Node(NodeSpec):
        opmap = _opmap
    class Terminal(TerminalSpec, Node):
        pass
    class Operator(OperatorSpec, Node):
        pass
    class Unary(Operator):
        pass
    class Binary(BinarySpec, Operator):
        pass
    class Nodes(
            NodesSpec):
        pass
    nodes = Nodes()
    nodes.Node = Node
    nodes.Terminal = Terminal
    nodes.Operator = Operator
    nodes.Unary = Unary
    nodes.Binary = Binary
    return nodes


def make_fol_nodes(
        opmap:
            OpMap |
            None=None
        ) -> NodesSpec:
    """AST classes for fragment of first-order logic."""
    nodes = make_nodes(opmap)
    class Var(VarSpec, nodes.Terminal):
        pass
    class Bool(BoolSpec, nodes.Terminal):
        pass
    class Num(NumSpec, nodes.Terminal):
        pass
    class Str(StrSpec, nodes.Terminal):
        pass
    class Comparator(ComparatorSpec, nodes.Binary):
        pass
    class Arithmetic(ArithmeticSpec, nodes.Binary):
        pass
    nodes.Var = Var
    nodes.Bool = Bool
    nodes.Num = Num
    nodes.Str = Str
    nodes.Comparator = Comparator
    nodes.Arithmetic = Arithmetic
    return nodes


nodes = make_fol_nodes()


def _comma(
        items:
            _abc.Iterable
        ) -> str:
    """Catenate with comma in-between.

    Between the `__str__`-representations
    of `items`, the string `', '` is
    placed, in the returned value.
    """
    return ', '.join(items)
