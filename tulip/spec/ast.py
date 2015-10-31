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
http://spot.lip6.fr/wiki/LtlSyntax
"""
import logging
logger = logging.getLogger(__name__)
from abc import ABCMeta, abstractmethod


# prototype for flattening to a "canonical" string
OPMAP = {
    'False': 'False', 'True': 'True',
    '!': '!',
    '|': '|', '&': '&', '->': '->', '<->': '<->', '^': '^', 'ite': 'ite',
    'X': 'X', 'G': 'G', 'F': 'F',
    'U': 'U', 'W': 'W', 'V': 'V',
    '<': '<', '<=': '<=', '=': '=', '>=': '>=', '>': '>', '!=': '!=',
    '+': '+', '-': '-', '<<>>': '<<>>'  # arithmetic
}

# this mapping is based on SPIN documentation:
#   http://spinroot.com/spin/Man/ltl.html
FULL_OPERATOR_NAMES = {
    'next': 'X',
    'always': '[]',
    'eventually': '<>',
    'until': 'U',
    'stronguntil': 'U',
    'weakuntil': 'W',
    'unless': 'W',  # see Baier - Katoen
    'release': 'V',
    'implies': '->',
    'equivalent': '<->',
    'not': '!',
    'and': '&&',
    'or': '||',
}


def make_nodes(opmap=None):
    """Return class with attributes the AST node classes.

    The tree is defined recursively,
    not with a graph data structure.
    L{Tree} is a graph data structure for that purpose.
    """
    if opmap is None:
        opmap = OPMAP

    class Node(object):
        """Base class for AST nodes."""

        # Caution
        # =======
        # Do **NOT** implement C{__hash__}, because you
        # will unintendently identify different AST nodes !
        # Only leaf nodes can be safely identified.
        #
        # The default for user-defined classes is
        # C{__hash__ == _}
        __metaclass__ = ABCMeta
        opmap = None

        @abstractmethod
        def __init__(self):
            pass

        @abstractmethod
        def __repr__(self):
            pass

        @abstractmethod
        def flatten(self):
            pass

    Node.opmap = opmap

    # Do not confuse "term" with the abbreviation of "terminal".
    # A "term" in FOL can comprise of terminals,
    # for example a function together with parentheses and its args.
    class Terminal(Node):
        """Terminal symbols of grammar.

        Include:

          - 0-ary function constants (numbers, strings)
          - 0-ary function variables (integer or string variable)
          - 0-ary connectives (Boolean constants)
          - 0-ary predicate constants
          - 0-ary predicate variables
        """

        def __init__(self, value):
            if not isinstance(value, basestring):
                raise TypeError(
                    'value must be a string, got: {v}'.format(
                        v=value))
            self.type = 'terminal'
            self.value = value

        def __repr__(self):
            return '{t}({v})'.format(t=type(self).__name__,
                                     v=repr(self.value))

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

    class Operator(Node):
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
                    'operator must be string, got: {op}'.format(
                        op=operator))
            self.type = 'operator'
            self.operator = operator
            self.operands = list(operands)

        # ''.join would be faster, but __repr__ is for debugging,
        # not for flattening, so readability takes precedence
        def __repr__(self):
            return '{t}({op}, {xyz})'.format(
                t=type(self).__name__,
                op=repr(self.operator),
                xyz=', '.join(repr(x) for x in self.operands))

        # more readable recursive counterpart of __repr__
        # depth allows limiting recursion to see a shallower view
        def __str__(self, depth=None):
            if depth is not None:
                depth = depth - 1
            if depth == 0:
                return '...'
            return '({op} {xyz})'.format(
                op=self.operator,
                xyz=' '.join(x.__str__(depth=depth)
                             for x in self.operands))

        def __len__(self):
            return 1 + sum(len(x) for x in self.operands)

        def flatten(self, *arg, **kw):
            return ' '.join([
                '(',
                self.opmap[self.operator],
                ', '.join(x.flatten(*arg, **kw) for x in self.operands),
                ')'])

    # Distinguish operators by arity
    class Unary(Operator):
        pass

    class Binary(Operator):
        def flatten(self, *arg, **kw):
            """Infix flattener for consistency with parser.

            Override it if you want prefix or postfix.
            """
            return ' '.join([
                '(',
                self.operands[0].flatten(*arg, **kw),
                self.opmap[self.operator],
                self.operands[1].flatten(*arg, **kw),
                ')'])

    class Nodes(object):
        """AST nodes for a generic grammar."""

    nodes = Nodes()
    nodes.Node = Node
    nodes.Terminal = Terminal
    nodes.Operator = Operator
    nodes.Unary = Unary
    nodes.Binary = Binary
    return nodes


def make_fol_nodes(opmap=None):
    """AST classes for fragment of first-order logic."""
    nodes = make_nodes(opmap)

    class Var(nodes.Terminal):
        """A 0-ary variable.

        Two cases:

          - 0-ary function variable (integer or string variable)
          - 0-ary propositional variable (atomic proposition)
        """

        def __init__(self, value):
            super(Var, self).__init__(value)
            self.type = 'var'

    class Bool(nodes.Terminal):
        """A 0-ary connective."""

        def __init__(self, value):
            if not isinstance(value, basestring):
                raise TypeError(
                    'value must be string, got: {v}'.format(v=value))
            if value.lower() not in {'true', 'false'}:
                raise TypeError(
                    'value must be "true" or "false" '
                    '(case insensitive), got: {v}'.format(v=value))
            self.value = 'True' if (value.lower() == 'true') else 'False'
            self.type = 'bool'

        def flatten(self, *arg, **kw):
            return self.opmap[self.value]

    class Num(nodes.Terminal):
        """A 0-ary function."""
        # self.value is str,
        # use int(self.value) if you need to

        def __init__(self, value):
            super(Num, self).__init__(value)
            self.type = 'num'

    class Str(nodes.Terminal):
        """A 0-ary function."""
        # parser ensures that value has no quotes

        def __init__(self, value):
            super(Str, self).__init__(value)
            self.type = 'str'

    class Comparator(nodes.Binary):
        """Binary relational operator (2-ary predicate)."""

    class Arithmetic(nodes.Binary):
        """Binary function.

        Maps terms to terms.
        """

    nodes.Var = Var
    nodes.Bool = Bool
    nodes.Num = Num
    nodes.Str = Str
    nodes.Comparator = Comparator
    nodes.Arithmetic = Arithmetic
    return nodes


nodes = make_fol_nodes()
