# Copyright (c) 2011-2014 by California Institute of Technology
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
    '|': '|', '&': '&', '->': '->', '<->': '<->', '^': '^',
    'X': 'X', 'G': 'G', 'F': 'F',
    'U': 'U', 'V': 'R', 'R': 'R',
    '<': '<', '<=': '<=', '=': '=', '>=': '>=', '>': '>', '!=': '!=',
    '+': '+', '-': '-'  # linear arithmetic
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
            self.value = value

        def __repr__(self):
            return '{t}({v})'.format(t=type(self).__name__,
                                     v=repr(self.value))

        def __str__(self):
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

        def __str__(self):
            return self.operator

    # Distinguish operators by arity
    class Unary(Operator):
        def __init__(self, operator, operand):
            if not isinstance(operator, basestring):
                raise TypeError(
                    'operator must be a string, got: {op}'.format(
                        op=operator))
            self.operator = operator
            self.operand = operand

        def __repr__(self):
            return '{t}({op}, {x})'.format(
                t=type(self).__name__,
                op=repr(self.operator),
                x=repr(self.operand))

        def __len__(self):
            return 1 + len(self.operand)

        def flatten(self, *arg, **kw):
            return '( {op} {x} )'.format(
                op=self.opmap[self.operator],
                x=self.operand.flatten(*arg, **kw))

    class Binary(Operator):
        def __init__(self, operator, left, right):
            if not isinstance(operator, basestring):
                raise TypeError(
                    'operator must be a string, got: {op}'.format(
                        op=operator))
            self.operator = operator
            self.left = left
            self.right = right

        def __repr__(self):
            return '{t}({op}, {x}, {y})'.format(
                t=type(self).__name__,
                op=repr(self.operator),
                x=repr(self.left),
                y=repr(self.right))

        def __len__(self):
            return 1 + len(self.left) + len(self.right)

        def flatten(self, *arg, **kw):
            """Infix flattener.

            Override it if you want prefix or postfix.
            """
            return '( {x} {op} {y} )'.format(
                op=self.opmap[self.operator],
                x=self.left.flatten(*arg, **kw),
                y=self.right.flatten(*arg, **kw))

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

        def flatten(self, *arg, **kw):
            return self.opmap[self.value]

    class Num(nodes.Terminal):
        """A 0-ary function."""
        # self.value is str,
        # use int(self.value) if you need to

    class Str(nodes.Terminal):
        """A 0-ary function."""
        # parser ensures that value has no quotes

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
