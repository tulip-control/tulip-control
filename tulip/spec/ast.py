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
"""
Abstract Syntax Tree classes for LTL,


Syntax taken originally roughly from:
http://spot.lip6.fr/wiki/LtlSyntax
"""
import logging
logger = logging.getLogger(__name__)

import os
import re
import networkx as nx

OP_MAP = {
    'False': 'False', 'True': 'True',
    '!':'!',
    '|': '|', '&': '&', '->': '->', '<->': '<->',
    '[]': 'G', 'G': 'G',
    'F': 'F', '<>': 'F',
    'X': 'X', 'next': 'X', "'": 'X',
    'U': 'U', 'V': 'R', 'R': 'R',
    '==': '=',
    '<': '<', '<=': '<=', '=': '=', '>=': '>=', '>': '>', '!=': '!='
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



class Node(object):
    """Base class for deriving AST nodes."""
    # Caution
    # =======
    # Do **NOT** implement C{__hash__}, because you
    # will unintendently identify different AST nodes !
    #
    # The default for user-defined classes is
    # C{__hash__ == _}
    def to_string(self, *arg, **kw):
        return self.flatten('string', *arg, **kw)


class Term(Node):
    def __init__(self, t):
        self.val = t

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.val == other.val

    def __repr__(self):
        return str(self.val)

    def flatten(self, *arg, **kw):
        return self.val

    def eval(self, d):
        return self.val

class Num(Term):
    def __init__(self, t):
        self.val = int(t)

    def flatten(self, *arg, **kw):
        return str(self.val)

class Var(Term):

    def eval(self, d):
        return d[self.val]

class Const(Term):
    def __init__(self, t):
        self.val = re.sub(r'^"|"$', '', t)

    def __repr__(self):
        return r'"%s"' % self.val


class Bool(Term):
    def __init__(self, t):
        self.val = (t.upper() == 'TRUE')
        self.str = 'True' if self.val else 'False'

    def __repr__(self):
        return self.str

    def flatten(self, lang, **kw):
        return maps[lang][str(self)]

class Operator(Node):
    def __init__(self, operator):
        self.operator = OP_MAP.get(operator, operator)

    def __repr__(self):
        return self.op

    @property
    def op(self):
        return self.operator

class Unary(Operator):
    def flatten(self, lang, x, **kw):
        return '( %s %s )' % (maps[lang][self.op], x)

class Not(Unary):
    @property
    def op(self):
        return '!'

    def eval(self, d):
        return not self.operand.eval(d)

class UnTempOp(Unary):
    def context(self):
        return self.op == 'X'

class Binary(Operator):
    def flatten(self, lang, l, r, **kw):
        return '( %s %s %s )' % (l, maps[lang][self.op], r)

    def eval(self, stack, d):
        try:
            l, r = self._consume(stack)
            return self._eval(l, r)
        except AttributeError:
            raise LTLException()

class And(Binary):
    @property
    def op(self):
        return '&'

    def _eval(self, l, r):
        return l and r

class Or(Binary):
    @property
    def op(self):
        return '|'

    def _eval(self, l, r):
        return l or r

class Xor(Binary):
    @property
    def op(self):
        return 'xor'

    def _eval(self, l, r):
        return l ^ r

class Imp(Binary):
    @property
    def op(self):
        return '->'

    def _eval(self, l, r):
        return not l or r

class BiImp(Binary):
    @property
    def op(self):
        return '<->'

    def _eval(self, l, r):
        return l == r

class BiTempOp(Binary):
    pass

class Comparator(Binary):
    pass

class Arithmetic(Binary):
    pass

