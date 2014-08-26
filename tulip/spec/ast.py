# Copyright (c) 2011-2013 by California Institute of Technology
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

supporting JTLV, SPIN, SMV, and gr1c syntax

Syntax taken originally roughly from:
http://spot.lip6.fr/wiki/LtlSyntax
"""
import logging
logger = logging.getLogger(__name__)

import subprocess

# inline:
#
# import networkx

TEMPORAL_OP_MAP = {
    'G':'G', 'F':'F', 'X':'X',
    '[]':'G', '<>':'F', 'next':'X',
    'U':'U', 'V':'R', 'R':'R', 
    "'":'X', 'FALSE':'False', 'TRUE':'True'
}

JTLV_MAP = {
    'G':'[]', 'F':'<>', 'X':'next',
    'U':'U', '||':'||', '&&':'&&',
    'False':'FALSE', 'True':'TRUE'
}

GR1C_MAP = {
    'G':'[]', 'F':'<>', 'X':"'", '||':'|', '&&':'&',
    'False':'False', 'True':'True'
}

SMV_MAP = {'G':'G', 'F':'F', 'X':'X', 'U':'U', 'R':'V'}

SPIN_MAP = {'G':'[]', 'F':'<>', 'U':'U', 'R':'V'}

# this mapping is based on SPIN documentation:
#   http://spinroot.com/spin/Man/ltl.html
FULL_OPERATOR_NAMES = {
    'next':'X',
    'always':'[]',
    'eventually':'<>',
    'until':'U',
    'stronguntil':'U',
    'weakuntil':'W',
    'unless':'W', # see Baier - Katoen
    'release':'V',
    'implies':'->',
    'equivalent':'<->',
    'not':'!',
    'and':'&&',
    'or':'||',
}

class LTLException(Exception):
    pass

def to_nx(ast):
    """Convert AST to C{NetworkX.DiGraph}.
    
    For example, the return value of a successful call to L{parser}.
    
    @param ast: L{ASTNode}
    
    @rtype: C{networkx.DiGraph}
    """
    try:
        import networkx as nx
    except ImportError:
        logger.error('failed to import networkx')
        return
    
    g = nx.DiGraph()
    ast.to_nx(g)
    return g

def dump_dot(ast, filename, detailed=False):
    """Create GraphViz dot string from given AST.
    
    @type ast: L{ASTNode}
    
    @rtype: str
    """
    try:
        import networkx as nx
    except ImportError:
        logger.error('failed to import networkx')
        return
    
    g = to_nx(ast)
    
    # show both repr and AST node class in each vertex
    if detailed:
        for u, d in g.nodes_iter(data=True):
            lb = d['label']
            nd = d['node']
            g.node[u]['label'] = str(lb) + '\n' + str(type(nd).__name__)
    
    nx.write_dot(g, filename)

def write_pdf(ast, filename, detailed=False):
    """Layout AST and save result in PDF file.
    """
    dump_dot(ast, filename, detailed)
    subprocess.call(['dot', '-Tpdf', '-O', filename])

# Flattener helpers
def _flatten_gr1c(node, **args):
    return node.to_gr1c(**args)

def _flatten_JTLV(node):
    return node.to_jtlv()

def _flatten_SMV(node):
    return node.to_smv()

def _flatten_Promela(node):
    return node.to_promela()

class Node(object):
    def to_gr1c(self, primed=False):
        return self.flatten(_flatten_gr1c, primed=primed)
    
    def to_jtlv(self):
        return self.flatten(_flatten_JTLV)
    
    def to_smv(self):
        return self.flatten(_flatten_SMV)
    
    def to_promela(self):
        return self.flatten(_flatten_Promela)
    
    def map(self, f):
        n = self.__class__([str(self.val)])
        return f(n)
    
    def __len__(self):
        return 1
    
    def to_nx(self, g):
        u = id(self)
        g.add_node(u, label=self.val, node=self)
        return u
    
class Num(Node):
    def __init__(self, t):
        self.val = int(t[0])
    
    def __repr__(self):
        return str(self.val)
    
    def flatten(self, flattener=None, op=None, **args):
        return str(self)

class Var(Node):
    def __init__(self, t):
        self.val = t[0]
    
    def __repr__(self):
        return self.val
    
    def flatten(self, flattener=str, op=None, **args):
        # just return variable name (quoted?)
        return str(self)
        
    def to_gr1c(self, primed=False):
        if primed:
            return str(self)+"'"
        else:
            return str(self)
        
    def to_jtlv(self):
        return '(' + str(self) + ')'
        
    def to_smv(self):
        return str(self)
        
class Bool(Node):
    def __init__(self, t):
        if t[0].upper() == 'TRUE':
            self.val = True
        else:
            self.val = False
    
    def __repr__(self):
        if self.val:
            return 'True'
        else:
            return 'False'
    
    def flatten(self, flattener=None, op=None, **args):
        return str(self)
    
    def to_gr1c(self, primed=False):
        try:
            return GR1C_MAP[str(self)]
        except KeyError:
            raise LTLException(
                'Reserved word "' + self.op +
                '" not supported in gr1c syntax map'
            )
    
    def to_jtlv(self):
        try:
            return JTLV_MAP[str(self)]
        except KeyError:
            raise LTLException(
                'Reserved word "' + self.op +
                '" not supported in JTLV syntax map'
            )

class Unary(Node):
    @classmethod
    def new(cls, op_node, operator=None):
        return cls([operator, op_node])
    
    def __init__(self, operator, operand):
        self.operator = operator
        self.operand = operand
    
    def __repr__(self):
        return ' '.join(['(', self.op, str(self.operand), ')'])
    
    def flatten(self, flattener=str, op=None, **args):
        if op is None:
            op = self.op
        try:
            o = flattener(self.operand, **args)
        except AttributeError:
            o = str(self.operand)
        return ' '.join(['(', op, o, ')'])
    
    def to_nx(self, g):
        u = id(self)
        g.add_node(u, label=self.op, node=self)
        
        v = self.operand.to_nx(g)
        g.add_edge(u, v)
        return u
    
    def map(self, f):
        n = self.__class__.new(self.operand.map(f), self.op)
        return f(n)
    
    def __len__(self):
        return 1 + len(self.operand)

class Not(Unary):
    @property
    def op(self):
        return '!'

class UnTempOp(Unary):
    def __init__(self, operator, operand):
        self.operator = TEMPORAL_OP_MAP[operator]
        self.operand = operand
    
    @property
    def op(self):
        return self.operator
    
    def to_promela(self):
        try:
            return self.flatten(_flatten_Promela, SPIN_MAP[self.op])
        except KeyError:
            raise LTLException(
                'Operator ' + self.op +
                ' not supported in Promela syntax map'
            )
    
    def to_gr1c(self, primed=False):
        if self.op == 'X':
            return self.flatten(_flatten_gr1c, '', primed=True)
        else:
            try:
                return self.flatten(
                    _flatten_gr1c, GR1C_MAP[self.op], primed=primed
                )
            except KeyError:
                raise LTLException(
                    'Operator ' + self.op +
                    ' not supported in gr1c syntax map'
                )
    
    def to_jtlv(self):
        try:
            return self.flatten(_flatten_JTLV, JTLV_MAP[self.op])
        except KeyError:
            raise LTLException(
                'Operator ' + self.op +
                ' not supported in JTLV syntax map'
            )
    
    def to_smv(self):
        return self.flatten(_flatten_SMV, SMV_MAP[self.op])

class Binary(Node):
    @classmethod
    def new(cls, op_l, op_r, operator=None):
        return cls([op_l, operator, op_r])
    
    def __init__(self, operator, x, y):
        self.operator = operator
        self.op_l = x
        self.op_r = y
        
    def __repr__(self):
        return ' '.join (['(', str(self.op_l), self.op, str(self.op_r), ')'])
    
    def flatten(self, flattener=str, op=None, **args):
        if not op:
            op = self.op
        
        try:
            l = flattener(self.op_l, **args)
        except AttributeError:
            l = str(self.op_l)
        try:
            r = flattener(self.op_r, **args)
        except AttributeError:
            r = str(self.op_r)
        return ' '.join (['(', l, op, r, ')'])
    
    def to_nx(self, g):
        u = id(self)
        
        v = self.op_l.to_nx(g)
        w = self.op_r.to_nx(g)
        
        g.add_node(u, label=self.op, node=self)
        g.add_edge(u, v)
        g.add_edge(u, w)
        
        return u
    
    def map(self, f):
        n = self.__class__.new(self.op_l.map(f), self.op_r.map(f), self.op)
        return f(n)
    
    def __len__(self):
        return 1 + len(self.op_l) + len(self.op_r)

class And(Binary):
    @property
    def op(self):
        return '&'
    
    def to_jtlv(self):
        return self.flatten(_flatten_JTLV, '&&')
    
    def to_promela(self):
        return self.flatten(_flatten_Promela, '&&')

class Or(Binary):
    @property
    def op(self):
        return '|'
    
    def to_jtlv(self):
        return self.flatten(_flatten_JTLV, '||')
    
    def to_promela(self):
        return self.flatten(_flatten_Promela, '||')

class Xor(Binary):
    @property
    def op(self):
        return 'xor'
    
class Imp(Binary):
    @property
    def op(self):
        return '->'

class BiImp(Binary):
    def op(self):
        return '<->'

class BiTempOp(Binary):
    def __init__(self, operator, x, y):
        print('operator is: '  + str(operator))
        super(BiTempOp, self).__init__(operator, x, y)
        
        self.operator = TEMPORAL_OP_MAP[self.operator]
    
    @property
    def op(self):
        return self.operator
    
    def to_promela(self):
        try:
            return self.flatten(_flatten_Promela, SPIN_MAP[self.op])
        except KeyError:
            raise LTLException(
                'Operator ' + self.op +
                ' not supported in Promela syntax map'
            )
    
    def to_gr1c(self, primed=False):
        try:
            return self.flatten(_flatten_gr1c, GR1C_MAP[self.op])
        except KeyError:
            raise LTLException(
                'Operator ' + self.op +
                ' not supported in gr1c syntax map'
            )
    
    def to_jtlv(self):
        try:
            return self.flatten(_flatten_JTLV, JTLV_MAP[self.op])
        except KeyError:
            raise LTLException(
                'Operator ' + self.op +
                ' not supported in JTLV syntax map'
            )
    
    def to_smv(self):
        return self.flatten(_flatten_SMV, SMV_MAP[self.op])
    
class Comparator(Binary):
    def __init__(self, operator, x, y):
        super(Comparator, self).__init__(operator, x, y)
        
        if self.operator is '==':
            self.operator = '='
    
    @property
    def op(self):
        return self.operator
    
    def to_promela(self):
        if self.operator == '=':
            return self.flatten(_flatten_Promela, '==')
        else:
            return self.flatten(_flatten_Promela)
    
class Arithmetic(Binary):
    @property
    def op(self):
        return self.operator

def get_vars(ast):
    var = set()
    Q = {ast}
    #tree S = {ast}
    while Q:
        x = Q.pop()
        logger.debug('visiting: ' + str(type(x) ) )
        
        if isinstance(x, Unary):
            Q.add(x.operand)
        elif isinstance(x, Binary):
            Q.add(x.op_l)
            Q.add(x.op_r)
        elif isinstance(x, Var):
            var.add(x)
    return var
