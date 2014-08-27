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

import os
import re
import networkx as nx

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

PYTHON_MAP = {'&&':'and', '||':'or', '!':'not', 'xor':'^'}

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

def ast_to_labeled_graph(tree, detailed):
    """Convert AST to C{NetworkX.DiGraph} for graphics.
    
    @param ast: Abstract syntax tree
    
    @rtype: C{networkx.DiGraph}
    """
    g = nx.DiGraph()
    
    for u, d in tree.nodes_iter(data=True):
        nd = d['ast_node']
        
        if isinstance(nd, (Unary, Binary)):
            label = nd.op
        elif isinstance(nd, Node):
            label = str(nd)
        else:
            raise TypeError('ast_node must be or subclass Node.')
        
        # show both repr and AST node class in each vertex
        if detailed:
            label += '\n' + str(type(nd).__name__)
        
        g.add_node(u, label=label)
    
    for u, v in tree.edges_iter():
        g.add_edge(u, v)
    
    return g

class LTL_AST(nx.DiGraph):
    """Abstract Syntax Tree of LTL.
    
    The tree's root node is C{self.root}.
    """
    def __init__(self):
        self.root = None
        super(LTL_AST, self).__init__()
    
    def to_pydot(self, detailed=False):
        """Create GraphViz dot string from given AST.
        
        @type ast: L{ASTNode}
        
        @rtype: str
        """
        g = ast_to_labeled_graph(self, detailed)
        return nx.to_pydot(g)
    
    def write(self, filename, detailed=False):
        """Layout AST and save result in PDF file.
        """
        fname, fext = os.path.splitext(filename)
        fext = fext[1:] # drop .
        p = self.to_pydot(detailed)
        p.write(filename, format=fext)
    
    def get_vars(self):
        """Return the set of variables in C{tree}.
        
        @rtype: C{set} of L{Var}
        """
        return {d['ast_node'] for u, d in self.nodes_iter(data=True)
                              if isinstance(d['ast_node'], Var)}
    
    def sub_values(self, var_values):
        """Substitute given values for variables.
        
        @param tree: AST
        
        @type var_values: C{dict}
        
        @return: AST with L{Var} nodes replaces by
            L{Num}, L{Const}, or L{Bool}
        """
        for u, d in self.nodes_iter(data=True):
            old = d['ast_node']
            
            if not isinstance(old, Var):
                continue
            
            val = var_values[str(old)]
            
            # instantiate appropriate value type
            if isinstance(val, bool):
                nd = Bool(val, g=None)
            elif isinstance(val, int):
                nd = Num(val, g=None)
            elif isinstance(val, str):
                nd = Const(val, g=None)
            
            # replace variable by value,
            # don't touch the underlying graph
            nd.id = old.id
            nd.graph = self
            
            d['ast_node'] = nd
    
    def sub_constants(self, var_const2int):
        """Replace string constants by integers.
        
        To be used for converting arbitrary finite domains
        to integer domains prior to calling gr1c.
        
        @param const2int: {'varname':['const_val0', ...], ...}
        @type const2int: C{dict} of C{list}
        """
        logger.info('substitute ints for constants in ' + str(self))
        
        for u, d in self.nodes_iter(data=True):
            nd = d['ast_node']
            
            if not isinstance(nd, Const):
                continue
            
            # find parent Binary operator
            b = nd
            while True:
                old = b
                bid = self.predecessors(old.id)[0]
                b = self.node[bid]['ast_node']
                
                if isinstance(b, Binary):
                    break
            
            succ = self.successors(b.id)
            
            var_branch = succ[0] if succ[1] == old.id else succ[1]
            
            # go down until var found
            # assuming correct syntax for gr1c
            v = self.node[var_branch]['ast_node']
            while True:
                if isinstance(v, Var):
                    break
                
                old = v
                vid = self.successors(old.id)[0]
                v = self.node[vid]['ast_node']
            
            # now: b, is the operator and: v, the variable
            const2int = var_const2int[str(v)]
            x = const2int.index(nd.val)
            
            val = Num(x, None)
            
            # replace Const with Num
            # dn't touch the underlying graph
            val.id = nd.id
            val.graph = self
            
            d['ast_node'] = val
    
    def to_gr1c(self):
        return self.root.to_gr1c()
    
    def to_jtlv(self):
        return self.root.to_jtlv()
    
    def to_promela(self):
        return self.root.to_promela()
    
    def to_python(self):
        return self.root.to_python()

class Node(object):
    def __init__(self, graph):
        # skip addition ?
        if graph is None:
            return
        
        u = id(self)
        self.id = u
        graph.add_node(u, ast_node=self)
        self.graph = graph
    
    def to_gr1c(self, primed=False):
        return self.flatten(_flatten_gr1c, primed=primed)
    
    def to_jtlv(self):
        return self.flatten(_flatten_JTLV)
    
    def to_smv(self):
        return self.flatten(_flatten_SMV)
    
    def to_promela(self):
        return self.flatten(_flatten_Promela)
    
    def to_python(self):
        return self.flatten(_flatten_python)
    
    def map(self, f):
        n = self.__class__([str(self.val)])
        return f(n)
    
    def __len__(self):
        return 1

class Num(Node):
    def __init__(self, t, g):
        super(Num, self).__init__(g)
        self.val = int(t)
    
    def __repr__(self):
        return str(self.val)
    
    def flatten(self, flattener=None, op=None, **args):
        return str(self)

class Var(Node):
    def __init__(self, t, g):
        super(Var, self).__init__(g)
        self.val = t
    
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

class Const(Node):
    def __init__(self, t, g):
        super(Const, self).__init__(g)
        self.val = re.sub(r'^"|"$', '', t)
    
    def __repr__(self):
        return  r'"' + self.val + r'"'
    
    def flatten(self, flattener=str, op=None, **args):
        return str(self)
        
    def to_gr1c(self, primed=False):
        if primed:
            return str(self) + "'"
        else:
            return str(self)
        
    def to_jtlv(self):
        return '(' + str(self) + ')'
        
    def to_smv(self):
        return str(self)

class Bool(Node):
    def __init__(self, t, g):
        super(Bool, self).__init__(g)
        
        if t.upper() == 'TRUE':
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
        return cls(operator, op_node)
    
    def __init__(self, operator, x, g):
        super(Unary, self).__init__(g)
        self.operator = operator
        self.graph.add_edge(self.id, x.id)
    
    def __repr__(self):
        return ' '.join(['(', self.op, str(self.operand), ')'])
    
    @property
    def operand(self):
        u = self.id
        
        n = len(self.graph.succ[u])
        if n != 1:
            logger.error('Unary AST node has %d children.' % n)
        
        v = set(self.graph.succ[u]).pop()
        
        return self.graph.node[v]['ast_node']
    
    def flatten(self, flattener=str, op=None, **args):
        if op is None:
            op = self.op
        try:
            o = flattener(self.operand, **args)
        except AttributeError:
            o = str(self.operand)
        return ' '.join(['(', op, o, ')'])
    
    def map(self, f):
        n = self.__class__.new(self.operand.map(f), self.op)
        return f(n)
    
    def __len__(self):
        return 1 + len(self.operand)

class Not(Unary):
    @property
    def op(self):
        return '!'
    
    def to_python(self):
        return PYTHON_MAP['!']

class UnTempOp(Unary):
    def __init__(self, operator, x, g):
        operator = TEMPORAL_OP_MAP[operator]
        super(UnTempOp, self).__init__(operator, x, g)
    
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
    def new(cls, x, y, operator=None):
        return cls(x, operator, y)
    
    def __init__(self, operator, x, y, g):
        super(Binary, self).__init__(g)
        self.operator = operator
        self.graph.add_edge(self.id, x.id, pos='left')
        self.graph.add_edge(self.id, y.id, pos='right')
        
    def __repr__(self):
        return ' '.join (['(', str(self.op_l), self.op, str(self.op_r), ')'])
    
    @property
    def op_l(self):
        return self._child('left')
    
    @property
    def op_r(self):
        return self._child('right')
    
    def _child(self, pos):
        u = self.id
        
        n = len(self.graph.successors(u))
        if n != 2:
            logger.error('Binary AST node has %d children.' % n)
        
        for u_, v, d in self.graph.edges_iter([u], data=True):
            if d['pos'] == pos:
                return self.graph.node[v]['ast_node']
    
    def flatten(self, flattener=str, op=None, **args):
        if op is None:
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
    
    def to_python(self):
        return self.flatten(_flatten_python, 'and')

class Or(Binary):
    @property
    def op(self):
        return '|'
    
    def to_jtlv(self):
        return self.flatten(_flatten_JTLV, '||')
    
    def to_promela(self):
        return self.flatten(_flatten_Promela, '||')
    
    def to_python(self):
        return self.flatten(_flatten_python, 'or')

class Xor(Binary):
    @property
    def op(self):
        return 'xor'
    
    def to_python(self):
        return self.flatten(_flatten_python, '^')
    
class Imp(Binary):
    @property
    def op(self):
        return '->'
    
    def to_python(self, flattener=str, op=None, **args):
        try:
            l = flattener(self.op_l, **args)
        except AttributeError:
            l = str(self.op_l)
        try:
            r = flattener(self.op_r, **args)
        except AttributeError:
            r = str(self.op_r)
        return '( (not (' + l + ')) or ' + r + ')'

class BiImp(Binary):
    @property
    def op(self):
        return '<->'
    
    def to_python(self, flattener=str, op=None, **args):
        try:
            l = flattener(self.op_l, **args)
        except AttributeError:
            l = str(self.op_l)
        try:
            r = flattener(self.op_r, **args)
        except AttributeError:
            r = str(self.op_r)
        return '( ' + l + ' and ' + r + ' ) or not ( ' + l + ' or ' + r + ' )'

class BiTempOp(Binary):
    def __init__(self, operator, x, y, g):
        operator = TEMPORAL_OP_MAP[operator]
        super(BiTempOp, self).__init__(operator, x, y, g)
    
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
    def __init__(self, operator, x, y, g):
        super(Comparator, self).__init__(operator, x, y, g)
        
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
    
    def to_python(self):
        if self.operator == '=':
            return self.flatten(_flatten_Promela, '==')
        else:
            return self.flatten(_flatten_Promela)
    
class Arithmetic(Binary):
    @property
    def op(self):
        return self.operator

# Flattener helpers
def _flatten_gr1c(node, **args):
    return node.to_gr1c(**args)

def _flatten_JTLV(node):
    return node.to_jtlv()

def _flatten_SMV(node):
    return node.to_smv()

def _flatten_Promela(node):
    return node.to_promela()

def _flatten_python(node):
    return node.to_python()
