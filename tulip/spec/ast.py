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

supporting syntax of:
    gr1c: http://slivingston.github.io/gr1c/md_spc_format.html
    JTLV
    SMV: http://nusmv.fbk.eu/NuSMV/userman/v21/nusmv_3.html
    SPIN: http://spinroot.com/spin/Man/ltl.html
    python (Boolean formulas only)

Syntax taken originally roughly from:
http://spot.lip6.fr/wiki/LtlSyntax
"""
import logging
logger = logging.getLogger(__name__)

import os
import re
import networkx as nx

TEMPORAL_OP_MAP = {
    '[]': 'G', 'G': 'G',
    'F': 'F', '<>': 'F',
    'X': 'X', 'next': 'X', "'": 'X',
    'U': 'U', 'V': 'R', 'R': 'R'
}

JTLV_MAP = {
    'False': 'FALSE', 'True': 'TRUE',
    '!': '!',
    '|': '|', '&': '&', '->': '->', '<->': '<->',
    'G': '[]', 'F': '<>', 'X': 'next',
    'U': 'U',
    '<=': '<=', '=': '=', '>=': '>=', '>': '>', '!=': '!='
}

GR1C_MAP = {
    'False': 'False', 'True': 'True',
    '!': '!',
    '|': '|', '&': '&', '->': '->', '<->': '<->',
    'G': '[]', 'F': '<>', 'X': "'",
    '<=': '<=', '=': '=', '>=': '>=', '>': '>'
}

SMV_MAP = {'G': 'G', 'F': 'F', 'X': 'X', 'U': 'U', 'R': 'V'}

SPIN_MAP = {
    'True': 'true', 'False': 'false',
    '!': '!',
    '|': '||', '&': '&&', '->': '->', '<->': '<->',
    'G': '[]', 'F': '<>', 'U': 'U', 'R': 'V'
}

PYTHON_MAP = {'!': 'not', '&': 'and', '|': 'or', 'xor': '^'}

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

flatteners = {'gr1c': _flatten_gr1c,
              'jtlv': _flatten_JTLV,
              'smv': _flatten_SMV,
              'spin': _flatten_Promela,
              'python': _flatten_python}

maps = {'gr1c': GR1C_MAP,
        'jtlv': JTLV_MAP,
        'smv': SMV_MAP,
        'spin': SPIN_MAP,
        'python': PYTHON_MAP}

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
    
    def __repr__(self):
        return repr(self.root)
    
    def __str__(self):
        # need to define __str__ only
        # to override networkx.DiGraph.__str__
        return repr(self)
    
    def replace_node(self, old, new):
        """Replace AST node, w/o touching the underlying graph.
        
        @type old: L{Node}
        
        @type new: L{Node}
        """
        new.id = old.id
        new.graph = old.graph
        
        self.node[old.id]['ast_node'] = new
    
    def add_subtree(self, ast_node, tree):
        """Return the C{tree} at node C{nd}.
        
        @type nd: L{Node}
        
        @param tree: to be added, w/o copying AST nodes.
        @type tree: L{LTL_AST}
        """
        # nd must be a leaf
        assert(not self.successors(ast_node.id))
        
        # add new nodes
        for v, d in tree.nodes_iter(data=True):
            nd = d['ast_node']
            nd.graph = self
            self.add_node(v, ast_node=nd)
        
        # add edges
        for u, v, d in tree.edges_iter(data=True):
            if 'pos' in d:
                self.add_edge(u, v, pos=d['pos'])
            else:
                self.add_edge(u, v)
        
        # replace old leaf with subtree root
        u = ast_node.id
        pred = self.predecessors(u)
        if pred:
            parent = next(iter(pred))
            pos = self[parent][u].get('pos')
            self.remove_node(u)
            if pos:
                self.add_edge(parent, tree.root.id, pos=pos)
            else:
                self.add_edge(parent, tree.root.id)
        else:
            self.remove_node(u)
            self.root = tree.root
        
        if logger.getEffectiveLevel() <= logging.DEBUG:
            for u, d in self.nodes_iter(data=True):
                assert(d['ast_node'].graph is self)
    
    def get_vars(self):
        """Return the set of variables in C{tree}.
        
        @rtype: C{set} of L{Var}
        """
        return {d['ast_node']
                for u, d in self.nodes_iter(data=True)
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
            
            # replace variable by value
            self.replace_node(old, nd)
    
    def sub_constants(self, var_const2int):
        """Replace string constants by integers.
        
        To be used for converting arbitrary finite domains
        to integer domains prior to calling gr1c.
        
        @param const2int: {'varname':['const_val0', ...], ...}
        @type const2int: C{dict} of C{list}
        """
        logger.info('substitute ints for constants in:\n\t' + str(self))
        
        for u, d in self.nodes_iter(data=True):
            nd = d['ast_node']
            
            if not isinstance(nd, Const):
                continue
            
            v, c = pair_node_to_var(self, nd)
            
            # now: c, is the operator and: v, the variable
            const2int = var_const2int[str(v)]
            x = const2int.index(nd.val)
            
            val = Num(x, None)
            
            # replace Const with Num
            self.replace_node(nd, val)
        
        logger.info('result after substitution:\n\t' + str(self) + '\n')
    
    def eval(self, d):
        """Evaluate over variable valuation C{d}.
        
        @param d: assignment of values to variables.
            Available types are Boolean, integer, and string.
        @type d: C{dict}
        
        @return: value of formula for the given valuation C{d} (model).
        @rtype: C{bool}
        """
        return self.root.eval(d)
    
    def to_gr1c(self):
        return self.root.to_gr1c()
    
    def to_jtlv(self):
        return self.root.to_jtlv()
    
    def to_promela(self):
        return self.root.to_promela()
    
    def to_python(self):
        return self.root.to_python()
    
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
        fext = fext[1:]  # drop .
        p = self.to_pydot(detailed)
        p.write(filename, format=fext)

def sub_bool_with_subtree(tree, bool2subtree):
    """Replace selected Boolean variables with given AST.
    
    @type tree: L{LTL_AST}
    
    @param bool2form: map from each Boolean variable to some
        equivalent formula. A subset of Boolean varibles may be used.
        
        Note that the types of variables in C{tree}
        are defined by C{bool2form}.
    @type bool2form: C{dict} from C{str} to L{LTL_AST}
    """
    for u in tree.nodes():
        nd = tree.node[u]['ast_node']
        
        if not isinstance(nd, Var):
            continue
        
        # not Boolean var ?
        if nd.val not in bool2subtree:
            continue
        
        #tree.write(str(id(tree)) + '_before.png')
        tree.add_subtree(nd, bool2subtree[nd.val])
        #tree.write(str(id(tree)) + '_after.png')

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
    
    def eval(self, d):
        return self.val

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
            return str(self) + "'"
        else:
            return str(self)
        
    def to_jtlv(self):
        return '(' + str(self) + ')'
        
    def to_smv(self):
        return str(self)
    
    def eval(self, d):
        return d[self.val]

class Const(Node):
    def __init__(self, t, g):
        super(Const, self).__init__(g)
        self.val = re.sub(r'^"|"$', '', t)
    
    def __repr__(self):
        return r'"' + self.val + r'"'
    
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
    
    def eval(self, d):
        return self.val

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
    
    def eval(self, d):
        return self.val

class Operator(Node):
    def to_promela(self):
        return _to_lang(self, 'spin')
    
    def to_gr1c(self, primed=False):
        if self.op == 'X':
            return self.flatten(_flatten_gr1c, '', primed=True)
        
        return _to_lang(self, 'gr1c', primed=primed)
    
    def to_jtlv(self):
        return _to_lang(self, 'jtlv')
    
    def to_python(self):
        return _to_lang(self, 'python')
    
    def to_smv(self):
        return _to_lang(self, 'smv')

def _to_lang(node, lang, **k):
    try:
        return node.flatten(flatteners[lang], maps[lang][node.op], **k)
    except KeyError:
        raise LTLException(
            'Operator ' + node.op +
            ' not supported in ' + lang + ' syntax map'
        )

class Unary(Operator):
    @classmethod
    def new(cls, op_node, operator=None):
        return cls(operator, op_node)
    
    def __init__(self, operator, x, g):
        super(Unary, self).__init__(g)
        self.operator = operator
        self.graph.add_edge(self.id, x.id)
    
    def __repr__(self):
        return '( %s %s )' % (self.op, str(self.operand))
    
    @property
    def operand(self):
        assert(len(self.graph.succ[self.id]) == 1)
        v = next(iter(self.graph.succ[self.id]))
        return self.graph.node[v]['ast_node']
    
    def flatten(self, flattener=str, op=None, **args):
        if op is None:
            op = self.op
        
        try:
            o = flattener(self.operand, **args)
        except AttributeError:
            o = str(self.operand)
        
        return '( %s %s )' % (op, o)
    
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
        return self.flatten(_flatten_python, PYTHON_MAP['!'])
    
    def eval(self, d):
        return not self.operand.eval(d)

class UnTempOp(Unary):
    def __init__(self, operator, x, g):
        operator = TEMPORAL_OP_MAP[operator]
        super(UnTempOp, self).__init__(operator, x, g)
    
    @property
    def op(self):
        return self.operator

class Binary(Operator):
    @classmethod
    def new(cls, x, y, operator=None):
        return cls(x, operator, y)
    
    def __init__(self, operator, x, y, g):
        super(Binary, self).__init__(g)
        self.operator = operator
        self.graph.add_edge(self.id, x.id, pos='left')
        self.graph.add_edge(self.id, y.id, pos='right')
        
    def __repr__(self):
        return '( %s %s %s )' % (str(self.op_l), self.op, str(self.op_r))
    
    @property
    def op_l(self):
        return self._child('left')
    
    @property
    def op_r(self):
        return self._child('right')
    
    def _child(self, pos):
        assert(len(self.graph.succ[self.id]) == 2)
        
        for v, d in self.graph.succ[self.id].iteritems():
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
        return '( %s %s %s )' % (l, op, r)
    
    def map(self, f):
        n = self.__class__.new(self.op_l.map(f), self.op_r.map(f), self.op)
        return f(n)
    
    def __len__(self):
        return 1 + len(self.op_l) + len(self.op_r)

class And(Binary):
    @property
    def op(self):
        return '&'
    
    def eval(self, d):
        return self.op_l.eval(d) and self.op_r.eval(d)

class Or(Binary):
    @property
    def op(self):
        return '|'
    
    def eval(self, d):
        return self.op_l.eval(d) or self.op_r.eval(d)

class Xor(Binary):
    @property
    def op(self):
        return 'xor'
    
    def eval(self, d):
        return self.op_l.eval(self, d) ^ self.op_r.eval(d)
    
class Imp(Binary):
    @property
    def op(self):
        return '->'
    
    def eval(self, d):
        return not self.op_l.eval(d) or self.op_r.eval(d)
    
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
    
    def eval(self, d):
        return self.op_l.eval(d) == self.op_r.eval(d)
    
    def to_python(self, flattener=str, op=None, **args):
        try:
            l = flattener(self.op_l, **args)
        except AttributeError:
            l = str(self.op_l)
        try:
            r = flattener(self.op_r, **args)
        except AttributeError:
            r = str(self.op_r)
        return '( ' + l + ' == ' + r + ' )'

class BiTempOp(Binary):
    def __init__(self, operator, x, y, g):
        operator = TEMPORAL_OP_MAP[operator]
        super(BiTempOp, self).__init__(operator, x, y, g)
    
    @property
    def op(self):
        return self.operator

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
            return self.flatten(_flatten_python, '==')
        else:
            return self.flatten(_flatten_python)
    
class Arithmetic(Binary):
    @property
    def op(self):
        return self.operator

def check_for_undefined_identifiers(tree, domains):
    """Check that types in C{tree} are incompatible with C{domains}.
    
    Raise a C{ValueError} if C{tree} either:
    
      - contains a variable missing from C{domains}
      - binary operator between variable and
        invalid value for that variable.
    
    @type tree: L{LTL_AST}
    
    @param domains: variable definitions:
        
        C{{'varname': domain}}
        
        See L{GRSpec} for more details of available domain types.
    @type domains: C{dict}
    """
    for u, d in tree.nodes_iter(data=True):
        nd = d['ast_node']
        print(nd)
        
        if isinstance(nd, Var) and nd.val not in domains:
            var = nd.val
            raise ValueError('undefined variable: ' + str(var) +
                             ', in subformula:\n\t' + str(tree))
        
        if not isinstance(nd, (Const, Num)):
            continue
        
        # is a Const or Num
        var, c = pair_node_to_var(tree, nd)
        
        if isinstance(c, Const):
            dom = domains[var]
            
            if not isinstance(dom, list):
                raise Exception(
                    'String constant: ' + str(c) +
                    ', assigned to non-string variable: ' +
                    str(var) + ', whose domain is:\n\t' + str(dom)
                )
            
            if c.val not in domains[var.val]:
                raise ValueError(
                    'String constant: ' + str(c) +
                    ', is not in the domain of variable: ' + str(var)
                )
        
        if isinstance(c, Num):
            dom = domains[var]
            
            if not isinstance(dom, tuple):
                raise Exception(
                    'Number: ' + str(c) +
                    ', assigned to non-integer variable: ' +
                    str(var) + ', whose domain is:\n\t' + str(dom)
                )
            
            if not dom[0] <= c.val <= dom[1]:
                raise Exception(
                    'Integer variable: ' + str(var) +
                    ', is assigned the value: ' + str(c) +
                    ', that is out of its range: %d ... %d ' % dom
                )
        
def pair_node_to_var(tree, c):
    """Find variable under L{Binary} operator above given node.
    
    First move up from C{nd}, stop at first L{Binary} node.
    Then move down, until first C{Var}.
    This assumes that only L{Unary} operators appear between a
    L{Binary} and its variable and constant operands.
    
    May be extended in the future, depending on what the
    tools support and is thus needed here.
    
    @type tree: L{LTL_AST}
    
    @type L{nd}: L{Const} or L{Num}
    
    @return: variable, constant
    @rtype: C{(L{Var}, L{Const})}
    """
    # find parent Binary operator
    while True:
        old = c
        bid = tree.predecessors(old.id)[0]
        c = tree.node[bid]['ast_node']
        
        if isinstance(c, Binary):
            break
    
    succ = tree.successors(c.id)
    
    var_branch = succ[0] if succ[1] == old.id else succ[1]
    
    # go down until var found
    # assuming correct syntax for gr1c
    v = tree.node[var_branch]['ast_node']
    while True:
        if isinstance(v, Var):
            break
        
        old = v
        vid = tree.successors(old.id)[0]
        v = tree.node[vid]['ast_node']
    
    # now: b, is the operator and: v, the variable
    return v, c
