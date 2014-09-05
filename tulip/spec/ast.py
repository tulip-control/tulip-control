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

OP_MAP = {
    '!':'!',
    '|': '|', '&': '&', '->': '->', '<->': '<->',
    '[]': 'G', 'G': 'G',
    'F': 'F', '<>': 'F',
    'X': 'X', 'next': 'X', "'": 'X',
    'U': 'U', 'V': 'R', 'R': 'R',
    '==': '=',
    '<=': '<=', '=': '=', '>=': '>=', '>': '>', '!=': '!='
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
    'G': '[]', 'F': '<>', 'X': '',
    '<=': '<=', '=': '=', '>=': '>=', '>': '>'
}

SMV_MAP = {'G': 'G', 'F': 'F', 'X': 'X', 'U': 'U', 'R': 'V'}

SPIN_MAP = {
    'True': 'true', 'False': 'false',
    '!': '!',
    '|': '||', '&': '&&', '->': '->', '<->': '<->',
    'G': '[]', 'F': '<>', 'U': 'U', 'R': 'V',
    '=': '=='
}

PYTHON_MAP = {'!': 'not', '&': 'and', '|': 'or', 'xor': '^', '=': '=='}

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

maps = {
    'string':OP_MAP,
    'gr1c': GR1C_MAP,
    'jtlv': JTLV_MAP,
    'smv': SMV_MAP,
    'spin': SPIN_MAP,
    'python': PYTHON_MAP
}

class LTLException(Exception):
    pass

def ast_to_labeled_graph(tree, detailed):
    """Convert AST to C{NetworkX.DiGraph} for graphics.
    
    @param ast: Abstract syntax tree
    
    @rtype: C{networkx.DiGraph}
    """
    g = nx.DiGraph()
    
    for u in tree:
        if isinstance(u, Operator):
            label = u.op
        elif isinstance(u, Term):
            label = str(u.val)
        else:
            raise TypeError(
                'AST node must be Operator or Identifier, '
                'got instead: ' + str(u) +
                ', of type: ' + str(type(u))
            )
        
        # show both repr and AST node class in each vertex
        if detailed:
            label += '\n' + str(type(u).__name__)
        
        g.add_node(id(u), label=label)
    
    for u, v, d in tree.edges_iter(data=True):
        g.add_edge(id(u), id(v), label=d['pos'])
    
    return g

class LTL_AST(nx.DiGraph):
    """Abstract Syntax Tree of LTL.
    
    The tree's root node is a L{Node} at C{self.root}.
    """
    def __init__(self):
        self.root = None
        super(LTL_AST, self).__init__()
    
    def __repr__(self):
        return flatten(self, self.root, 'string')
    
    def __str__(self):
        # need to define __str__ only
        # to override networkx.DiGraph.__str__
        return repr(self)
    
    def add_identifier(self, identifier):
        self.add_node(identifier)
    
    def add_unary(self, operator, operand):
        self.add_edge(operator, operand, pos=None)
    
    def add_binary(self, operator, left, right):
        self.add_edge(operator, left, pos='left')
        self.add_edge(operator, right, pos='right')
    
    def children(self, u):
        succ = self.succ[u]
        if not succ:
            return []
        elif len(succ) == 1:
            return succ
        elif len(succ) == 2:
            l, r = succ
            if self[u][r]['pos'] != 'right':
                return r, l
            else:
                return l, r
        else:
            raise Exception('Cannot have > 2 successors.')
    
    def add_subtree(self, leaf, tree):
        """Return the C{tree} at node C{nd}.
        
        @type nd: L{Node}
        
        @param tree: to be added, w/o copying AST nodes.
        @type tree: L{LTL_AST}
        """
        # nd must be a leaf
        assert(not self.successors(leaf))
        
        # add new nodes
        for v in tree:
            self.add_node(v)
        
        # add edges
        for u, v, d in tree.edges_iter(data=True):
            self.add_edge(u, v, pos=d['pos'])
        
        # replace old leaf with subtree root
        u = leaf
        pred = self.predecessors(u)
        if pred:
            parent = next(iter(pred))
            pos = self[parent][u].get('pos')
            self.remove_node(u)
            self.add_edge(parent, tree.root, pos=pos)
        else:
            self.remove_node(u)
            self.root = tree.root
    
    def get_vars(self):
        """Return the set of variables in C{tree}.
        
        @rtype: C{set} of L{Var}
        """
        return {u for u in self if isinstance(u, Var)}
    
    def sub_values(self, var_values):
        """Substitute given values for variables.
        
        @param tree: AST
        
        @type var_values: C{dict}
        
        @return: AST with L{Var} nodes replaces by
            L{Num}, L{Const}, or L{Bool}
        """
        old2new = dict()
        
        for u in self.nodes_iter():
            if not isinstance(u, Var):
                continue
            
            val = var_values[str(u)]
            
            # instantiate appropriate value type
            if isinstance(val, bool):
                v = Bool(val)
            elif isinstance(val, int):
                v = Num(val)
            elif isinstance(val, str):
                v = Const(val)
            
            old2new[u] = v
            
        # replace variable by value
        nx.relabel_nodes(self, old2new, copy=False)
    
    def sub_constants(self, var_const2int):
        """Replace string constants by integers.
        
        To be used for converting arbitrary finite domains
        to integer domains prior to calling gr1c.
        
        @param const2int: {'varname':['const_val0', ...], ...}
        @type const2int: C{dict} of C{list}
        """
        logger.info('substitute ints for constants in:\n\t' + str(self))
        
        old2new = dict()
        
        for u in self.nodes_iter():
            if not isinstance(u, Const):
                continue
            
            var, op = pair_node_to_var(self, u)
            
            # now: c, is the operator and: v, the variable
            const2int = var_const2int[str(var)]
            x = const2int.index(u.val)
            
            num = Num(x)
            
            # replace Const with Num
            old2new[u] = num
        
        nx.relabel_nodes(self, old2new, copy=False)
        
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
        return flatten(self, self.root, 'gr1c')
    
    def to_jtlv(self):
        return flatten(self, self.root, 'jtlv')
    
    def to_promela(self):
        return flatten(self, self.root, 'spin')
    
    def to_python(self):
        return flatten(self, self.root, 'python')
    
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
        p.set_ordering('out')
        p.write(filename, format=fext)

def flatten(tree, node, lang):
    flattener = 'to_' + lang
    
    Q = [node]
    S = {node}
    
    stack = []
    context = {'prime':0}
    
    while Q:
        u = Q[-1]
        #logger.debug('visit: ' + str(u))
        #logger.debug('current stack:\n\t' + str(stack))
        
        not_expanded = False
        for s in tree.succ[u]:
            if s not in S:
                not_expanded = True
                break
        
        if not_expanded:
            # for example: prime
            if hasattr(u, 'apply_context'):
                u.apply_context(context)
            #logger.debug('current context: ' + str(context))
            
            succ = tree.succ[u]
            if len(succ) == 2:
                l, r = succ
                if succ[r]['pos'] == 'left':
                    l, r = r, l
            
            Q.extend(succ)
            S.update(succ)
            
            #logger.debug('not expanded yet')
            continue
        
        # eval only if fully expanded
        if hasattr(u, flattener):
            r = getattr(u, flattener)(stack, context)
        else:
            try:
                r = u.flatten(lang, stack, context)
            except KeyError:
                raise LTLException(
                    'Operator ' + node.op +
                    ' not supported in ' + lang + ' syntax map'
                )
        
        #logger.debug('push to stack: ' + str(r))
        stack.append(r)
        
        if hasattr(u, 'clear_context'):
            u.clear_context(context)
        
        Q.pop(-1)
    
    return stack[0]

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
        if not isinstance(u, Var):
            continue
        
        # not Boolean var ?
        if u.val not in bool2subtree:
            continue
        
        #tree.write(str(id(tree)) + '_before.png')
        tree.add_subtree(u, bool2subtree[u.val])
        #tree.write(str(id(tree)) + '_after.png')

class Node(object):
    """Base class for deriving AST nodes.
    """
    # Caution
    # =======
    # Do **NOT** implement C{__hash__}, because you
    # will unintendently identify different AST nodes !
    #
    # The default for user-defined classes is
    # C{__hash__ == _}

class Term(Node):
    def __init__(self, t):
        self.val = t
    
    def __eq__(self, other):
        return isinstance(other, type(self)) and self.val == other.val
    
    def __repr__(self):
        return str(self.val)
    
    def flatten(self, lang, stack, context):
        return str(self)
    
    def eval(self, d):
        return self.val

class Num(Term):
    def __init__(self, t):
        self.val = int(t)

class Var(Term):
    def to_gr1c(self, stack, context):
        if context.get('prime') > 0:
            return str(self) + "'"
        else:
            return str(self)
        
    def to_jtlv(self, *args):
        return '(%s)' % str(self)
    
    def eval(self, d):
        return d[self.val]

class Const(Term):
    def __init__(self, t):
        self.val = re.sub(r'^"|"$', '', t)
    
    def __repr__(self):
        return r'"%s"' % self.val
        
    def to_jtlv(self, *args):
        return '(%s)' % str(self)

class Bool(Term):
    def __init__(self, t):
        if t.upper() == 'TRUE':
            self.val = True
        else:
            self.val = False
    
    def __repr__(self):
        if self.val:
            return 'True'
        else:
            return 'False'
    
    def to_gr1c(self, *args):
        try:
            return GR1C_MAP[str(self)]
        except KeyError:
            raise LTLException(
                'Reserved word "' + self.op +
                '" not supported in gr1c syntax map'
            )
    
    def to_jtlv(self, *args):
        try:
            return JTLV_MAP[str(self)]
        except KeyError:
            raise LTLException(
                'Reserved word "' + self.op +
                '" not supported in JTLV syntax map'
            )

class Operator(Node):
    def __init__(self, operator):
        if operator in OP_MAP:
            operator = OP_MAP[operator]
        self.operator = operator
    
    def __hash__(self):
        return hash(self.op)
    
    def __repr__(self):
        return self.op
    
    @property
    def op(self):
        return self.operator
    
    def to_lang(self, lang):
        return maps[lang][self.op]

class Unary(Operator):
    def operand(self, graph):
        return next(iter(graph.succ[self]))
    
    def flatten(self, lang, stack, context):
        op = self.to_lang(lang)
        x = stack.pop(-1)
        return '( %s %s )' % (op, x)

class Not(Unary):
    @property
    def op(self):
        return '!'
    
    def eval(self, d):
        return not self.operand.eval(d)

class UnTempOp(Unary):
    def apply_context(self, context):
        if self.op is 'X':
            context['prime'] += 1
        
    def clear_context(self, context):
        if self.op is 'X':
            context['prime'] -= 1

class Binary(Operator):
    def _consume(self, stack):
        l = stack.pop(-1)
        r = stack.pop(-1)
        return l, r
    
    def flatten(self, lang, stack, context):
        op = self.to_lang(lang)
        l, r = self._consume(stack)
        return '( %s %s %s )' % (l, op, r)
    
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
    
    def to_python(self, stack):
        l, r = self._consume(stack)
        return '( (not (' + l + ')) or ' + r + ')'

class BiImp(Binary):
    @property
    def op(self):
        return '<->'
    
    def _eval(self, l, r):
        return l == r
    
    def to_python(self, stack):
        l, r = self._consume(stack)
        return '( ' + l + ' == ' + r + ' )'

class BiTempOp(Binary):
    pass

class Comparator(Binary):
    pass
    
class Arithmetic(Binary):
    pass

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
    for u in tree:
        if isinstance(u, Var) and u.val not in domains:
            var = u.val
            raise ValueError('undefined variable: ' + str(var) +
                             ', in subformula:\n\t' + str(tree))
        
        if not isinstance(u, (Const, Num)):
            continue
        
        # is a Const or Num
        var, c = pair_node_to_var(tree, u)
        
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
        c = next(iter(tree.predecessors(c)))
        
        if isinstance(c, Binary):
            break
    
    print('Binary: ' + str(c))
    succ = tree.successors(c)
    
    v = succ[0] if succ[1] == old else succ[1]
    
    # go down until var found
    # assuming correct syntax for gr1c
    while True:
        print('v: ' + str(v))
        if isinstance(v, Var):
            break
        
        v = next(iter(tree.successors(v)))
    
    # now: b, is the operator and: v, the variable
    return v, c
