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

Syntax taken originally roughly from http://spot.lip6.fr/wiki/LtlSyntax
"""
TEMPORAL_OP_MAP = \
        { "G" : "G", "F" : "F", "X" : "X",
        "[]" : "G", "<>" : "F", "next" : "X",
        "U" : "U", "V" : "R", "R" : "R", 
          "'" : "X", "FALSE" : "False", "TRUE" : "True"}

JTLV_MAP = { "G" : "[]", "F" : "<>", "X" : "next",
             "U" : "U", "||" : "||", "&&" : "&&",
             "False" : "FALSE", "True" : "TRUE" }

GR1C_MAP = {"G" : "[]", "F" : "<>", "X" : "'", "||" : "|", "&&" : "&",
            "False" : "False", "True" : "True"}

SMV_MAP = { "G" : "G", "F" : "F", "X" : "X",
        "U" : "U", "R" : "V" }

SPIN_MAP = { "G" : "[]", "F" : "<>", "U" : "U",
        "R" : "V" }

class LTLException(Exception):
    pass

def dump_dot(ast):
    """Create Graphiz DOT string from given AST.

    @param ast: L{ASTNode}, etc., that has a dump_dot() method; for
        example, the return value of a successful call to L{parser}.
    """
    return "digraph AST {\n"+ast.dump_dot()+"}\n"

# Flattener helpers
def _flatten_gr1c(node, **args):
    return node.to_gr1c(**args)

def _flatten_JTLV(node):
    return node.to_jtlv()

def _flatten_SMV(node):
    return node.to_smv()

def _flatten_Promela(node):
    return node.to_promela()

class ASTNode(object):
    def __init__(self, s, l, t):
        # t can be a list or a list of lists, handle both
        try:
            tok = sum(t.asList(), [])
        except:
            try:
                tok = t.asList()
            except:
                # not a ParseResult
                tok = t
        self.init(tok)
    def to_gr1c(self, primed=False):
        return self.flatten(_flatten_gr1c, primed=primed)
    
    def to_jtlv(self):
        return self.flatten(_flatten_JTLV)
    
    def to_smv(self):
        return self.flatten(_flatten_SMV)
    
    def to_promela(self):
        return self.flatten(_flatten_Promela)
    
    def map(self, f):
        n = self.__class__(None, None, [str(self.val)])
        return f(n)
    
    def __len__(self):
        return 1
    
class ASTNum(ASTNode):
    def init(self, t):
        self.val = int(t[0])
    
    def __repr__(self):
        return str(self.val)
    
    def flatten(self, flattener=None, op=None, **args):
        return str(self)
    
    def dump_dot(self):
        return str(id(self)) + "\n" + \
               str(id(self)) + " [label=\"" + str(self.val) + "\"]\n"

class ASTVar(ASTNode):
    def init(self, t):
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
        return "(" + str(self) + ")"
        
    def to_smv(self):
        return str(self)
        
    def dump_dot(self):
        return str(id(self)) + "\n" + \
               str(id(self)) + " [label=\"" + str(self.val) + "\"]\n"
        
class ASTBool(ASTNode):
    def init(self, t):
        if t[0].upper() == "TRUE":
            self.val = True
        else:
            self.val = False
    
    def __repr__(self):
        if self.val:
            return "True"
        else:
            return "False"
    
    def flatten(self, flattener=None, op=None, **args):
        return str(self)
    
    def to_gr1c(self, primed=False):
        try:
            return GR1C_MAP[str(self)]
        except KeyError:
            raise LTLException(
                "Reserved word \"" + self.op() +
                "\" not supported in gr1c syntax map"
            )
    
    def to_jtlv(self):
        try:
            return JTLV_MAP[str(self)]
        except KeyError:
            raise LTLException(
                "Reserved word \"" + self.op() +
                "\" not supported in JTLV syntax map"
            )
    
    def dump_dot(self):
        return str(id(self)) + "\n" + \
               str(id(self)) + " [label=\"" + str(self.val) + "\"]\n"

class ASTUnary(ASTNode):
    @classmethod
    def new(cls, op_node, operator=None):
        return cls(None, None, [operator, op_node])
    
    def init(self, tok):
        if tok[1] == "'":
            if len(tok) > 2:
                # handle left-associative chains, e.g. Y''
                t = self.__class__(None, None, tok[:-1])
                tok = [t, tok[-1]]
            self.operand = tok[0]
            self.operator = "X"
        else:
            self.operand = tok[1]
            if isinstance(self, ASTUnTempOp):
                self.operator = TEMPORAL_OP_MAP[tok[0]]
    def __repr__(self):
        return ' '.join(['(', self.op(), str(self.operand), ')'])
    
    def flatten(self, flattener=str, op=None, **args):
        if op is None:
            op = self.op()
        try:
            o = flattener(self.operand, **args)
        except AttributeError:
            o = str(self.operand)
        return ' '.join(['(', op, o, ')'])
    
    def dump_dot(self):
        return (str(id(self))+"\n"
                + str(id(self))+" [label=\""+str(self.op())+"\"]\n"
                + str(id(self))+" -> "+self.operand.dump_dot())
    
    def map(self, f):
        n = self.__class__.new(self.operand.map(f), self.op())
        return f(n)
    
    def __len__(self):
        return 1 + len(self.operand)

class ASTNot(ASTUnary):
    def op(self):
        return "!"

class ASTUnTempOp(ASTUnary):
    def op(self):
        return self.operator
    
    def to_promela(self):
        try:
            return self.flatten(_flatten_Promela, SPIN_MAP[self.op()])
        except KeyError:
            raise LTLException(
                "Operator " + self.op() +
                " not supported in Promela syntax map"
            )
    
    def to_gr1c(self, primed=False):
        if self.op() == "X":
            return self.flatten(_flatten_gr1c, "", primed=True)
        else:
            try:
                return self.flatten(
                    _flatten_gr1c, GR1C_MAP[self.op()], primed=primed
                )
            except KeyError:
                raise LTLException(
                    "Operator " + self.op() +
                    " not supported in gr1c syntax map"
                )
    
    def to_jtlv(self):
        try:
            return self.flatten(_flatten_JTLV, JTLV_MAP[self.op()])
        except KeyError:
            raise LTLException(
                "Operator " + self.op() +
                " not supported in JTLV syntax map"
            )
    
    def to_smv(self):
        return self.flatten(_flatten_SMV, SMV_MAP[self.op()])

class ASTBinary(ASTNode):
    @classmethod
    def new(cls, op_l, op_r, operator=None):
        return cls(None, None, [op_l, operator, op_r])
    
    def init(self, tok):
        # handle left-associative chains e.g. x && y && z
        if len(tok) > 3:
            t = self.__class__(None, None, tok[:-2])
            tok = [t, tok[-2], tok[-1]]
        self.op_l = tok[0]
        self.op_r = tok[2]
        # generalise temporal operator
        if isinstance(self, ASTBiTempOp):
            self.operator = TEMPORAL_OP_MAP[tok[1]]
        elif isinstance(self, ASTComparator) or isinstance(self, ASTArithmetic):
            if tok[1] == "==":
                self.operator = "="
            else:
                self.operator = tok[1]
    def __repr__(self):
        return ' '.join (['(', str(self.op_l), self.op(), str(self.op_r), ')'])
    
    def flatten(self, flattener=str, op=None, **args):
        if not op: op = self.op()
        try:
            l = flattener(self.op_l, **args)
        except AttributeError:
            l = str(self.op_l)
        try:
            r = flattener(self.op_r, **args)
        except AttributeError:
            r = str(self.op_r)
        return ' '.join (['(', l, op, r, ')'])
    
    def dump_dot(self):
        return (str(id(self))+"\n"
                + str(id(self))+" [label=\""+str(self.op())+"\"]\n"
                + str(id(self))+" -> "+self.op_l.dump_dot()
                + str(id(self))+" -> "+self.op_r.dump_dot())
    
    def map(self, f):
        n = self.__class__.new(self.op_l.map(f), self.op_r.map(f), self.op())
        return f(n)
    
    def __len__(self):
        return 1 + len(self.op_l) + len(self.op_r)

class ASTAnd(ASTBinary):
    def op(self):
        return "&"
    
    def to_jtlv(self):
        return self.flatten(_flatten_JTLV, "&&")
    
    def to_promela(self):
        return self.flatten(_flatten_Promela, "&&")

class ASTOr(ASTBinary):
    def op(self):
        return "|"
    
    def to_jtlv(self):
        return self.flatten(_flatten_JTLV, "||")
    
    def to_promela(self):
        return self.flatten(_flatten_Promela, "||")

class ASTXor(ASTBinary):
    def op(self):
        return "xor"
    
class ASTImp(ASTBinary):
    def op(self):
        return "->"

class ASTBiImp(ASTBinary):
    def op(self):
        return "<->"

class ASTBiTempOp(ASTBinary):
    def op(self):
        return self.operator
    
    def to_promela(self):
        try:
            return self.flatten(_flatten_Promela, SPIN_MAP[self.op()])
        except KeyError:
            raise LTLException(
                "Operator " + self.op() +
                " not supported in Promela syntax map"
            )
    
    def to_gr1c(self, primed=False):
        try:
            return self.flatten(_flatten_gr1c, GR1C_MAP[self.op()])
        except KeyError:
            raise LTLException(
                "Operator " + self.op() +
                " not supported in gr1c syntax map"
            )
    
    def to_jtlv(self):
        try:
            return self.flatten(_flatten_JTLV, JTLV_MAP[self.op()])
        except KeyError:
            raise LTLException(
                "Operator " + self.op() +
                " not supported in JTLV syntax map"
            )
    
    def to_smv(self):
        return self.flatten(_flatten_SMV, SMV_MAP[self.op()])
    
class ASTComparator(ASTBinary):
    def op(self):
        return self.operator
    
    def to_promela(self):
        if self.operator == "=":
            return self.flatten(_flatten_Promela, "==")
        else:
            return self.flatten(_flatten_Promela)
    
class ASTArithmetic(ASTBinary):
    def op(self):
        return self.operator
