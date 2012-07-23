#!/usr/bin/env python
#
# Copyright (c) 2011, 2012 by California Institute of Technology
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
# $Id$
# LTL parser supporting JTLV, SPIN and SMV syntax (and mixtures thereof!)
# Syntax taken roughly from http://spot.lip6.fr/wiki/LtlSyntax
from pyparsing import *
import sys

# Packrat parsing - it's much faster
ParserElement.enablePackrat()

TEMPORAL_OP_MAP = \
        { "G" : "GLOBALLY", "F" : "FINALLY", "X" : "NEXT",
        "[]" : "GLOBALLY", "<>" : "FINALLY", "next" : "NEXT",
        "U" : "UNTIL", "V" : "RELEASE", "R" : "RELEASE", 
        "'" : "NEXT" }

JTLV_MAP = { "GLOBALLY" : "[]", "FINALLY" : "<>", "NEXT" : "next",
        "UNTIL" : "U" }

SMV_MAP = { "GLOBALLY" : "G", "FINALLY" : "F", "NEXT" : "X",
        "UNTIL" : "U", "RELEASE" : "V" }

SPIN_MAP = { "GLOBALLY" : "[]", "FINALLY" : "<>", "UNTIL" : "U",
        "RELEASE" : "V" }

class LTLException(Exception):
    pass
    
# Flattener helpers
def flatten_JTLV(node): return node.toJTLV()
def flatten_SMV(node): return node.toSMV()
def flatten_Promela(node): return node.toPromela()

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
    def toJTLV(self): return self.flatten(flatten_JTLV)
    def toSMV(self): return self.flatten(flatten_SMV)
    def toPromela(self): return self.flatten(flatten_Promela)
    def map(self, f): return f(self)
    
class ASTNum(ASTNode):
    def init(self, t):
        self.val = int(t[0])
    def __repr__(self):
        return str(self.val)
    def flatten(self, flattener=None, op=None):
        return str(self)

class ASTVar(ASTNode):
    def init(self, t):
        self.val = t[0]
    def __repr__(self):
        return self.val
    def flatten(self, flattener=str, op=None):
        # just return variable name (quoted?)
        return str(self)
    def toJTLV(self):
        return "(" + str(self) + ")"
    def toSMV(self):
        return str(self)
        
class ASTBool(ASTNode):
    def init(self, t):
        if t[0].upper() == "TRUE":
            self.val = True
        else:
            self.val = False
    def __repr__(self):
        if self.val: return "TRUE"
        else: return "FALSE"
    def flatten(self, flattener=None, op=None):
        return str(self)

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
            self.operator = "NEXT"
        else:
            self.operand = tok[1]
            if isinstance(self, ASTUnTempOp):
                self.operator = TEMPORAL_OP_MAP[tok[0]]
    def __repr__(self):
        return ' '.join(['(', self.op(), str(self.operand), ')'])
    def flatten(self, flattener=str, op=None):
        if not op: op = self.op()
        try:
            o = flattener(self.operand)
        except AttributeError:
            o = str(self.operand)
        return ' '.join(['(', op, o, ')'])
    def map(self, f):
        self.operand = self.operand.map(f)
        return f(self)

class ASTNot(ASTUnary):
    def op(self): return "!"
class ASTUnTempOp(ASTUnary):
    def op(self): return self.operator
    def toPromela(self):
        try:
            return self.flatten(flatten_Promela, SPIN_MAP[self.op()])
        except KeyError:
            raise LTLException("Operator " + self.op() + " not supported in Promela")
    def toJTLV(self):
        try:
            return self.flatten(flatten_JTLV, JTLV_MAP[self.op()])
        except KeyError:
            raise LTLException("Operator " + self.op() + " not supported in JTLV")
    def toSMV(self):
        return self.flatten(flatten_SMV, SMV_MAP[self.op()])

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
            self.operator = tok[1]
    def __repr__(self):
        return ' '.join (['(', str(self.op_l), self.op(), str(self.op_r), ')'])
    def flatten(self, flattener=str, op=None):
        if not op: op = self.op()
        try:
            l = flattener(self.op_l)
        except AttributeError:
            l = str(self.op_l)
        try:
            r = flattener(self.op_r)
        except AttributeError:
            r = str(self.op_r)
        return ' '.join (['(', l, op, r, ')'])
    def map(self, f):
        self.op_l = self.op_l.map(f)
        self.op_r = self.op_r.map(f)
        return f(self)


class ASTAnd(ASTBinary):
    def op(self): return "&"
    def toPromela(self):
        return self.flatten(flatten_Promela, "&&")
class ASTOr(ASTBinary):
    def op(self): return "|"
    def toPromela(self):
        return self.flatten(flatten_Promela, "||")
class ASTXor(ASTBinary):
    def op(self): return "xor"
class ASTImp(ASTBinary):
    def op(self): return "->"
class ASTBiImp(ASTBinary):
    def op(self): return "<->"
class ASTBiTempOp(ASTBinary):
    def op(self): return self.operator
    def toPromela(self):
        try:
            return self.flatten(flatten_Promela, SPIN_MAP[self.op()])
        except KeyError:
            raise LTLException("Operator " + self.op() + " not supported in Promela")
    def toJTLV(self):
        try:
            return self.flatten(flatten_JTLV, JTLV_MAP[self.op()])
        except KeyError:
            raise LTLException("Operator " + self.op() + " not supported in JTLV")
    def toSMV(self):
        return self.flatten(flatten_SMV, SMV_MAP[self.op()])
class ASTComparator(ASTBinary):
    def op(self): return self.operator
    def toPromela(self):
        if self.operator == "=":
            return self.flatten(flatten_Promela, "==")
class ASTArithmetic(ASTBinary):
    def op(self): return self.operator

# Literals cannot start with G, F or X unless quoted
restricted_alphas = filter(lambda x: x not in "GFX", alphas)
# Quirk: allow literals of the form (G|F|X)[0-9_][A-Za-z0-9._]* so we can have X0 etc.
bool_keyword = CaselessKeyword("TRUE") | CaselessKeyword("FALSE")
var = ~bool_keyword + (Word(restricted_alphas, alphanums + "." + "_") | \
        Regex("[A-Za-z][0-9_][A-Za-z0-9._]*") | QuotedString('"')).setParseAction(ASTVar)
atom = var | bool_keyword.setParseAction(ASTBool)
number = var | Word(nums).setParseAction(ASTNum)

# simple expression - no LTL operators
'''simple_expr = operatorPrecedence(atom,
        [("!", 1, opAssoc.RIGHT, ASTNot),
        (oneOf("& &&"), 2, opAssoc.LEFT, ASTAnd),
        (oneOf("| ||"), 2, opAssoc.LEFT, ASTOr),
        (oneOf("xor ^"), 2, opAssoc.LEFT, ASTXor),
        ("->", 2, opAssoc.RIGHT, ASTImp),
        ("<->", 2, opAssoc.RIGHT, ASTBiImp),
        (oneOf("< <= > >= != ="), 2, opAssoc.LEFT, ASTComparator),
        (oneOf("* /"), 2, opAssoc.LEFT, ASTArithmetic),
        (oneOf("+ -"), 2, opAssoc.LEFT, ASTArithmetic),
        ("mod", 2, opAssoc.LEFT, ASTArithmetic)
        ])'''

# arithmetic expression
arith_expr = operatorPrecedence(number,
        [(oneOf("* /"), 2, opAssoc.LEFT, ASTArithmetic),
        (oneOf("+ -"), 2, opAssoc.LEFT, ASTArithmetic),
        ("mod", 2, opAssoc.LEFT, ASTArithmetic)
        ])

# integer comparison expression
comparison_expr = Group(arith_expr + oneOf("< <= > >= != =") + arith_expr).setParseAction(ASTComparator)

proposition = comparison_expr | atom

# hack so G/F/X doesn't mess with keywords (i.e. FALSE) or variables like X0, X_0_1
UnaryTempOps = ~bool_keyword + oneOf("G F X [] <> next") + ~Word(nums + "_")

# LTL expression
ltl_expr = operatorPrecedence(proposition,
        [("'", 1, opAssoc.LEFT, ASTUnTempOp),
        ("!", 1, opAssoc.RIGHT, ASTNot),
        (UnaryTempOps, 1, opAssoc.RIGHT, ASTUnTempOp),
        (oneOf("& &&"), 2, opAssoc.LEFT, ASTAnd),
        (oneOf("| ||"), 2, opAssoc.LEFT, ASTOr),
        (oneOf("xor ^"), 2, opAssoc.LEFT, ASTXor),
        ("->", 2, opAssoc.RIGHT, ASTImp),
        ("<->", 2, opAssoc.RIGHT, ASTBiImp),
        (oneOf("= !="), 2, opAssoc.RIGHT, ASTComparator),
        (oneOf("U V R"), 2, opAssoc.RIGHT, ASTBiTempOp),
        ])
ltl_expr.ignore(LineStart() + "--" + restOfLine)

def extractVars(tree):
    v = []
    def f(t):
        if isinstance(t, ASTVar):
            v.append(t.val)
        return t
    tree.map(f)
    return v

def parse(formula):
    # Increase recursion limit for complex formulae
    sys.setrecursionlimit(2000)
    try:
        return ltl_expr.parseString(formula, parseAll=True)[0]
    except RuntimeError:
        raise ParseException("Maximum recursion depth exceeded, could not parse")

if __name__ == "__main__":
    try:
        ast = parse(sys.argv[1])
    except ParseException as e:
        print "Parse error: " + str(e)
        sys.exit(1)
    print "Parsed expression:", ast
    print "Variables:", extractVars(ast)
    try:
        print "JTLV syntax:", ast.toJTLV()
        print "SMV syntax:", ast.toSMV()
        print "Promela syntax:", ast.toPromela()
    except LTLException as e:
        print e.message
