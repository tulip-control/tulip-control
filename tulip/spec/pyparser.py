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
Pyparsing-based parser for TuLiP LTL syntax,
using AST classes from spec.ast

Syntax taken originally roughly from http://spot.lip6.fr/wiki/LtlSyntax
"""
import pyparsing as pp
import sys

from .ast import (ASTVar, ASTNum, ASTBool, ASTArithmetic,
    ASTComparator, ASTUnTempOp, ASTBiTempOp,
    ASTNot, ASTAnd, ASTOr, ASTXor, ASTImp, ASTBiImp)

# Packrat parsing - it's much faster
pp.ParserElement.enablePackrat()

# Literals cannot start with G, F or X unless quoted
_restricted_alphas = filter(lambda x: x not in "GFX", pp.alphas)

# Quirk: allow literals of the form:
#   (G|F|X)[0-9_][A-Za-z0-9._]*
# so we can have X0 etc.
_bool_keyword = pp.CaselessKeyword("TRUE") | pp.CaselessKeyword("FALSE")

_var = ~_bool_keyword + (
    pp.Word(_restricted_alphas, pp.alphanums + "._:") | \
    pp.Regex("[A-Za-z][0-9_][A-Za-z0-9._:]*") | pp.QuotedString('"')
).setParseAction(ASTVar)

_atom = _var | _bool_keyword.setParseAction(ASTBool)
_number = _var | pp.Word(pp.nums).setParseAction(ASTNum)

# arithmetic expression
_arith_expr = pp.operatorPrecedence(
    _number,
    [
        (pp.oneOf("* /"), 2, pp.opAssoc.LEFT, ASTArithmetic),
        (pp.oneOf("+ -"), 2, pp.opAssoc.LEFT, ASTArithmetic),
        ("mod", 2, pp.opAssoc.LEFT, ASTArithmetic)
    ]
)

# integer comparison expression
_comparison_expr = pp.Group(
    _arith_expr + pp.oneOf("< <= > >= != = ==") + _arith_expr
).setParseAction(ASTComparator)

_proposition = _comparison_expr | _atom

# hack so G/F/X doesn't mess with keywords
#(i.e. FALSE) or variables like X0, X_0_1
_UnaryTempOps = ~_bool_keyword + \
    pp.oneOf("G F X [] <> next") + ~pp.Word(pp.nums + "_")

def parse(formula):
    """Parse formula string and create abstract syntax tree (AST).
    """
    # LTL expression
    _ltl_expr = pp.operatorPrecedence(
        _proposition,
       [("'", 1, pp.opAssoc.LEFT, ASTUnTempOp),
        ("!", 1, pp.opAssoc.RIGHT, ASTNot),
        (_UnaryTempOps, 1, pp.opAssoc.RIGHT, ASTUnTempOp),
        (pp.oneOf("& &&"), 2, pp.opAssoc.LEFT, ASTAnd),
        (pp.oneOf("| ||"), 2, pp.opAssoc.LEFT, ASTOr),
        (pp.oneOf("xor ^"), 2, pp.opAssoc.LEFT, ASTXor),
        ("->", 2, pp.opAssoc.RIGHT, ASTImp),
        ("<->", 2, pp.opAssoc.RIGHT, ASTBiImp),
        (pp.oneOf("= == !="), 2, pp.opAssoc.RIGHT, ASTComparator),
        (pp.oneOf("U V R"), 2, pp.opAssoc.RIGHT, ASTBiTempOp)]
    )
    _ltl_expr.ignore(pp.LineStart() + "--" + pp.restOfLine)

    # Increase recursion limit for complex formulae
    sys.setrecursionlimit(2000)
    try:
        return _ltl_expr.parseString(formula, parseAll=True)[0]
    except RuntimeError:
        raise pp.ParseException(
            "Maximum recursion depth exceeded,"
            "could not parse"
        )
