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
LTL parser supporting JTLV, SPIN, SMV, and gr1c syntax
"""
import sys

from .ast import LTLException, ASTVar, ASTUnTempOp, \
    ASTBiTempOp, ASTUnary, ASTBinary

def extract_vars(tree):
    v = []
    def f(t):
        if isinstance(t, ASTVar):
            v.append(t.val)
        return t
    tree.map(f)
    return v
    
# Crude test for safety spec
def issafety(tree):
    def f(t):
        if isinstance(t, ASTUnTempOp) and not t.operator == "G":
            return False
        if isinstance(t, ASTBiTempOp):
            return False
        if isinstance(t, ASTUnary):
            return t.operand
        if isinstance(t, ASTBinary):
            return (t.op_l and t.op_r)
        return True
    return tree.map(f)

def parse(formula, parser='ply'):
    """Parse formula string and create abstract syntax tree (AST).
    
    Both PyParsing and PLY are available for the parsing.
    For large formulae and repeated parsing PLY is faster.
    
    @param parser: python package to use for generating lexer and parser
    @type parser: 'pyparsing' | 'ply'
    """
    if parser == 'pyparsing':
        from .pyparser import parse as pyparse
        spec = pyparse(formula)
    elif parser == 'ply':
        from .plyparser import parse as plyparse
        spec = plyparse(formula)
    else:
        raise ValueError(
            'Unknown parser: ' + str(parser) + '\n' +
            "Available options: 'ply' | 'pyparsing'"
        )
    
    # did ply fail merely printing warnings ?
    if spec is None:
        raise Exception('Parsing formula:\n' +
                        str(formula) + 'failed.')
    return spec

if __name__ == "__main__":
    try:
        from .pyparser import parse as pyparse
        ast = pyparse(sys.argv[1])
    except Exception as e:
        print("Parse error: " + str(e) )
        sys.exit(1)
    print("Parsed expression: " + str(ast) )
    print("Length: " +str( len(ast) ) )
    print("Variables: " + str(extract_vars(ast) ) )
    print("Safety: " +str(issafety(ast) ) )
    try:
        print("JTLV syntax: " +str(ast.to_jtlv() ) )
        print("SMV syntax: " +str(ast.to_smv() ) )
        print("Promela syntax: " +str(ast.to_promela() ) )
    except LTLException as e:
        print(e.message)
