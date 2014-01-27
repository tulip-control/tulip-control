# Copyright (c) 2014 by California Institute of Technology
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
PLY-based parser for TuLiP LTL syntax,
using AST classes from spec.ast
"""
import logging
logger = logging.getLogger(__name__)
from warnings import warn

from ast import ASTVar, ASTNum, ASTBool, ASTArithmetic, \
    ASTComparator, ASTUnTempOp, ASTBiTempOp, \
    ASTNot, ASTAnd, ASTOr, ASTXor, ASTImp, ASTBiImp

tokens = (
    'TRUE', 'FALSE',
    'NAME','NUMBER',
    'NOT', 'AND','OR', 'XOR', 'IMP', 'BIMP',
    'EQUALS', 'NEQUALS', 'LT', 'LE', 'GT', 'GE',
    'PRIME', 'ALWAYS', 'EVENTUALLY', 'NEXT',
    'UNTIL', 'RELEASE',
    'PLUS', 'MINUS', 'TIMES', 'DIV',
    'LPAREN','RPAREN',
)

# Tokens
t_TRUE = 'TRUE|True|true'
t_FALSE = 'FALSE|False|false'

t_NEXT = r'X'
t_PRIME  = r'\''
t_ALWAYS = r'\[\]|G'
t_EVENTUALLY = r'\<\>|F'

t_UNTIL = r'U'
t_RELEASE = r'R'

t_NOT = r'\!'
t_AND = r'\&\&'
t_OR = r'\|\|'
t_XOR = r'\^'

t_EQUALS = r'\=|\=\='
t_NEQUALS = r'\!\='
t_LT = r'\<'
t_LE = r'\<\='
t_GT = r'>\='
t_GE = r'>'

t_LPAREN = r'\('
t_RPAREN = r'\)'

t_NAME = r'[A-EH-QSTWYZa-z_][A-za-z0-9._:]*|[A-Za-z][0-9_][a-zA-Z0-9._:]*'
t_NUMBER = r'\d+'

t_IMP = '->'
t_BIMP = '\<->'

t_PLUS = r'\+'
t_MINUS = r'-'
t_TIMES = r'\*'
t_DIV = r'/'

# Ignored characters
t_ignore = " \t"

def t_newline(t):
    r'\n+'
    t.lexer.lineno += t.value.count("\n")
    
def t_error(t):
    warn("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)
    
# Build the lexer
import ply.lex as lex
lex.lex()

# lowest to highest
precedence = (
    ('right', 'UNTIL', 'RELEASE'),
    ('right', 'BIMP'),
    ('right', 'IMP'),
    ('left', 'XOR'),
    ('left', 'OR'),
    ('left', 'AND'),
    ('right', 'ALWAYS', 'EVENTUALLY'),
    ('right', 'NEXT'),
    ('right', 'NOT'),
    ('left', 'PRIME'),
    ('nonassoc', 'EQUALS', 'NEQUALS', 'LT', 'LE', 'GT', 'GE'),
    ('nonassoc', 'TIMES', 'DIV'),
    ('nonassoc', 'PLUS', 'MINUS'),
    ('nonassoc', 'TRUE', 'FALSE')
)

# dictionary of names
#names = {'var':'replacement'}

def p_arithmetic(t):
    """expression : expression TIMES expression
                  | expression DIV expression
                  | expression PLUS expression
                  | expression MINUS expression
    """
    t[0] = ASTArithmetic(None, None, t[1:4])

def p_comparator(t):
    """expression : expression EQUALS expression
                  | expression NEQUALS expression
                  | expression LT expression
                  | expression LE expression
                  | expression GT expression
                  | expression GE expression
    """
    t[0] = ASTComparator(None, None, t[1:4])

def p_and(t):
    """expression : expression AND expression
    """
    t[0] = ASTAnd(None, None, t[1:4])

def p_or(t):
    """expression : expression OR expression
    """
    t[0] = ASTOr(None, None, t[1:4])

def p_xor(t):
    """expression : expression XOR expression
    """
    t[0] = ASTXor(None, None, t[1:4])

def p_imp(t):
    """expression : expression IMP expression
    """
    t[0] = ASTImp(None, None, t[1:4])

def p_bimp(t):
    """expression : expression BIMP expression
    """
    t[0] = ASTBiImp(None, None, t[1:4])

def p_unary_temp_op(t):
    """expression : NEXT expression
                  | ALWAYS expression
                  | EVENTUALLY expression
    """
    t[0] = ASTUnTempOp(None, None, t[1:3])

def p_bin_temp_op(t):
    """expression : expression UNTIL expression
                  | expression RELEASE expression
    """
    t[0] = ASTBiTempOp(None, None, t[1:4])

def p_not(t):
    """expression : NOT expression
    """
    t[0] = ASTNot(None, None, t[1:3])

def p_group(t):
    """expression : LPAREN expression RPAREN
    """
    t[0] = t[2]

def p_number(t):
    """expression : NUMBER
    """
    t[0] = ASTNum(None, None, [t[1]])

def p_expression_name(t):
    """expression : NAME
    """
    t[0] = ASTVar(None, None, [t[1]])

def p_bool(t):
    """expression : TRUE
                  | FALSE
    """
    t[0] = ASTBool(None, None, [t[1]])

def p_error(t):
    warn("Syntax error at '%s'" % t.value)

import ply.yacc as yacc
yacc.yacc()

def parse(formula):
    """Parse formula string and create abstract syntax tree (AST).
    """
    return yacc.parse(formula)
    
if __name__ == '__main__':
    s = 'up && !(loc = 29) && X((u_in = 0) || (u_in = 2))'
    p = yacc.parse(s)
    
    print('Parsing result: ' + str(p.to_gr1c()))
