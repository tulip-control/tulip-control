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
from __future__ import absolute_import

import logging
logger = logging.getLogger(__name__)
from warnings import warn

import ply.lex as lex
import ply.yacc as yacc
import networkx as nx

from . import ast

tokens = (
    'TRUE', 'FALSE',
    'NAME','NUMBER',
    'NOT', 'AND','OR', 'XOR', 'IMP', 'BIMP',
    'EQUALS', 'NEQUALS', 'LT', 'LE', 'GT', 'GE',
    'PRIME', 'ALWAYS', 'EVENTUALLY', 'NEXT',
    'UNTIL', 'RELEASE',
    'PLUS', 'MINUS', 'TIMES', 'DIV',
    'LPAREN','RPAREN', 'DQUOTES'
)

# Tokens
t_TRUE = 'TRUE|True|true'
t_FALSE = 'FALSE|False|false'

t_NEXT = r'X|next'
t_PRIME  = r'\''
t_ALWAYS = r'\[\]|G'
t_EVENTUALLY = r'\<\>|F'

t_UNTIL = r'U'
t_RELEASE = r'R'

t_NOT = r'\!'
t_AND = r'\&\&|\&'
t_OR = r'\|\||\|'
t_XOR = r'\^'

t_EQUALS = r'\=|\=\='
t_NEQUALS = r'\!\='
t_LT = r'\<'
t_LE = r'\<\='
t_GT = r'>\='
t_GE = r'>'

t_LPAREN = r'\('
t_RPAREN = r'\)'

t_NAME = r'(?!next)([A-EH-QSTWYZa-z_][A-za-z0-9._:]*|[A-Za-z][0-9_][a-zA-Z0-9._:]*)'
t_NUMBER = r'\d+'

t_IMP = '->'
t_BIMP = '\<->'

t_PLUS = r'\+'
t_MINUS = r'-'
t_TIMES = r'\*'
t_DIV = r'/'

t_DQUOTES = r'\"'

# Ignored characters
t_ignore = " \t"

def t_newline(t):
    r'\n+'
    t.lexer.lineno += t.value.count("\n")
    
def t_error(t):
    warn("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)
    
# Build the lexer
lexer = lex.lex()

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

TABMODULE = 'tulip.spec.parsetab'

class LTLParser(object):
    def __init__(self):
        self.graph = None
        self.tokens = tokens
    
    def build(self):
        self.parser = yacc.yacc(
            tabmodule=TABMODULE, module=self,
            write_tables=False, debug=False
        )
    
    def rebuild_parsetab(self, tabmodule):
        yacc.yacc(
            tabmodule=tabmodule, module=self,
            write_tables=True, debug=True
        )
    
    def parse(self, formula):
        """Parse formula string and create abstract syntax tree (AST).
        """
        self.build()
        g = nx.DiGraph()
        self.graph = g
        root_node = self.parser.parse(formula, lexer=lexer, debug=logger)
        
        # mark root
        self.graph.root = id(root_node)
        return root_node
    
    def p_arithmetic(self, p):
        """expression : expression TIMES expression
                      | expression DIV expression
                      | expression PLUS expression
                      | expression MINUS expression
        """
        p[0] = ast.Arithmetic(p[2], p[1], p[3], self.graph)
    
    def p_comparator(self, p):
        """expression : expression EQUALS expression
                      | expression NEQUALS expression
                      | expression LT expression
                      | expression LE expression
                      | expression GT expression
                      | expression GE expression
        """
        p[0] = ast.Comparator(p[2], p[1], p[3], self.graph)
    
    def p_and(self, p):
        """expression : expression AND expression
        """
        p[0] = ast.And(p[2], p[1], p[3], self.graph)
    
    def p_or(self, p):
        """expression : expression OR expression
        """
        p[0] = ast.Or(p[2], p[1], p[3], self.graph)
    
    def p_xor(self, p):
        """expression : expression XOR expression
        """
        p[0] = ast.Xor(p[2], p[1], p[3], self.graph)
    
    def p_imp(self, p):
        """expression : expression IMP expression
        """
        p[0] = ast.Imp(p[2], p[1], p[3], self.graph)
    
    def p_bimp(self, p):
        """expression : expression BIMP expression
        """
        p[0] = ast.BiImp(p[2], p[1], p[3], self.graph)
    
    def p_unary_temp_op(self, p):
        """expression : NEXT expression
                      | ALWAYS expression
                      | EVENTUALLY expression
        """
        p[0] = ast.UnTempOp(p[1], p[2], self.graph)
    
    def p_bin_temp_op(self, p):
        """expression : expression UNTIL expression
                      | expression RELEASE expression
        """
        p[0] = ast.BiTempOp(p[2], p[1], p[3], self.graph)
    
    def p_not(self, p):
        """expression : NOT expression
        """
        p[0] = ast.Not(p[1], p[2], self.graph)
    
    def p_group(self, p):
        """expression : LPAREN expression RPAREN
        """
        p[0] = p[2]
    
    def p_number(self, p):
        """expression : NUMBER
        """
        p[0] = ast.Num(p[1], self.graph)
    
    def p_expression_name(self, p):
        """expression : NAME
        """
        p[0] = ast.Var(p[1], self.graph)
    
    def p_expression_const(self, p):
        """expression : DQUOTES NAME DQUOTES
        """
        p[0] = ast.Const(p[2], self.graph)
    
    def p_bool(self, p):
        """expression : TRUE
                      | FALSE
        """
        p[0] = ast.Bool([p[1]], self.graph)
    
    def p_error(self, p):
        warn("Syntax error at '%s'" % p.value)

def parse(formula):
    parser = LTLParser()
    parser.parse(formula)

if __name__ == '__main__':
    s = 'up && !(loc = 29) && X((u_in = 0) || (u_in = 2))'
    parsed_formula = parse(s)
    
    print('Parsing result: ' + str(parsed_formula.to_gr1c()))
