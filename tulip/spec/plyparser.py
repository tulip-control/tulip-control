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

import warnings

import ply.lex
import ply.yacc

from . import ast


TABMODULE = 'tulip.spec.parsetab'

LEX_LOGGER = '{name}.lex_logger'.format(name=__name__)
YACC_LOGGER = '{name}.yacc_logger'.format(name=__name__)
PARSER_LOGGER = '{name}.parser_logger'.format(name=__name__)


def _format_docstring(**kwargs):
    """Apply C{kwargs} to function docstring using C{format}."""

    def dec(func):
        func.__doc__ = func.__doc__.format(**kwargs)
        return func

    return dec


class Lexer(object):
    """Token rules to build LTL lexer."""

    def __init__(self, debug=False):
        # for setting the logger, call build explicitly
        self.build(debug=debug)

    tokens = (
        'TRUE', 'FALSE',
        'NAME', 'NUMBER',
        'NOT', 'AND', 'OR', 'XOR', 'IMP', 'BIMP',
        'EQUALS', 'NEQUALS', 'LT', 'LE', 'GT', 'GE',
        'ALWAYS', 'EVENTUALLY', 'NEXT',
        'UNTIL', 'RELEASE',
        'PLUS', 'MINUS', 'TIMES', 'DIV',
        'LPAREN', 'RPAREN', 'DQUOTES', 'PRIME',
        'COMMENT', 'NEWLINE'
    )

    # Tokens
    t_TRUE = 'TRUE|True|true'
    t_FALSE = 'FALSE|False|false'

    t_NEXT = r'X|next'
    # t_PRIME  = r'\''
    t_ALWAYS = r'\[\]|G'
    t_EVENTUALLY = r'\<\>|F'

    t_UNTIL = r'U'
    t_RELEASE = r'R'

    t_NOT = r'\!'
    t_AND = r'\&\&|\&'
    t_OR = r'\|\||\|'
    t_XOR = r'\^'

    t_EQUALS = r'\=\=|\='
    t_NEQUALS = r'\!\='
    t_LT = r'\<'
    t_LE = r'\<\='
    t_GT = r'>\='
    t_GE = r'>'

    t_LPAREN = r'\('
    t_RPAREN = r'\)'

    t_NAME = (r'(?!next)([A-EH-QSTWYZa-z_][A-za-z0-9._:]*|'
              r'[A-Za-z][0-9_][a-zA-Z0-9._:]*)')
    t_NUMBER = r'\d+'

    t_IMP = '->'
    t_BIMP = '\<->'

    t_PLUS = r'\+'
    t_MINUS = r'-'
    t_TIMES = r'\*'
    t_DIV = r'/'

    t_DQUOTES = r'\"'
    t_PRIME = r"\'"

    # Ignored characters
    t_ignore = " \t"

    def t_COMMENT(self, t):
        r'\#.*'
        return

    def t_NEWLINE(self, t):
        r'\n+'
        t.lexer.lineno += t.value.count("\n")

    def t_error(self, t):
        warnings.warn('Illegal character "{t}"'.format(t=t.value[0]))
        t.lexer.skip(1)

    @_format_docstring(logger=LEX_LOGGER)
    def build(self, debug=False, debuglog=None, **kwargs):
        """Create a lexer.

        @param kwargs: Same arguments as C{{ply.lex.lex}}:

          - except for C{{module}} (fixed to C{{self}})
          - C{{debuglog}} defaults to the logger C{{"{logger}"}}.
        """
        if debug and debuglog is None:
            debuglog = logging.getLogger(LEX_LOGGER)

        self.lexer = ply.lex.lex(
            module=self,
            debug=debug,
            debuglog=debuglog,
            **kwargs
        )


class Parser(object):
    """Production rules to build LTL parser."""

    # lowest to highest
    precedence = (
        ('right', 'UNTIL', 'RELEASE'),
        ('right', 'BIMP'),
        ('right', 'IMP'),
        ('left', 'XOR'),
        ('left', 'OR'),
        ('left', 'AND'),
        ('right', 'ALWAYS', 'EVENTUALLY'),
        ('right', 'NOT'),
        ('left', 'EQUALS', 'NEQUALS'),
        ('left', 'LT', 'LE', 'GT', 'GE'),
        ('left', 'PLUS', 'MINUS'),
        ('left', 'TIMES', 'DIV'),
        ('right', 'NEXT'),
        ('left', 'PRIME'),
        ('nonassoc', 'TRUE', 'FALSE')
    )

    def __init__(self):
        self.graph = None

        self.lexer = Lexer()
        self.tokens = self.lexer.tokens

        self.build()

    def build(self):
        self.parser = ply.yacc.yacc(
            module=self,
            tabmodule=TABMODULE,
            write_tables=False,
            debug=False
        )

    @_format_docstring(logger=YACC_LOGGER)
    def rebuild_parsetab(self, tabmodule, outputdir='',
                         debug=True, debuglog=None):
        """Rebuild parsetable in debug mode.

        @param tabmodule: name of table file
        @type tabmodule: C{{str}}

        @param outputdir: save C{{tabmodule}} in this directory.
        @type outputdir: c{{str}}

        @param debuglog: defaults to logger C{{"{logger}"}}.
        @type debuglog: C{{logging.Logger}}
        """
        if debug and debuglog is None:
            debuglog = logging.getLogger(YACC_LOGGER)

        self.lexer.build(debug=debug)

        self.parser = ply.yacc.yacc(
            module=self,
            tabmodule=tabmodule,
            outputdir=outputdir,
            write_tables=True,
            debug=debug,
            debuglog=debuglog
        )

    @_format_docstring(logger=PARSER_LOGGER)
    def parse(self, formula, debuglog=None):
        """Parse formula string and create abstract syntax tree (AST).

        @param logger: defaults to logger C{{"{logger}"}}.
        @type logger: C{{logging.Logger}}
        """
        if debuglog is None:
            debuglog = logging.getLogger(PARSER_LOGGER)

        g = ast.LTL_AST()
        self.graph = g
        root = self.parser.parse(
            formula,
            lexer=self.lexer.lexer,
            debug=debuglog
        )

        if root is None:
            raise Exception('failed to parse:\n\t{f}'.format(f=formula))

        g.root = root
        return g

    def add_identifier(self, Type, x):
        identifier = Type(x)
        self.graph.add_identifier(identifier)
        return identifier

    def add_unary(self, Type, p):
        operator = Type(p[1])
        self.graph.add_unary(operator, p[2])
        return operator

    def add_binary(self, Type, p):
        operator = Type(p[2])
        self.graph.add_binary(operator, p[1], p[3])
        return operator

    def p_arithmetic(self, p):
        """expression : expression TIMES expression
                      | expression DIV expression
                      | expression PLUS expression
                      | expression MINUS expression
        """
        p[0] = self.add_binary(ast.Arithmetic, p)

    def p_comparator(self, p):
        """expression : expression EQUALS expression
                      | expression NEQUALS expression
                      | expression LT expression
                      | expression LE expression
                      | expression GT expression
                      | expression GE expression
        """
        p[0] = self.add_binary(ast.Comparator, p)

    def p_and(self, p):
        """expression : expression AND expression"""
        p[0] = self.add_binary(ast.And, p)

    def p_or(self, p):
        """expression : expression OR expression"""
        p[0] = self.add_binary(ast.Or, p)

    def p_xor(self, p):
        """expression : expression XOR expression"""
        p[0] = self.add_binary(ast.Xor, p)

    def p_imp(self, p):
        """expression : expression IMP expression"""
        p[0] = self.add_binary(ast.Imp, p)

    def p_bimp(self, p):
        """expression : expression BIMP expression"""
        p[0] = self.add_binary(ast.BiImp, p)

    def p_unary_temp_op(self, p):
        """expression : NEXT expression
                      | ALWAYS expression
                      | EVENTUALLY expression
        """
        p[0] = self.add_unary(ast.UnTempOp, p)

    def p_postfix_next(self, p):
        """expression : expression PRIME"""
        p[0] = self.add_unary(ast.UnTempOp, [None, 'X', p[1]])

    def p_bin_temp_op(self, p):
        """expression : expression UNTIL expression
                      | expression RELEASE expression
        """
        p[0] = self.add_binary(ast.BiTempOp, p)

    def p_not(self, p):
        """expression : NOT expression"""
        p[0] = self.add_unary(ast.Not, p)

    def p_group(self, p):
        """expression : LPAREN expression RPAREN"""
        p[0] = p[2]

    def p_number(self, p):
        """expression : NUMBER"""
        p[0] = self.add_identifier(ast.Num, p[1])

    def p_expression_name(self, p):
        """expression : NAME"""
        p[0] = self.add_identifier(ast.Var, p[1])

    def p_expression_const(self, p):
        """expression : DQUOTES NAME DQUOTES"""
        p[0] = self.add_identifier(ast.Const, p[2])

    def p_bool(self, p):
        """expression : TRUE
                      | FALSE
        """
        p[0] = self.add_identifier(ast.Bool, p[1])

    def p_error(self, p):
        print('Syntax error at "{p}"'.format(p=p.value))


def parse(formula):
    warnings.warn('Deprecated: Better to instantiate a Parser once only.')
    parser = Parser()
    return parser.parse(formula)


if __name__ == '__main__':
    s = 'up && !(loc = 29) && X((u_in = 0) || (u_in = 2))'
    parsed_formula = parse(s)
    print('Parsing result: {s}'.format(s=parsed_formula.to_gr1c()))
