# Copyright (c) 2014, 2015 by California Institute of Technology
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
"""PLY-based parser for TuLiP LTL syntax.

This parser uses syntax-tree classes from
the module `tulip.spec.ast`.
"""
import logging
import warnings

import ply.lex
import ply.yacc

import tulip.spec.ast as _ast


_logger = logging.getLogger(__name__)
TABMODULE = 'tulip.spec.ltl_parsetab'
LEX_LOGGER = 'tulip.ltl_lex_log'
YACC_LOGGER = 'tulip.ltl_yacc_log'
PARSER_LOGGER = 'tulip.ltl_parser_log'
# TODO: add past fragment of LTL


class Lexer:
    """Token rules to build LTL lexer."""

    def __init__(
            self,
            debug:
                bool=False):
        self.reserved = {
            'ite':
                'ITE',
            'next':
                'NEXT',
            'X':
                'NEXT',
            'FALSE':
                'FALSE',
            'TRUE':
                'TRUE',
            'G':
                'ALWAYS',
            'F':
                'EVENTUALLY',
            'U':
                'UNTIL',
            'W':
                'WEAK_UNTIL',
            'V':
                'RELEASE'}
        self.values = {
            'next':
                'X'}
        self.delimiters = [
            'LPAREN',
            'RPAREN',
            'DQUOTES',
            'COMMA']
        self.operators = [
            # logic
            'NOT',
            'AND',
            'OR',
            'XOR',
            'IMP',
            'BIMP',
            # set theory
            'EQUALS',
            'NEQUALS',
            # arithmetic
            'LT',
            'LE',
            'GT',
            'GE',
            'PLUS',
            'MINUS',
            'TIMES',
            'DIV',
            'TRUNCATE',
            # action operators
            'PRIME']
        self.misc = ['NAME', 'NUMBER']
        # for setting the logger,
        # directly call the method `build`
        self.tokens = (
            self.delimiters +
            self.operators +
            self.misc +
            list(set(self.reserved.values())))
        self.build(debug=debug)

    def t_NAME(self, t):
        r"""
        [A-Za-z_]
        [A-za-z0-9._:]*
        """
        t.value = self.values.get(t.value, t.value)
        t.type = self.reserved.get(t.value, 'NAME')
        # special treatment
        if t.value.upper() in {'FALSE', 'TRUE'}:
            t.type = self.reserved[t.value.upper()]
        return t

    def t_ALWAYS(self, t):
        r' \[ \] '
        t.value = 'G'
        return t

    def t_EVENTUALLY(self, t):
        r' <> '
        t.value = 'F'
        return t

    def t_AND(self, t):
        r"""
          \& \&
        | \&
        | / \\
        """
        t.value = '&'
        return t

    def t_OR(self, t):
        r"""
          \| \|
        | \|
        | \\ /
        """
        t.value = '|'
        return t

    t_NOT = r'''
          !
        | \~
        '''
    t_XOR = r' \^ '
    t_EQUALS = r' = '
        # a declarative language
        # has no assignment
    t_NEQUALS = r'''
          !=
        | /=
        '''
    t_LT = r' < '
    t_LE = r'''
          <=
        | =<
        '''
    t_GT = r' >= '
    t_GE = r' > '
    t_LPAREN = r' \( '
    t_RPAREN = r' \) '
    t_NUMBER = r' \d+ '
    t_IMP = r'''
          \- >
        | =>
        '''
    t_BIMP = r'''
          < \- >
        | <=>
        '''
    t_PLUS = r' \+ '
    t_MINUS = r' \- '
    t_TIMES = r' \* '
    t_DIV = r' / '
    t_TRUNCATE = r' <<>> '
    t_COMMA = r' , '
    t_DQUOTES = r' " '
    t_PRIME = r" ' "
    # skipped characters
    t_ignore = ''.join(['\x20', '\t'])

    def t_comment(self, t):
        r' \# .* '
        return

    def t_newline(self, t):
        r' \n+ '
        t.lexer.lineno += t.value.count('\n')

    def t_error(self, t):
        warnings.warn(
            'Unknown character '
            f'`{t.value[0]}`')
        t.lexer.skip(1)

    def build(
            self,
            debug:
                bool=False,
            debuglog:
                logging.Logger |
                None=None,
            **kwargs):
        """Create a lexer.

        @param kwargs:
            same arguments as
            for the function `ply.lex.lex`,
            - except for `module`
              (fixed to `self`)
            - `debuglog` defaults to the
              logger `"ltl_lex_log"`
        """
        if debug and debuglog is None:
            debuglog = logging.getLogger(
                LEX_LOGGER)
        self.lexer = ply.lex.lex(
            module=self,
            debug=debug,
            debuglog=debuglog,
            **kwargs)


class Parser:
    """Production rules to build LTL parser."""

    def __init__(
            self,
            ast=None,
            lexer=None):
        if ast is None:
            self.ast = _ast.nodes
        else:
            self.ast = ast
        if lexer is None:
            self.lexer = Lexer()
        else:
            self.lexer = lexer
        self.tabmodule = TABMODULE
        self.start = 'expr'
        # lowest to highest
        # closely follows `spin.y`
        self.precedence = (
            ('left',
                'BIMP'),
            ('left',
                'IMP'),
            ('left',
                'XOR'),
            ('left',
                'OR'),
            ('left',
                'AND'),
            ('left',
                'ALWAYS',
                'EVENTUALLY'),
            ('left',
                'UNTIL',
                'WEAK_UNTIL',
                'RELEASE'),
            ('left',
                'EQUALS',
                'NEQUALS'),
            ('left',
                'LT',
                'LE',
                'GT',
                'GE'),
            ('left',
                'PLUS',
                'MINUS'),
            ('left',
                'TIMES',
                'DIV'),
            ('right',
                'NOT',
                'UMINUS'),
            ('right',
                'NEXT'),
            ('left',
                'PRIME'),
            ('nonassoc',
                'TRUE',
                'FALSE'))
        self.tokens = self.lexer.tokens
        self.build()

    def build(
            self,
            tabmodule:
                str |
                None=None,
            outputdir:
                str='',
            write_tables:
                bool=False,
            debug:
                bool=False,
            debuglog:
                logging.Logger |
                None=None):
        """Build parser using `ply.yacc`.

        The default table module is
        as defined in `self.tabmodule`.
        The default logger is `YACC_LOGGER`.
        """
        if tabmodule is None:
            tabmodule = self.tabmodule
        if debug and debuglog is None:
            debuglog = logging.getLogger(
                YACC_LOGGER)
        self.parser = ply.yacc.yacc(
            method='LALR',
            module=self,
            start=self.start,
            tabmodule=tabmodule,
            outputdir=outputdir,
            write_tables=write_tables,
            debug=debug,
            debuglog=debuglog)

    def parse(
            self,
            formula:
                str,
            debuglog:
                logging.Logger |
                None=None
            ) -> _ast.NodeSpec:
        """Return syntax tree for `formula`.

        @param formula:
            logic formula
        @param debuglog:
            logger passed as
            keyword parameter `debuglog` to
            the method
            `ply.yacc.LRParser.parse`.
            The default value is the logger
            with name `PARSER_LOGGER`.
        @return:
            abstract syntax tree that
            results from parsing `formula`
            (the return type depends on
            whether the parameter `ast` was
            passed to `self.__init__`)
        """
        if debuglog is None:
            debuglog = logging.getLogger(
                PARSER_LOGGER)
        root = self.parser.parse(
            formula,
            lexer=self.lexer.lexer,
            debug=debuglog)
        if root is None:
            raise Exception(
                'failed to parse:\n'
                f'\t{formula}')
        return root

    def p_nullary_connective(self, p):
        """expr : TRUE
                | FALSE
        """
        p[0] = self.ast.Bool(p[1])

    def p_unary_connective(self, p):
        """expr : NOT expr
                | ALWAYS expr
                | EVENTUALLY expr
                | NEXT expr
        """
        p[0] = self.ast.Unary(p[1], p[2])

    # both function and connective
    def p_postfix_next(self, p):
        """expr : expr PRIME"""
        p[0] = self.ast.Unary('X', p[1])

    def p_binary_connective(self, p):
        """expr : expr AND expr
                | expr OR expr
                | expr XOR expr
                | expr IMP expr
                | expr BIMP expr
                | expr UNTIL expr
                | expr WEAK_UNTIL expr
                | expr RELEASE expr
        """
        p[0] = self.ast.Binary(p[2], p[1], p[3])

    # both function and connective
    def p_ternary_conditional(self, p):
        ("""expr : LPAREN ITE expr """
         """COMMA expr COMMA expr RPAREN""")
        p[0] = self.ast.Operator(
            p[2], p[3], p[5], p[7])

    def p_binary_predicate(self, p):
        """expr : expr EQUALS expr
                | expr NEQUALS expr
                | expr LT expr
                | expr LE expr
                | expr GT expr
                | expr GE expr
        """
        p[0] = self.ast.Comparator(
            p[2], p[1], p[3])

    def p_truncator(self, p):
        """expr : expr TRUNCATE number"""
        p[0] = self.ast.Arithmetic(
            p[2], p[1], p[3])

    def p_binary_function(self, p):
        """expr : expr TIMES expr
                | expr DIV expr
                | expr PLUS expr
                | expr MINUS expr
        """
        p[0] = self.ast.Arithmetic(
            p[2], p[1], p[3])

    def p_paren(self, p):
        """expr : LPAREN expr RPAREN"""
        p[0] = p[2]

    def p_var(self, p):
        """expr : NAME"""
        p[0] = self.ast.Var(p[1])

    def p_number_expr(self, p):
        """expr : number"""
        p[0] = p[1]

    def p_number(self, p):
        """number : NUMBER"""
        p[0] = self.ast.Num(p[1])

    def p_negative_number(self, p):
        """expr : MINUS NUMBER %prec UMINUS"""
        p[0] = self.ast.Num('-' + p[2])

    def p_string(self, p):
        """expr : DQUOTES NAME DQUOTES"""
        p[0] = self.ast.Str(p[2])

    def p_error(self, p):
        s = list()
        while True:
            tok = self.parser.token()
            if tok is None:
                break
            s.append(tok.value)
        s = ' '.join(s)
        raise Exception(
            f'Syntax error at "{p.value}"\n' +
            f'remaining input:\n{s}\n')


def parse(
        formula:
            str
        ) -> _ast.NodeSpec:
    warnings.warn(
        'Deprecated: Better to '
        'instantiate a Parser once only.')
    parser = Parser()
    return parser.parse(formula)


def _main():
    h = logging.FileHandler(
        'log.txt',
        mode='w')
    h.setLevel(logging.DEBUG)
    log = logging.getLogger(YACC_LOGGER)
    log.setLevel(logging.DEBUG)
    log.addHandler(h)
    import os
    tabmodule = TABMODULE.split('.')[-1]
    outputdir = './'
    tablepy = f'{tabmodule}.py'
    tablepyc = f'{tabmodule}.pyc'
    try:
        os.remove(tablepy)
    except:
        print(f'no "{tablepy}" found')
    try:
        os.remove(tablepyc)
    except:
        print(f'no "{tablepyc}" found')
    parser = Parser()
    parser.build(
        tabmodule,
        outputdir=outputdir,
        write_tables=True,
        debug=True)


if __name__ == '__main__':
    _main()
