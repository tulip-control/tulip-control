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
"""Interface to ltl2ba"""
import logging
logger = logging.getLogger(__name__)
import subprocess
import networkx as nx
import ply.lex
import ply.yacc


TABMODULE = 'ltl2ba_parsetab'


class Lexer(object):
    """Token rules to build lexer for ltl2ba output."""

    reserved = {
        'goto': 'GOTO',
        'if': 'IF',
        'fi': 'FI',
        'never': 'NEVER',
        'skip': 'SKIP'}

    tokens = (
        'TRUE', 'FALSE',
        'NUMBER',
        'NOT', 'AND', 'OR', 'XOR', 'IMP', 'BIMP',
        'EQUALS', 'NEQUALS', 'LT', 'LE', 'GT', 'GE',
        'PLUS', 'MINUS', 'TIMES', 'DIV',
        'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE',
        'COLON', 'COLON2', 'SEMI',
        'COMMENT', 'NAME') + tuple(reserved.values())

    # Tokens
    t_TRUE = 'TRUE|True|true'
    t_FALSE = 'FALSE|False|false'

    t_COMMENT = '/\*.*\*/'
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
    t_LBRACE = r'\{'
    t_RBRACE = r'\}'
    t_SEMI = r';'
    t_COLON2 = r'::'
    t_COLON = r':'

    t_NUMBER = r'\d+'

    t_IMP = r'->'

    # Ignored characters
    t_ignore = ' \t'

    def __init__(self):
        self.lexer = ply.lex.lex(module=self, debug=False)

    def t_newline(self, t):
        r'\n+'
        t.lexer.lineno += t.value.count("\n")

    def t_error(self, t):
        logger.warn("Illegal character '%s'" % t.value[0])
        t.lexer.skip(1)

    def t_name(self, t):
        r'[A-Za-z_][a-zA-Z0-9_]*'
        t.type = self.reserved.get(t.value, 'NAME')
        return t


class Parser(object):
    """Production rules to build parser for ltl2ba output."""

    precedence = (
        # ('right', 'UNTIL', 'RELEASE'),
        ('right', 'BIMP'),
        ('right', 'IMP'),
        ('left', 'XOR'),
        ('left', 'OR'),
        ('left', 'AND'),
        # ('right', 'ALWAYS', 'EVENTUALLY'),
        # ('right', 'NEXT'),
        ('right', 'NOT'),
        # ('left', 'PRIME'),
        ('nonassoc', 'EQUALS', 'NEQUALS', 'LT', 'LE', 'GT', 'GE'),
        ('nonassoc', 'TIMES', 'DIV'),
        ('nonassoc', 'PLUS', 'MINUS'),
        ('nonassoc', 'TRUE', 'FALSE'))

    def __init__(self):
        """Build lexer and parser."""
        self.lexer = Lexer()
        self.tokens = self.lexer.tokens
        self.parser = ply.yacc.yacc(module=self, tabmodule=TABMODULE,
                                    write_tables=True, debug=False)
        self.g = None
        self.initial_nodes = None
        self.accepting_nodes = None
        self.symbols = None

    def parse(self, ltl2ba_output):
        """Return a Buchi automaton from parsing C{ltl2ba_output}.

        @return: Buchi automaton as a 3-`tuple` containing:
          - `dict` mapping symbols to types (all `"boolean"`)
          - `networkx.MultiDiGraph`, each edge labeled with a
            `"guard"` that is a Boolean formula as `str`
          - `set` of initial nodes
          - `set` of accepting nodes
        """
        self.g = nx.MultiDiGraph()
        # flat is better than nested
        self.initial_nodes = set()
        self.accepting_nodes = set()
        self.symbols = dict()
        self.parser.parse(ltl2ba_output, lexer=self.lexer.lexer)
        return (self.symbols, self.g,
                self.initial_nodes, self.accepting_nodes)

    def p_claim(self, p):
        """claim : NEVER LBRACE COMMENT clauses RBRACE"""
        p[0] = p[4]

    def p_clauses(self, p):
        """clauses : clauses clause"""
        p[0] = p[1] + [p[2]]

    def p_clauses_end(self, p):
        """clauses : clause"""
        p[0] = [p[1]]

    def p_clause(self, p):
        """clause : if_clause"""
        p[0] = p[1]

    def p_clause_labeled(self, p):
        """clause : state COLON if_clause"""
        u = p[1]
        if_clause = p[3]
        for guard, v in if_clause:
            self.g.add_edge(u, v, guard=guard)
        p[0] = (u, if_clause)

    def p_if_clause(self, p):
        """if_clause : IF cases FI SEMI"""
        p[0] = p[2]

    def p_cases(self, p):
        """cases : cases case"""
        p[0] = p[1] + [p[2]]

    def p_cases_end(self, p):
        """cases : case"""
        p[0] = [p[1]]

    def p_case(self, p):
        """case : COLON2 expr IMP goto"""
        p[0] = (p[2], p[4])

    def p_expr_paren(self, p):
        """expr : LPAREN expr RPAREN"""
        p[0] = '({expr})'.format(expr=p[2])

    def p_and(self, p):
        """expr : expr AND expr"""
        p[0] = '({x} and {y})'.format(x=p[1], y=p[3])

    def p_or(self, p):
        """expr : expr OR expr"""
        p[0] = '({x} or {y})'.format(x=p[1], y=p[3])

    def p_not(self, p):
        """expr : NOT expr"""
        p[0] = '(not {expr})'.format(expr=p[2])

    def p_number(self, p):
        """expr : NUMBER"""
        p[0] = int(p[1])

    def p_expr_name(self, p):
        """expr : NAME"""
        self.symbols[p[1]] = 'boolean'
        p[0] = p[1]

    def p_goto(self, p):
        """goto : GOTO state"""
        p[0] = p[2]

    def p_state(self, p):
        """state : NAME"""
        state = p[1]
        self.g.add_node(state)
        if 'init' in state:
            self.initial_nodes.add(state)
        elif 'accept' in state:
            self.accepting_nodes.add(state)
        p[0] = p[1]

    def p_empty(self, p):
        """empty :"""

    def p_error(self, p):
        logger.error('Syntax error at ' + p.value)


def call_ltl2ba(formula, prefix=''):
    """Load a Buchi Automaton from a Never Claim.

    TODO
    ====
    Make sure guard semantics properly accounted for:
    'prop' | '!prop' | '1' and skip

    Depends
    =======
    ltl2ba: http://www.lsv.ens-cachan.fr/~gastin/ltl2ba/

    @param formula: LTL formula for input to ltl2ba
    @type formula: str

    @return: Buchi Automaton
    @rtype: tulip.transys.BA
    """
    try:
        subprocess.call(['ltl2ba', '-h'], stdout=subprocess.PIPE)
    except OSError:
        raise Exception('cannot find ltl2ba on path')
    p = subprocess.Popen(
        [prefix + 'ltl2ba', '-f', '"{f}"'.format(f=formula)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    p.wait()
    ltl2ba_output = p.stdout.read()
    logger.info('ltl2ba output:\n\n{s}\n'.format(s=ltl2ba_output))
    if p.returncode != 0:
        raise Exception('Error when converting LTL to Buchi.')
    return ltl2ba_output


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger.setLevel(level=logging.DEBUG)
    parser = Parser()
    f = '[] !s && <> (a || c) && <> (b U c)'
    out = call_ltl2ba(f)
    symbols, g, initial, accepting = parser.parse(out)
    g.save('ba.pdf')
