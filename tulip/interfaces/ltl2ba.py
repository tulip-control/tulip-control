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
"""Interface to `ltl2ba`."""
import logging
import subprocess

import networkx as nx
import ply.lex
import ply.yacc

import tulip.transys as _trs


TABMODULE = 'tulip.interfaces.ltl2ba_parsetab'
_logger = logging.getLogger(__name__)


class Lexer:
    """Token rules to build lexer for `ltl2ba` output."""

    # Tokens
    t_TRUE = '''
          TRUE
        | True
        | true
        '''
    t_FALSE = '''
          FALSE
        | False
        | false
        '''

    t_COMMENT = r'''
        / \*
        .*
        \* /
        '''
    t_NOT = r'!'
    t_AND = r'''
          \& \&
        | \&
        '''
    t_OR = r'''
          \| \|
        | \|
        '''
    t_XOR = r' \^ '

    t_EQUALS = r'''
          =
        | ==
        '''
    t_NEQUALS = r' != '
    t_LT = r' < '
    t_LE = r' <= '
    t_GT = r' >= '
    t_GE = r' > '

    t_LPAREN = r' \( '
    t_RPAREN = r' \) '
    t_LBRACE = r' \{ '
    t_RBRACE = r' \} '
    t_SEMI = r' ; '
    t_COLON2 = r' :: '
    t_COLON = r' : '

    t_NUMBER = r' \d+ '

    t_IMP = r' \- > '

    # Ignored characters
    t_ignore = ''.join(['\x20', '\t'])

    def __init__(self):
        self.reserved = {
            'goto':
                'GOTO',
            'if':
                'IF',
            'fi':
                'FI',
            'never':
                'NEVER',
            'skip':
                'SKIP'}
        self.tokens = (
            'TRUE', 'FALSE',
            'NUMBER',
            'NOT', 'AND', 'OR', 'XOR', 'IMP', 'BIMP',
            'EQUALS', 'NEQUALS', 'LT', 'LE', 'GT', 'GE',
            'PLUS', 'MINUS', 'TIMES', 'DIV',
            'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE',
            'COLON', 'COLON2', 'SEMI',
            'COMMENT', 'NAME') + tuple(
                self.reserved.values())
        self.lexer = ply.lex.lex(
            module=self,
            debug=False)

    def t_newline(self, t):
        r' \n+ '
        t.lexer.lineno += t.value.count('\n')

    def t_error(self, t):
        _logger.warning(
            f'Illegal character `{t.value[0]}`')
        t.lexer.skip(1)

    def t_name(self, t):
        r"""
        [A-Za-z_]
        [a-zA-Z0-9_]*
        """
        t.type = self.reserved.get(t.value, 'NAME')
        return t


class Parser:
    """Production rules to build parser for `ltl2ba` output."""

    def __init__(self):
        """Build lexer and parser."""
        self.precedence = (
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
            ('nonassoc', 'EQUALS', 'NEQUALS',
             'LT', 'LE', 'GT', 'GE'),
            ('nonassoc', 'TIMES', 'DIV'),
            ('nonassoc', 'PLUS', 'MINUS'),
            ('nonassoc', 'TRUE', 'FALSE'))
        self.lexer = Lexer()
        self.start = 'claim'
        self.tokens = self.lexer.tokens
        self.tabmodule = TABMODULE
        self.build(
            write_tables=True,
            debug=False)
        self.g = None
        self.initial_nodes = None
        self.accepting_nodes = None
        self.symbols = None

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
        """Cache LALR(1) state machine.

        Using `ply.yacc`.

        The default Python module where the
        state-machine transitions are stored is
        as defined in the attribute
        `self.tabmodule`.

        This module's logger is used as default.
        """
        if tabmodule is None:
            tabmodule = self.tabmodule
        if debug and debuglog is None:
            debuglog = _logger
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
            ltl2ba_output:
                str
            ) -> tuple[
                dict[str, str],
                nx.DiGraph,
                set[str],
                set[str]]:
        """Return a Buchi automaton from parsing `ltl2ba_output`.

        @return:
            Buchi automaton as a 4-`tuple` containing:

            ```python
            (mapping_symbols_to_types,
             graph,
             initial_nodes,
             accepting_nodes)
            ```

            where:
            - all values in the mapping are `"boolean"`
            - each graph edge is labeled with
              the key `"guard"`, with a Boolean
              formula as value (the formula as `str`)
        """
        self.g = nx.MultiDiGraph()
        # flat is better than nested
        self.initial_nodes = set()
        self.accepting_nodes = set()
        self.symbols = dict()
        self.parser.parse(ltl2ba_output, lexer=self.lexer.lexer)
        return (
            self.symbols,
            self.g,
            self.initial_nodes,
            self.accepting_nodes)

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
        p[0] = f'({p[2]})'

    def p_and(self, p):
        """expr : expr AND expr"""
        p[0] = f'({p[1]} and {p[3]})'

    def p_or(self, p):
        """expr : expr OR expr"""
        p[0] = f'({p[1]} or {p[3]})'

    def p_not(self, p):
        """expr : NOT expr"""
        p[0] = f'(not {p[2]})'

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
        _logger.error(
            f'Syntax error at {p.value}')


def ltl2ba(
        formula:
            str
        ) -> _trs.BuchiAutomaton:
    """Convert LTL formula to Buchi Automaton using `ltl2ba`.

    @param formula:
        `str(formula)` must be admissible `ltl2ba` input
    @return:
        Buchi automaton whose edges are annotated
        with Boolean formulas as `str`
    """
    ltl2ba_out = call_ltl2ba(str(formula))
    parser = ltl2baint.Parser()
    symbols, g, initial, accepting = parser.parse(ltl2ba_out)
    ba = _trs.BuchiAutomaton()
    ba.atomic_propositions.update(symbols)
    ba.add_nodes_from(g)
    ba.add_edges_from(g.edges(data=True))
    ba.initial_nodes = initial
    ba.accepting_sets = accepting
    _logger.info(f'Resulting automaton:\n\n{ba}\n')
    return ba


def call_ltl2ba(
        formula:
            str,
        prefix:
            str=''
        ) -> str:
    """Load a Buchi Automaton from a Never Claim.

    TODO
    ====
    Make sure guard semantics properly accounted for:
    'prop' | '!prop' | '1' and skip

    Depends
    =======
    `ltl2ba`: <http://www.lsv.ens-cachan.fr/~gastin/ltl2ba/>

    @param formula:
        LTL formula for input to `ltl2ba`
    @return:
        Buchi Automaton description
    """
    try:
        subprocess.call(
            ['ltl2ba', '-h'],
            stdout=subprocess.PIPE)
    except OSError:
        raise Exception('cannot find ltl2ba on path')
    p = subprocess.Popen(
        [f'{prefix}ltl2ba',
         '-f',
         f'"{formula}"'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True)
    p.wait()
    ltl2ba_output = p.stdout.read()
    _logger.info(
        f'ltl2ba output:\n\n{ltl2ba_output}\n')
    if p.returncode != 0:
        raise Exception(
            'Error when converting LTL to Buchi.')
    return ltl2ba_output


def _main():
    logging.basicConfig(level=logging.DEBUG)
    _logger.setLevel(level=logging.DEBUG)
    parser = Parser()
    f = '[] !s && <> (a || c) && <> (b U c)'
    out = call_ltl2ba(f)
    symbols, g, initial, accepting = parser.parse(out)
    g.save('ba.pdf')


if __name__ == '__main__':
    _main()
