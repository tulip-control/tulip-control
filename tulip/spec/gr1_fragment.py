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
"""Test if given formula belongs to an LTL fragment that
is convertible to deterministic Buchi Automata
(readily expressible in GR(1) ).

reference
=========
1. Andreas Morgenstern and Klaus Schneider,
   A LTL Fragment for GR(1)-Synthesis,
   in Proceedings First International Workshop on
   Interactions, Games and Protocols (iWIGP),
   Electronic Proceedings in Theoretical Computer Science (EPTCS),
   50, pp. 33--45, 2011,
   http://doi.org/10.4204/EPTCS.50.3
"""
from __future__ import absolute_import

import logging
logger = logging.getLogger(__name__)

from tulip import transys as trs
from . import plyparser
from . import ast as sast

def check(formula):
    """Parse formula string and create abstract syntax tree (AST).
    """
    ast = plyparser.parse(formula)

    dfa = trs.automata.FiniteWordAutomaton(atomic_proposition_based=False,
                                           deterministic=True)

    dfa.alphabet |= {'!', 'W', 'U', 'G', 'F',
                     'U_left', 'U_right',
                     'W_left', 'W_right'}

    dfa.states.add_from({'gf', 'fg', 'g', 'f'})
    dfa.states.initial.add('gf')

    dfa.transitions.add('gf', 'fg', letter='!')
    dfa.transitions.add('fg', 'gf', letter='!')
    dfa.transitions.add('g', 'f', letter='!')
    dfa.transitions.add('f', 'g', letter='!')

    dfa.transitions.add('gf', 'gf', letter='W')
    dfa.transitions.add('gf', 'gf', letter='U_left')
    dfa.transitions.add('gf', 'gf', letter='G')

    dfa.transitions.add('fg', 'fg', letter='U')
    dfa.transitions.add('fg', 'fg', letter='F')
    dfa.transitions.add('fg', 'fg', letter='W_right')

    dfa.transitions.add('gf', 'f', letter='U_right')
    dfa.transitions.add('gf', 'f', letter='F')

    dfa.transitions.add('fg', 'g', letter='W_left')
    dfa.transitions.add('fg', 'g', letter='G')

    dfa.transitions.add('g', 'g', letter='W')
    dfa.transitions.add('g', 'g', letter='G')

    dfa.transitions.add('f', 'f', letter='U')
    dfa.transitions.add('f', 'f', letter='F')

    # plot tree automaton
    # dfa.save('dfa.pdf')

    # plot parse tree
    sast.dump_dot(ast, 'ast.dot')

    # sync product of AST with DFA,
    # to check acceptance
    Q = [(ast, 'gf')]
    while Q:
        s, q = Q.pop()
        logger.info('visiting: ' + str(s) + ', ' + str(q))

        if isinstance(s, sast.Unary):
            op = s.operator

            if op in {'!', 'G', 'F'}:
                t = dfa.transitions.find(q, letter=op)

                if not t:
                    raise Exception('not in fragment')

                qi, qj, w = t[0]

                Q.append((s.operand, qj))
            else:
                # ignore
                Q.append((s.operand, q))
        elif isinstance(s, sast.Binary):
            op = s.operator

            if op in {'W', 'U'}:
                t = dfa.transitions.find(q, letter=op)
                if t:
                    qi, qj, w = t[0]
                    Q.append((s.op_l, qj))
                    Q.append((s.op_r, qj))
                else:
                    t = dfa.transitions.find(q, letter=op + '_left')

                    if not t:
                        raise Exception('not in fragment')

                    qi, qj, w = t[0]

                    Q.append((s.op_l, qj))

                    t = dfa.transitions.find(q, letter=op + '_right')

                    if not t:
                        raise Exception('not in fragment')

                    qi, qj, w = t[0]

                    Q.append((s.op_r, qj))
            else:
                # ignore
                Q.append((s.op_l, q))
                Q.append((s.op_r, q))
        elif isinstance(s, sast.Var):
            print('reached var')

    return ast

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    s = '(a U b) && []a && <>a && <>a && []<>(<>z)'
    parsed_formula = check(s)

    print('Parsing result: ' + str(parsed_formula))
