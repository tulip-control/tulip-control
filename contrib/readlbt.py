#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 by Scott C. Livingston
# Copyright (c) 2013, 2015 by California Institute of Technology
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
"""Load output of scheck or lbt into a NetworkX DiGraph

The official website of scheck is
http://tcs.legacy.ics.tkk.fi/users/tlatvala/scheck/
It uses the output format of LBT, which is defined at
http://www.tcs.hut.fi/Software/maria/tools/lbt/

scheck is also available at https://github.com/slivingston/scheck
where a fork of it is maintained.

* Parsing is done by hand; consider changing to use pyparsing.
  N.B., the current implementation is lax about the input file, i.e.,
  it accepts valid output of `scheck` along with variants.

* The GBAutomaton class (GB abbreviates generalized Büchi) is a very
  light derivative of networkx.DiGraph; in other words, if you prefer
  to only work with a DiGraph, it is easy to modify the existing code
  to do so.

* readlbt.py is a commandline utility.  It expects to be given the
  name of file from which to read (previously recorded) output of
  scheck.  Alternatively, use "-" to read from stdin.  E.g.,

      echo '| F p1 F p2' | scheck2 -s -d -- | ./readlbt.py -


SCL; 2017.
"""
from __future__ import print_function

import sys
import networkx as nx


class GBAutomaton(nx.DiGraph):
    def __init__(self, number_of_acceptance_sets):
        nx.DiGraph.__init__(self, number_of_acceptance_sets=0)
        self.number_of_acceptance_sets = number_of_acceptance_sets

    def dumpdot(self):
        output = 'digraph A {\n'
        for node, ndata in self.nodes(data=True):
            output += (
                f'{node} [label="{node}\\n'
                'acceptance sets: {' +
                ','.join([str(ac) for ac in ndata['acceptance_sets']]) +
                '}"]\n')
        for u, v, edata in self.edges(data=True):
            label = edata['gate']
            output += (
                f'{u} -> {v}'
                f' [label="{label}"]\n')
        output += '}\n'
        return output


def _split2(x):
    x = x.strip()
    space = x.find(' ')
    newline = x.find('\n')
    if newline == -1:
        nexts = space
    elif space == -1:
        nexts = newline
    else:
        nexts = min(space, newline)
    if nexts == -1:
        return [x]
    first = x[:nexts]
    x = x[nexts:].strip()
    space = x.find(' ')
    newline = x.find('\n')
    if newline == -1:
        nexts = space
    elif space == -1:
        nexts = newline
    else:
        nexts = min(space, newline)
    if nexts == -1:
        return [first, x]
    second = x[:nexts]
    return [first, second, x[nexts:]]


def readlbt(gbastr):
    """Construct automaton from scheck output.

    `gbastr` is a string that defines the generalized Büchi automaton.
    """
    parts = _split2(gbastr)
    state_count = int(parts[0])

    A = GBAutomaton(int(parts[1]))

    state_parts = [part for part in parts[2].split('-1')
                   if len(part.strip()) > 0]

    for (ii, state_part) in enumerate(state_parts):
        if ii % 2 == 0:
            x = _split2(state_part)
            state_name = int(x[0])
            initial = True if x[1] == '1' else False
            if len(x) > 2:
                acceptance_sets=[int(ac) for ac in x[2].split()]
            else:
                acceptance_sets = []
            A.add_node(state_name,
                       initial=initial,
                       acceptance_sets=acceptance_sets)

        else:  # Transitions
            succ = None
            next_succ = None
            gate = ''
            for transitions_part in state_part.split():
                try:
                    next_succ = int(transitions_part)
                except ValueError:
                    next_succ = None
                if succ is None and next_succ is not None:
                    succ = next_succ
                    next_succ = None
                    continue
                if next_succ is not None:
                    A.add_edge(state_name, succ, gate=gate)
                    succ = next_succ
                    next_succ = None
                    gate = ''
                else:
                    if len(gate) > 0:
                        gate += ' '
                    gate += transitions_part
            if succ is not None:
                A.add_edge(state_name, succ, gate=gate)

    return A


if __name__ == "__main__":
    if len(sys.argv) < 2 or "-h" in sys.argv:
        print(f'Usage: {sys.argv[0]} FILE')
        exit(1)

    if sys.argv[1] == "-":  # Read from stdin
        gbastr = sys.stdin.read()
    else:
        f = open(sys.argv[1], "r")
        gbastr = f.read()

    gba = readlbt(gbastr)
    print(gba.dumpdot())
