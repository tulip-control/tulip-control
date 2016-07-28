#!/usr/bin/env python
#
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
"""
Load output of LTL2DSTAR into a NetworkX DiGraph

The official website of LTL2DSTAR is http://ltl2dstar.de/ , where a
definition of its output format can be found.  Use "-" in place of
FILE to read from stdin.

* Parsing is done by hand; consider changing to use pyparsing.  N.B.,
  the current implementation is lax about the input file, i.e., it
  accepts valid LTL2DSTAR output along with variants.

* The Automaton class is a very light derivative of networkx.DiGraph;
  in other words, if you prefer to only work with a DiGraph, it is
  easy to modify the existing code to do so.  Expect the name to
  change later, especially if (when) it becomes integrated into TuLiP
  and nTLP.

* readdstar.py is a commandline utility.  It expects to be given the
  name of file from which to read (previously recorded) output of
  LTL2DSTAR.  Alternatively, use "-" to read from stdin.  E.g., if
  ltl2dstar and ltl2ba are in the current directory, try

    $ echo 'U a b' | ltl2dstar --ltl2nba=spin:ltl2ba - -|./readdstar.py -

* Each edge (transition) is labeled with two things:
  - "formula" : a disjunctive normal form expression for when the edge
                should be taken; and
  - "subsets of AP": a list of sets of atomic propositions.

  E.g., if AP = {p, q}, and (1,3) is an edge in the Automaton object A,
  then

    A.edge[1][3]["subsets of AP"] = [set([]), set(['p'])]

  means that the transition (1,3) should be taken (assuming current
  execution has led to state 1) if the none of the atomic propositions
  are true (i.e., !p & !q holds), or precisely "p" is true (i.e., p &
  !q holds).


SCL; 2013, 2015.
"""

import sys
import networkx as nx
import matplotlib.pyplot as plt


class AcceptancePair(object):
    def __init__(self, L=None, U=None):
        if L is None:
            self.L = set()
        else:
            self.L = set(L)
        if U is None:
            self.U = set()
        else:
            self.U = set(U)


class Automaton(nx.DiGraph):
    def __init__(self, aut_type=None):
        nx.DiGraph.__init__(self)
        self.aut_type = aut_type

    def __str__(self):
        output = "Type: "
        if self.aut_type == "DRA":
            output += "deterministic Rabin\n"
        elif self.aut_type == "DSA":
            output += "deterministic Streett\n"
        output += "AP = "+str(self.ap)+"\n"
        output += "Transitions:"+"\n"
        output += "\n".join(["\t("+str(u)+", "+str(v)+") :\n\t\tformula: "+str(d["formula"])+"\n\t\tsubsets of AP: "+str(d["subsets of AP"]) for (u,v,d) in self.edges_iter(data=True)])+"\n"
        output += "Acceptance Pairs (each line is of the form (L, U)):"+"\n"
        output += "\n".join(["\t("+str(Fi.L)+", "+str(Fi.U)+")" for Fi in self.F])
        return output


def gen_apformula(AP, intrep):
    """Generate conjunction formula

    >>> gen_apformula(AP=("p", "q"), intrep=2)
    '!p & q'
    """
    return " & ".join([AP[i] if ((intrep >> i) & 1) != 0 else "!"+AP[i] for i in range(len(AP))])

def gen_apsubset(AP, intrep):
    """Generate set of atomic propositions corresponding to integer

    >>> gen_apsubset(AP=("p", "q"), intrep=2)
    set(['q'])
    """
    return set([AP[i] for i in range(len(AP)) if ((intrep >> i) & 1) != 0])


def readdstar(getline):
    """Construct automaton from LTL2DSTAR output.

    getline is any method that can yield successive lines of output
    from LTL2DSTAR.  E.g., a file could be opened and then its
    readline() method passed to readdstar.
    """
    A = None
    aut_type = None
    comments = []

    last_state = -1  # -1 indicates unset
    try:
        while True:
            line = getline()
            if len(line) == 0:
                break  # file.readline() returns empty string at EOF
            parts = line.split()
            if len(parts) == 0:
                continue  # Ignore blank lines

            if not parts[0].endswith(":") and len(parts) == 3:  # id line
                aut_type = parts[0]
                version = parts[1]
                edge_type = parts[2]

                A = Automaton(aut_type=aut_type)
            elif parts[0] == "Comment:":
                comments.append(" ".join(parts[1:]))
            elif parts[0] == "States:":
                num_states = int(parts[1])
            elif parts[0] == "Acceptance-Pairs:":
                num_pairs = int(parts[1])
                A.F = [None for i in range(num_pairs)]
            elif parts[0] == "Start:":
                A.start_state = int(parts[1])
            elif parts[0] == "AP:":
                ap_len = int(parts[1])
                A.ap = tuple([prop.strip("\"").rstrip("\"") for prop in  parts[2:]])
                assert ap_len == len(A.ap)
            elif parts[0] == "State:":
                last_state = int(parts[1])
                apsubset_counter = 0
                A.add_node(last_state)
            elif parts[0] == "Acc-Sig:":
                for accsig in parts[1:]:
                    accsig_index = int(accsig[1:])
                    if A.F[accsig_index] is None:
                        A.F[accsig_index] = AcceptancePair()
                    if accsig[0] == "+":
                        A.F[accsig_index].L.add(last_state)
                    elif accsig[0] == "-":
                        A.F[accsig_index].U.add(last_state)
            elif last_state >= 0 and parts[0] != "---":
                to_state = int(parts[0])
                if not A.has_edge(last_state, to_state):
                    A.add_edge(last_state, to_state)
                    A.edge[last_state][to_state]["formula"] = "("+gen_apformula(A.ap, apsubset_counter)+")"
                    A.edge[last_state][to_state]["subsets of AP"] = [gen_apsubset(A.ap, apsubset_counter)]
                else:
                    A.edge[last_state][to_state]["formula"] += " | ("+gen_apformula(A.ap, apsubset_counter)+")"
                    A.edge[last_state][to_state]["subsets of AP"].append(gen_apsubset(A.ap, apsubset_counter))
                apsubset_counter += 1

    except EOFError:
        pass  # raw_input() throws this at end-of-file

    return A


if __name__ == "__main__":
    if len(sys.argv) < 2 or "-h" in sys.argv:
        print("Usage: "+sys.argv[0]+" FILE")
        exit(1)

    if sys.argv[1] == "-":  # Read from stdin
        getline = raw_input
    else:
        f = open(sys.argv[1], "r")
        getline = f.readline

    print(readdstar(getline) )
