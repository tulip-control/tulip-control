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
"""
Interface to Lily that solves LTL games.

Requires pydot for networkx to load the Moore strategy graph.

Relevant links:
  - U{Lily<http://www.ist.tugraz.at/staff/jobstmann/lily/>}
"""
import logging
import os
import re
import subprocess

import networkx as nx
import pydot

from tulip.spec.parser import parse
from tulip.spec.translation import translate
from tulip.spec.translation import translate_ast
from tulip.transys import MooreMachine
from tulip.spec.form import GRSpec
from tulip.spec.form import LTL


logger = logging.getLogger(__name__)
LILY = 'lily.pl'
IO_PARTITION_FILE = 'io_partition.txt'
DOTFILE = 'ltl2vl-synthesis.dot'


def synthesize(formula, env_vars=None, sys_vars=None):
    """Return Moore transducer if C{formula} is realizable.

    Variable C{dict}s have variable names as keys and their type as
    value. The types should be 'boolean'. These parameters are only
    used if formula is of type C{str}. Else, the variable dictionaries
    associated with the L{LTL} or L{GRSpec} object are used.

    @param formula: linear temporal logic formula
    @type formula: C{str}, L{LTL}, or L{GRSpec}

    @param env_vars: uncontrolled variables (inputs); used only if
        C{formula} is of type C{str}
    @type env_vars: C{dict} or None

    @param sys_vars: controlled variables (outputs); used only if
        C{formula} is of type C{str}
    @type sys_vars: C{dict} or None

    @return: symbolic Moore transducer
        (guards are conjunctions, not sets)
    @rtype: L{MooreMachine}
    """
    if isinstance(formula, GRSpec):
        env_vars = formula.env_vars
        sys_vars = formula.sys_vars
    elif isinstance(formula, LTL):
        env_vars = formula.input_variables
        sys_vars = formula.output_variables

    all_vars = dict(env_vars)
    all_vars.update(sys_vars)
    if not all(v == 'boolean' for v in all_vars.itervalues()):
        raise TypeError(
            'all variables should be Boolean:\n{v}'.format(v=all_vars))

    if isinstance(formula, GRSpec):
        f = translate(formula, 'wring')
    else:
        T = parse(str(formula))
        f = translate_ast(T, 'wring').flatten(env_vars=env_vars,
                                              sys_vars=sys_vars)

    # dump partition file
    s = '.inputs {inp}\n.outputs {out}'.format(
        inp=' '.join(env_vars),
        out=' '.join(sys_vars)
    )
    with open(IO_PARTITION_FILE, 'w') as fid:
        fid.write(s)

    # call lily
    try:
        p = subprocess.Popen([LILY, '-f', f, '-syn', IO_PARTITION_FILE],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = p.stdout.read()
    except OSError as e:
        if e.errno == os.errno.ENOENT:
            raise Exception(
                'lily.pl not found in path.\n'
                'See the Lily docs for setting PERL5LIB and PATH.')
        else:
            raise

    dotf = open(DOTFILE, 'r')
    fail_msg = 'Formula is not realizable'
    if dotf.read(len(fail_msg)) == fail_msg:
        return None
    dotf.seek(0)
    data = dotf.read()
    (pd,) = pydot.graph_from_dot_data(data)
    g = nx.drawing.nx_pydot.from_pydot(pd)
    dotf.close()
    moore = lily_strategy2moore(g, env_vars, sys_vars)
    return moore


def lily_strategy2moore(g, env_vars, sys_vars):
    """Return Moore transducer from Lily strategy graph C{g}.

    Caution
    =======
    The resulting transducer is symboic,
    in that the guards denote conjunctions,
    *not* subsets of ports.

    @param g: Moore strategy game graph as output by Lily
    @type g: C{networkx.MultiDiGraph}

    @rtype: L{MooreMachine}
    """
    g.remove_node('title')
    phantom_init = {x for x in g if x.startswith('init-')}
    game_nodes = {x for x in g if x not in phantom_init}
    sys_nodes = {x for x in game_nodes if g.node[x].get('shape') is None}
    mapping = {k: i for i, k in enumerate(sys_nodes)}
    # avoid mapping MooreMachine, because it raises errors
    h = nx.relabel_nodes(g, mapping)

    m = MooreMachine()
    inports = {k: {False, True} for k in env_vars}
    outports = {k: {False, True} for k in sys_vars}
    m.add_inputs(inports)
    m.add_outputs(outports)
    m.add_nodes_from(mapping[x] for x in sys_nodes)

    # add initial states
    for u in phantom_init:
        for v in g.successors_iter(u):
            m.states.initial.add(mapping[v])

    # label vertices with output values
    for u in m:
        oute = h.out_edges(u, data=True)
        assert(len(oute) == 1)
        u_, v, attr = oute[0]
        assert(u_ is u)
        d = _parse_label(attr['label'])
        m.add_node(u, **d)

        # input doesn't matter for this reaction ?
        if v in m:
            m.add_edge(u, v)
            continue

        # label edges with input values that matter
        for v_, w, attr in h.out_edges_iter(v, data=True):
            assert(v_ is v)
            assert(w in m)
            d = _parse_label(attr['label'])
            m.add_edge(u, w, **d)
    return m


def _parse_label(s):
    """Return C{dict} from variable conjunction.

    @type s: C{str}

    @rtype: C{dict}
    """
    l = re.findall('(\w+)=(0|1)', s)
    return {k: bool(int(v)) for k, v in l}
