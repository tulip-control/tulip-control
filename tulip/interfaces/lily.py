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
"""Interface to Lily, which solves LTL games.

Relevant links:
  - [Lily](http://www.ist.tugraz.at/staff/jobstmann/lily/)
"""
import collections as _cl
import itertools as _itr
import logging
import errno
import os
import re
import subprocess

from tulip.spec.form import GRSpec
from tulip.spec.form import LTL
from tulip.spec.parser import parse
from tulip.spec.translation import translate
from tulip.spec.translation import translate_ast
from tulip.transys import MooreMachine


logger = logging.getLogger(__name__)
LILY = 'lily.pl'
IO_PARTITION_FILE = 'io_partition.txt'
DOTFILE = 'ltl2vl-synthesis.dot'


def synthesize(
        formula,
        env_vars=None,
        sys_vars=None):
    """Return Moore transducer if `formula` is realizable.

    Variable `dict`s have variable names as keys and their type as
    value. The types should be 'boolean'. These parameters are only
    used if formula is of type `str`. Else, the variable dictionaries
    associated with the `LTL` or `GRSpec` object are used.

    @param formula:
        linear temporal logic formula
    @type formula:
        `str`,
        `LTL`, or
        `GRSpec`
    @param env_vars:
        uncontrolled variables (inputs); used only if
        `formula` is of type `str`
    @type env_vars:
        `dict` or
        None
    @param sys_vars:
        controlled variables (outputs); used only if
        `formula` is of type `str`
    @type sys_vars:
        `dict` or
        None
    @return:
        symbolic Moore transducer
        (guards are conjunctions, not sets)
    @rtype:
        `MooreMachine`
    """
    if isinstance(formula, GRSpec):
        env_vars = formula.env_vars
        sys_vars = formula.sys_vars
    elif isinstance(formula, LTL):
        env_vars = formula.input_variables
        sys_vars = formula.output_variables
    all_vars = dict(env_vars)
    all_vars.update(sys_vars)
    if not all(v == 'boolean' for v in all_vars.values()):
        raise TypeError(
            f'all variables should be Boolean:\n{all_vars}')
    if isinstance(formula, GRSpec):
        f = translate(formula, 'wring')
    else:
        T = parse(str(formula))
        f = translate_ast(T, 'wring').flatten(
            env_vars=env_vars,
            sys_vars=sys_vars)
    # dump partition file
    inp = ' '.join(env_vars)
    out = ' '.join(sys_vars)
    s = f'.inputs {inp}\n.outputs {out}'
    with open(IO_PARTITION_FILE, 'w') as fid:
        fid.write(s)
    # call lily
    try:
        p = subprocess.Popen(
            [LILY, '-f', f, '-syn', IO_PARTITION_FILE],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True)
        out = p.stdout.read()
    except OSError as e:
        if e.errno == errno.ENOENT:
            raise Exception(
                '`lily.pl` not found in path.\n'
                'See the Lily docs for '
                'setting `PERL5LIB` and `PATH`.')
        else:
            raise
    with open(DOTFILE, 'r') as dotf:
        text = dotf.read()
    fail_msg = 'Formula is not realizable'
    if text.startswith(fail_msg):
        return None
    moore = _lily_strategy2moore(
        text, env_vars, sys_vars)
    return moore


def _lily_strategy2moore(
        text,
        env_vars,
        sys_vars):
    """Return Moore transducer from Lily strategy.

    Caution
    =======

    The resulting transducer is symboic,
    in that the guards denote conjunctions,
    *not* subsets of ports.

    @param text:
        Moore strategy game graph,
        described in output from Lily
    @rtype:
        `MooreMachine`
    """
    lines = text.splitlines()
    def is_node_or_edge(line):
        return line.startswith('"')
    lines = filter(is_node_or_edge, lines)
    def is_title_line(line):
        return line.startswith('"title"')
    lines = list(filter(
        is_title_line, lines))
    def line_is_edge(line):
        return '->' in line
    edge_lines = set(filter(
        line_is_edge, lines))
    node_lines = set(_itr.filterfalse(
        line_is_edge, lines))
    # collect nodes and edges
    nodes = set()
    edges = list()
    for line in edge_lines:
        edge, annot = line.split('[label="')
        # edge
        endpoints = edge.split('->')
        start, end = list(map(
            str.strip, endpoints))
        edge = (start, end, annot)
        edges.append(edge)
        nodes.update((start, end))
    def is_init(node):
        return node.startswith('init-')
    phantom_init = set(filter(
        is_init, nodes))
    game_nodes = set(_itr.filterfalse(
        is_init, nodes))
    env_nodes = set()
    for line in node_lines:
        node, annot = line.split('[')
        node = node.strip()
        if 'shape=' in annot:
            continue
        env_nodes.add(node)
    def is_sys_node(node):
        return node not in env_nodes
    sys_nodes = set(filter(
        is_sys_node, game_nodes))
    # make machine
    mapping = {k: i for i, k in enumerate(sys_nodes)}
    successors = _cl.defaultdict(set)
    edge_labels = dict()
    for start, end, annot in edges:
        start = mapping.get(start, start)
        end = mapping.get(end, end)
        successors[start].add(end)
        edge_labels[start, end] = annot
    machine = MooreMachine()
    inports = {
        name: {False, True}
        for name in env_vars}
    outports = {
        name: {False, True}
        for name in sys_vars}
    machine.add_inputs(inports)
    machine.add_outputs(outports)
    machine.add_nodes_from(
        mapping[x]
        for x in sys_nodes)
    # add initial states
    for u in phantom_init:
        for v in successors[u]:
            machine.states.initial.add(v)
    # label vertices with output values
    for u in machine:
        succ = successors[u]
        if len(succ) != 1:
            raise AssertionError(succ)
        v, = succ
        attr = edge_labels[u, v]
        d = _parse_label(attr['label'])
        machine.add_node(u, **d)
        # input does not matter for this reaction ?
        if v in machine:
            machine.add_edge(u, v)
            continue
        # label edges with input values that matter
        for w in successors[v]:
            assert w in machine, w
            attr = edge_labels[v, w]
            d = _parse_label(attr['label'])
            machine.add_edge(u, w, **d)
    return machine


def _parse_label(s):
    """Return `dict` from variable conjunction.

    @type s:
        `str`
    @rtype:
        `dict`
    """
    l = re.findall(
        r' (\w+) = (0 | 1) ',
        s,
        flags=re.VERBOSE)
    return {k: bool(int(v)) for k, v in l}
