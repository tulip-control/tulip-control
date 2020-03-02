# Copyright (c) 2013 by California Institute of Technology
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
TuLiP toolbox

The Temporal Logic Planning (TuLiP) Toolbox provides functions
for verifying and constructing control protocols.

Notes
=====
Citations are used throughout the documentation.  References
corresponding to these citations are defined in doc/bibliography.rst
of the TuLiP source distribution.  E.g., [BK08] used in various
docstrings is listed in doc/bibliography.rst as the book "Principles
of Model Checking" by Baier and Katoen (2008).
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = None

import tulip.abstract
from tulip.abstract import (
    # tulip.abstract.discretization
    AbstractSwitched, AbstractPwa,
    discretize, discretize_switched,
    multiproc_discretize, multiproc_discretize_switched,
    # tulip.abstract.feasible
    is_feasible, solve_feasible,
    # tulip.abstract.find_controller
    get_input, find_discrete_state,
    # tulip.abstract.plot
    plot_partition, plot_transition_arrow,
    plot_abstraction_scc, plot_ts_on_partition,
    project_strategy_on_partition, plot_strategy,
    plot_trajectory,
    # tulip.abstract.prop2partition
    prop2part, part2convex,
    pwa_partition, add_grid,
    PropPreservingPartition, PPP, ppp2ts)

import tulip.dumpsmach
from tulip.dumpsmach import write_python_case, python_case

import tulip.graphics
from tulip.graphics import dimension, newax, dom2vec, quiver

import tulip.gridworld
from tulip.gridworld import (
    GridWorld, random_world, narrow_passage, unoccupied, add_trolls,
    extract_coord, animate_paths)

import tulip.hybrid
from tulip.hybrid import LtiSysDyn, PwaSysDyn, SwitchedSysDyn

from tulip.interfaces import print_env

import tulip.spec
from tulip.spec import (
    # tulip.spec.form
    LTL, GRSpec, replace_dependent_vars,
    # tulip.spec.gr1_fragment
    check, str_to_grspec, split_gr1, has_operator, stability_to_gr1,
    response_to_gr1, eventually_to_gr1, until_to_gr1,
    # tulip.spec.parser
    parse,
    # tulip.spec.transformation
    Tree, ast_to_labeled_graph, check_for_undefined_identifiers,
    sub_values, sub_constants, sub_bool_with_subtree,
    pair_node_to_var, infer_constants, check_var_name_conflict,
    collect_primed_vars,
    # tulip.spec.translation
    translate)

import tulip.synth
from tulip.synth import (
    mutex, exactly_one, sys_to_spec, env_to_spec,
    build_dependent_var_table,
    synthesize_many, synthesize, is_realizable,
    strategy2mealy, mask_outputs, determinize_machine_init)

import tulip.transys
from tulip.transys import (
    # tulip.transys.algorithms
    ltl2ba,
    # tulip.transys.automata
    FiniteStateAutomaton, BuchiAutomaton, BA, tuple2ba,
    RabinAutomaton, DRA,
    ParityGame,
    # tulip.transys.labeled_graphs
    LabeledDiGraph, prepend_with,
    # tulip.transys.machines
    create_machine_ports, MooreMachine, MealyMachine,
    guided_run, random_run, interactive_run,
    moore2mealy, mealy2moore,
    # tulip.transys.mathset
    MathSet, SubSet, PowerSet, TypedDict,
    # tulip.transys.transys
    KripkeStructure, FiniteTransitionSystem, FTS,
    LabeledGameGraph,
    tuple2fts, line_labeled_with, cycle_labeled_with,
    simu_abstract)
