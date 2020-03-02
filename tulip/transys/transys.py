# Copyright (c) 2013-2015 by California Institute of Technology
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
"""Transition System Module"""
from __future__ import absolute_import
import logging
from collections import Iterable
from pprint import pformat
from tulip.transys.labeled_graphs import (
    LabeledDiGraph, str2singleton, prepend_with)
from tulip.transys.mathset import PowerSet, MathSet
from tulip.transys.cost import ValidTransitionCost
from networkx import MultiDiGraph
import numpy as np
# inline imports
#
# from tulip.transys.export import graph2promela


logger = logging.getLogger(__name__)
_hl = 40 * '-'


class KripkeStructure(LabeledDiGraph):
    """Directed graph with labeled vertices and initial vertices.

    References
    ==========
    1. Kripke S.
      Semantical Considerations on Modal Logic
      Acta Philosophica Fennica, 16, pp. 83-94, 1963

    2. Clarke E.M.; Grumberg O.; Peled D.A.
      Model Checking, MIT Press, 1999, p.14

    3. Schneider K.
      Verification of Reactive Systems
      Springer, 2004, Def. 2.1, p.45
    """

    def __init__(self):
        ap_labels = PowerSet()
        node_label_types = [
            {'name': 'ap',
             'values': ap_labels,
             'setter': ap_labels.math_set,
             'default': set()}]
        super(KripkeStructure, self).__init__(node_label_types)
        self.atomic_propositions = self.ap
        # dot formatting
        self._state_dot_label_format = {
            'ap': '',
            'type?label': '',
            'separator': r'\\n'}
        self.dot_node_shape = {'normal': 'rectangle'}
        self._state_dot_label_format = {
            'ap': '',
            'type?label': '',
            'separator': r'\\n'}
        self._transition_dot_label_format = {
            'type?label': ':',
            'separator': r'\\n'}
        self._transition_dot_mask = dict()

    def __str__(self):
        s = (
            'Kripke Structure: ' + self.name + '\n' +
            _hl + '\n' +
            'Atomic Propositions (APs):\n\t' +
            pformat(self.atomic_propositions, indent=3) + 2 * '\n' +
            'States labeled with sets of APs:\n' +
            _dumps_states(self) + 2 * '\n' +
            'Initial States:\n' +
            pformat(self.states.initial, indent=3) + 2 * '\n' +
            'Transitions:\n' +
            pformat(self.transitions(), indent=3) +
            '\n' + _hl + '\n')
        return s


class WeightedKripkeStructure(KripkeStructure):
    """KripkeStructure with weight/cost on the transitions
    """

    def __init__(self):
        edge_label_types = [
            {'name': 'cost',
             'values': ValidTransitionCost(),
             'setter': True}
        ]
        super(WeightedKripkeStructure, self).__init__()
        super(WeightedKripkeStructure, self).add_label_types(edge_label_types, True)


class FiniteTransitionSystem(LabeledDiGraph):
    """Kripke structure with labeled states and edges.

    Who controls the state
    ======================
    To define who "moves the token" between vertices in
    the graph, set the attribute:

    >>> g = FiniteTransitionSystem()
    >>> g.owner = 'sys'

    This means that when there are more than one transition
    enabled, then the system picks the next state.

    The other option is:

    >>> g.owner = 'env'

    so the environment picks the next state.

    State labeling
    ==============
    The state labels are sets of atomic propositions,
    similar to a L{KripkeStructure}.

    In principle some of the propositions that label states
    could be controlled by either of the players,
    but this would lead to less straightforward semantics.

    You can achieve the same effect by using actions of
    the opponent.

    It is a matter of future experimentation whether
    this capability will be introduced, by partitioning
    the props into C{env_props} and C{sys_props}
    (similar to C{env_vars}, C{sys_vars} in L{GRSpec}).

    Edge labeling
    =============
    Edge labels are called "actions".

    The edge labeling is syntactic sugar for
    labels that are shifted to the target states of
    those edges. So edge labeling is not an essential
    difference from Kripke structures.

    Not to be confused with the term:
    "Labeled Transition System"
    found in the literature.

    Also, it differs from the definition in Baier-Katoen
    in that actions are not mere reading aid,
    but are interpreted as propositions as explained above.

    Besides, edge labeling usually allows for
    graphs with fewer vertices than the corresponding
    Kripke structure.

    Open vs Closed
    ==============
    The essential difference from Kripke structures
    is the partition of atomic propositions into
    input/output sets.

    If the set of inputs is empty, then the system is closed.
    Otherwise it is an open system.
    Open systems have an environment, closed don't.

    Alternatively, FTS can be thought of as a shorthand
    for defining a vertex-labeled game graph,
    or equivalently a game structure.

    System and environment actions
    ==============================
    The only significant difference is in transition labeling.
    For closed systems, each transition is labeled with a system action.
    So each transition label comprises of a single sublabel,
    the system action.

    For open systems, each transition is labeled with 2 sublabels:
        - The first sublabel is a system action,
        - the second an environment action.

    Mutual exclusion of actions
    ===========================
    Constraints on actions can be defined
    similarly to L{FTS} actions by setting the fields:

        - C{ofts.env_actions_must}
        - C{ofts.sys_actions_must}

    The default constraint is 'xor'.

    sys.sys_actions_must: select constraint on actions. Options:

        - C{'mutex'}: at most 1 action True each time
        - C{'xor'}: exactly 1 action True each time
        - C{'none'}: no constraint on action values

    The xor constraint can prevent the environment from
    blocking the system by setting all its actions to False.

    The action are taken when traversing an edge.
    Each edge is annotated by a single action.
    If an edge (s1, s2) can be taken on two transitions,
    then 2 copies of that same edge are stored.
    Each copy is annotated using a different action,
    the actions must belong to the same action set.
    That action set is defined as a set instance.
    This description is a (closed) L{FTS}.

    The system and environment actions are associated with an edge
    of a reactive system. To store these, mutliple labels are used
    and their sets are encapsulated within the same C{FTS}.

    Example
    =======
    In the following C{None} represents the empty set, subset of AP.
    First create an empty transition system and add some states to it:

    >>> from tulip import transys as trs
    >>> ts = trs.FiniteTransitionSystem()
    >>> ts.states.add('s0')
    >>> ts.states.add_from(['s1', 's3', 'end', 5] )

    Set an initial state, which must already be in states:

    >>> ts.states.initial.add('s0')

    There can be more than one possible initial states:

    >>> ts.states.initial.add_from(['s0', 's3'] )

    To label the states, we need at least one atomic proposition,
    here C{'p'}:

    >>> ts.atomic_propositions |= ['p', None]
    >>> ts.states.add('s0', ap={'p'})
    >>> ts.states.add_from([('s1', {'ap':{'p'} }),
                            ('s3', {'ap':{} } )])

    If a state has already been added, its label of atomic
    propositions can be defined directly:

    >>> ts.states['s0']['ap'] = {'p'}

    Having added states, we can also add some labeled transitions:

    >>> ts.actions |= ['think', 'write']
    >>> ts.transitions.add('s0', 's1', actions='think')
    >>> ts.transitions.add('s1', 5, actions='write')

    Note that an unlabeled transition:

    >>> ts.transitions.add('s0', 's3')

    is considered as different from a labeled one and to avoid
    unintended duplication, after adding an unlabeled transition,
    any attempt to add a labeled transition between the same states
    will raise an exception, unless the unlabeled transition is
    removed before adding the labeled transition.

    The user can still invoke NetworkX functions to set custom node
    and edge labels, in addition to the above ones.
    For example:

    >>> ts.states.add('s0')
    >>> ts.nodes['s0']['my_cost'] = 5

    The difference is that atomic proposition and action labels
    are checked to make sure they are elements of the system's
    AP and Action sets.

    It is not advisable to use C{MultiDiGraph.add_node} and
    C{MultiDiGraph.add_edge} directly,
    because that can result in an inconsistent system,
    since it skips all checks performed by L{transys}.

    Note
    ====
    The attributes atomic_propositions and aps are equal.
    When you want to produce readable code, use atomic_propositions.
    Otherwise, aps offers shorthand access to the APs.

    Reference
    =========
    For closed systems this corresponds to Def. 2.1, p.20 U{[BK08]
    <https://tulip-control.sourceforge.io/doc/bibliography.html#bk08>}:
        - states (instance of L{States}) = S
        - states.initial = S_0 \subseteq S
        - atomic_propositions = AP
        - actions = Act
        - transitions (instance of L{Transitions})::
              the transition relation ->
                = edge set + edge labeling function
                (labels \in actions)
        Unlabeled edges are defined using:
            - sys.transitions.add
            - sys.transitions.add_from
            - sys.transitions.add_adj
        and accessed using:
            - sys.transitions.find
        - the state labeling function::
                L: S-> 2^AP
        can be defined using:
            - sys.states.add
            - sys.states.add_from
        and accessed using methods:
            - sys.states(data=True)
            - sys.states.find

    See Also
    ========
    L{KripkeStructure}, L{tuple2fts},
    L{line_labeled_with}, L{cycle_labeled_with}
    """

    def __init__(self, env_actions=None, sys_actions=None):
        """Instantiate finite transition system.

        @param env_actions: environment (uncontrolled) actions,
            defined as C{edge_label_types} in L{LabeledDiGraph.__init__}

        @param sys_actions: system (controlled) actions, defined as
            C{edge_label_types} in L{LabeledDiGraph.__init__}
        """
        self._owner = 'sys'

        if env_actions is None:
            env_actions = [
                {'name': 'env_actions',
                 'values': MathSet(),
                 'setter': True}]
        if sys_actions is None:
            sys_actions = [
                {'name': 'sys_actions',
                 'values': MathSet(),
                 'setter': True}]
        # note: "sys_actions" used to be "actions"
        # in closed systems (old FTS)
        action_types = env_actions + sys_actions
        edge_label_types = action_types
        ap_labels = PowerSet()
        node_label_types = [
            {'name': 'ap',
             'values': ap_labels,
             'setter': ap_labels.math_set,
             'default': set()}]
        super(FiniteTransitionSystem, self).__init__(
            node_label_types, edge_label_types)
        # make them available also via an "actions" dicts
        # name, codomain, *rest = x
        actions = {x['name']: x['values'] for x in edge_label_types}
        if 'actions' in actions:
            msg = '"actions" cannot be used as an action type name,\n'
            msg += 'because if an attribute for this action type'
            msg += 'is requested,\n then it will conflict with '
            msg += 'the dict storing all action types.'
            raise ValueError(msg)
        self.actions = actions
        self.atomic_propositions = self.ap
        self.aps = self.atomic_propositions  # shortcut
        # action constraint used in synth.synthesize
        self.env_actions_must = 'xor'
        self.sys_actions_must = 'xor'
        # dot formatting
        self._state_dot_label_format = {
            'ap': '',
            'type?label': '',
            'separator': r'\\n'}
        self._transition_dot_label_format = {
            'sys_actions': 'sys',  # todo: '' if no env
            'env_actions': 'env',
            'type?label': ':',  # todo: '' if no env
            'separator': r'\\n'}
        self._transition_dot_mask = dict()
        self.dot_node_shape = {'normal': 'box'}  # todo: rectangle if no env
        self.default_export_fname = 'fts'

    def __str__(self):
        isopen = (
            ('sys' and any({'env' in x for x in self.actions})) or
            ('env' and any({'sys' in x for x in self.actions})))
        if isopen:
            t = 'open'
        else:
            t = 'closed'
        s = (
            _hl + '\nFinite Transition System (' + t + '): ' +
            self.name + '\n' + _hl + '\n' +
            'Atomic Propositions (APs):\n' +
            pformat(self.atomic_propositions, indent=3) + 2 * '\n' +
            'States labeled with sets of APs:\n' +
            _dumps_states(self) + 2 * '\n' +
            'Initial States:\n' +
            pformat(self.states.initial, indent=3) + 2 * '\n')

        for action_type, codomain in self.actions.items():
            if 'sys' in action_type:
                s += (
                    'System Action Type: ' + str(action_type) +
                    ', with possible values: ' + str(codomain) + '\n' +
                    pformat(codomain, indent=3) + 2 * '\n')
            elif 'env' in action_type:
                s += (
                    'Environment Action Type: ' + str(action_type) +
                    ', with possible values:\n\t' + str(codomain) + '\n' +
                    pformat(codomain, indent=3) + 2 * '\n')
            else:
                s += (
                    'Action type controlled by neither env nor sys\n'
                    ' (will cause you errors later)'
                    ', with possible values:\n\t' +
                    pformat(codomain, indent=3) + 2 * '\n')
        s += (
            'Transitions labeled with sys and env actions:\n' +
            pformat(self.transitions(data=True), indent=3) +
            '\n' + _hl + '\n')
        return s

    @property
    def owner(self):
        return self._owner

    @owner.setter
    def owner(self, x):
        if x not in {'env', 'sys'}:
            raise ValueError("The owner can be either 'sys' or 'env'.")
        self._owner = x

    def _save(self, path, fileformat):
        """Export options available only for closed systems.

        Provides: pml (Promela)

        See Also
        ========
        L{save}, L{plot}
        """
        if fileformat not in {'promela', 'Promela', 'pml'}:
            return False
        # closed ?
        if self.env_vars:
            return False
        from tulip.transys.export import graph2promela
        s = graph2promela.fts2promela(self, self.name)
        # dump to file
        f = open(path, 'w')
        f.write(s)
        f.close()
        return True


class FTS(FiniteTransitionSystem):
    """Alias to L{FiniteTransitionSystem}."""


def tuple2fts(S, S0, AP, L, Act, trans, name='fts',
              prepend_str=None):
    """Create a Finite Transition System from a tuple of fields.

    Hint
    ====
    To remember the arg order:

    1) it starts with states (S0 requires S before it is defined)

    2) continues with the pair (AP, L), because states are more
    fundamental than transitions
    (transitions require states to be defined)
    and because the state labeling L requires AP to be defined.

    3) ends with the pair (Act, trans), because transitions in trans
    require actions in Act to be defined.

    See Also
    ========
    L{tuple2ba}

    @param S: set of states
    @type S: iterable of hashables

    @param S0: set of initial states, must be \\subset S
    @type S0: iterable of elements from S

    @param AP: set of Atomic Propositions for state labeling:
            L: S-> 2^AP
    @type AP: iterable of hashables

    @param L: state labeling definition
    @type L: iterable of (state, AP_label) pairs:
        [(state0, {'p'} ), ...]
        | None, to skip state labeling.

    @param Act: set of Actions for edge labeling:
            R: E-> Act
    @type Act: iterable of hashables

    @param trans: transition relation
    @type trans: list of triples: [(from_state, to_state, act), ...]
        where act \\in Act

    @param name: used for file export
    @type name: str
    """
    def pair_labels_with_states(states, state_labeling):
        if state_labeling is None:
            return
        if not isinstance(state_labeling, Iterable):
            raise TypeError('State labeling function: L->2^AP must be '
                            'defined using an Iterable.')
        state_label_pairs = True
        # cannot be caught by try below
        if isinstance(state_labeling[0], str):
            state_label_pairs = False
        if state_labeling[0] is None:
            state_label_pairs = False
        try:
            (state, ap_label) = state_labeling[0]
        except:
            state_label_pairs = False
        if state_label_pairs:
            return state_labeling
        logger.debug('State labeling L not tuples (state, ap_label),\n'
                     'zipping with states S...\n')
        state_labeling = zip(states, state_labeling)
        return state_labeling
    # args
    if not isinstance(S, Iterable):
        raise TypeError('States S must be iterable, even for single state.')
    # convention
    if not isinstance(S0, Iterable) or isinstance(S0, str):
        S0 = [S0]
    # comprehensive names
    states = S
    initial_states = S0
    ap = AP
    state_labeling = pair_labels_with_states(states, L)
    actions = Act
    transitions = trans
    # prepending states with given str
    if prepend_str:
        logger.debug('Given string:\n\t' + str(prepend_str) + '\n' +
                     'will be prepended to all states.')
    states = prepend_with(states, prepend_str)
    initial_states = prepend_with(initial_states, prepend_str)

    ts = FTS()
    ts.name = name

    ts.states.add_from(states)
    ts.states.initial |= initial_states

    ts.atomic_propositions |= ap

    # note: verbosity before actions below
    # to avoid screening by possible error caused by action

    # state labeling assigned ?
    if state_labeling is not None:
        for state, ap_label in state_labeling:
            if ap_label is None:
                ap_label = set()
            ap_label = str2singleton(ap_label)
            state = prepend_str + str(state)
            logger.debug('Labeling state:\n\t' + str(state) + '\n' +
                         'with label:\n\t' + str(ap_label) + '\n')
            ts.states[state]['ap'] = ap_label
    # any transition labeling ?
    if actions is None:
        for from_state, to_state in transitions:
            (from_state, to_state) = prepend_with([from_state, to_state],
                                                  prepend_str)
            logger.debug('Added unlabeled edge:\n\t' + str(from_state) +
                         '--->' + str(to_state) + '\n')
            ts.transitions.add(from_state, to_state)
    else:
        ts.actions |= actions
        for from_state, to_state, act in transitions:
            (from_state, to_state) = prepend_with([from_state, to_state],
                                                  prepend_str)
            logger.debug(
                'Added labeled edge (=transition):\n\t' +
                str(from_state) + '---[' + str(act) + ']--->' +
                str(to_state) + '\n')
            ts.transitions.add(from_state, to_state, actions=act)
    return ts


def line_labeled_with(L, m=0):
    """Return linear FTS with given labeling.

    The resulting system will be a terminating sequence::
        s0-> s1-> ... -> sN
    where: N = C{len(L) -1}.

    See Also
    ========
    L{cycle_labeled_with}

    @param L: state labeling
    @type L: iterable of state labels, e.g.,::
            [{'p', '!p', 'q',...]
    Single strings are identified with singleton Atomic Propositions,
    so [..., 'p',...] and [...,{'p'},...] are equivalent.

    @param m: starting index
    @type m: int

    @return: L{FTS} with:
        - states ['s0', ..., 'sN'], where N = len(L) -1
        - state labels defined by L, so s0 is labeled with L[0], etc.
        - transitions forming a sequence:
            - s_{i} ---> s_{i+1}, for: 0 <= i < N
    """
    n = len(L)
    S = range(m, m + n)
    S0 = []  # user will define them
    AP = {True}
    for ap_subset in L:
        # skip empty label ?
        if ap_subset is None:
            continue
        AP |= set(ap_subset)
    Act = None
    from_states = range(m, m + n - 1)
    to_states = range(m + 1, m + n)
    trans = zip(from_states, to_states)
    ts = tuple2fts(S, S0, AP, L, Act, trans, prepend_str='s')
    return ts


def cycle_labeled_with(L):
    """Return cycle FTS with given labeling.

    The resulting system will be a cycle::
        s0-> s1-> ... -> sN -> s0
    where: N = C{len(L) -1}.

    See Also
    ========
    L{line_labeled_with}

    @param L: state labeling
    @type L: iterable of state labels, e.g., [{'p', 'q'}, ...]
        Single strings are identified with singleton Atomic Propositions,
        so [..., 'p',...] and [...,{'p'},...] are equivalent.

    @return: L{FTS} with:
        - states ['s0', ..., 'sN'], where N = len(L) -1
        - state labels defined by L, so s0 is labeled with L[0], etc.
        - transitions forming a cycle:
            - s_{i} ---> s_{i+1}, for: 0 <= i < N
            - s_N ---> s_0
    """
    ts = line_labeled_with(L)
    last_state = 's' + str(len(L) - 1)
    ts.transitions.add(last_state, 's0')
    # trans += [(n-1, 0)] # close cycle
    return ts


def add_initial_states(ts, ap_labels):
    """Make initial any state of ts labeled with any label in ap_labels.

    For example if isinstance(ofts, FTS):

      >>> from tulip.transys.transys import add_initial_states
      >>> initial_labels = [{'home'}]
      >>> add_initial_states(ofts, initial_labels)

    @type ts: L{FiniteTransitionSystem}

    @param ap_labels: labels, each comprised of atomic propositions
    @type ap_labels: iterable of sets of elements from
        ts.atomic_propositions
    """
    for label in ap_labels:
        new_init_states = ts.states.find(ap='label')
        ts.states.initial |= new_init_states


def _dumps_states(g):
    """Dump string of transition system states.

    @type g: L{FTS}
    """
    nodes = g
    a = []
    for u in nodes:
        s = '\t State: {u}, AP: {ap}\n'.format(
            u=u, ap=g.nodes[u]['ap']) + ', '.join([
                '{k}: {v}'.format(k=k, v=v)
                for k, v in g.nodes[u].items()
                if k is not 'ap'])
        a.append(s)
    return ''.join(a)


class GameGraph(LabeledDiGraph):
    """Store a game graph.

    When adding states, you have to say
    which player controls the outgoing transitions.
    Use C{networkx} state labels for that:

      >>> g = GameGraph()
      >>> g.states.add('s0', player=0)

    See also
    ========
    L{automata.ParityGame}

    Reference
    =========
    1. Chatterjee K.; Henzinger T.A.; Jobstmann B.
       Environment Assumptions for Synthesis
       CONCUR'08, LNCS 5201, pp. 147-161, 2008
    """

    def __init__(self, node_label_types, edge_label_types):
        node_label_types += [{
            'name': 'player',
            'values': {0, 1},
            'default': 0}]
        super(GameGraph, self).__init__(node_label_types,
                                        edge_label_types)

    def player_states(self, n):
        """Return states controlled by player C{n}.

        'controlled' means that player C{n}
        gets to decide the successor state.

        @param n: player index (id number)
        @type n: 0 or 1

        @return: set of states
        @rtype: C{set}
        """
        return {x for x in self if self.nodes[x]['player'] == n}

    def edge_controlled_by(self, e):
        """Return the index of the player controlling edge C{e}.

        @type e: 2-tuple of nodes C{(n1, n2)}

        @rtype: integer 0 or 1
        """
        from_state = e[0]
        return self.nodes[from_state]['player']


class LabeledGameGraph(GameGraph):
    """Game graph with labeled states.

    Its contraction is a Kripke structure.
    Given a Kripke structure and a partition of propositions,
    then the corresponding labeled game graph
    can be obtained by graph expansion.

    Reference
    =========
    1. Chatterjee K.; Henzinger T.A.; Piterman N.
       Strategy Logic
       UCB/EECS-2007-78
    """

    def __init__(self):
        ap_labels = PowerSet()
        node_label_types = [
            {'name': 'ap',
             'values': ap_labels,
             'setter': ap_labels.math_set,
             'default': set()}]
        super(LabeledGameGraph, self).__init__(node_label_types)
        self.atomic_propositions = self.ap
        # dot formatting
        self._state_dot_label_format = {
            'ap': '',
            'type?label': '',
            'separator': r'\\n'}
        self.dot_node_shape = {'normal': 'rectangle'}


def _pre(graph, list_n):
    """Find union of predecessors of nodes in graph defined by C{MultiDiGraph}.

    @param graph: a graph structure corresponding to a FTS.
    @type graph: C{networkx.MultiDiGraph}
    @param list_n: list of nodes whose predecessors need to be returned
    @type list_n: list of nodes in C{graph}
    @return: set of predecessors of C{list_n}
    @rtype: C{set}
    """
    pre_set = set()
    for n in list_n:
        pre_set = pre_set.union(graph.predecessors(n))
    return pre_set


def _output_fts(ts, transitions, sol):
    """Convert the Part from C{nx.MultiDiGraph} to FTS.

    The returned FTS does not contain any edge attribute in the original FTS.
    All the transitions are assumed to be controllable.

    @param ts: the input finite transition system
    @type ts: L{FTS}
    @param Part: the final partition
    @type Part: C{networkx.MultiDiGraph}

    @return: the bi/dual simulation abstraction, and the
        partition of states in input ts
    @rtype: L{FTS}, and C{dict} with keys C{ts2simu} and C{simu2ts}.
        C{ts2simu} maps nodes from input FTS to output FTS and C{simu2ts}
        maps the nodes of two FTS in the other direction.
    """
    env_actions = [
        dict(name='env_actions',
             values=ts.env_actions,
             setter=True)]
    sys_actions = [
        dict(name='sys_actions',
             values=ts.sys_actions,
             setter=True)]
    ts_simu = FTS(env_actions, sys_actions)
    ts2simu = dict()
    simu2ts = dict()
    Part_hash = dict(ts2simu=ts2simu, simu2ts=simu2ts)
    n_cells = len(sol)
    for i in range(n_cells):
        simu2ts[i] = sol[i]
        for j in sol[i]:
            if j in ts2simu:
                ts2simu[j].append(i)
            else:
                ts2simu[j] = [i]
    S = range(n_cells)
    S0 = set()
    for i in ts.states.initial:
        [S0.add(j) for j in ts2simu[i]]
    ts_simu.states.add_from(S)
    ts_simu.states.initial.add_from(S0)
    AP = ts.aps
    ts_simu.atomic_propositions.add_from(AP)
    for i in range(n_cells):
        ts_simu.states.add(i, ap=ts.nodes[next(iter(sol[i]))]['ap'])
    for i in range(n_cells):
        for j in range(n_cells):
            if transitions[j, i]:
                ts_simu.transitions.add(i, j)
    return ts_simu, Part_hash


def simu_abstract(ts, simu_type):
    """Create a bi/dual-simulation abstraction for a Finite Transition System.

    @param ts: input finite transition system, the one you want to get
                    its bi/dual-simulation abstraction.
    @type ts: L{FTS}
    @param simu_type: string 'bi'/'dual', flag used to switch b.w.
                      bisimulation algorithm and dual-simulation algorithm.
    @return: the bi/dual simulation, and the corresponding partition.
    @rtype: L{FTS}, C{dict}


    References
    ==========

    1. Wagenmaker, A. J.; Ozay, N.
       "A Bisimulation-like Algorithm for Abstracting Control Systems."
       54th Annual Allerton Conference on CCC 2016
    """
    # create MultiDiGraph instance from the input FTS
    G = MultiDiGraph(ts)
    # build coarsest partition
    S0 = dict()
    Part = MultiDiGraph()  # a graph associated with the new partition
    n_cells = 0
    hash_ap = dict()  # map ap to cells in Part
    for node in G:
        ap = repr(G.nodes[node]['ap'])
        if ap not in S0:
            S0[ap] = set()
            hash_ap[ap] = set()
            Part.add_node(n_cells, ap=ap, cov=S0[ap])  # hash table S0--->G
            n_cells += 1
        S0[ap].add(node)

    sol = []
    for ap in S0:
        sol.append(S0[ap])

    IJ = np.ones([n_cells, n_cells])
    transitions = np.zeros([n_cells, n_cells])
    while np.sum(IJ) > 0:
        # get i,j from IJ matrix, i--->j
        ind = np.nonzero(IJ)
        i = ind[1][0]
        j = ind[0][0]
        IJ[j, i] = 0
        si = sol[i]
        sj = sol[j]
        pre_j = _pre(G, sj)
        if si.issubset(pre_j):
            transitions[j, i] = 1
        else:
            isect = si.intersection(pre_j)
            if isect == set():
                continue
            else:
                # check if the isect has existed
                check_isect = False
                if simu_type == 'dual':
                    assert len(sol) == n_cells
                    for k in range(n_cells):
                        if sol[k] == isect:
                            check_isect = True
                            break
                if not check_isect:
                    # assume that i != j
                    sol.append(isect)
                    # update transition matrix
                    transitions = np.pad(transitions, (0, 1), 'constant')
                    transitions[n_cells, :] = 0
                    transitions[:, n_cells] = transitions[:, i]
                    transitions[j, n_cells] = 1
                    if simu_type == 'bi':
                        sol[i] = sol[i].difference(sol[n_cells])
                        transitions[i, :] = 0
                        if i == j:
                            transitions[j, n_cells] = 0
                    # update IJ matrix
                    IJ = np.pad(IJ, (0, 1), 'constant', constant_values=1)
                    if simu_type == 'bi':
                        IJ[i, :] = 1
                        IJ[:, i] = 1
                    n_cells += 1
                else:
                    transitions[j, k] = 1
        IJ = ((IJ - transitions) > 0).astype(int)
    [ts_simu, part_hash] = _output_fts(ts, transitions, sol)
    return ts_simu, part_hash
