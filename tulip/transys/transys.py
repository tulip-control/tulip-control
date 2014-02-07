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
Transition System Module
"""
import logging
from collections import Iterable, OrderedDict
from pprint import pformat
import copy
import warnings

from .labeled_graphs import LabeledStateDiGraph, str2singleton
from .labeled_graphs import prepend_with, vprint
from .mathset import PowerSet, MathSet
from .executions import FTSSim
from .export import graph2promela

hl = 40 *'-'

logger = logging.getLogger(__name__)

class FiniteTransitionSystem(LabeledStateDiGraph):
    """Finite Transition System modeling a closed system.
    
    Implements Def. 2.1, p.20 [Baier]:
        - states = S
        - states.initial = S_0 \subseteq S
        - atomic_propositions = AP
        - actions = Act
        - transitions = the transition relation ->
                = edge set + edge labeling function
                (labels \in actions)
            Unlabeled edges are defined using:
                - sys.transitions.add
                - sys.transitions.add_from
                - sys.transitions.add_adj
            Labeled edges (=transitions) are defined using:
                - sys.transitions.add_labeled
                - sys.transitions.add_labeled_from
                - sys.transitions.add_labeled_adj
            and accessed using:
                - sys.transitions.find
                - sys.transitions.between
            To convert an unlabeled edge to a labeled edge use:
                - sys.transitions.label
            and to change the label of an existing labeled edge:
                - sys.transitions.relabel
        - the state labeling function
                L: S-> 2^AP
            can be defined using:
                - sys.states.label
                - sys.states.labels
            and accessed using methods:
                - sys.states(data=True)
                - sys.states.label_of
                - sys.states.labeled_with
                - sys.states.find
    
    The state labels are subsets of atomic_propositions, so \in 2^AP.
    The transition labels are actions.
    
    sys.actions_must: select constraint on actions. Options:
        
        - 'mutex': at most 1 action True each time
        - 'xor': exactly 1 action True each time
        - 'none': no constraint on action values
    
    The xor constraint can prevent the environment from
    blocking the system by setting all its actions to False.
    
    example
    -------
    In the following C{None} represents the empty set, subset of AP.
    First create an empty transition system and add some states to it:
    
    >>> from tulip import transys as trs
    >>> ts = trs.FiniteTransitionSystem()
    >>> ts.states.add('s0')
    >>> ts.states.add_from(['s1', 's3', 'end', 5] )
    
    Set an initial state, must already be in states:
    
    >>> ts.states.initial.add('s0')
    
    There can be more than one possible initial states:
    
    >>> ts.states.initial.add_from(['s0', 's3'] )
    
    To label the states, we need at least one atomic proposition,
    here 'p':
    
    >>> ts.atomic_propositions |= ['p', None]
    >>> ts.states.label('s0', {'p'} )
    >>> ts.states.labels(['s1', 's3'], [{'p'}, {None} ] )
    >>> ts.states.labels([('end', {'p'} ), (5, {None} ) ] )
    
    For singleton subsets of AP passing the atomic proposition
    itself, instead of the singleton, will also work:
    
    >>> ts.states.label('s0', 'p')
    
    Having added states, we can also add some labeled transitions:
    
    >>> ts.actions |= ['think', 'write']
    >>> ts.transitions.add_labeled('s0', 's1', 'think')
    >>> ts.transitions.add_labeled('s1', 5, 'write')
    
    Note that an unlabeled transition:
    
    >>> ts.transitions.add('s0', 's3')
    
    is considered as different from a labeled one and to avoid
    unintended duplication, after adding an unlabeled transition,
    any attempt to add a labeled transition between the same states
    will raise an exception, unless the unlabeled transition is
    either removed before adding the labeled transition,
    or method C{sys.transitions.label} is used instead,
    to convert the unlabeled into a labeled transition.
    
    Using C{tuple2fts} offers a more convenient constructor
    for transition systems.
    
    The user can still invoke NetworkX functions to set custom node
    and edge labels, in addition to the above ones.
    For example:
    
    >>> ts.states.add('s0')
    >>> ts.node['s0']['my_cost'] = 5
    
    The difference is that atomic proposition and action labels
    are checked to make sure they are elements of the system's
    AP and Action sets.
    
    It is not advisable to use NetworkX C{add_node} and C{add_edge}
    directly, because that can result in an inconsistent system,
    since it skips all checks performed by transys.
    
    dot export
    ----------
    Format transition labels using C{_transition_dot_label_format}
    which is a dict with values:
        - 'actions' (=name of transitions attribute):
            type before separator
        - 'type?label': separator between label type and value
        - 'separator': between labels for different sets of actions
            (e.g. sys, env). Not used for closed FTS,
            because it has single set of actions.
    
    note
    ----
    The attributes atomic_propositions and aps are equal.
    When you want to produce readable code, use atomic_propositions.
    Otherwise, aps offers shorthand access to the APs.
    
    see also
    --------
    OpenFTS, tuple2fts, line_labeled_with, cycle_labeled_with
    """
    def __init__(self, *args, **kwargs):
        """Initialize Finite Transition System.
        
        @param atomic_propositions: state are labeled with
            subsets of atomic propositions (\in 2^AP)
        @type atomic_propositions: iterable of str
        
        @param actions: actions used to label transitions
        @type actions: iterable of str
        
        For other arguments, see LabeledStateDiGraph
        """
        atomic_propositions = []
        actions = []
        
        # state labels
        self._state_label_def = OrderedDict(
            [['ap', PowerSet(atomic_propositions) ]]
        )
        self.atomic_propositions = self._state_label_def['ap'].math_set
        self.aps = self.atomic_propositions # shortcut
        self._state_dot_label_format = {'ap':'',
                                           'type?label':'',
                                           'separator':'\\n'}
        
        # edge labels comprised of sublabels (here single sublabel)
        self._transition_label_def = OrderedDict(
            [['actions', MathSet(actions)]]
        )
        self.actions = self._transition_label_def['actions']
        self._transition_dot_label_format = {'actions':'',
                                                'type?label':'',
                                                'separator':'\\n'}
        self._transition_dot_mask = dict()
        
        self.actions_must = 'xor'
        
        LabeledStateDiGraph.__init__(self, *args, **kwargs)
        
        self.dot_node_shape = {'normal':'box'}
        self.default_export_fname = 'fts'

    def __str__(self):
        s = hl +'\nFinite Transition System (closed) : '
        s += self.name +'\n' +hl +'\n'
        s += 'Atomic Propositions:\n\t'
        s += pformat(self.atomic_propositions, indent=3) +2*'\n'
        s += 'States and State Labels (\in 2^AP):\n'
        s += pformat(self.states(data=True), indent=3) +2*'\n'
        s += 'Initial States:\n'
        s += pformat(self.states.initial, indent=3) +2*'\n'
        s += 'Actions:\n\t' +str(self.actions) +2*'\n'
        s += 'Transitions & Labels:\n'
        s += pformat(self.transitions(labeled=True), indent=3)
        s += '\n' +hl +'\n'
        
        return s
    
    def __mul__(self, ts_or_ba):
        """Synchronous product TS * BA or TS * TS2.
        
        TS is this transition system, TS2 another one and
        BA a Buchi Automaton.
        
        @rtype: FiniteTransitionSystem
        
        see also
        --------
        self.sync_prod
        """
        return self.sync_prod(ts_or_ba)
    
    def __add__(self, other):
        """Merge two Finite Transition Systems.
        
        States, Initial States, Actions, Atomic Propositions and
        State labels and Transitions of the second Transition System
        are merged into the first and take precedence, overwriting
        existing labeling.
        
        example
        -------
        This can be useful to construct systems quickly by
        creating standard "pieces" using the functions:
            line_labeled_with, cycle_labeled_with
        
        >>> n = 4
        >>> L = n*['p'] # state labeling
        >>> ts1 = line_labeled_with(L, n-1)
        >>> ts1.plot()
        >>> 
        >>> L = n*['p']
        >>> ts2 = cycle_labeled_with(L)
        >>> ts2.states.label('s3', '!p')
        >>> ts2.plot()
        >>> 
        >>> ts3 = ts1 +ts2
        >>> ts3.transitions.add('s'+str(n-1), 's'+str(n) )
        >>> ts3.default_layout = 'circo'
        >>> ts3.plot()
        
        see also
        --------
        line_labeled_with, cycle_labeled_with
        
        @param other: system to merge with
        @type other: C{FiniteTransitionSystem}
        
        @return: merge of C{self} with C{other}, union of states,
            initial states, atomic propositions, actions, edges and
            labelings, those of C{other} taking precedence over C{self}.
        @rtype: FiniteTransitionSystem
        """
        if not isinstance(other, FiniteTransitionSystem):
            msg = 'other class must be FiniteTransitionSystem.\n'
            msg += 'Got instead:\n\t' +str(other)
            msg += '\nof type:\n\t' +str(type(other) )
            raise TypeError(msg)
        
        self.atomic_propositions |= other.atomic_propositions
        self.actions |= other.actions
        
        # add extra states & their labels
        for state, label in other.states.find():
            if state not in self.states:
                self.states.add(state)
            
            if label:
                self.states.label(state, label['ap'] )
        
        self.states.initial |= other.states.initial()
        
        # copy extra transitions (be careful w/ labeling)
        for (from_state, to_state, label_dict) in \
            other.transitions.find():
            # labeled edge ?
            if not label_dict:
                self.transitions.add(from_state, to_state)
            else:
                sublabel_value = label_dict['actions']
                self.transitions.add_labeled(
                    from_state, to_state, sublabel_value
                )
        
        return copy.copy(self)
    
    def __or__(self, ts):
        """Asynchronous product self * ts.
        
        see also
        --------
        async_prod
        
        @type ts: FiniteTransitionSystem
        """
        return self.async_prod(ts)
    
    def sync_prod(self, ts_or_ba):
        """Synchronous product TS * BA or TS1 * TS2.
        
        Returns a Finite Transition System, because TS is
        the first term in the product.
        
        Changing term order, i.e., BA * TS, returns the
        synchronous product as a BA.
        
        see also
        --------
        __mul__, async_prod, BuchiAutomaton.sync_prod, tensor_product
        Def. 2.42, pp. 75--76 [Baier 2008]
        Def. 4.62, p.200 [Baier 2008]
        
        @param ts_or_ba: system with which to take synchronous product
        @type ts_or_ba: FiniteTransitionSystem or BuchiAutomaton
        
        @return: synchronous product C{self} x C{ts_or_ba}
        @rtype: FiniteTransitionSystem
        """
        if isinstance(ts_or_ba, FiniteTransitionSystem):
            ts = ts_or_ba
            return self._sync_prod(ts)
        else:
            ba = ts_or_ba
            return _ts_ba_sync_prod(self, ba)
    
    def _sync_prod(self, ts):
        """Synchronous (tensor) product with other FTS.
        
        @param ts: other FTS with which to take synchronous product
        @type ts: FiniteTransitionSystem
        """
        # type check done by caller: sync_prod
        if self.states.mutants or ts.states.mutants:
            mutable = True
        else:
            mutable = False
        
        prod_ts = FiniteTransitionSystem(mutable=mutable)
        
        # union of AP sets
        prod_ts.atomic_propositions |= \
            self.atomic_propositions | ts.atomic_propositions
        
        # for synchronous product: Cartesian product of action sets
        prod_ts.actions |= self.actions * ts.actions
        
        prod_ts = super(FiniteTransitionSystem, self).tensor_product(
            ts, prod_sys=prod_ts
        )
        
        return prod_ts
    
    def async_prod(self, ts):
        """Asynchronous product TS1 x TS2 between FT Systems.
        
        see also
        --------
        __or__, sync_prod, cartesian_product
        Def. 2.18, p.38 [Baier 2008]
        """
        if not isinstance(ts, FiniteTransitionSystem):
            raise TypeError('ts must be a FiniteTransitionSystem.')
        
        if self.states.mutants or ts.states.mutants:
            mutable = True
        else:
            mutable = False
        
        # union of AP sets
        prod_ts = FiniteTransitionSystem(mutable=mutable)
        prod_ts.atomic_propositions |= \
            self.atomic_propositions | ts.atomic_propositions
        
        # for parallel product: union of action sets
        prod_ts.actions |= self.actions | ts.actions
        
        prod_ts = super(FiniteTransitionSystem, self).cartesian_product(
            ts, prod_sys=prod_ts
        )
        
        return prod_ts

    # operations between transition systems    
    def intersection(self):
        """Conjunction with another FTS.
        """
        raise NotImplementedError
        
    def difference(self):
        """Remove a sub-FTS.
        """
        raise NotImplementedError

    def composition(self):
        """Compositions of FTS, with state replaced by another FTS.
        """
        raise NotImplementedError
    
    def project(self, factor=None):
        """Project onto subgraph or factor graph.
        
        @param factor: on what to project:
            - If C{None}, then project on subgraph, i.e., using states
            - If C{int}, then project on the designated element of
                the tuple comprising each state
        @type factor: None | int
        """
        raise NotImplementedError
    
    def simulate(self, state_sequence="random"):
        """Simulate Finite Transition System.
        
        @type state_sequence: inputs="random"
            | given array
        
        see also
        --------
        is_simulation
        """
        raise NotImplementedError
    
    def is_simulation(self, simulation=FTSSim() ):
        """Check path, execution, trace or simulation given.
        
        terminology
        -----------
        - A path is a sequence of states.
        - An execution is a sequence of alternating states, actions.
        - A trace is the lift of a path by the labeling.
            (Caution: a single trace may project on multiple paths)
        - A bundle of the above is here called a simulation.
        
        see also
        --------
        simulate
        """
        raise NotImplementedError
    
    def loadSPINAut():
        raise NotImplementedError
    
    def _save(self, path, fileformat):
        """Export options available only for FTS systems.
        
        Provides: pml (Promela)
        
        see also
        --------
        save, plot
        """
        if fileformat not in {'promela', 'Promela', 'pml'}:
            return False
        
        s = graph2promela.fts2promela(self, self.name)
        
        # dump to file
        f = open(path, 'w')
        f.write(s)
        f.close()
        return True

class FTS(FiniteTransitionSystem):
    """Alias to FiniteTransitionSystem.
    """    
    def __init__(self, *args, **kwargs):
        FiniteTransitionSystem.__init__(self, *args, **kwargs)

class OpenFiniteTransitionSystem(LabeledStateDiGraph):
    """Open Finite Transitin System modeling an open system.
    
    Analogous to FTS, but for open systems comprised of
    the system and its environment.
    
    Please refer to FiniteTransitionSystem for usage details.
    
    The only significant difference is in transition labeling.
    For closed systems, each transition is labeled with a system action.
    So each transition label comprises of a single sublabel,
    the system action.
    
    For open systems, each transition is labeled with 2 sublabels:
        - The first sublabel is a system action,
        - the second an environment action.
    
    Constrains on actions can be defined
    similarly to FTS actions by setting the fields:
    
        - ofts.env_actions_must
        - ofts.sys_actions_must
    
    The default constraint is 'xor'.
    For more details see FTS.
    
    see also
    --------
    FiniteTransitionSystem
    """
    def __init__(self, atomic_propositions=[], sys_actions=[],
                 env_actions=[], **args):
        # state labeling
        self._state_label_def = OrderedDict(
            [['ap', PowerSet(atomic_propositions) ]]
        )
        self.atomic_propositions = self._state_label_def['ap'].math_set
        self.aps = self.atomic_propositions
        self._state_dot_label_format = {'ap':'',
                                           'type?label':'',
                                           'separator':'\\n'}
        
        # edge labeling (here 2 sublabels)
        self._transition_label_def = OrderedDict([
            ['sys_actions', MathSet(sys_actions) ],
            ['env_actions', MathSet(env_actions) ]
        ])
        self.sys_actions = self._transition_label_def['sys_actions']
        self.env_actions = self._transition_label_def['env_actions']
        self._transition_dot_label_format = {'sys_actions':'sys',
                                                'env_actions':'env',
                                                'type?label':':',
                                                'separator':'\\n'}
        # action constraint used in synth.synthesize
        self.env_actions_must = 'xor'
        self.sys_actions_must = 'xor'
        
        LabeledStateDiGraph.__init__(self, **args)
        
        self.dot_node_shape = {'normal':'box'}
        self.default_export_fname = 'ofts'
        
    def __str__(self):
        s = hl +'\nFinite Transition System (open) : '
        s += self.name +'\n' +hl +'\n'
        s += 'Atomic Propositions:\n'
        s += pformat(self.atomic_propositions, indent=3) +2*'\n'
        s += 'States & State Labels (\in 2^AP):\n'
        s += pformat(self.states(data=True), indent=3) +2*'\n'
        s += 'Initial States:\n'
        s += pformat(self.states.initial, indent=3) +2*'\n'
        s += 'System Actions:\n'
        s += pformat(self.sys_actions, indent=3) +2*'\n'
        s += 'Environment Actions:\n'
        s += pformat(self.env_actions, indent=3) +2*'\n'
        s += 'Transitions & Labeling w/ Sys, Env Actions:\n'
        s += pformat(self.transitions(labeled=True), indent=3)
        s += '\n' +hl +'\n'
        
        return s

class OpenFTS(OpenFiniteTransitionSystem):
    """Alias to transys.OpenFiniteTransitionSystem.
    """
    def __init__(self, **args):
        OpenFiniteTransitionSystem.__init__(self, **args)

def tuple2fts(S, S0, AP, L, Act, trans, name='fts',
              prepend_str=None, verbose=False):
    """Create a Finite Transition System from a tuple of fields.

    hint
    ----
    To rememeber the arg order:

    1) it starts with states (S0 requires S before it is defined)

    2) continues with the pair (AP, L), because states are more
    fundamental than transitions
    (transitions require states to be defined)
    and because the state labeling L requires AP to be defined.

    3) ends with the pair (Act, trans), because transitions in trans
    require actions in Act to be defined.

    see also
    --------
    L{tuple2ba}

    @param S: set of states
    @type S: iterable of hashables
    
    @param S_0: set of initial states, must be \\subset S
    @type S_0: iterable of elements from S
    
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
        
        vprint('State labeling L not tuples (state, ap_label),\n'
                   'zipping with states S...\n', verbose)
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
        vprint('Given string:\n\t' +str(prepend_str) +'\n' +
               'will be prepended to all states.', verbose)
    states = prepend_with(states, prepend_str)
    initial_states = prepend_with(initial_states, prepend_str)
    
    ts = FTS(name=name)
    
    ts.states.add_from(states)
    ts.states.initial |= initial_states
    
    ts.atomic_propositions |= ap
    
    # note: verbosity before actions below
    # to avoid screening by possible error caused by action
    
    # state labeling assigned ?
    if state_labeling is not None:
        for (state, ap_label) in state_labeling:
            ap_label = str2singleton(ap_label, verbose=verbose)
            (state,) = prepend_with([state], prepend_str)
            
            vprint('Labeling state:\n\t' +str(state) +'\n' +
                  'with label:\n\t' +str(ap_label) +'\n', verbose)
            ts.states.label(state, ap_label)
    
    # any transition labeling ?
    if actions is None:
        for (from_state, to_state) in transitions:
            (from_state, to_state) = prepend_with([from_state, to_state],
                                                  prepend_str)
            vprint('Added unlabeled edge:\n\t' +str(from_state) +
                   '--->' +str(to_state) +'\n', verbose)
            ts.transitions.add(from_state, to_state)
    else:
        ts.actions |= actions
        for (from_state, to_state, act) in transitions:
            (from_state, to_state) = prepend_with([from_state, to_state],
                                                  prepend_str)
            vprint('Added labeled edge (=transition):\n\t' +
                   str(from_state) +'---[' +str(act) +']--->' +
                   str(to_state) +'\n', verbose)
            ts.transitions.add_labeled(from_state, to_state, act)
    
    return ts

def line_labeled_with(L, m=0):
    """Return linear FTS with given labeling.
    
    The resulting system will be a terminating sequence:
        s0-> s1-> ... -> sN
    where: N = C{len(L) -1}.
    
    see also
    --------
    cycle_labeled_with
    
    @param L: state labeling
    @type L: iterable of state labels, e.g., [{'p', '!p', 'q',...]
        Single strings are identified with singleton Atomic Propositions,
        so [..., 'p',...] and [...,{'p'},...] are equivalent.
    
    @param m: starting index
    @type m: int
    
    @return: FTS with:
        - states ['s0', ..., 'sN'], where N = len(L) -1
        - state labels defined by L, so s0 is labeled with L[0], etc.
        - transitions forming a sequence:
            - s_{i} ---> s_{i+1}, for: 0 <= i < N
    """
    n = len(L)
    S = range(m, m+n)
    S0 = [] # user will define them
    AP = {True}
    for ap_subset in L:
        # skip empty label ?
        if ap_subset is None:
            continue
        AP |= set(ap_subset)
    Act = None
    from_states = range(m, m+n-1)
    to_states = range(m+1, m+n)
    trans = zip(from_states, to_states)
    
    ts = tuple2fts(S, S0, AP, L, Act, trans, prepend_str='s')
    return ts

def cycle_labeled_with(L):
    """Return cycle FTS with given labeling.
    
    The resulting system will be a cycle:
        s0-> s1-> ... -> sN -> s0
    where: N = C{len(L) -1}.
    
    see also
    --------
    line_labeled_with
    
    @param L: state labeling
    @type L: iterable of state labels, e.g., [{'p', 'q'}, ...]
        Single strings are identified with singleton Atomic Propositions,
        so [..., 'p',...] and [...,{'p'},...] are equivalent.
    
    @return: FTS with:
        - states ['s0', ..., 'sN'], where N = len(L) -1
        - state labels defined by L, so s0 is labeled with L[0], etc.
        - transitions forming a cycle:
            - s_{i} ---> s_{i+1}, for: 0 <= i < N
            - s_N ---> s_0
    """
    ts = line_labeled_with(L)
    last_state = 's' +str(len(L)-1)
    ts.transitions.add(last_state, 's0')
    
    #trans += [(n-1, 0)] # close cycle
    return ts

def add_initial_states(ts, ap_labels):
    """Make initial any state of ts labeled with any label in ap_labels.
    
    For example if isinstance(disc_dynamics, AbstractSysDyn):
    
    >>> from tulip.transys.transys import add_initial_states
    >>> initial_labels = [{'home'}]
    >>> add_initial_states(disc_dynamics.ofts, initial_labels)
    
    @type ts: transys.FiniteTransitionSystem,
        transys.OpenFiniteTransitionSystem
    
    @param ap_labels: labels, each comprised of atomic propositions
    @type ap_labels: iterable of sets of elements from
        ts.atomic_propositions
    """
    for label in ap_labels:
        new_init_states = ts.states.labeled_with({'ap':label})
        ts.states.initial |= new_init_states

def _ts_ba_sync_prod(transition_system, buchi_automaton):
    """Construct transition system for the synchronous product TS * BA.
    
    Def. 4.62, p.200 [Baier]
    
    erratum
    -------
    note the erratum: P_{pers}(A) is ^_{q\in F} !q, verified from:
        http://www-i2.informatik.rwth-aachen.de/~katoen/errata.pdf
    
    see also
    --------
    _ba_ts_sync_prod, FiniteTransitionSystem.sync_prod
    
    @return: C{(product_ts, persistent_states)}, where:
        - C{product_ts} is the synchronous product TS * BA
        - C{persistent_states} are those in TS * BA which
            project on accepting states of BA.
    @rtype:
        - C{product_TS} is a FiniteTransitionSystem
        - C{persistent_states} is the set of states which project
            on accepting states of the Buchi Automaton BA.
    """
    def convert_ts2ba_label(state_label_dict):
        """Replace 'ap' key with 'in_alphabet'.
        
        @param state_label_dict: FTS state label, its value \\in 2^AP
        @type state_label_dict: dict {'ap' : state_label_value}
        
        @return: BA edge label, its value \\in 2^AP
            (same value with state_label_dict)
        @rtype: dict {'in_alphabet' : edge_label_value}
            Note: edge_label_value is the BA edge "guard"
        """
        logger.debug('Ls0:\t' +str(state_label_dict) )
        
        (s0_, label_dict) = state_label_dict[0]
        Sigma_dict = {'in_alphabet': label_dict['ap'] }
        
        logger.debug('State label of: ' +str(s0) +
                      ', is: ' +str(Sigma_dict) )
        
        return Sigma_dict
    
    if not isinstance(transition_system, FiniteTransitionSystem):
        msg = 'transition_system not transys.FiniteTransitionSystem.\n'
        msg += 'Actual type passed: ' +str(type(transition_system) )
        raise TypeError(msg)
    
    if not hasattr(buchi_automaton, 'alphabet'):
        msg = 'transition_system not transys.BuchiAutomaton.\n'
        msg += 'Actual type passed: ' +str(type(buchi_automaton) )
        raise TypeError(msg)
    
    if not buchi_automaton.atomic_proposition_based:
        msg = """Buchi automaton not stored as Atomic Proposition-based.
                synchronous product with Finite Transition System
                is not well-defined."""
        raise Exception(msg)
    
    fts = transition_system
    ba = buchi_automaton
    
    prodts_name = fts.name +'*' +ba.name
    
    if fts.states.mutants or ba.states.mutants:
        mutable = True
    else:
        mutable = False
    
    # using set() destroys order
    prodts = FiniteTransitionSystem(
        name=prodts_name, mutable=mutable
    )
    prodts.states.add_from(set() )
    prodts.atomic_propositions.add_from(ba.states() )
    prodts.actions.add_from(fts.actions)

    # construct initial states of product automaton
    s0s = fts.states.initial()
    q0s = ba.states.initial()
    
    accepting_states_preimage = MathSet()
    
    logger.debug(hl +'\n' +' Product TS construction:\n' +hl +'\n')
    
    if not s0s:
        msg = 'Transition System has no initial states !\n'
        msg += '=> Empty product system.\n'
        msg += 'Did you forget to define initial states ?'
        warnings.warn(msg)
    
    for s0 in s0s:
        logger.debug('Checking initial state:\t' +str(s0) )
        
        Ls0 = fts.states.find(s0)
        
        # desired input letter for BA
        Sigma_dict = convert_ts2ba_label(Ls0)
        
        for q0 in q0s:
            enabled_ba_trans = ba.transitions.find(
                [q0], desired_label=Sigma_dict
            )
            
            # q0 blocked ?
            if not enabled_ba_trans:
                logger.debug('blocked q0 = ' +str(q0) )
                continue
            
            # which q next ?     (note: curq0 = q0)
            logger.debug('enabled_ba_trans = ' +str(enabled_ba_trans) )
            for (curq0, q, sublabels) in enabled_ba_trans:
                new_sq0 = (s0, q)                
                prodts.states.add(new_sq0)
                prodts.states.initial.add(new_sq0)
                prodts.states.label(new_sq0, {q} )
                
                # accepting state ?
                if q in ba.states.accepting:
                    accepting_states_preimage.add(new_sq0)
    
    logger.debug(prodts)
    
    # start visiting reachable in DFS or BFS way
    # (doesn't matter if we are going to store the result)    
    queue = MathSet(prodts.states.initial() )
    visited = MathSet()
    while queue:
        sq = queue.pop()
        visited.add(sq)
        (s, q) = sq
        
        logger.debug('Current product state:\n\t' +str(sq) )
        
        # get next states
        next_ss = fts.states.post(s)
        next_sqs = MathSet()
        for next_s in next_ss:
            logger.debug('Next state:\n\t' +str(next_s) )
            
            Ls = fts.states.find(next_s)
            if not Ls:
                raise Exception(
                    'No AP label for FTS state: ' +str(next_s) +
                     '\n Did you forget labeing it ?'
                )
            
            Sigma_dict = convert_ts2ba_label(Ls)
            logger.debug("Next state's label:\n\t" +str(Sigma_dict) )
            
            enabled_ba_trans = ba.transitions.find(
                [q], desired_label=Sigma_dict
            )
            logger.debug('Enabled BA transitions:\n\t' +
                          str(enabled_ba_trans) )
            
            if not enabled_ba_trans:
                continue
            
            for (q, next_q, sublabels) in enabled_ba_trans:
                new_sq = (next_s, next_q)
                next_sqs.add(new_sq)
                logger.debug('Adding state:\n\t' + str(new_sq) )
                
                prodts.states.add(new_sq)
                
                if next_q in ba.states.accepting:
                    accepting_states_preimage.add(new_sq)
                    logger.debug(str(new_sq) +
                                  ' contains an accepting state.')
                
                prodts.states.label(new_sq, {next_q} )
                
                logger.debug('Adding transitions:\n\t' +
                              str(sq) + '--->' + str(new_sq) )
                
                # is fts transition labeled with an action ?
                ts_enabled_trans = fts.transitions.find(
                    [s], to_states=[next_s],
                    desired_label='any', as_dict=False
                )
                for (from_s, to_s, sublabel_values) in ts_enabled_trans:
                    assert(from_s == s)
                    assert(to_s == next_s)
                    logger.debug('Sublabel value:\n\t' +
                                  str(sublabel_values) )
                    
                    # labeled transition ?
                    if not sublabel_values:
                        prodts.transitions.add(sq, new_sq)
                    else:
                        #TODO open FTS
                        prodts.transitions.add_labeled(
                            sq, new_sq, sublabel_values[0]
                        )
        
        # discard visited & push them to queue
        new_sqs = MathSet()
        for next_sq in next_sqs:
            if next_sq not in visited:
                new_sqs.add(next_sq)
                queue.add(next_sq)
    
    return (prodts, accepting_states_preimage)
