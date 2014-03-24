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
Base classes for labeled directed graphs
"""
import logging
logger = logging.getLogger(__name__)

import os
from pprint import pformat
from collections import Iterable
import warnings
import copy

import networkx as nx

from .mathset import SubSet, TypedDict, is_subset
from .export import save_d3, graph2dot

def label_is_desired(attr_dict, desired_dict):
    """Return True if all labels match.
    
    Supports symbolic evaluation, if label type is callable.
    """
    if not isinstance(attr_dict, TypedDict):
        raise Exception('attr_dict must be TypedDict' +
                        ', instead: ' + str(type(attr_dict) ))
    
    if attr_dict == desired_dict:
        return True
    
    # different keys ?
    mismatched_keys = set(attr_dict).symmetric_difference(desired_dict)
    if mismatched_keys:
        return False
    
    # any labels have symbolic semantics ?
    label_def = attr_dict.allowed_values
    for type_name, value in attr_dict.iteritems():
        logger.debug('Checking label type:\n\t' +str(type_name))
        
        type_def = label_def[type_name]
        desired_value = desired_dict[type_name]
        
        if hasattr(type_def, '__call__'):
            logger.debug('Found label semantics:\n\t' + str(type_def))
            
            # value = guard
            if not type_def(value, desired_value):
                return False
            else:
                continue
        
        # no guard semantics given,
        # then by convention: guard is singleton {cur_val},
        if not value == desired_value:
            test_common_bug(value, desired_value)
            return False
    return True

def test_common_bug(value, desired_value):
    logger.debug('Label value:\n\t' +str(value) )
    logger.debug('Desired value:\n\t' +str(desired_value) )
    
    if isinstance(value, (set, list) ) and \
    isinstance(desired_value, (set, list) ) and \
    value.__class__ != desired_value.__class__:
       msg = 'Set SubLabel:\n\t' +str(value)
       msg += 'compared to list SubLabel:\n\t' +str(desired_value)
       msg += 'Did you mix sets & lists when setting AP labels ?'
       raise Exception(msg)

class States(object):
    """Methods to manage states, initial states, current state.
    """
    def __init__(self, graph):
        self.graph = graph
        self.initial = []
        self.current = []
    
    def __get__(self):
        return self.__call__()
    
    def __getitem__(self, state):
        return self.graph.node[state]
    
    def __call__(self, *args, **kwargs):
        """Return list of states.
        
        For more details see L{LabeledDiGraph.nodes}
        """
        return self.graph.nodes(*args, **kwargs)
    
    def __str__(self):
        return 'States:\n' +pformat(self(data=False) )
    
    def __len__(self):
        """Total number of states.
        """
        return self.graph.number_of_nodes()
    
    def __iter__(self):
        return iter(self() )
        
    def __ior__(self, new_states):
        #TODO carefully test this
        self.add_from(new_states)
        return self
    
    def __contains__(self, state):
        """Return True if state in states.
        """
        return state in self.graph
    
    @property
    def initial(self):
        """ Return SubSet of initial states.
        """
        return self._initial
    
    @initial.setter
    def initial(self, states):
        s = SubSet(self)
        s |= states
        self._initial = s
    
    @property
    def current(self):
        """Return SubSet of current states.
        
        Non-deterministic automata can have multiple current states.
        """
        return self._current
    
    @current.setter
    def current(self, states):
        """Set current states.
        
        If state not in states, exception raised.
        """
        s = SubSet(self)
        s |= states
        self._current = s
    
    def _warn_if_state_exists(self, state):
        if state in self():
            if self.list is not None:
                raise Exception('State exists and ordering enabled: ambiguous.')
            else:
                logger.debug('State already exists.')
                return
    
    def _single_state2singleton(self, state):
        """Convert to a singleton list, if argument is a single state.
        
        Otherwise return given argument.
        """
        if state in self:
            states = [state]
        else:
            states = state
        return states
    
    def add(self, new_state, attr_dict=None, check=True, **attr):
        """Create or label single state.
        
        For details see L{LabeledDiGraph.add_node}.
        """
        new_state_id = self._mutant2int(new_state)
        self._warn_if_state_exists(new_state)
        
        logger.debug('Adding new id: ' +str(new_state_id) )
        self.graph.add_node(new_state_id, attr_dict, check, **attr)
    
    def add_from(self, new_states, destroy_order=False):
        """Create or label multiple states from iterable container.
        
        For details see L{LabeledDiGraph.add_nodes_from}.
        """
        # iteration used for comprehensible error message
        for new_state in new_states:
            self._warn_if_state_exists(new_state)
        
        self.graph.add_nodes_from(new_states)
    
    def remove(self, state):
        """Remove single state.
        """
        if state in self.initial:
            self.initial.remove(state)
        
        self.graph.remove_node(state)
    
    def remove_from(self, states):
        """Remove a list of states.
        """
        for state in states:
            self.remove(state)
    
    def post(self, states):
        """Direct successor set (1-hop) for given states.
        
        Over all actions or letters, i.e., edge labeling ignored
        by states.pre, because it may be undefined. Only classes
        which have an action set, alphabet, or other transition
        labeling set provide a pre(state, label) method, as for
        example pre(state, action) in the case of closed transition
        systems.
        
        If multiple stats provided,
        then union Post(s) for s in states provided.
        
        See Also
        ========
          - L{pre}
          - Def. 2.3, p.23 U{[BK08]
            <http://tulip-control.sourceforge.net/doc/bibliography.html#bk08>}
        """
        states = self._single_state2singleton(states)
        
        successors = list()
        for state in state:
            successors += self.graph.successors(state)
        return successors
    
    def pre(self, states):
        """Return direct predecessors (1-hop) of given state.
        
        See Also
        ========
          - L{post}
          - Def. 2.3, p.23 U{[BK08]
            <http://tulip-control.sourceforge.net/doc/bibliography.html#bk08>}
        """
        states = self._single_state2singleton(states)
        
        predecessors = list()
        for state in states:
            predecessors += self.graph.predecessors(state)
        
        return predecessors
    
    def forward_reachable(self, state):
        """Return states reachable from given state.
        
        Iterated post(), a wrapper of networkx.descendants.
        """
        descendants = nx.descendants(self, state)
        return descendants
    
    def backward_reachable(self, state):
        """Return states from which the given state can be reached.
        """
        ancestors = nx.ancestors(self, state)
        return ancestors
    
    def paint(self, state, color):
        """Color the given state.
        
        The state is filled with given color,
        rendered with dot when plotting and saving.
        
        @param state: valid system state
        
        @param color: with which to paint C{state}
        @type color: str of valid dot color
        """
        self.graph.node[state]['style'] = 'filled'
        self.graph.node[state]['fillcolor'] = color
    
    def find(self, states='any', with_attr_dict=None, **with_attr):
        """Filter by desired states and by desired state labels.
        
        Examples
        ========
        Assume that the system is:
        
        >>> import transys as trs
        >>> ts = trs.FTS()
        >>> ts.atomic_propositions.add('p')
        >>> ts.states.add('s0')
        >>> ts.states.label('s0', {'p'} )
        
          - To find the label of a single state C{'s0'}:

              >>> a = ts.states.find(['s0'] )
              >>> (s0_, label) = a[0]
              >>> print(label)
              {'ap': set(['p'])}

              equivalently, but asking for a list instead of a dict:

              >>> a = ts.states.find(['s0'], as_dict=False)
              >>> (s0_, label) = a[0]
              >>> print(label)
              [set(['p'])]

              Calling C{label_of} is a shortcut for the above.

          - To find all states with a specific label C{{'p'}}:

              >>> ts.states.label('s1', {'p'}, check=False)
              >>> b = ts.states.find(with_attr_dict={'ap':{'p'} } )
              >>> states = [state for (state, label_) in b]
              >>> print(set(states) )
              {'s0', 's1'}

              Calling C{labeled_with} is a shortcut for the above.

          - To find all states in subset C{M} labeled with C{{'p'}}:

              >>> ts.states.label('s2', {'p'}, check=False)
              >>> M = {'s0', 's2'}
              >>> b = ts.states.find(M, {'ap': {'p'} } )
              >>> states = [state for (state, label_) in b]
              >>> print(set(states) )
              {'s0', 's2'}
        
        See Also
        ========
        L{label_of}, L{labeled_with}, L{label}, L{labels},
        L{LabeledTransitions.find}
        
        @param states: subset of states over which to search
        @type states: 'any' (default)
            | iterable of valid states
            | single valid state
        
        @param with_attr_dict: label with which to filter the states
        @type with_attr_dict: {sublabel_type : desired_sublabel_value, ...}
            | leave empty, to allow any state label (default)
        
        @param with_attr: label key-value pairs which take
            precedence over C{with_attr_dict}.
        
        @rtype: list of labeled states
        @return: [(C{state}, C{label}),...]
            where:
                - C{state} \\in C{states}
                - C{label}: dict
        """
        if with_attr_dict is None:
            with_attr_dict = with_attr
        else:
            try:
                with_attr_dict.update(with_attr)
            except AttributeError:
                raise Exception('with_attr_dict must be a dict')
        
        if states is not 'any':
            # singleton check
            if states in self:
                state = states
                msg = 'LabeledStates.find got single state: ' +str(state) +'\n'
                msg += 'instead of Iterable of states.\n'
                states = [state]
                msg += 'Replaced given states = ' +str(state)
                msg += ' with states = ' +str(states)
                logger.debug(msg)
        
        found_state_label_pairs = []
        for state, attr_dict in self.graph.nodes_iter(data=True):
            logger.debug('Checking state_id = ' +str(state) +
                          ', with attr_dict = ' +str(attr_dict) )
            
            if states is not 'any':
                if state not in states:
                    logger.debug('state_id = ' +str(state) +', not desired.')
                    continue
            
            msg = 'Checking state label:\n\t attr_dict = '
            msg += str(attr_dict)
            msg += '\n vs:\n\t desired_label = ' + str(with_attr_dict)
            logger.debug(msg)
            
            if not with_attr_dict:
                logger.debug('Any label acceptable.')
                ok = True
            else:
                ok = label_is_desired(attr_dict, with_attr_dict)
            
            if ok:
                logger.debug('Label Matched:\n\t' +str(attr_dict) +
                              ' == ' +str(with_attr_dict) )
                state_label_pair = (state, dict(attr_dict))
                
                found_state_label_pairs.append(state_label_pair)
            else:
                logger.debug('No match for label---> state discarded.')
        
        return found_state_label_pairs
        
        #except KeyError:
        #warnings.warn("State: " +str(state) +", doesn't have AP label.")
        #return None
    
    def is_terminal(self, state):
        """Check if state has no outgoing transitions.
        
        See Also
        ========
        Def. 2.4, p.23 U{[BK08]
        <http://tulip-control.sourceforge.net/doc/bibliography.html#bk08>}
        """
        successors = self.post(state)
        if successors:
            return False
        else:
            return True
    
    def is_blocking(self, state):
        """Check if state has outgoing transitions for each label.
        """
        raise NotImplementedError

class Transitions(object):
    """Methods for handling labeled transitions.
    
    Note that a directed edge is an ordered set of nodes.
    Unlike an edge, a transition is a labeled edge.
    """
    def __init__(self, graph, deterministic=False):
        self.graph = graph
        self._deterministic = deterministic
    
    def __call__(self, **kwargs):
        """Return list of transitions.
        
        Wraps L{LabeledDiGraph.edges}.
        """
        return self.graph.edges(**kwargs)
    
    def __str__(self):
        return 'Transitions:\n' +pformat(self() )
    
    def __len__(self):
        """Count transitions."""
        return self.graph.number_of_edges()
    
    def _breaks_determinism(self, from_state, sublabels):
        """Return True if adding transition conserves determinism.
        """
        if not self._deterministic:
            return
        
        if not from_state in self.graph.states:
            raise Exception('from_state \notin graph')
        
        same_labeled = self.find([from_state], with_attr_dict=sublabels)        
        
        if same_labeled:
            msg = 'Candidate transition violates determinism.\n'
            msg += 'Existing transitions with same label:\n'
            msg += str(same_labeled)
            raise Exception(msg)
    
    def add(self, from_state, to_state, attr_dict=None, check=True, **attr):
        """Wrapper for L{LabeledDiGraph.add_edge}.
        """
        #self._breaks_determinism(from_state, labels)
        self.graph.add_edge(from_state, to_state, attr_dict, check, **attr)
    
    def add_from(self, from_states, to_states, check_states=True):
        """Add non-deterministic transition.
        
        No labeling at this level of structuring.
                
        L{LabeledTransitions.label}, L{LabeledTransitions.relabel},
        L{LabeledTransitions.add_labeled} manipulate labeled
        transitions.
        
        They become available only if set of actions, or an alphabet
        are defined, so can be used only in FTS, open FTS, automaton, etc.
        """
        if not check_states:
            self.graph.states.add_from(from_states)
            self.graph.states.add_from(to_states)
        
        if not is_subset(from_states, self.graph.states() ):
            raise Exception('from_states \\not\\subseteq states.')
        
        if not is_subset(to_states, self.graph.states() ):
            raise Exception('to_states \\not\\subseteq states.')
        
        for from_state in from_states:
            for to_state in to_states:
                self.graph.add_edge(from_state, to_state)
    
    def remove(self, from_state, to_state, attr_dict=None, **attr):
        """Remove single transition.
        
        If only the states are passed,
        then all transitions between them are removed.
        
        If C{attr_dict}, C{attr} are also passed,
        then only transitions annotated with those labels are removed.
        
        Wraps L{LabeledDiGraph.remove_labeled_edge}.
        """
        self.graph.remove_labeled_edge(from_state, to_state, attr_dict, **attr)
    
    def remove_from(self, transitions):
        """Remove list of transitions.
        
        Each transition is either a:
        
          - 2-tuple: (u, v), or a
          - 3-tuple: (u, v, data)
        """
        self.graph.remove_labeled_edges(transitions)
    
    def add_adj(self, adj, adj2states):
        """Add multiple transitions from adjacency matrix.
        
        These transitions are not labeled.
        To label then, use either L{LabeledTransitions.relabel},
        or L{remove} and then L{LabeledTransitions.add_labeled_adj}.

        See Also
        ========
        L{States.add}, L{States.add_from},
        L{LabeledTransitions.add_labeled_adj}
        
        @param adj: new transitions, represented by the
            non-zero elements of an adjacency matrix.
            Note that adjacency here is in the sense of nodes
            and not spatial.
        @type adj: scipy.sparse.lil (list of lists)
        
        @param adj2states: correspondence between adjacency matrix
            nodes and existing states.
            
            For example the 1st state in adj2states corresponds to
            the first node in C{adj}.
            
            States must have been added using:
            
                - sys.states.add, or
                - sys.states.add_from
                
            If adj2states includes a state not in sys.states,
            no transition is added and an exception raised.
        @type adj2states: list of valid states
        """
        # square ?
        if adj.shape[0] != adj.shape[1]:
            raise Exception('Adjacency matrix must be square.')
        
        # check states exist, before adding any transitions
        for state in adj2states:
            if state not in self.graph.states:
                raise Exception(
                    'State: ' +str(state) +' not found.'
                    ' Consider adding it with sys.states.add'
                )
        
        # convert to format friendly for edge iteration
        nx_adj = nx.from_scipy_sparse_matrix(
            adj, create_using=nx.DiGraph()
        )
        
        # add each edge using existing checks
        for edge in nx_adj.edges_iter():
            (from_idx, to_idx) = edge
            
            from_state = adj2states[from_idx]
            to_state = adj2states[to_idx]
            
            self.add(from_state, to_state)
    
    def add_labeled_adj(
            self, adj, adj2states,
            labels, check_labels=True
        ):
        """Add multiple transitions from adjacency matrix.
        
        These transitions are enabled when the given guard is active.

        See Also
        ========
        L{add_labeled}, L{Transitions.add_adj}
        
        @param adj: new transitions represented by adjacency matrix.
        @type adj: scipy.sparse.lil (list of lists)
        
        @param adj2states: correspondence between adjacency matrix
            nodes and existing states.
            
            For example the 1st state in adj2states corresponds to
            the first node in C{adj}.
            
            States must have been added using:
            
                - sys.states.add, or
                - sys.states.add_from
                
            If adj2states includes a state not in sys.states,
            no transition is added and an exception raised.
        @type adj2states: list of valid states
        
        @param labels: combination of labels with which to annotate each of
            the new transitions created from matrix adj.
            Each label value must be already in one of the
            transition labeling sets.
        @type labels: tuple of valid transition labels
        
        @param check_labels: check validity of labels,
            or just add them as new
        @type check_labels: bool
        """
        # square ?
        if adj.shape[0] != adj.shape[1]:
            raise Exception('Adjacency matrix must be square.')
        
        # check states exist, before adding any transitions
        for state in adj2states:
            if state not in self.graph.states:
                raise Exception(
                    'State: ' +str(state) +' not found.'
                    ' Consider adding it with sys.states.add'
                )
        
        # convert to format friendly for edge iteration
        nx_adj = nx.from_scipy_sparse_matrix(
            adj, create_using=nx.DiGraph()
        )
        
        # add each edge using existing checks
        for edge in nx_adj.edges_iter():
            (from_idx, to_idx) = edge
            
            from_state = adj2states[from_idx]
            to_state = adj2states[to_idx]
            
            self.add_labeled(from_state, to_state, labels,
                             check=check_labels)
        
        # TODO add overwriting (=delete_labeled +add once more) capability
    
    def find(self, from_states='any', to_states='any',
             with_attr_dict=None, typed_only=False, **with_attr):
        """Find all edges from_state to_states, annotated with given label.
        
        Instead of having two separate methods to:
            - find all labels of edges between given states (s1, s2)
            - find all transitions (s1, s2, L) with given label L,
                possibly from some given state s1,
                i.e., the edges leading to the successor states
                Post(s1, a) = Post(s1) restricted by action a
        this method provides both functionalities,
        attempting to reduce duplication of effort by the user.
        
        Preimage under edge labeling function L of given label,
        intersected with given subset of edges::
            L^{-1}(desired_label) \\cap (from_states x to_states)
        
        Note
        ====
          -  L{__call__}

          - If called with C{from_states} = all states,
            then the labels annotating returned edges are those which
            appear at least once as edge annotations.
            This may not be the set of all possible
            labels, in case there valid but yet unused edge labels.

          - find could have been named ".from...", but it would
            elongate its name w/o adding information. Since you search
            for transitions, there are underlying states and this
            function naturally provides the option to restrict those
            states to a subset of the possible ones.
        
        See Also
        ========
        L{label}, L{relabel}, L{add_labeled}, L{add_labeled_adj}, L{__call__}
        
        @param from_states: subset of states from which transition must start
        @type from_states: 'any' (default)
            | iterable of valid states
            | single valid state
        
        @param to_states: set of states to which the transitions must lead
        @type to_states: 'any' (default)
            | iterable of valid states
            | single valid state
        
        @param with_attr_dict: label with which to filter the transitions
        @type with_attr_dict: {sublabel_type : desired_sublabel_value, ...}
            | leave empty, to allow any label (default)
        
        @param with_attr: label type-value pairs,
            take precedence over C{desired_label}.
        
        @return: set of transitions = labeled edges::
                (C{from_state}, C{to_state}, label)
            such that::
                (C{from_state}, C{to_state} )
                \\in C{from_states} x C{to_states}
                
        @rtype: list of transitions::
                = list of labeled edges
                = [(C{from_state}, C{to_state}, C{label}),...]
            where:
                - C{from_state} \\in C{from_states}
                - C{to_state} \\in C{to_states}
                - C{label}: dict
        """
        if with_attr_dict is None:
            with_attr_dict = with_attr 
        try:
            with_attr_dict.update(with_attr)
        except:
            raise TypeError('with_attr_dict must be a dict')
        
        found_transitions = []
        
        if from_states is 'any':
            from_states = self.graph.nodes()
        
        for from_state, to_state, attr_dict in self.graph.edges_iter(
            from_states, data=True, keys=False
        ):
            if to_states is not 'any':
                if to_state not in to_states:
                    continue
            
            if not with_attr_dict:
                logger.debug('Any label is allowed.')
                ok = True
            elif not attr_dict:
                logger.debug('No guard defined.')
                ok = True
            else:
                logger.debug('Checking guard.')
                ok = label_is_desired(attr_dict, with_attr_dict)
            
            if ok:
                logger.debug('Transition label matched desired label.')
                transition = (from_state, to_state, dict(attr_dict))
                
                found_transitions.append(transition)
            
        return found_transitions

class LabeledDiGraph(nx.MultiDiGraph):
    """Directed multi-graph with constrained labeling.
    
    Provided facilities to define labeling functions on
    vertices and edges, with given co-domains,
    so that labels are type-checked, not arbitrary.
    
    Each state (or edge) is annotated with labels.
    Before removing a value from a label type,
    first make sure no state (or edge) is labeled with it.
    
    Multiple edges with the same C{attr_dict} are not possible.
    So the difference from C{networkx.MultiDigraph} is that
    the C{dict} of edges between u,v is a bijection.
    
    Between two nodes either:
    
      - a single unlabeled edge exists (no labeling constraints), or
      - labeled edges exist
      
    but mixing labeled with unlabeled edges for the same
    edge is not allowed, to simplifiy and avoid confusion.
    
    Example
    =======
    The action taken when traversing an edge.
    Each edge is annotated by a single action.
    If an edge (s1, s2) can be taken on two transitions,
    then 2 copies of that same edge are stored.
    Each copy is annotated using a different action,
    the actions must belong to the same action set.
    That action set is defined as a ser instance.
    This description is a (closed) L{FTS}.
    
    The system and environment actions associated with an edge
    of a reactive system. To store these, 2 sub-labels are used
    and their sets are encapsulated within the same (open) L{FTS}.
    
    Example
    =======
    Initialize and define one label type called C{'fruits'}.
    This also creates a field C{g.fruits}.
    
    >>> from tulip.transys import LabeledDiGraph
    >>> node_label_types = [('fruits', set(), True)]
    >>> g = LabeledDiGraph(node_label_types=node_label_types)
    
    Add some value to the codomain of type C{'fruits'}.
    
    >>> g.fruits |= ['apple', 'lemon']
    
    The label key 'day' will be untyped,
    so the class accepts 'Jan', though incorrect.
    
    >>> g.add_nodes_from([(1, {'fruit':'apple'}), \
                          (2, {'fruit':'lemon', 'day':'Jan'})])
    
    Note
    ====
    1. Edge labeling implies the "multi",
       so it is omitted from the class name.
    
    2. From its name, C{networkx} targets networks, so vertices
       are called nodes. Here a general graph-theoretic
       viewpoint is more appropriate, so vertices would be
       preferred. But to maintain uniform terminology,
       the term node is used (plus it is shorter to write).
    
    3. Label ordering will be disabled (i.e., internal use of OrderedDict),
       because it is unreliable, so user will be encouraged to use dicts,
       or key-value pairs. For figure saving purposes it will be
       possible to define an order independently, as an export filter.
    
    Credits
    =======
    Some code in overriden methods of networkx.MultiDiGraph
    is adapted from networkx, which is distributed under the BSD license.
    """
    def __init__(
        self,
        node_label_types=None,
        edge_label_types=None,
        max_outdegree=None,
        **kwargs
    ):
        """Initialize the types of labelings on states and edges.
        
        @param node_label_types: defines the state labeling functions:
            
                L_i : V -> D_i
            
            each from vertices C{V} to some co-domain C{D_i}.
            
            Each labeling function is defined by
            a tuple C{(L_i, D_i, setter)}:
            
                - C{L_i} is a C{str} naming the labeling function.
                
                - C{D_i} implements C{__contains__}
                    to enable checking label validity.
                    If you want co-domain C{D_i} to be extensible,
                    it must implement C{add}.
                
                - C{setter}: 3 cases:
                    
                    - if 2-tuple C{(L_i, D_i)} provided,
                      then no C{setter} attributes created
                    
                    - if C{setter} is C{True},
                      then an attribute C{self.L_i} is created
                      pointing at the given co-domain C{D_i}
                    
                    - Otherwise an attribute C{self.Li}
                      is created pointing at the given C{setter}.
            
            Be careful to avoid name conflicts with existing
            networkx C{MultiDiGraph} attributes.
        @type state_label_types: C{[(L_i, D_i, setter), ...]}
        
        @param edge_label_types: labeling functions for edges,
            defined similarly to C{state_label_types}.
        
        @param max_outdegree: upper bound on the outdegree of each node.
            Labels are ignored while counting edges,
            so edges with different labels count as two edges.
        @type max_outdegree: int
        """
        #todo
        """
        @param max_outdegree_per_label: like C{max_outdegree},
            but outgoing edges are counted separately for each
            labeling function.
        @type max_outdegree_per_label: int
        """
        self._state_label_def = self._init_labeling(node_label_types)
        self._transition_label_def = self._init_labeling(edge_label_types)
            
        # temporary hack until rename
        self._node_label_types = self._state_label_def
        self._edge_label_types = self._transition_label_def
        
        nx.MultiDiGraph.__init__(self, **kwargs)
        
        self.states = States(self)
        
        #todo: handle accepting states separately
        if max_outdegree == 1:
            deterministic = True
        else:
            deterministic = False
        self.transitions = Transitions(self, deterministic)
        
        # export properties
        self.dot_node_shape = {'normal':'circle'}
        self.default_export_path = './'
        self.default_export_fname = 'out'
        self.default_layout = 'dot'
        
    def _init_labeling(self, label_types):
        """
        Note
        ====
        'state' will be renamed to 'node' in the future
        'transition' will be renamed to 'edge' in the future
        
        @type domain: 'state' | 'transition'
        
        @param label_types: see L{__init__}.
        """
        labeling = dict()
        
        if label_types is None:
            logger.debug('no label types passed')
            return labeling
        
        if not label_types:
            logger.debug('label types absent (given: ' +
                         str(label_types) + ')')
            return labeling
        
        label_types = list(label_types)
        
        for label_type in label_types:
            if len(label_type) == 2:
                type_name, codomain = label_type
                setter = None
            elif len(label_type) == 3:
                type_name, codomain, setter = label_type
            else:
                msg = 'label_type can be 2 or 3-tuple, '
                msg += 'got instead:\n\t' + str(label_type)
                raise ValueError(msg)
            
            labeling[type_name] = codomain
            
            if setter is None:
                # don't create attribute unless told to do so
                pass
            elif setter is True:
                # create attribute, point it directly to D_i
                setattr(self, type_name, labeling[type_name])
            else:
                # custom setter
                setattr(self, type_name, setter)
        return labeling
    
    def _check_for_untyped_keys(self, typed_attr, type_defs, check):
        untyped_keys = set(typed_attr).difference(type_defs)
        
        msg = 'checking for untyped keys...\n'
        msg += 'attribute dict: ' + str(typed_attr) + '\n'
        msg += 'type definitions: ' + str(type_defs) + '\n'
        msg += 'untyped_keys: ' + str(untyped_keys)
        logger.debug(msg)
        
        if untyped_keys:
            msg = 'Given untyped edge attributes:\n\t' +\
                  str({k:typed_attr[k] for k in untyped_keys}) +'\n\t'
            if check:
                msg += '\nTo allow untyped annotation, pass: check = False'
                raise AttributeError(msg)
            else:
                msg += 'Allowed because you passed: check = True'
                logger.warning(msg)
        else:
            logger.debug('no untyped keys.')
    
    def _update_attr_dict_with_attr(attr_dict, attr):
        if attr_dict is None:
            attr_dict = attr
        else:
            try:
                attr_dict.update(attr)
            except AttributeError:
                msg = 'The attr_dict argument must be a dictionary.'
                raise nx.NetworkXError(msg)
        return attr_dict
    
    def add_node(self, n, attr_dict=None, check=True, **attr):
        """Use a L{TypedDict} as attribute dict.
        
        Log warning if node already exists.
        
        All other functionality remains the same.
        
        @param check: if True and untyped keys are passed,
            then raise C{AttributeError}.
        """
        # avoid multiple additions
        if n in self:
            logger.warn('Graph alreay has node: ' + str(n))
        
        attr_dict = self._update_attr_dict_with_attr(attr_dict, attr)
        
        typed_attr = TypedDict()
        typed_attr.set_types(self._node_label_types)
        typed_attr.update(attr_dict) # type checking happens here
        
        logger.debug('node typed_attr: ' + str(typed_attr))
        
        self._check_for_untyped_keys(typed_attr, self._node_label_types, check)
        
        nx.MultiDiGraph.add_node(self, n, attr_dict=typed_attr)
    
    def add_nodes_from(self, nodes, check=True, **attr):
        """Create or label multiple nodes.
        
        For details see L{add_node} and
        C{networkx.MultiDiGraph.add_nodes_from}
        """
        for n in nodes:
            try:
                n not in self.succ
                node = n
                attr_dict = attr
            except TypeError:
                node, ndict = n
                attr_dict = attr.copy()
                attr_dict.update(ndict)
            
            self.add_node(node, attr_dict=attr_dict, check=check)
    
    def add_edge(self, u, v, attr_dict=None, check=True, **attr):
        """Use a L{TypedDict} as attribute dict.
        
          - Raise ValueError if C{u} or C{v} are not already nodes.
          - Raise Exception if edge (u, v, {}).
          - Log warning if edge (u, v, attr_dict) exists.
          - Raise ValueError if C{attr_dict} contains typed key with invalid value.
          - Raise AttributeError if C{attr_dict} contains untyped keys,
            unless C{check=True}.
        
        Each label defines a different labeled edge.
        So to "change" the label, either:
        
            - remove the edge with this label, then add a new one, or
            - find the edge key, then use subscript notation:
                
                C{G[i][j][key]['attr_name'] = attr_value}
        
        For more details see L{networkx.MultiDiGraph.add_edge}.
        
        Notes
        =====
        1. Argument C{key} has been removed compared to
           L{networkx.MutliDigraph.add_edge}, because edges are defined
           by their labeling, i.e., multiple edges with same labeling
           are not allowed.
        
        @param check: raise C{AttributeError} if C{attr_dict}
            has untyped attribute keys, otherwise warn
        """
        #TODO: (from_state_id, to_state_id) = self._mutant2int(from_state, to_state)
        
        # legacy
        if 'check_states' in attr:
            msg = 'saw keyword argument: check_states ' +\
                  'which is no longer available, ' +\
                  'firstly add the new nodes.'
            logger.warning(msg)
        
        # check nodes exist
        if u not in self.succ:
            raise ValueError('Graph does not have node u: ' + str(u))
        if v not in self.succ:
            raise ValueError('Graph does not have node v: ' + str(v))
        
        attr_dict = self._update_attr_dict_with_attr(attr_dict, attr)
        
        typed_attr = TypedDict()
        typed_attr.set_types(self._edge_label_types)
        typed_attr.update(attr_dict) # type checking happens here
        
        logger.debug('Given: attr_dict = ' + str(attr_dict))
        logger.debug('Stored in: typed_attr = ' + str(typed_attr))
        
        # may be possible to speedup using .succ
        existing_u_v = self.get_edge_data(u, v, default={})
        
        if dict() in existing_u_v.values():
            msg = 'Unlabeled transition: '
            msg += 'from_state-> to_state already exists,\n'
            msg += 'where:\t from_state = ' +str(u) +'\n'
            msg += 'and:\t to_state = ' +str(v) +'\n'
            raise Exception(msg)
        
        # check if same labeled transition exists
        if attr_dict in existing_u_v.values():
            msg = 'Same labeled transition:\n'
            msg += 'from_state---[label]---> to_state\n'
            msg += 'already exists, where:\n'
            msg += '\t from_state = ' +str(u) +'\n'
            msg += '\t to_state = ' +str(v) +'\n'
            msg += '\t label = ' +str(typed_attr) +'\n'
            warnings.warn(msg)
            logger.warning(msg)
            return
        
        #self._breaks_determinism(from_state, labels)
        
        self._check_for_untyped_keys(typed_attr,
                                     self._edge_label_types,
                                     check)
        
        # the only change from nx in this clause is using TypedDict
        logger.debug('adding edge: ' + str(u) + ' ---> ' + str(v))
        if v in self.succ[u]:
            msg = 'there already exist directed edges with ' +\
                  'same end-points'
            logger.debug(msg)
            
            keydict = self.adj[u][v]
            # find a unique integer key
            key = len(keydict)
            while key in keydict:
                key-=1
            
            datadict = keydict.get(key, typed_attr)
            datadict.update(typed_attr)
            
            keydict[key] = datadict
        else:
            logger.debug('first directed edge between these nodes')
            # selfloops work this way without special treatment
            key = 0
            keydict = {key:typed_attr}
            self.succ[u][v] = keydict
            self.pred[v][u] = keydict
    
    def remove_labeled_edge(self, u, v, attr_dict=None, **attr):
        """Remove single labeled edge.
        
        @param: attr_dict 
        """
        attr_dict = self._update_attr_dict_with_attr(attr_dict, attr)
        
        rm_keys = {key for key, data in self[u][v]
                   if data == attr_dict}
        for key in rm_keys:
            self.remove_edge(u, v, key=key)
    
    def remove_labeled_edges_from(self, labeled_ebunch):
        """Remove labeled edges.
        
        Example
        =======
        >>> g = LabeledDiGraph()
        >>> g.add_edge(1, 2, day='Mon')
        >>> g.add_edge(1, 2, day='Tue')
        >>> edges = [(1, 2, {'day':'Mon'}), \
                     (1, 2, {'day':'Tue'})]
        >>> g.remove_edges_from(edges)
        
        @param labeled_ebunch: iterable container of edge tuples
            Each edge tuple can be:
            
              - 2-tuple: (u, v) All edges between u and v are removed.
              - 3-tuple: (u, v, attr_dict) all edges between u and v
                  annotated with that C{attr_dict} are removed.
        """
        for u, v, attr_dict in labeled_ebunch:
            self.remove_edge(u, v, **attr_dict)
    
    def dot_str(self, wrap=10):
        """Return dot string.
        
        Requires pydot.        
        """
        return graph2dot.graph2dot_str(self, wrap)
    
    def save(self, filename='default', fileformat=None,
             add_missing_extension=True, rankdir='LR', prog=None,
             wrap=10):
        """Save image to file.
        
        Recommended: pdf, html, svg (can render LaTeX labels with inkscape export)
        
        Caution
        =======
        rankdir experimental argument
        
        Depends
        =======
        dot, pydot
        
        See Also
        ========
        L{plot}, pydot.Dot.write
        
        @param fileformat: type of image file
        @type fileformat: str = 'dot' | 'pdf'| 'png'| 'svg' | 'gif' | 'eps' 
            (for more, see pydot.write)
            | 'scxml' | 'html' (using d3.js for animation)
        
        @param filename: path to image
            (extension C{.fileformat} appened if missing and
            C{add_missing_extension==True} )
            Default:
            
              - If C{self.name} is not set and no C{path} given,
                  then use C{self.default_export_fname} prepended with
                  C{self.default_export_fname}.
              - If C{self.name} is set, but no C{path} given,
                  then use C{self.name} prepended with
                  C{self.default_export_fname}.
              - If C{path} is given, use that.
        @type filename: str
        
        @param add_missing_extension: if extension C{.fileformat} missing,
            it is appended
        @type add_missing_extension: bool
        
        @param rankdir: direction for dot layout
        @type rankdir: str = 'TB' | 'LR'
            (i.e., Top->Bottom | Left->Right)
        
        @param prog: executable to call
        @type prog: dot | circo | ... see pydot.Dot.write

        @rtype: bool
        @return: True if saving completed successfully, False otherwise.
        """
        if fileformat is None:
            fname, fextension = os.path.splitext(filename)
            if not fextension:
                fextension = '.pdf'
            fileformat = fextension[1:]
        
        path = self._export_fname(filename, fileformat,
                                  addext=add_missing_extension)
        
        if fileformat is 'html':
            return save_d3.labeled_digraph2d3(self, path)
        
        # subclass has extra export formats ?
        if hasattr(self, '_save'):
            if self._save(path, fileformat):
                return True
        
        if prog is None:
            prog = self.default_layout
        
        graph2dot.save_dot(self, path, fileformat, rankdir, prog, wrap)
        
        return True
    
    def _add_missing_extension(self, path, file_type):
        filename, file_extension = os.path.splitext(path)
        desired_extension = os.path.extsep +file_type
        if file_extension != desired_extension:
            path = filename +desired_extension
        return path
    
    def _export_fname(self, path, file_type, addext):
        if path == 'default':
            if self.name == '':
                path = self.default_export_path +self.default_export_fname
            else:
                path = self.default_export_path +self.name
        
        if addext:
            path = self._add_missing_extension(path, file_type)
        
        return path
    
    def plot(self, rankdir='LR', prog=None, wrap=10, ax=None):
        """Plot image using dot.
        
        No file I/O involved.
        Requires GraphViz dot and either Matplotlib or IPython.
        
        NetworkX does not yet support plotting multiple edges between 2 nodes.
        This method fixes that issue, so users don't need to look at files
        in a separate viewer during development.
        
        See Also
        ========
        L{save}
        
        Depends
        =======
        dot and either of IPython or Matplotlib
        """
        # anything to plot ?
        if not self.states:
            print(60*'!'+"\nThe system doesn't have any states to plot.\n"+60*'!')
            return
        
        if prog is None:
            prog = self.default_layout
        
        return graph2dot.plot_pydot(self, prog, rankdir, wrap, ax=ax)

class LabeledStateDiGraph(nx.MultiDiGraph):
    """Species: System & Automaton.
    
    For dot export subclasses must define:
        
        - _state_label_def
        - _state_dot_label_format
        
        - _transition_label_def
        - _transition_dot_label_format
        - _transition_dot_mask
    
    Note: this interface will be improved in the future.
    """
    def __init__(
            self, name='', mutable=False,
            deterministic=False,
            accepting_states_type=None,
        ):
        """Initialize labeled digraph.
        
        @param name: system name, default for save and __str__
        @type name: str
        
        @param mutable: if C{True}, then mutable C{states} allowed
        @type mutable: bool
        
        @param deterministic: if C{True}, then each transition
            added is checked to maintain edge-label-determinism
        @type deterministic: bool
        
        @param accepting_states_type:
            accepting states use this class,
            f C{None}, then no accepting states initialized
        @type accepting_states_type: class definition
        """
        nx.MultiDiGraph.__init__(self, name=name)
        
        self.states = States(self)
        self.transitions = Transitions(self, deterministic)

        self.dot_node_shape = {'normal':'circle'}
        self.default_export_path = './'
        self.default_export_fname = 'out'
        self.default_layout = 'dot'
    
    def copy(self):
        return copy.deepcopy(self)
    
    def _multiply_mutable_states(self, other, prod_graph, prod_sys):
        def prod_ids2states(prod_state_id, self, other):
            (idx1, idx2) = prod_state_id
            state1 = self.states._int2mutant(idx1)
            state2 = other.states._int2mutant(idx2)
            prod_state = (state1, state2)
            
            return prod_state
        
        def label_union(nx_label):
            (v1, v2) = nx_label
            
            if v1 is None or v2 is None:
                raise Exception(
                    'At least one factor has unlabeled state, '+
                    "or the state sublabel types don't match."
                )
            
            try:
                return v1 | v2
            except:
                pass
            
            try:
                return v2 +v2
            except:
                raise TypeError(
                    'The state sublabel types should support ' +
                    'either | or + for labeled system products.'
                )
        
        def state_label_union(attr_dict):
            prod_attr_dict = dict()
            for k,v in attr_dict.iteritems():
                prod_attr_dict[k] = label_union(v)
            return prod_attr_dict
        
        # union of state labels from the networkx tuples
        for prod_state_id, attr_dict in prod_graph.nodes_iter(data=True):
            prod_attr_dict = state_label_union(attr_dict)
            prod_state = prod_ids2states(prod_state_id, self, other)
            
            prod_sys.states.add(prod_state)
            prod_sys.states.label(prod_state, prod_attr_dict)
        print(prod_sys.states)
        
        # prod of initial states
        inits1 = self.states.initial
        inits2 = other.states.initial
        
        prod_init = []
        for (init1, init2) in zip(inits1, inits2):
            new_init = (init1, init2)
            prod_init.append(new_init)
        prod_sys.states.initial |= prod_init
        
        """
        # multiply mutable states (only the reachable added)
        if self.states.mutants or other.states.mutants:
            for idx, prod_state_id in enumerate(prod_graph.nodes_iter() ):
                prod_state = prod_ids2states(prod_state_id, self, other)
                prod_sys.states.mutants[idx] = prod_state
            
            prod_sys.states.min_free_id = idx +1
        # no else needed: otherwise self already not mutant
        """
        
        # action labeling is taken care by nx,
        # since transition taken at a time
        for from_state_id, to_state_id, edge_dict in \
        prod_graph.edges_iter(data=True):            
            from_state = prod_ids2states(from_state_id, self, other)
            to_state = prod_ids2states(to_state_id, self, other)
            
            prod_sys.transitions.add_labeled(
                from_state, to_state, edge_dict
            )
        return prod_sys
    
    # binary operators (for magic binary operators: see above)
    def tensor_product(self, other, prod_sys=None):
        """Return strong product with given graph.
        
        Reference
        =========
        http://en.wikipedia.org/wiki/Strong_product_of_graphs
        nx.algorithms.operators.product.strong_product
        """
        prod_graph = nx.product.tensor_product(self, other)
        
        # not populating ?
        if prod_sys is None:
            if self.states.mutants or other.states.mutants:
                mutable = True
            else:
                mutable = False
            prod_sys = LabeledStateDiGraph(mutable=mutable)
        
        prod_sys = self._multiply_mutable_states(
            other, prod_graph, prod_sys
        )
        
        return prod_sys
        
    def cartesian_product(self, other, prod_sys=None):
        """Return Cartesian product with given graph.
        
        If u,v are nodes in C{self} and z,w nodes in C{other},
        then ((u,v), (z,w) ) is an edge in the Cartesian product of
        self with other if and only if:
            - (u == v) and (z,w) is an edge of C{other}
            OR
            - (u,v) is an edge in C{self} and (z == w)
            
        In system-theoretic terms, the Cartesian product
        is the interleaving where at each step,
        only one system/process/player makes a move/executes.
        
        So it is a type of parallel system.
        
        This is an important distinction with the C{strong_product},
        because that includes "diagonal" transitions, i.e., two
        processes executing truly concurrently.
        
        Note that a Cartesian interleaving is different from a
        strong interleaving, because the latter can skip states
        and transition directly along the diagonal.
        
        For a model of computation, strong interleaving
        would accurately model the existence of multiple cores,
        not just multiple processes executing on a single core.
        
        References
        ==========
          - U{http://en.wikipedia.org/wiki/Cartesian_product_of_graphs}
          - networkx.algorithms.operators.product.cartesian_product
        """
        prod_graph = nx.product.cartesian_product(self, other)
        
        # not populating ?
        if prod_sys is None:
            if self.states.mutants or other.states.mutants:
                mutable = True
            else:
                mutable = False
            prod_sys = LabeledStateDiGraph(mutable=mutable)
        
        prod_sys = self._multiply_mutable_states(
            other, prod_graph, prod_sys
        )
        
        return prod_sys
    
    def strong_product(self, other):
        """Return strong product with given graph.
        
        Reference
        =========
          - U{http://en.wikipedia.org/wiki/Strong_product_of_graphs}
          - networkx.algorithms.operators.product.strong_product
        """
        raise NotImplementedError
        # An issue here is that transitions are possible both
        # in sequence and simultaneously. So the actions set
        # is the product of the factor ones and an empty action
        # should also be introduced
        
    def is_blocking(self):
        """Does each state have at least one outgoing transition ?
        
        Note that edge labels are NOT checked, i.e.,
        it is not checked whether for each state and each possible symbol/letter
        in the input alphabet, there exists at least one transition.
        
        The reason is that edge labels do not have any semantics at this level,
        so they are not yet regarded as guards.
        For more semantics, use a L{FiniteStateMachine}.
        """
        for state in self.states():
            if self.states.is_terminal(state):
                return True
        return False

def str2singleton(ap_label):
        """If string, convert to set(string).
        
        Convention: singleton str {'*'}
        can be passed as str '*' instead.
        """
        if isinstance(ap_label, str):
            logger.debug('Saw str state label:\n\t' +str(ap_label))
            ap_label = {ap_label}
            logger.debug('Replaced with singleton:\n\t' +str(ap_label) +'\n')
        return ap_label

def prepend_with(states, prepend_str):
    """Prepend items with given string.
    
    Example
    =======
    states = [0, 1]
    prepend_str = 's'
    states = prepend_with(states, prepend_str)
    assert(states == ['s0', 's1'] )
    
    See Also
    ========
    L{tuple2ba}, L{tuple2fts}
    
    @param states: items prepended with string C{prepend_str}
    @type states: iterable
    
    @param prepend_str: text prepended to C{states}
    @type prepend_str: str
    """
    if not isinstance(states, Iterable):
        raise TypeError('states must be Iterable. Got:\n\t' +
                        str(states) +'\ninstead.')
    if not isinstance(prepend_str, str) and prepend_str is not None:
        raise TypeError('prepend_str must be Iterable. Got:\n\t' +
                        str(prepend_str) +'\ninstead.')
    
    if prepend_str is None:
        return states
    
    return [prepend_str +str(s) for s in states]
