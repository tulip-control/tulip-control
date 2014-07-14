# Copyright (c) 2013-2014 by California Institute of Technology
# and 2014 The Regents of the University of Michigan
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
# 3. Neither the name of the copyright holder(s) nor the names of its 
#    contributors may be used to endorse or promote products derived 
#    from this software without specific prior written permission.
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

import networkx as nx

from .mathset import SubSet, TypedDict
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
        """

        @type graph: L{LabeledDiGraph}
        """
        self.graph = graph
        self.initial = []
        self.current = []
    
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
        """ Return L{SubSet} of initial states.
        """
        return self._initial
    
    @initial.setter
    def initial(self, states):
        s = SubSet(self)
        s |= states
        self._initial = s
    
    @property
    def current(self):
        """Return L{SubSet} of current states.
        
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
        if state in self:
            logger.debug('State already exists.')
    
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
        """Wraps L{LabeledDiGraph.add_node}.
        """
        self._warn_if_state_exists(new_state)
        
        logger.debug('Adding new id: ' +str(new_state) )
        self.graph.add_node(new_state, attr_dict, check, **attr)
    
    def add_from(self, new_states, check=True, **attr):
        """Wraps L{LabeledDiGraph.add_nodes_from}.
        """
        self.graph.add_nodes_from(new_states, check, **attr)
    
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
        
        Over all actions or letters, i.e., edge labeling is ignored,
        because it may be undefined. Only classes which have an action
        set, alphabet, or other transition labeling set provide a
        pre(state, label) method, as for example pre(state, action) in
        the case of closed transition systems.
        
        If multiple states provided,
        then union Post(s) for s in states provided.
        
        See Also
        ========
          - L{pre}
          - Def. 2.3, p.23 U{[BK08]
            <http://tulip-control.sourceforge.net/doc/bibliography.html#bk08>}

        @rtype: set
        """
        states = self._single_state2singleton(states)
        
        successors = set()
        for state in states:
            successors |= set(self.graph.successors(state))
        return successors
    
    def pre(self, states):
        """Return direct predecessors (1-hop) of given state.
        
        See Also
        ========
          - L{post}
          - Def. 2.3, p.23 U{[BK08]
            <http://tulip-control.sourceforge.net/doc/bibliography.html#bk08>}

        @rtype: set
        """
        states = self._single_state2singleton(states)
        
        predecessors = set()
        for state in states:
            predecessors |= set(self.graph.predecessors(state))
        
        return predecessors
    
    def forward_reachable(self, state):
        """Return states reachable from given state.
        
        Iterated post(), a wrapper of networkx.descendants.
        """
        descendants = nx.descendants(self, state)
        return descendants
    
    def backward_reachable(self, state):
        """Return states from which the given state can be reached.

        A wrapper of networkx.ancestors.
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
    
    def find(self, states=None, with_attr_dict=None, **with_attr):
        """Filter by desired states and by desired state labels.
        
        Examples
        ========
        Assume that the system is:
        
        >>> import transys as trs
        >>> ts = trs.FTS()
        >>> ts.atomic_propositions.add('p')
        >>> ts.states.add('s0', ap={'p'})
        
          - To find the label of a single state C{'s0'}:

              >>> a = ts.states.find(['s0'] )
              >>> (s0_, label) = a[0]
              >>> print(label)
              {'ap': set(['p'])}
              
          - To find all states with a specific label C{{'p'}}:

              >>> ts.states.add('s1', ap={'p'})
              >>> b = ts.states.find(with_attr_dict={'ap':{'p'} } )
              >>> states = [state for (state, label_) in b]
              >>> print(set(states) )
              {'s0', 's1'}

          - To find all states in subset C{M} labeled with C{{'p'}}:

              >>> ts.states.add('s2', ap={'p'})
              >>> M = {'s0', 's2'}
              >>> b = ts.states.find(M, {'ap': {'p'} } )
              >>> states = [state for (state, label_) in b]
              >>> print(set(states) )
              {'s0', 's2'}
        
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
        
        if states is not None:
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
            
            if states is not None:
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

        UNDER DEVELOPMENT; function signature may change without
        notice.  Calling will result in NotImplementedError.
        """
        raise NotImplementedError

class Transitions(object):
    """Methods for handling labeled transitions.
    
    Note that a directed edge is an ordered set of nodes.
    Unlike an edge, a transition is a labeled edge.
    """
    def __init__(self, graph, deterministic=False):
        """

        @type graph: L{LabeledDiGraph}
        """
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
        """Wrapper of L{LabeledDiGraph.add_edge}.
        """
        #self._breaks_determinism(from_state, labels)
        self.graph.add_edge(from_state, to_state, attr_dict, check, **attr)
    
    def add_from(self, transitions, attr_dict=None, check=True, **attr):
        """Wrapper of L{LabeledDiGraph.add_edges_from}.
        """
        self.graph.add_edges_from(transitions, attr_dict=attr_dict,
                                  check=check, **attr)
    
    def add_comb(self, from_states, to_states, attr_dict=None,
                 check=True, **attr):
        """Add an edge for each combination C{(u, v)},
        
        for C{u} in C{from_states} for C{v} in C{to_states}.
        """
        for u in from_states:
            for v in to_states:
                self.graph.add_edge(u, v, attr_dict=attr_dict,
                                    check=check, **attr)
    
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
        self.graph.remove_labeled_edges_from(transitions)
    
    def add_adj(
            self, adj, adj2states, attr_dict=None,
            check=True, **attr
        ):
        """Add multiple labeled transitions from adjacency matrix.
        
        The label can be empty.
        For more details see L{add}.
        
        @param adj: new transitions represented by adjacency matrix.
        @type adj: scipy.sparse.lil (list of lists)
        
        @param adj2states: map from adjacency matrix indices to states.
            If value not a state, raise Exception.
            Use L{States.add}, L{States.add_from} to add states first.
            
            For example the 1st state in adj2states corresponds to
            the first node in C{adj}.
            
            States must have been added using:

               - sys.states.add, or
               - sys.states.add_from

            If C{adj2states} includes a state not in sys.states,
            no transition is added and an exception raised.
        @type adj2states: list of existing states
        """
        # square ?
        if adj.shape[0] != adj.shape[1]:
            raise Exception('Adjacency matrix must be square.')
        
        # check states exist, before adding any transitions
        for state in adj2states:
            if state not in self.graph:
                raise Exception(
                    'State: ' +str(state) +' not found.'
                    ' Consider adding it with sys.states.add'
                )
        
        # convert to format friendly for edge iteration
        nx_adj = nx.from_scipy_sparse_matrix(
            adj, create_using=nx.DiGraph()
        )
        
        # add each edge using existing checks
        for i, j in nx_adj.edges_iter():
            si = adj2states[i]
            sj = adj2states[j]
            
            self.add(si, sj, attr_dict, check, **attr)
    
    def find(self, from_states=None, to_states=None,
             with_attr_dict=None, typed_only=False, **with_attr):
        """Find all edges between given states with given labels.
        
        Instead of having two separate methods to:
          
          - find all labels of edges between given states (s1, s2)
          
          - find all transitions (s1, s2, L) with given label L,
                possibly from some given state s1,
                i.e., the edges leading to the successor states
                Post(s1, a) = Post(s1) restricted by action a
        
        this method provides both functionalities.
        
        Preimage under edge labeling function L of given label,
        intersected with given subset of edges::
            L^{-1}(desired_label) \\cap (from_states x to_states)
        
        See Also
        ========
        L{add}, L{add_adj}
        
        @param from_states: edges must start from this subset of states
        @type from_states:
            - iterable of existing states, or
            - None (no constraint, default)
        
        @param to_states: edges must end in this subset of states
        @type to_states:
            - iterable of existing states, or
            - None (no constraint, default)
        
        @param with_attr_dict: edges must be annotated with these labels
        @type with_attr_dict:
            - {label_type : desired_label_value, ...}, or
            - None (no constraint, default)
        
        @param with_attr: label type-value pairs,
            take precedence over C{desired_label}.
        
        @return: set of transitions = labeled edges::
                (from_state, to_state, label)
        such that::
                (from_state, to_state )
                in from_states x to_states
                
        @rtype: list of transitions::
                = list of labeled edges
                = [(from_state, to_state, label),...]
        where:
          - C{from_state} in C{from_states}
          - C{to_state} in C{to_states}
          - C{label}: dict
        """
        if with_attr_dict is None:
            with_attr_dict = with_attr 
        try:
            with_attr_dict.update(with_attr)
        except:
            raise TypeError('with_attr_dict must be a dict')
        
        found_transitions = []
        
        u_v_edges = self.graph.edges_iter(nbunch=from_states, data=True)
        
        if to_states is not None:
            u_v_edges = [(u,v,d) for u,v,d in u_v_edges
                                 if v in to_states]
        
        for u, v, attr_dict in u_v_edges:
            ok = True
            if not with_attr_dict:
                logger.debug('Any label is allowed.')
            elif not attr_dict:
                logger.debug('No labels defined.')
            else:
                logger.debug('Checking guard.')
                ok = label_is_desired(attr_dict, with_attr_dict)
            
            if ok:
                logger.debug('Transition label matched desired label.')
                transition = (u, v, dict(attr_dict))
                
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
    So the difference from C{networkx.MultiDiGraph} is that
    the C{dict} of edges between u,v is a bijection.
    
    Between two nodes either:
    
      - a single unlabeled edge exists (no labeling constraints), or
      - labeled edges exist
      
    but mixing labeled with unlabeled edges for the same
    edge is not allowed, to simplify and avoid confusion.
    
    For dot export subclasses must define:
        
        - _state_label_def
        - _state_dot_label_format
        
        - _transition_label_def
        - _transition_dot_label_format
        - _transition_dot_mask
    
    Note: this interface will be improved in the future.
    
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
    
    >>> g.add_nodes_from([(1, {'fruit':'apple'}),
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
    Some code in overridden methods of networkx.MultiDiGraph
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
        
        @param node_label_types: defines the state labeling functions::
            
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
        @type node_label_types: C{[(L_i, D_i, setter), ...]}
        
        @param edge_label_types: labeling functions for edges,
            defined similarly to C{node_label_types}.
        
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
        self.default_layout = 'dot'
        
    def _init_labeling(self, label_types):
        """
        Note
        ====
        'state' will be renamed to 'node' in the future
        'transition' will be renamed to 'edge' in the future
        
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
            msg = 'The following edge attributes:\n' +\
                  str({k:typed_attr[k] for k in untyped_keys}) +'\n' +\
                  'are not allowed.\n' +\
                  'Currently the allowed attributes are:' +\
                  ', '.join([str(x) for x in type_defs])
            if check:
                msg += '\nTo set attributes not included '+\
                       'in the existing types, pass: check = False'
                raise AttributeError(msg)
            else:
                msg += '\nAllowed because you passed: check = True'
                logger.warning(msg)
        else:
            logger.debug('no untyped keys.')
    
    def _update_attr_dict_with_attr(self, attr_dict, attr):
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
        
        self._check_for_untyped_keys(typed_attr,
                                     self._node_label_types,
                                     check)
        
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
          - Raise Exception if edge (u, v, {}) exists.
          - Log warning if edge (u, v, attr_dict) exists.
          - Raise ValueError if C{attr_dict} contains typed key with invalid value.
          - Raise AttributeError if C{attr_dict} contains untyped keys,
            unless C{check=False}.
        
        Each label defines a different labeled edge.
        So to "change" the label, either:
        
            - remove the edge with this label, then add a new one, or
            - find the edge key, then use subscript notation:
                
                C{G[i][j][key]['attr_name'] = attr_value}
        
        For more details see C{networkx.MultiDiGraph.add_edge}.
        
        Notes
        =====
        1. Argument C{key} has been removed compared to
           C{networkx.MultiDiGraph.add_edge}, because edges are defined
           by their labeling, i.e., multiple edges with same labeling
           are not allowed.
        
        @param check: raise C{AttributeError} if C{attr_dict}
            has untyped attribute keys, otherwise warn
        """
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
    
    def add_edges_from(self, labeled_ebunch, attr_dict=None,
                       check=True, **attr):
        """Add multiple labeled edges.
        
        For details see C{networkx.MultiDiGraph.add_edges_from}.
        Only difference is that only 2 and 3-tuple edges allowed.
        Keys cannot be specified, because a bijection is maintained.
        
        @param labeled_ebunch: iterable container of:
        
            - 2-tuples: (u, v), or
            - 3-tuples: (u, v, label)
          
          See also L{remove_labeled_edges_from}.
        """
        attr_dict = self._update_attr_dict_with_attr(attr_dict, attr)
       
        # process ebunch
        for e in labeled_ebunch:
            datadict = dict(attr_dict)
            
            ne = len(e)
            if ne == 3:
                u, v, dd = e
                datadict.update(dd)
            elif ne == 2:
                u, v = e
            else:
                raise ValueError(\
                    "Edge tuple %s must be a 2- or 3-tuple ."%(e,))
            
            self.add_edge(u, v, attr_dict=datadict, check=check)
    
    def remove_labeled_edge(self, u, v, attr_dict=None, **attr):
        """Remove single labeled edge.
        
        @param attr_dict: attributes with which to identify the edge.
        @type attr_dict: dict
        
        @param attr: keyword arguments with which to update C{attr_dict}.
        """
        if u not in self:
            return
        if v not in self[u]:
            return
        
        attr_dict = self._update_attr_dict_with_attr(attr_dict, attr)
        
        rm_keys = {key for key, data in self[u][v].iteritems()
                   if data == attr_dict}
        for key in rm_keys:
            self.remove_edge(u, v, key=key)
    
    def remove_labeled_edges_from(self, labeled_ebunch, attr_dict=None, **attr):
        """Remove labeled edges.
        
        Example
        =======
        >>> g = LabeledDiGraph()
        >>> g.add_edge(1, 2, day='Mon')
        >>> g.add_edge(1, 2, day='Tue')
        >>> edges = [(1, 2, {'day':'Mon'}),
                     (1, 2, {'day':'Tue'})]
        >>> g.remove_edges_from(edges)
        
        @param labeled_ebunch: iterable container of edge tuples
            Each edge tuple can be:
            
              - 2-tuple: (u, v) All edges between u and v are removed.
              - 3-tuple: (u, v, attr_dict) all edges between u and v
                  annotated with that C{attr_dict} are removed.
        """
        attr_dict = self._update_attr_dict_with_attr(attr_dict, attr)
        
        for e in labeled_ebunch:
            datadict = dict(attr_dict)
            
            ne = len(e)
            if ne == 3:
                u, v, dd = e
                datadict.update(dd)
            elif ne == 2:
                u, v = e
            else:
                raise ValueError(\
                    "Edge tuple %s must be a 2- or 3-tuple ."%(e,))
            
            self.remove_labeled_edge(u, v, attr_dict=datadict)

    def remove_deadends(self):
        """Recursively delete nodes with no outgoing transitions.
        """
        s = {1}
        while s:
            s = {n for n in self if not self.succ[n]}
            self.states.remove_from(s)
                    
    def dot_str(self, wrap=10):
        """Return dot string.
        
        Requires pydot.        
        """
        return graph2dot.graph2dot_str(self, wrap)
    
    def save(self, filename=None, fileformat=None,
             rankdir='LR', prog=None,
             wrap=10, latex=False):
        """Save image to file.
        
        Recommended file formats:
        
            - pdf, eps
            - png, gif
            - svg (can render LaTeX labels with inkscape export)
            - dot
        
        Any other format supported by C{pydot.write} is available.
        
        Experimental:
        
            - html (uses d3.js)
            - 'scxml'
        
        Requires
        ========
        dot, pydot
        
        See Also
        ========
        L{plot}, C{pydot.Dot.write}
        
        @param filename: file path to save image to
            Default is C{self.name}, unless C{name} is empty,
            then use 'out.pdf'.
            
            If extension is missing '.pdf' is used.
        @type filename: str
        
        @param fileformat: replace the extension of C{filename}
            with this. For example::
                
                filename = 'fig.pdf'
                fileformat = 'svg'
            
            result in saving 'fig.svg'
        
        @param rankdir: direction for dot layout
        @type rankdir: str = 'TB' | 'LR'
            (i.e., Top->Bottom | Left->Right)
        
        @param prog: executable to call
        @type prog: dot | circo | ... see pydot.Dot.write
        
        @param wrap: max width of node strings
        @type wrap: int
        
        @param latex: when printing states,
            prepend underscores to numbers that follow letters,
            enclose the string is $ to create a math environment.
        
        @rtype: bool
        @return: True if saving completed successfully, False otherwise.
        """
        if filename is None:
            if not self.name:
                filename = 'out'
            else:
                filename = self.name
        
        fname, fextension = os.path.splitext(filename)
        
        # default extension
        if not fextension or fextension is '.':
            fextension = '.pdf'
        
        if fileformat:
            fextension = '.' + fileformat
        
        filename = fname + fextension
        
        # drop '.'
        fileformat = fextension[1:]    
        
        if fileformat is 'html':
            return save_d3.labeled_digraph2d3(self, filename)
        
        # subclass has extra export formats ?
        if hasattr(self, '_save'):
            if self._save(filename, fileformat):
                return True
        
        if prog is None:
            prog = self.default_layout
        
        graph2dot.save_dot(self, filename, fileformat, rankdir,
                           prog, wrap, latex)
        
        return True
    
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

class _LabeledStateDiGraph(nx.MultiDiGraph):
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
            prod_sys.states.add(prod_state, **prod_attr_dict)
        print(prod_sys.states)
        
        # prod of initial states
        inits1 = self.states.initial
        inits2 = other.states.initial
        
        prod_init = []
        for (init1, init2) in zip(inits1, inits2):
            new_init = (init1, init2)
            prod_init.append(new_init)
        prod_sys.states.initial |= prod_init
        
        # # multiply mutable states (only the reachable added)
        # if self.states.mutants or other.states.mutants:
        #     for idx, prod_state_id in enumerate(prod_graph.nodes_iter() ):
        #         prod_state = prod_ids2states(prod_state_id, self, other)
        #         prod_sys.states.mutants[idx] = prod_state
        #
        #     prod_sys.states.min_free_id = idx +1
        # # no else needed: otherwise self already not mutant
        
        # action labeling is taken care by nx,
        # since transition taken at a time
        for from_state_id, to_state_id, edge_dict in \
        prod_graph.edges_iter(data=True):            
            from_state = prod_ids2states(from_state_id, self, other)
            to_state = prod_ids2states(to_state_id, self, other)
            
            prod_sys.transitions.add(
                from_state, to_state, **edge_dict
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
            prod_sys = LabeledDiGraph(mutable=mutable)
        
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
            prod_sys = LabeledDiGraph(mutable=mutable)
        
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

        UNDER DEVELOPMENT; function signature may change without
        notice.  Calling will result in NotImplementedError.
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
    >>> states = [0, 1]
    >>> prepend_str = 's'
    >>> states = prepend_with(states, prepend_str)
    >>> assert(states == ['s0', 's1'] )
    
    See Also
    ========
    L{tuple2ba}, L{tuple2fts}
    
    @param states: items prepended with string C{prepend_str}
    @type states: iterable
    
    @param prepend_str: text prepended to C{states}.  If None, then
        C{states} is returned without modification
    @type prepend_str: str or None
    """
    if not isinstance(states, Iterable):
        raise TypeError('states must be Iterable. Got:\n\t' +
                        str(states) +'\ninstead.')
    if not isinstance(prepend_str, str) and prepend_str is not None:
        raise TypeError('prepend_str must be of type str. Got:\n\t' +
                        str(prepend_str) +'\ninstead.')
    
    if prepend_str is None:
        return states
    
    return [prepend_str +str(s) for s in states]
