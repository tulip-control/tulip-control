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

"""
incorporating code from:
    Automaton class and supporting methods (Scott Livingston)
and
    Automaton Module (TuLiP distribution v0.3c)

@author: Ioannis Filippidis
"""

"""
targets
-------
 timed automata

todo
----
    same filtering for state annot

 import from
   string/text file
   promela
   xml

 simulation
   random
   via matlab
   transducer mode

 save to
   xml, dot-> svg-> pdf

 conversions between automata types
   either internally or
   by calling external converters (e.g. ltl2dstar)
 operations between trasition systms and automata or game graphs

 dependent on other modules
   ltl2ba: uses also spec class
   
 dot export based on networkx uses pygraphviz, recently ported to python3
     https://groups.google.com/forum/#!topic/pygraphviz-discuss/mbK5voZ9-hs
"""

import networkx as nx
from scipy.sparse import lil_matrix # is this really needed ?
import warnings
import copy
from pprint import pformat
from itertools import chain, combinations
from collections import Iterable, Hashable

def powerset(iterable):
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    
    From:
        http://docs.python.org/2/library/itertools.html,
    also in:
        https://pypi.python.org/pypi/more-itertools
    """
    
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def contains_multiple(iterable):
    return len(iterable) != len(set(iterable) )

try:
    import pydot
except ImportError:
    warnings.warn('pydot package not found.\nHence dot export not unavailable.')
    # python-graph package not found. Disable dependent methods.
    pydot = None

def dprint(s):
    """Debug mode print."""
    #print(s)

class States(object):
    """Methods to manage states, initial states, current state.
        
    add, remove, count, test membership
    
    see also
    --------
    LabeledStateDiGraph
    """
    def __init__(self, graph, states=[], initial_states=[], current_state=None):
        self.graph = graph
        self.initial = set()
        self.list = list() # None when list disabled
        
        self.add_from(states)
        self.add_initial_from(initial_states)
        self.set_current(current_state)
    
    def __call__(self, data=False, listed=False):
        """Return set of states.
        
        Default: state annotation not returned.
        To obtain that use argumet data=True.
        
        @param data: include annotation dict of each state
        @param listed: return list of states (instead of set)        
        
        @returns: set of states if C{data=False}
                list of tuples of states with annotation dict if C{data=False}
        """
        if data:
            return self.graph.nodes(data=True)
        else:
            if listed:
                if self.list is None:
                    raise Exception('State ordering not maintained.')
                return self.graph.nodes(data=False)
            else:
                return set(self.graph.nodes(data=False) )
    
    def __str__(self):
        return 'States:\n\t' +pformat(self(data=False) )    
    
    def __eq__(self, other):
        return self.graph.nodes(data=False) == other
        
    def __ne__(self, other):
        return self.graph.nodes(data=False) != other
    
    def __lt__(self, other):
        return self.graph.nodes(data=False) < other
    
    def __gt__(self, other):
        return self.graph.nodes(data=False) > other
    
    def __le__(self, other):
        return self.graph.nodes(data=False) <= other
    
    def __ge__(self, other):
        return self.graph.nodes(data=False) >= other
    
    def __contains__(self, state):
        """Check if single state \\in set_of_states."""
        return self.graph.has_node(state)    
    
    def __exist_labels__(self):
        """State labeling defined ?"""
        if hasattr(self.graph, '__state_label_def__'):
            return True
        else:
            msg = 'No state labeling defined for class: '
            msg += str(type(self.graph) )
            dprint(msg)
            return False
    
    def __exist_final_states__(self):
        """Check if system has final states."""
        if not hasattr(self.graph, 'final_states'):
            warnings.warn('System does not have final states.')
            return False
        else:
            return True
    
    def __dot_str__(self, to_pydot_graph):
        """Copy nodes to given graph, with attributes for dot export."""
        
        def if_initial_add_incoming_edge(g, state, initial_states):
            if state in initial_states:
                phantom_node = 'phantominit' +str(state)
                
                g.add_node(phantom_node, label='""', shape='none')
                g.add_edge(phantom_node, state)
        
        def form_node_label(state, state_data, label_def, label_format):
            sep_label_sets = label_format['separator']
            node_dot_label = '"' +str(state) +'\\n'
            for (label_type, label_value) in state_data.iteritems():
                if label_type in label_def:
                    # label formatting
                    type_name = label_format[label_type]
                    sep_type_value = label_format['type?label']
                    
                    # avoid turning strings to lists,
                    # or non-iterables to lists
                    if isinstance(label_value, str):
                        label_str = label_value
                    elif isinstance(label_value, Iterable): # and not str
                        label_str = str(list(label_value) )
                    else:
                        label_str = str(label_value)
                    
                    node_dot_label += type_name +sep_type_value
                    node_dot_label += label_str +sep_label_sets
            node_dot_label += '"'
            
            return node_dot_label  
        
        def decide_node_shape(graph, state):
            node_shape = graph.dot_node_shape['normal']
            
            # check if final states defined
            if not hasattr(graph, 'final_states'):
                return node_shape
            
            # check for final states
            if state in graph.final_states:
                node_shape = graph.dot_node_shape['final']
                
            return node_shape
        
        # get labeling def
        
        if self.__exist_labels__():
            label_def = self.graph.__state_label_def__
            label_format = self.graph.__state_dot_label_format__
        
        for (state, state_data) in self.graph.nodes_iter(data=True):
            if_initial_add_incoming_edge(to_pydot_graph, state, self.initial)
            node_shape = decide_node_shape(self.graph, state)
            
            if self.__exist_labels__():
                node_dot_label = form_node_label(state, state_data, label_def, label_format)
            else:
                node_dot_label = str(state)
            
            # TODO replace with int to reduce size
            to_pydot_graph.add_node(state, label=node_dot_label, shape=node_shape,
                                    style='rounded')
    
    def __warn_if_state_exists__(self, state):
        if state in self:
            if self.list is not None:
                raise Exception('State exists and ordering enabled: ambiguous.')
            else:
                warnings.warn('State already exists.')
                return
    
    # states
    def add(self, new_state):
        """Create single state.
        
        C{state} can be any hashable object except None (see nx add_node below)
        
        For annotating a state with a subset of atomic propositions,
        or other (custom) annotation, use the functions provided by
        AtomicPropositions, or directly the add_node function of networkx.
        
        see also
        --------
        networkx.MultiDiGraph.add_node
        """
        self.__warn_if_state_exists__(new_state)
        self.graph.add_node(new_state)
        
        # list maintained ?
        if self.list is not None:
            self.list.append(new_state)
    
    def add_from(self, new_states, destroy_order=False):
        """Add multiple states from iterable container states.
        
        see also
        --------
        networkx.MultiDiGraph.add_nodes_from.
        """
        if not isinstance(new_states, list):
            if not isinstance(new_states, Iterable):
                raise Exception('New set of states must be iterable container.')
            
            # order currently maintained ?
            if self.list is not None:
                # no states stored ?
                if len(self.list) == 0:
                    warnings.warn("Added non-list to empty system with ordering."+
                                  "Won't remember state order from now on.")
                    self.list = None
                else:
                    # cancel existing ordering ?
                    if destroy_order:
                        warnings.warn('Added non-list of new states.'+
                                      'Existing state order forgotten.')
                        self.list = None
                    else:
                        raise Exception('Ordered states maintained.'+
                                        'Please add list of states instead.')
        
        # iteration used for comprehensible error message
        for new_state in new_states:
            self.__warn_if_state_exists__(new_state)
        
        self.graph.add_nodes_from(new_states)
        
        # list maintained ?
        if self.list is not None:
            self.list = self.list +new_states
    
    def number(self):
        """Total number of states."""
        return self.graph.number_of_nodes()
    
    def remove(self, state):
        """Remove single state."""
        self.graph.remove_node(state)
        
        # no ordering maintained ?
        if self.list is None:
            return
        
        self.list.remove(state)
    
    def remove_from(self, states):
        """Remove a list of states."""
        self.graph.remove_nodes_from(states)
        
        # no ordering maintained ?
        if self.list is None:
            return
        
        for state in states:
            self.list.remove(state)
    
    def set_current(self, state):
        """Select current state.
        
        State membership is checked.
        If state \\notin states, exception raised.
        
        None is possible.
        """
        if state is None:
            self.current = None
            return
        
        if state not in self:
            raise Exception('Current state given is not in set of states.\n'+
                            'Cannot set current state to given state.')
        
        self.current = state
    
	# initial states
    def add_initial(self, new_initial_state):
        """Add state to set of initial states.
        
        C{new_initial_state} should already be a state.
        First use states.add to include it in set of states,
        then states.add_initial.
        """
        if not new_initial_state in self.graph.nodes():
            raise Exception('New initial state \\notin States.')
        else:
            self.initial.add(new_initial_state)

    def add_initial_from(self, new_initial_states):
        """Add multiple initial states.
        
        Should already bein set of states.
        """
        new_initial_states = set(new_initial_states)     
        
        if len(new_initial_states) is 0:
            return
        
        if not new_initial_states <= set(self.graph.nodes() ):
            raise Exception('New Initial States \\notsubset States.')
        else:
            self.initial |= set(new_initial_states)
        
    def number_of_initial(self):
        """Count initial states."""
        return len(self.initial)
    
    def remove_initial(self, rm_initial_state):
        """Delete single state from set of initial states."""
        self.initial.remove(rm_initial_state)
    
    def remove_initial_from(self, rm_initial_states):
        """Delete multiple states from set of initial states."""
        self.initial = self.initial.difference(rm_initial_states)
    
    def is_initial(self, state):
        return state in self.initial
    
    def is_final(self, state):       
        """Check if state \\in final states."""
        if not self.__exist_final_states__():
            return
        
        return state in self.graph.final_states
    
    def is_accepting(self, state):
        """Alias to is_final()."""
        return self.is_final(state)
    
    def check(self):
        """Check sanity of various state sets.
        
        Checks if:
            Initial states \\subseteq states
            Current state is set
            Current state \\subseteq states
        """
        if not self.initial <= set(self.graph.nodes() ):
            warnings.warn('Ininital states \\not\\subseteq states.')
        
        if self.current is None:
            warnings.warn('Current state unset.')
            return
        
        if self.current not in self.graph.nodes():
            warnings.warn('Current state \\notin states.')
        
        print('States are ok.')
    
    def post_single(self, state):
        """Direct successors of a single state.
        
        post_single() exists to contrast with post().
        
        post() cannot guess when it is passed a single state, or multiple states.
        Reason is that a state may happen to be anything,
        so possibly something iterable.
        """
        return self.post({state} )
    
    def post(self, states, actions='all'):
        """Direct successor set (1-hop) for given states.
        
        Over all actions or letters, i.e., edge labeling ignored by states.pre,
        because it may be undefined. Only classes which have an action set,
        alphabet, or other transition labeling set provide a pre(state, label)
        method, as for example pre(state, action) in the case of closed transition
        systems.
        
        Def. 2.3, p.23 [Baier] (and similar for automata)
            Post(s)
        If multiple stats provided, then union Post(s) for s in states provided.
        """
        
        if not states <= self():
            raise Exception('Not all states given are in the set of states.')
        
        successors = set()
        for state in states:
            successors |= set(self.graph.successors(state) )
            
        if actions == 'all':
            return successors
        
        for state in successors:
            pass
    
    def pre_single(self, state):
        """Direct predecessors of single state.
        
        pre_single() exists to contrast with pre().
        
        see also
        --------
        post() vs post_single().
        """
        return self.pre({state} )
    
    def pre(self, states):
        """Predecessor set (1-hop) for given state.
        """
        if not states <= self():
            raise Exception('Not all states given are in the set of states.')
        
        predecessors = set()
        for state in states:
            predecessors |= set(self.graph.predecessors(state) )
        return predecessors
    
    def add_final(self, state):
        """Convenience for FSA.add_final_state().
        
        see also
        --------
        self.add_final_from  
        """
        if not self.__exist_final_states__():
            return
        
        self.graph.add_final_state(state)
    
    def add_final_from(self, states):
        """Convenience for FSA.add_final_states_from().
        
        see also
        --------
        self.add_final
        """
        if not self.__exist_final_states__():
            return
        
        self.graph.add_final_states_from(states)
    
    def rename(self, new_states_dict):
        """Map states in place, based on dict.
        
        input
        -----
        - C{new_states_dict}: {old_state : new_state}
        (partial allowed, i.e., projection)
        
        See also
        --------
        networkx.relabel_nodes
        """
        return nx.relabel_nodes(self.graph, new_states_dict, copy=False)

class Transitions(object):
    """Building block for managing transitions.
    
    Note that a directed edge is an ordered set of nodes.
    Unlike an edge, a transition is a labeled edge.
    However, labelings may vary, so they are defined separately and methods for
    working with labeled transitions are defined in the respective classes.
    """
    def __init__(self, graph):
        self.graph = graph
    
    def __call__(self, data=True):
        """Return list of transitions."""
        
        return self.graph.edges(data=data)
    
    def __str__(self):
        return 'Transitions:\n' +pformat(self(data=True) )
    
    def add(self, from_state, to_state, check_states=True):
        """Add unlabeled transition, checking states \\in set of states.
        
        If either state not in set of states, raise exception.
        
        Argument check_states = False can override the check.
        If check_states = False, and states not already in set of states,
        then they are added.
        """
        graph = self.graph
        
        if not check_states:
            graph.states.add_from({from_state, to_state} )
        
        if from_state not in graph.states:
            raise Exception('from_state \\notin states.')
        
        if to_state not in graph.states:
            raise Exception('to_state \\notin states.')
        
        # if another un/labeled edge already exists between these nodes,
        # then avoid duplication of edges
        if not graph.has_edge(from_state, to_state):
            graph.add_edge(from_state, to_state)
    
    def add_from(self, from_states, to_states, check_states=True):
        """Add non-deterministic transition.
        
        No labeling at this level of structuring.
                
        label(), relabel(), add_labeled() manipulate labeled transitions.
        
        They become available only if set of actions, or an alphabet are defined,
        so can be used only in FTS, open FTS, automaton, etc.
        """
        if not check_states:
            self.graph.states.add_from(from_states)
            self.graph.states.add_from(to_states)
        
        if not from_states <= self.graph.states():
            raise Exception('from_states \\not\\subseteq states.')
        
        if not to_states <= self.graph.states():
            raise Exception('to_states \\not\\subseteq states.')
        
        for from_state in from_states:
            for to_state in to_states:
                self.graph.add_edge(from_state, to_state)
    
    def number(self):
        """Count transitions."""
        return self.graph.number_of_edges()
    
    def remove(self, from_state, to_state):
        """Delete all unlabeled transitions between two given states.
        
        MultiDigraph identifies different edges between same nodes
        by an additional id. When created here, no such id is passed,
        because edge labeling is not yet used.
        
        Use instead the appropriate transition labeling function
        provided by the alphabet or action classes.
        Those identify transitions by their action or input letter labels.
        """
        edge_set = copy.copy(self.graph.get_edge_data(from_state, to_state) )
        for (edge_key, label) in edge_set.iteritems():
            if label == {}:
                self.graph.remove_edge(from_state, to_state, key=edge_key)
    
    def remove_from(self, from_states, to_states):
        """Delete all unlabeled transitions between multiple state pairs.
        
        See also remove().        
        """
        for from_state in from_states:
            for to_state in to_states:
                self.remove(from_state, to_state)

class LabeledTransitions(Transitions):
    """Superclass for open/closed FTS, FSA, FSM.
    
    In more detail, the following classes inherit from this one:
        FiniteTransitionSystem (closed)
        OpenFiniteTransitionSystem
        FiniteStateAutomaton
        FiniteStateMachine
    """
    
    def __init__(self, graph):
        Transitions.__init__(self, graph)
    
    def __exist_labels__(self):
        """Labeling defined ?"""
        if not hasattr(self.graph, '__transition_label_order__') or \
           not hasattr(self.graph, '__transition_label_def__'):
            raise Exception('No transition labeling defined for this class.')
    
    def __check_states__(self, from_state, to_state, check=True):
        """Are from_state, to_state \\in states.
        
        If check == False, then add them.
        """
        if not check:
            # attempt adding only if not already in set of states
            # to avoid ordering-related exceptions
            if from_state not in self.graph:
                self.graph.states.add(from_state)
            if to_state not in self.graph:
                self.graph.states.add(to_state)
        
        if from_state not in self.graph.states:
            msg = str(from_state) +' = from_state \\notin state'
            raise Exception(msg)
        
        if to_state not in self.graph.states:
            msg = str(to_state) +' = to_state \\notin state'
            raise Exception(msg)
    
    def __get_labeling__(self, labels, check_label=True):
        self.__exist_labels__()
        
        # get labeling def
        label_order = self.graph.__transition_label_order__
        label_def = self.graph.__transition_label_def__
        
        # single label ?
        if len(label_order) == 1:
            labels = [labels]
        
        # constuct label dict
        edge_label = dict()
        if isinstance(labels, list) or isinstance(labels, tuple):
            for i in range(len(label_order) ):
                cur_name = label_order[i]
                cur_label = labels[i]
                
                edge_label[cur_name] = cur_label
        elif isinstance(labels, dict):
            edge_label = labels
        else:
            raise Exception('Bug')
        
        # check if dict is consistent with label defs
        for (typename, label) in edge_label.iteritems():
            possible_labels = label_def[typename]
            if not check_label:
                possible_labels.add(label)
            elif label not in possible_labels:
                msg = 'Given label:\n\t' +str(label) +'\n'
                msg += 'not in set of transition labels:\n\t' +str(possible_labels)
                raise Exception(msg)
                
        return edge_label
        
    def __dot_str__(self, to_pydot_graph):
        """Return label for dot export.
        """        
        def form_edge_label(edge_data, label_def, label_format):
            edge_dot_label = '"'
            sep_label_sets = label_format['separator']
            for (label_type, label_value) in edge_data.iteritems():
                if label_type in label_def:
                    # label formatting
                    type_name = label_format[label_type]
                    sep_type_value = label_format['type?label']
                    
                    # avoid turning strings to lists
                    if isinstance(label_value, str):
                        label_str = label_value
                    else:
                        label_str = str(list(label_value) )
                    
                    edge_dot_label += type_name +sep_type_value
                    edge_dot_label += label_str +sep_label_sets
            edge_dot_label += '"'
            
            return edge_dot_label
        
        self.__exist_labels__()     
        
        # get labeling def
        label_def = self.graph.__transition_label_def__
        label_format = self.graph.__transition_dot_label_format__
        
        for (u, v, key, edge_data) in self.graph.edges_iter(data=True, keys=True):
            edge_dot_label = form_edge_label(edge_data, label_def, label_format)
            to_pydot_graph.add_edge(u, v, key=key, label=edge_dot_label)
            
    def remove_labeled(self, from_state, to_state, label):
        self.__exist_labels__()
        self.__check_states__(from_state, to_state, check=True)
        edge_label = self.__get_labeling__(label, check_label=True)
        
        # get all transitions with given label
        edge_set = copy.copy(self.graph.get_edge_data(from_state, to_state,
                                                      default={} ) )
        
        found_one = 0
        for (edge_key, label) in edge_set.iteritems():
            dprint('Checking edge with:\n\t key = ' +str(edge_key) +'\n')
            dprint('\n\t label = ' +str(label) +'\n')
            dprint('\n against: ' +str(edge_label) )
            
            if label == edge_label:
                dprint('Matched. Removing...')
                self.graph.remove_edge(from_state, to_state, key=edge_key)
                found_one = 1
        
        if not found_one:
            msg = 'No transition with specified labels found, none removed.'
            raise Exception(msg)
            
    def label(self, from_state, to_state, labels, check_label=True):
        """Add label to existing unlabeled transition.
        
        If unlabeled transition between the given nodes already exists, label it.
        Otherwise raise error.
        
        States checked anyway, because method assumes transition already exists.        
        
        Requires that action set or alphabet be defined.
        """
        self.__exist_labels__()
        self.__check_states__(from_state, to_state, check=True)
        
        # chek if same unlabeled transition exists
        trans_from_to = self.graph.get_edge_data(from_state, to_state, default={} )
        if {} not in trans_from_to.values():
            msg = "Unlabeled transition from_state-> to_state doesn't exist,\n"
            msg += 'where:\t from_state = ' +str(from_state) +'\n'
            msg += 'and:\t to_state = ' +str(to_state) +'\n'
            msg += 'So it cannot be labeled.\n'
            msg += 'Either add it first using: transitions.add(), then label it,\n'
            msg += 'or use transitions.add_labeled(), same with a single call.\n'
            raise Exception(msg)
        
        # label it
        self.remove(from_state, to_state)
        self.add_labeled(from_state, to_state, labels, check=check_label)
    
    def relabel(self, from_state, to_state, old_labels, new_labels, check=True):
        """Change the label of an existing labeled transition.
        
        TODO partial relabeling available
        
        Need to identify existing transition by providing old label.
        
        A labeled transition is (uniquely) identified by the list:
            [from_state, to_state, old_label]
        However disagrees will have to work directly using int IDs for edges,
        or any other type desired as edge key.
        
        The other option is to switch to DiGraph and then "manually" handle
        multiple edges with different labels by storing them as attribute info
        in a single graph edge, not very friendly.
        """
        self.__exist_labels__()
        self.__check_states__(from_state, to_state, check=True)
        
        self.remove_labeled(from_state, to_state, old_labels)
        self.add_labeled(from_state, to_state, new_labels, check=check)
        
    def add_labeled(self, from_state, to_state, labels, check=True):
        """Add new labeled transition, error if same exists.
        
        If edge between same nodes, either unlabeled or with same label
        already exists, then raise error.
        
        Checks states are already in set of states.
        Checks action is already in set of actions.
        If not, raises exception.
        
        To override, use check = False.
        Then given states are added to set of states,
        and given action is added to set of actions.
        
        input
        -----
            -C{labels} is single label, if single action set /alphabet defined,
            or if multiple action sets /alphabets, then either:
                list of labels in proper oder
                or dict of action_set_name : label pairs
        """
        self.__exist_labels__()
        self.__check_states__(from_state, to_state, check=check)
        
        # chek if same unlabeled transition exists
        trans_from_to = self.graph.get_edge_data(from_state, to_state, default={} )
        if {} in trans_from_to.values():
            msg = 'Unlabeled transition from_state-> to_state already exists,\n'
            msg += 'where:\t from_state = ' +str(from_state) +'\n'
            msg += 'and:\t to_state = ' +str(to_state) +'\n'
            raise Exception(msg)
        
        # note that first we add states, labels, if check =False,
        # then we check to see if same transition already exists
        #
        # if states were not previously in set of states,
        # then transition is certainly new, so we won't abort in the middle,
        # after adding states, but before adding transition,
        # due to finding an existing one, because that is impossible.
        #
        # if labels were not previously in label set,
        # then a similar issue can arise only with unlabeled transitions
        # pre-existing. This is avoided by first checking for an unlabeled trans.        
        edge_label = self.__get_labeling__(labels, check_label=check)
        
        # check if same labeled transition exists
        if edge_label in trans_from_to.values():
            msg = 'Same labeled transition:\n'
            msg += 'from_state---[label]---> to_state\n'
            msg += 'already exists, where:\n'
            msg += '\t from_state = ' +str(from_state) +'\n'
            msg += '\t to_state = ' +str(to_state) +'\n'
            msg += '\t label = ' +str(edge_label) +'\n'
            raise Exception('Same labeled transiion already exists.')
        
        # states, labels checked, no same unlabeled nor labeled,
        # so add it
        self.graph.add_edge(from_state, to_state, **edge_label)
    
    def add_labeled_from(self, from_states, to_states, labels, check=True):
        """Add multiple labeled transitions.
        
        Adds transitions between all states in set from_states,
        to all states in set to_states, annotating them with the same labels.
        For more details, see add_labeled().
        """
        for from_state in from_states:
            for to_state in to_states:
                self.add_labeled(from_state, to_state, labels, check=check)
    
    def add_labeled_adj(self, adj, labels, check_labels=True, state_map='ordered'):
        """Add multiple transitions from adjacency matrix.
        
        These transitions are enabled when the given guard is active.        
        """
        # state order maintained ?
        if self.graph.states.list is None:
            raise Exception('System must have ordered states to use add_labeled_adj.')
        
        # square ?
        if adj.shape[0] != adj.shape[1]:
            raise Exception('Adjacency matrix must be square.')
        
        n = adj.shape[0]
        
        # no existing states ?
        if len(self.graph.states() ) == 0:
            new_states = range(n)
            self.graph.states.add_from(new_states)
            print('Added ordered list of states: ' +str(self.graph.states.list) )
        
        # convert to format friendly for edge iteration
        nx_adj = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph())
        
        # add each edge using existing checks
        states_list = self.graph.states.list
        for edge in nx_adj.edges_iter():
            (from_idx, to_idx) = edge
            
            from_state = states_list[from_idx]
            to_state = states_list[to_idx]
            
            self.graph.transitions.add_labeled(from_state, to_state, labels,
                                               check=check_labels)
        
        # in-place replace nodes, based on map
        # compose graphs (vs union, vs disjoint union)
    
    def with_label(self, from_state, to_states='any', desired_label='any'):
        """Find all edges from_state to_states, annotated with guard_label.
        
        TODO support partial labels
        """
        def label_is_desired(cur_label, desired_label):
            for (label_type, desired_val) in desired_label.iteritems():
                dprint('Label type checked:\n\t' +str(label_type) )
                cur_val = cur_label[label_type]
                dprint('Label of checked transition:\n\t' +str(cur_val) )
                dprint('Desired label:\n\t' +str(desired_val) )
                if cur_val != desired_val and True not in cur_val:
                    return False
            return True
        
        out_edges = self.graph.edges(from_state, data=True, keys=True)
        
        found_edges = set()        
        for from_state, to_state, key, cur_label in out_edges:
            if to_state not in to_states and to_states is not 'any':
                continue
            
            # any guard ok ?
            if desired_label is 'any':
                ok = True
            else:
                dprint('checking guard')
                ok = label_is_desired(cur_label, desired_label)
            
            if ok:
                dprint('transition label matched desired label')
                found_edge = (from_state, to_state, key)
                found_edges.add(found_edge)
            
        return found_edges

class LabeledStateDiGraph(nx.MultiDiGraph):
    """Species: System & Automaton."""
    
    def __init__(self, name='', states=[], initial_states=[], current_state=None):
        nx.MultiDiGraph.__init__(self, name=name)
        
        self.states = States(self, states=states, initial_states=initial_states,
                             current_state=current_state)
        self.transitions = LabeledTransitions(self)

        self.dot_node_shape = {'normal':'circle'}
        self.default_export_path = './'
        self.default_export_fname = 'out'
        
    def __add_missing_extension__(self, path, file_type):
        import os
        filename, file_extension = os.path.splitext(path)
        desired_extension = os.path.extsep +file_type
        if file_extension != desired_extension:
            path = filename +desired_extension
        return path
    
    def __export_fname__(self, path, file_type, addext):
        if path == 'default':
            if self.name == '':
                path = self.default_export_path +self.default_export_fname
            else:
                path = self.default_export_path +self.name
        
        if addext:
            path = self.__add_missing_extension__(path, file_type)
        
        return path
    
    def __pydot_missing__(self):
        if pydot is None:
            msg = 'Attempted calling dump_dot.\n'
            msg += 'Unavailable due to pydot not installed.\n'
            warnings.warn(msg)
            return True
        
        return False
    
    def __to_pydot__(self):
        """Convert to properly annotated pydot graph."""
        if self.__pydot_missing__():
            return
        
        dummy_nx_graph = nx.MultiDiGraph()
        
        self.states.__dot_str__(dummy_nx_graph)
        self.transitions.__dot_str__(dummy_nx_graph)
        
        pydot_graph = nx.to_pydot(dummy_nx_graph)
        
        return pydot_graph
    
    def __eq__(self, other):
        """Check finite-transition system equality.
        
        A == B
        
        4 sets should match:
            1) nodes  IDs
            2) node attributes (include labels)
            3) transitions
            4) transition attributes (include labels)
        """
        raise NotImplementedError
    
    def __ne__(self, other):
        return not self.__eq__(other) 
    
    def __le__(self, other):
        """Check sub-finite-transition-system relationship.
        
        A <= B
        A is a sub-finite-transition-system of B
        
        A should have a subset of B's:
            1) node IDs
            2) node attributes (includes labels)
            2) transitions (between same node IDs)
            3) transition attributes (includes labels).
        """
        raise NotImplementedError
    
    def __lt__(self, other):
        return self.__le__(other) and self.__ne__(other)
        
    def __ge__(self, other):
        return other.__le__(self)
        
    def __gt__(self, other):
        return other.__lt__(self)

	# operations on single transitions system
    def reachable(self):
        """Return reachable subautomaton."""
        raise NotImplementedError
        
    def trim_dead(self):
        raise NotImplementedError
    
    def trim_unreachable(self):
        raise NotImplementedError
    
    # file i/o
    def load_xml(self):
        raise NotImplementedError
        
    def dump_xml(self):
        raise NotImplementedError
    
    def write_xml_file(self):
        raise NotImplementedError
    
    def dump_dot(self):
        """Return dot string.
        
        Requires pydot.        
        """
        pydot_graph = self.__to_pydot__()
        
        return pydot_graph.to_string()
    
    def dot_str(self):
        """Alias to dump_dot()."""
        return self.dump_dot()
    
    def save_dot(self, path='default', add_missing_extension=True):
        """Save .dot file.
        
        Requires pydot.        
        """
        path = self.__export_fname__(path, 'dot', addext=add_missing_extension)
        
        pydot_graph = self.__to_pydot__()
        pydot_graph.write_dot(path)
    
    def save_png(self, path='default', add_missing_extension=True):
        """Save .png file.
        
        Requires pydot.        
        """
        path = self.__export_fname__(path, 'png', addext=add_missing_extension)
        pydot_graph = self.__to_pydot__()
        pydot_graph.write_png(path)
    
    def save_pdf(self, path='default', add_missing_extension=True, rankdir='LR'):
        """Save .pdf file.
        
        @param path: path to image
            (extension .pdf appened if missing and add_missing_extension==True)
        @type path: str
        
        @param add_missing_extension: if extension .pdf missing, it is appended
        @type add_missing_extension: bool
        
        @param rankdir: direction for dot layout
        @type rankdir: str = 'TB' | 'LR'
            (i.e., Top->Bottom | Left->Right)
        
        caution
        -------
        rankdir is experimental argument
        
        See also
        --------
        save_dot, save_png
        
        depends
        -------
        pydot      
        """
        path = self.__export_fname__(path, 'pdf', addext=add_missing_extension)
        pydot_graph = self.__to_pydot__()
        pydot_graph.set_rankdir(rankdir)
        pydot_graph.write_pdf(path)
    
    def dump_dot_color(self):
        raise NotImplementedError
    
    def write_dot_color_file(self):
        raise NotImplementedError

class FiniteSequence(object):
    """Used to construct finite words."""
    def __init__(self, sequence):
        self.sequence = sequence
    
    def __str__(self):
        return str(self.sequence)
    
    def __call__(self):
        return self.sequence
    
    def steps(self):
        cur_seq = self.sequence[:-1]
        next_seq = self.sequence[1:]
        
        return cur_seq, next_seq

class InfiniteSequence(object):
    """Used to construct simulations."""
    def __init__(self, prefix=[], suffix=[]):
        self.set_prefix(prefix)
        self.set_suffix(suffix)
    
    def set_prefix(self, prefix):
        self.prefix = FiniteSequence(prefix)
    
    def get_prefix(self):
        return self.prefix()
    
    def set_suffix(self, suffix):
        self.suffix = FiniteSequence(suffix)
    
    def get_suffix(self):
        return self.suffix()
    
    def prefix_steps(self):
        return self.prefix.steps()
    
    def suffix_steps(self):
        return self.suffix.steps()
    
    def __str__(self):
        return 'Prefix = ' +str(self.prefix) +'\n' \
                +'Suffix = ' +str(self.suffix) +'\n'

class FiniteTransitionSystemSimulation(object):
    """Stores execution, path, trace.
    
    execution = s0, a1, s1, a1, ..., aN, sN (Prefix)
                sN, a(N+1), ..., aM, sN (Suffix)
    path = s0, s1, ..., sN (Prefix)
           sN, s(N+1), ..., sN (Suffix)
    trace = L(s0), L(s1), ..., L(sN) (Prefix)
            L(sN), L(s(N+1) ), ..., L(sN) (Suffix)
    
    where:
        sI \in States
        aI \in Actions (=Transition_Labels =Edge_Labels)
        L(sI) \in State_Labels
    
    Note: trace computation avoided because it requires definitin of
    the whole transition system
    """
    
    #todo:
    #    check consitency with actions and props
    
    def __init__(self, execution=InfiniteSequence(), trace=InfiniteSequence() ):
        self.execution = execution
        self.path = self.execution2path()
        self.trace = trace
        self.action_trace = self.execution2action_trace()
    
    def execution2path(self):
        """Return path by projecting execution on set of States.
        
        path of states = s0, s1, ..., sN     
        """
        
        # drop actions from between states
        execution = self.execution
        
        prefix = execution.get_prefix()[0::2]
        suffix = execution.get_suffix()[0::2]
        
        path = InfiniteSequence(prefix, suffix)
        
        return path
    
    def execution2action_trace(self):
        """Return trace of actions by projecting execution on set of Actions.
        
        trace of actions = a1, a2, ..., aN        
        """
        
        execution = self.execution        
        
        prefix = execution.get_prefix()[1::2]
        suffix = execution.get_suffix()[1::2]
        
        action_trace = InfiniteSequence(prefix, suffix)
        
        return action_trace
    
    def __str__(self):
        msg = "Finite Transition System\n\t Simulation Prefix:\n\t"
        
        path = self.path.prefix
        trace = self.trace.prefix
        action_trace = self.action_trace.prefix
        
        msg += self.__print__(path, trace, action_trace)
        
        msg += "\n\t Simulation Suffix:\n\t"
        
        path = self.path.suffix
        trace = self.trace.suffix
        action_trace = self.action_trace.suffix
        
        msg += self.__print__(path, trace, action_trace)
        
        return msg
        
    def __print__(self, path, trace, action_trace):
        cur_state_seq, next_state_seq = path.steps()
        cur_label_seq, next_label_seq = trace.steps()
        
        action_seq = action_trace.steps()[1::2]
        
        msg = ''
        for cur_state, cur_label, action, next_state, next_label in zip(
            cur_state_seq, cur_label_seq,
            action_seq, next_state_seq, next_label_seq
        ):
            msg += str(cur_state)+str(list(cur_label) ) \
                  +'--'+str(action)+'-->' \
                  +str(next_state)+str(list(next_label) ) +'\n'
        return msg
    
    def save(self):
        """Dump to file.
        
        We need to decide a format.
        """
        raise NotImplementedError

class FTSSim(FiniteTransitionSystemSimulation):
    """Alias for Finite Transition System Simulation."""
    
    def __init__(self):
        FiniteTransitionSystemSimulation.__init__(self)

class AtomicPropositions(object):
    """Store & print set of atomic propositions.

    Note that any transition system or automaton is just annotated by atomic
    propositions. They are either present or absent.
    Their interpretation is external to this module.
    That is, evaluating whether an AP is true or false, so present or absent as
    a member of a set of APs requires semantics defined and processed elsewhere.
    
    The simplest representation for APs stored here is a set of strings.
    """    
    
    # manipulate AP set (AP alphabet, not to be confused with input alphabet)
    def __init__(self, graph, name, atomic_propositions=[]):
        self.graph = graph
        self.name = name
        self.atomic_propositions = set(atomic_propositions)
    
    def __call__(self):
        return self.atomic_propositions    
    
    def __str__(self):
        return 'Atomic Propositions:\n\t' +pformat(self() )
    
    def __contains__(self, atomic_proposition):
        return atomic_proposition in self.atomic_propositions
    
    def __check_state__(self, state):
        if state not in self.graph.states():
            msg = 'State:\n\t' +str(state)
            msg += ' is not in set of states:\n\t:' +str(self.graph.states() )
            raise Exception(msg)
    
    def add(self, atomic_proposition, check_existing=True):
        """Add single atomic proposition.
        
        @type atomic_proposition: hashable
        """
        if not isinstance(atomic_proposition, Hashable):
            raise Exception('Atomic propositions stored in set, so must be hashable.')
        
        if atomic_proposition in self.atomic_propositions and check_existing:
            raise Exception('Atomic Proposition already in set of APs.')
        
        self.atomic_propositions.add(atomic_proposition)
    
    def add_from(self, atomic_propositions, check_existing=True):
        """Add multiple atomic propositions.
        
        @type atomic_propositions: iterable
        """
        if not isinstance(atomic_propositions, Iterable):
            raise Exception('Atomic Propositions must be provided in iterable.')
        
        for atomic_proposition in atomic_propositions:
            self.add(atomic_proposition, check_existing) # use checks
        
    def remove(self, atomic_proposition):
        node_ap = nx.get_node_attributes(self.graph, self.name)
        
        nodes_using_ap = set()
        for (node, ap_subset) in node_ap.iteritems():
            if atomic_proposition in ap_subset:
                nodes_using_ap.add(node)                
        
        if nodes_using_ap:
            msg = 'AP (=' +str(atomic_proposition) +') still used '
            msg += 'in label of nodes: ' +str(nodes_using_ap)
            raise Exception(msg)
        
        self.atomic_propositions = \
            self.atomic_propositions.difference({atomic_proposition} )
        
    def number(self):
        """Count atomic propositions."""
        return len(self.atomic_propositions)
    
    def add_labeled_state(self, state, ap_label, check=True):
        """Add single state with its label.
        
        @param state: defines element to be added to set of states S
                  = hashable object (int, str, etc)
        @type ap_label: iterable \\in 2^AP
        """
        self.graph.states.add(state)
        
        if not check:
            self.add_from(ap_label)
        
        if not set(ap_label) <= self.atomic_propositions:
            raise Exception('Label \\not\\subset AP.')
        
        kw = {self.name: ap_label}
        self.graph.add_node(state, **kw)
    
    def label_state(self, state, ap_label, check=True):
        """Label state with subset of AP (Atomic Propositions).
        
        State and AP label checked, override with check = False.        
        """
        if not check:
            self.add_labeled_state(state, ap_label, check=check)
            return
        
        self.__check_state__(state)
        
        if not set(ap_label) <= self.atomic_propositions:
            raise Exception('Label \\not\\subset AP.')
        
        kw = {self.name: ap_label}
        self.graph.add_node(state, **kw)
    
    def label_states(self, states, ap_label, check=True):
        """Label multiple states with the same AP label."""
        
        for state in states:
            self.label_state(state, ap_label, check=True)
    
    def delabel_state(self, state):
        """Alias for remove_label_from_state()."""
        
        raise NotImplementedError
    
    def of(self, state):
        """Get AP set labeling given state."""
        
        self.__check_state__(state)
        return self.graph.node[state][self.name]
        
    def list_states_with_labels(self):
        """Return list of labeled states.
        
        Each state is a tuple:
            (state, label)
        where:
            state \in States
            label \in 2^AP
        """
        return self.states(data=True)
    
    def remove_state_with_label(self, labels):
        """Find states with given label"""
        raise NotImplementedError
    
    def find_states_with_label(self, labels):
        """Return all states with label in given set."""
        raise NotImplementedError
    
    def remove_labels_from_states(self, states):
        raise NotImplementedError

# big issue: edge naming !
class Actions(object):
    """Store set of system or environment actions."""
    
    def __init__(self, graph, name, actions=[]):
        self.name = name
        self.actions = set(actions)
        self.graph = graph
    
    def __call__(self):
        return self.actions
    
    def __str__(self):
        n = str(self.number() )
        action_set_name = self.name
        
        msg = 'The ' +n +' Actions of type: ' +action_set_name +', are:\n\t'
        msg += str(self.actions)
        
        return msg
    
    def __contains__(self, action):
        return action in self.actions
    
    def add(self, action=[]):
        self.actions.add(action)
    
    def add_from(self, actions=[]):
        """Add multiple actions.
        
        @type actions: iterable
        """
        self.actions |= set(actions)
    
    def number(self):
        return len(self.actions)
    
    def remove(self, action):
        edges_action = nx.get_edge_attributes(self.graph, self.name)
        
        edges_using_action = set()
        for (edge, curaction) in edges_action.iteritems():
            if curaction == action:
                edges_using_action.add(edge)                
        
        if edges_using_action:
            msg = 'AP (=' +str(action) +') still used '
            msg += 'in label of nodes: ' +str(edges_using_action)
            raise Exception(msg)
        
        self.actions.remove(action)
        
    def of(self, from_state, to_state, edge_key):
        attr_dict = self.graph.get_edge_data(from_state, to_state, key=edge_key)
        
        if attr_dict is None:
            msg = 'No transition from state: ' +str(from_state)
            msg += ', to state: ' +str(to_state) +', with key: '
            msg += str(edge_key) +' exists.'
            warnings.warn(msg)
        
        label_order = self.graph.__transition_label_order__
        transition_label_values = set()
        for label_type in label_order:
            cur_label_value = attr_dict[label_type]
            transition_label_values.add(cur_label_value)
    
        return transition_label_values
        
class FiniteTransitionSystem(LabeledStateDiGraph):
    """Finite Transition System for modeling closed systems.
    
    Def. 2.1, p.20 [Baier]:
        S = states
        S_0 = initial states \\subseteq states
        
        AP = atomic proposition set (state labels \in 2^AP)
        Act = action set (edge labels)
        
        T = transition relation
          = edge set + edge labeling function
           (transitions labeled by Act)
        L = state labeing function
          : S-> 2^AP
    
    dot export
    ----------
    Format transition labels using C{__transition_dot_label_format__} which is a
    dict with values:
        - 'actions' (=name of transitions attribute): type before separator
        - 'type?label': separator between label type and value
        - 'separator': between labels for different sets of actions
            (e.g. sys, env). Not used for closed FTS, because it has single set
            of actions.
    """
    
    def __init__(self, name='', states=[], initial_states=[], current_state=None,
                 atomic_propositions=[], actions=[] ):
        """Note first sets of states in order of decreasing importance,
        then first state labeling, then transitin labeling (states more
        fundamentalthan transitions, because transitions need states in order to
        be defined).
        """
        LabeledStateDiGraph.__init__(
            self, name=name, states=states, initial_states=initial_states,
            current_state=current_state
        )
        
        self.atomic_propositions = AtomicPropositions(self, 'ap', atomic_propositions)
        self.actions = Actions(self, 'actions', actions)
        
        self.__state_label_def__ = {'ap': self.atomic_propositions}
        self.__state_dot_label_format__ = {'ap':'',
                                           'type?label':'',
                                           'separator':'\\n'}
        
        self.__transition_label_def__ = {'actions': self.actions}
        self.__transition_label_order__ = ['actions']
        self.__transition_dot_label_format__ = {'actions':'',
                                                'type?label':'',
                                                'separator':'\\n'}

        self.dot_node_shape = {'normal':'box'}
        self.default_export_fname = 'fts'

    def __str__(self):
        s = str(self.states) +'\nState Labels:\n' +pformat(self.states(data=True) )
        s += '\n' +str(self.transitions) +'\n' +str(self.actions) +'\n'
        s += str(self.atomic_propositions) +'\n'
        
        return s
    
    def __mul__(self, ts_or_ba):
        """Synchronous product of TS with TS or BA.
        
        see also
        --------
        self.sync_prod
        """
        return self.sync_prod(ts_or_ba)
    
    def __or__(self, ts):
        """Synchronous product between transition systems."""
        return self.async_prod(ts)
    
    def sync_prod(self, ts_or_ba):
        """Synchronous product TS x BA or TS1 x TS2.
        
        see also
        --------
        self.__mul__, self.async_prod, BuchiAutomaton.sync_prod
        """
        if isinstance(ts_or_ba, FiniteTransitionSystem):
            return self.__ts_ts_sync_prod__(ts_or_ba)
        elif isinstance(ts_or_ba, BuchiAutomaton):
            ba = ts_or_ba
            return __ts_ba_sync_prod__(self, ba)
        else:
            raise Exception('Argument must be TS or BA.')
    
    def async_prod(self, ts):
        """Asynchronous product TS1 x TS2 between Finite Transition Systems."""
        raise NotImplementedError
    
    def is_blocking(self):
        """Does each state have at least one outgoing transition ?
        
        Note that edge labels are NOT checked, i.e.,
        it is not checked whether for each state and each possible symbol/letter
        in the input alphabet, there exists at least one transition.
        
        The reason is that edge labels do not have any semantics at this level,
        so they are not yet regarded as guards.
        For more semantics, use a FiniteStateMachine.
        """
        raise NotImplementedError
    
    def merge_states(self):
        raise NotImplementedError

    # operations between transition systems
    def union(self):
        raise NotImplementedError
    
    def intersection(self):
        raise NotImplementedError
        
    def difference(self):
        raise NotImplementedError

    def composition(self):
        raise NotImplementedError
    
    def projection_on(self):
        raise NotImplementedError
    
    def simulate(self, state_sequence="random"):
        """
            simulate automaton
                inputs="random" | given array
                mode="acceptor" | "transfucer"
        """
        raise NotImplementedError
    
    def is_simulation(self, simulation=FTSSim() ):
        raise NotImplementedError
    
    def loadSPINAut():
        raise NotImplementedError

class FTS(FiniteTransitionSystem):
    """Alias to FiniteTransitionSystem."""
    
    def __init__(self, name='', states=[], initial_states=[], current_state=None,
                 atomic_propositions=[], actions=[] ):
        FiniteTransitionSystem.__init__(
            self, name=name, states=states, initial_states=initial_states,
            current_state=current_state, atomic_propositions=atomic_propositions,
            actions=actions
        )

class OpenFiniteTransitionSystem(LabeledStateDiGraph):
    """Analogous to FTS, but for open systems, with system and environment."""
    def __init__(self, name='', states=[], initial_states=[], current_state=None,
                 atomic_propositions=[], sys_actions=[], env_actions=[] ):
        LabeledStateDiGraph.__init__(
            self, name=name, states=[], initial_states=initial_states,
            current_state=current_state
        )
        
        self.sys_actions = Actions(self, 'sys_actions', sys_actions)
        self.env_actions = Actions(self, 'env_actions', env_actions)
        self.atomic_propositions = AtomicPropositions(self, 'ap', atomic_propositions)
        
        self.__state_label_def__ = {'ap': self.atomic_propositions}
        self.__state_dot_label_format__ = {'ap':'',
                                           'type?label':'',
                                           'separator':'\\n'}
        
        self.__transition_label_def__ = {'sys_actions': self.sys_actions,
                                         'env_actions': self.env_actions}
        self.__transition_label_order__ = ['sys_actions', 'env_actions']
        self.__transition_dot_label_format__ = {'sys_actions':'sys',
                                                'env_actions':'env',
                                                'type?label':':',
                                                'separator':'\\n'}
        self.dot_node_shape = {'normal':'box'}
        self.default_export_fname = 'ofts'
        
    def __str__(self):
        s = str(self.states) +'\nState Labels:\n' +pformat(self.states(data=True) )
        s += '\n' +str(self.transitions) +'\n'
        s += str(self.sys_actions) +'\n' +str(self.env_actions) +'\n'
        s += str(self.atomic_propositions) +'\n'
        
        return s

class oFTS(OpenFiniteTransitionSystem):
    """Alias to transys.OpenFiniteTransitionSystem."""
    def __init__(self, name='', states=[], initial_states=[], current_state=None,
                 atomic_propositions=[], sys_actions=[], env_actions=[] ):
        OpenFiniteTransitionSystem.__init__(
            self, name=name, states=states, initial_states=initial_states,
            current_state=None, atomic_propositions=atomic_propositions,
            sys_actions=sys_actions, env_actions=env_actions
        )

###########
# Automata
###########
class Alphabet(object):
    """Stores input letters annotating transitions of an automaton."""
    
    def __init__(self, graph, name, letters=[], atomic_proposition_based=False):
        self.name = name
        self.graph = graph
        self.atomic_proposition_based = atomic_proposition_based
        
        if self.atomic_proposition_based:
            self.atomic_propositions = AtomicPropositions(graph, 'ap for alphabet')
            self.alphabet = None
        else:
            self.alphabet = set(letters)
            self.atomic_propositions = None
        
        self.add_from(letters)
    
    def __str__(self):
        return 'Alphabet:\n' +str(self() )
    
    def __call__(self):
        if self.atomic_proposition_based:
            return set(powerset(self.atomic_propositions() ) )
        else:
            return self.alphabet
    
    def __contains__(self, letter):
        if self.atomic_proposition_based:
            return letter <= self.atomic_propositions()
        else:
            return letter in self.alphabet
    
    def add(self, new_letter):
        """Add single letter to alphabet.
        
        If C{atomic_proposition_based=False},
        then the letter is stored in C{alphabet}.
        
        If C{atomic_proposition_based=True},
        the the atomic propositions within the letter
        are stored in C{atomic_propositions}.
        
        @type new_input_letter:
            - hashable if C{atomic_proposition_based=False}
            - iterable if C{atomic_proposition_based=True}
        
        See also
        --------
        add_from, AtomicPropositions
        """
        if self.atomic_proposition_based:
            # check multiplicity
            if contains_multiple(new_letter):
                msg = """
                    Letter contains multiple Atomic Propositions.
                    Multiples will be discarded.
                    """
                warnings.warn(msg)
            
            self.atomic_propositions.add_from(new_letter, check_existing=False)
        else:
            self.alphabet.add(new_letter)
    
    def add_from(self, new_letters):
        """Add multiple letters to alphabet.
        
        If C{atomic_proposition_based=False},
        then these are stored in C{alphabet}.
        
        If C{atomic_proposition_based=True},
        then the atomic propositions within each letter
        are stored in C{atomic_propositions}.
        
        @type new_letters:
            - iterable of hashables,
              if C{atomic_proposition_based=False}
            - iterable of iterables of hashables,
              if C{atomic_proposition_based=True}
        
        See also
        --------
        add, AtomicPropositions
        """        
        if self.atomic_proposition_based:
            for new_letter in new_letters:
                self.add(new_letter)
        else:
            self.alphabet |= set(new_letters)
    
    def number(self):
        if self.atomic_proposition_based:
            return 2**self.atomic_propositions.number()
        else:
            return len(self.alphabet)        
    
    def remove(self, rm_input_letter):
        if self.atomic_proposition_based:
            msg = """Removing from an Atomic Proposition-based aphabet
                  not supported, because it can reduce the set of atomic
                  propositions so that other letters \\not\\in 2^AP any more.
                  """
            raise Exception(msg)
        else:
            self.alphabet.remove(rm_input_letter)
        
    def remove_from(self, rm_input_letters):
        self.alphabet.difference(rm_input_letters)

class InfiniteWord(InfiniteSequence):
    """Store word.
    
    Caution that first symbol corresponds to w1, not w0.
    
    word = w1, w2, ..., wN
    """
    
    def __init__(self, prefix=[], suffix=[]):
        InfiniteSequence.__init__(self, prefix, suffix)

class FiniteStateAutomatonSimulation(object):
    """Store automaton input word and run.

    input_word = w1, w2, ...wN (Prefix)
                 wN, ..., wM (Suffix)
    run = s0, s1, ..., sN (Prefix)
          sN, ..., sM (Suffix)
    
    s(i-1) --w(i)--> s(i)
    """
    
    def __init__(self, input_word=InfiniteWord(), run=InfiniteSequence() ):
        self.input_word = input_word
        self.run = run
    
    def __str__(self):
        msg = "Finite-State Automaton\n\t Simulation Prefix:\n\t"
        
        word = self.input_word.prefix
        run = self.run.prefix
        
        msg += self.__print__(word, run)
        
        msg += "\n\t Simulation Suffix:\n\t"
        
        word = self.input_word.suffix
        run = self.run.suffix
        
        msg += self.__print__(word, run)
        
        return msg
        
    def __print__(self, word, run):
        cur_state_seq, next_state_seq = run.steps()
        letter_seq = word.sequence
        
        msg = ''
        for cur_state, cur_letter, next_state in zip(
            cur_state_seq, letter_seq, next_state_seq
        ):
            msg += str(cur_state) \
                  +'--'+str(list(cur_letter) )+'-->' \
                  +str(next_state) +'\n\t'
        return msg
    
    def save(self):
        """Dump to file.close
        
        We need to decide a format.
        """
        
class FSASim(FiniteStateAutomatonSimulation):
    """Alias."""
    
    def __init__(self, input_word=InfiniteWord(), run=InfiniteSequence() ):
        FiniteStateAutomatonSimulation.__init__(
            self, input_word=input_word, run=run
        )

class FiniteStateAutomaton(LabeledStateDiGraph):
    """Generic automaton.
    
    1) states
    2) initial states
    3) final states
    
    4) input alphabet = set of input letters
    5) transition labels
    
    4) acceptor mode (i.e., you can ask is_accepted ?, but nothing more)
    5) for generator mode, use a synthesis algorithm
       To avoid misconceptions, NO SIMULATION MODE provided.
    
    Synthesis interpretation
    ------------------------
    A synthesis algorithm is applying model checking (is accepted ?)
    to each possible input word,
    normally represented by a deterministic finite transition system,
    but during acceptance checking implicitly by graph searching
    (assuming the model is everything an only the automaton is the constraint)
    
    Dually, a model checking algorithm is iteratively attempting synthesis
    for each possible input word.
    However, since synthesis is fundamentally operating by trying out words
    and seeing whether they get accepted, it follows that
    an automaton is operable ONLY in acceptor mode.
    
    Generator construction
    ----------------------
    For a "generator", we would need to convert the automaton to a
    (nontrivial) transition system producing maximal initial paths,
    as discussed below.
    
    The above algorithms return a single accepted input word, if found.
    That word is represented as a (deterministic) Finite Transition System.
    If we want to represent more than one accepted word (e.g. the whole
    language), we would need to find all possible such FTS and
    construct their "union".
    
    Open Systems
    ------------
    Finally, note that a Finite State Machine or transducer is an OPEN SYSTEM.
    As such, it does not represent an input word of an automaton.
    It can be used for game synthesis, where inputs and outputs make sense.
    
    Alternatively, only after closing a system can it be used
    (in the sense of having a program graph which can be unfolded).
    
    input
    -----
    
    returns
    -------
    
    see also
    --------    
    __dot_str__ of LabeledStateDiGraph
        
    """
    
    def __init__(self, name='', states=[], initial_states=[], final_states=[],
                 input_alphabet=[], atomic_proposition_based=True):
        LabeledStateDiGraph.__init__(
            self, name=name,
            states=states, initial_states=initial_states
        )
        
        self.final_states = set()
        self.add_final_states_from(final_states)
        
        self.alphabet = Alphabet(self, 'in_alphabet',
                                 atomic_proposition_based=atomic_proposition_based)
        self.alphabet.add_from(input_alphabet)
        
        self.__transition_label_def__ = {'in_alphabet': self.alphabet}
        self.__transition_label_order__ = ['in_alphabet']
        
        # used before label value
        self.__transition_dot_label_format__ = {'in_alphabet':'',
                                                'type?label':'',
                                                'separator':'\\n'}
        
        self.dot_node_shape = {'normal':'circle', 'final':'doublecircle'}
        self.default_export_fname = 'fsa'
        
    def __str__(self):
        s = str(self.states) +'\nState Labels:\n' +pformat(self.states(data=True) )
        s += '\n' +str(self.transitions) +'\n' +str(self.alphabet) +'\n'
        s += 'Final States:\n\t' +str(self.final_states)
        
        return s

    # final states
    def add_final_state(self, new_final_state):
        if not new_final_state in self.states():
            raise Exception('New final state \\notin States.')
        else:
            self.final_states.add(new_final_state)

    def add_final_states_from(self, new_final_states):
        new_final_states = set(new_final_states)     
        
        if not new_final_states <= set(self.states() ):
            raise Exception('New Final States \\notsubset States.')
        else:
            self.final_states |= set(new_final_states)
    
    def number_of_final_states(self):
        return len(self.final_states)
    
    def remove_final_state(self, rm_final_state):
        self.final_states.remove(rm_final_state)
    
    def remove_final_states_from(self, rm_final_states):
        self.final_states = self.final_states.difference(rm_final_states)
    
    def find_edges_between(self, start_state, end_state):
        """Return list of edges between given nodes.
        
        Each edge in the list is a tuple of:
            (start_state, end_state, edge_dict)
        """
        return self.edges(start_state, end_state)
    
    def find_guards_between(self, start_state, end_state):
        """Return sets of input letters labeling transtions between given states."""
        edges_between = self.find_edges_between(start_state, end_state)
        guards = []
        for edge in edges_between:
            guard = edge[2]['input_letter']
            guards.append(guard)
    
    def remove_transition(self, start_state, end_state, input_letter=[]):
        """Remove all edges connecting the given states."""
        self.remove_edges_from((start_state, end_state) )

    # checks
    def is_deterministic(self):
        """overloaded method."""
        
    def is_blocking(self):
        """overloaded method."""
    
    def is_accepted(self, input_word):
        """Check if input word is accepted."""
        sim = self.simulate(input_word)
        
        inf_states = set(sim.run.get_suffix() )
        
        if bool(inf_states & self.final_states):
            accept = True
        else:
            accept = False
        
        return accept
        
    def simulate(self, initial_state, input_word):
        """Returns an Omega Automaton Simulation, with prefix, suffix."""
        
        # should be implemented properly with nested depth-first search,
        # becaus of possible branching due to non-determinism
        
        for letter in input_word:
            dprint(letter)
            
            # blocked
        
        return FSASim()

    # operations on two automata
    def add_subautomaton(self):
        raise NotImplementedError

class StarAutomaton(FiniteStateAutomaton):
    """Finite-word finite-state automaton."""

class DeterninisticFiniteAutomaton(StarAutomaton):
    """Deterministic finite-word finite-state Automaton."""

    # check each initial state added
    # check each transition added
    
class DFA(DeterninisticFiniteAutomaton):
    """Alias for deterministic finite-word finite-state automaton."""

class NonDeterministicFiniteAutomaton(StarAutomaton):
    """"Non-deterministic finite-word finite-state automaton."""
    
    # note:
    #   is_deterministic still makes sense
    
class NFA(NonDeterministicFiniteAutomaton):
    """Alias for non-deterministic finite-word finite-state automaton."""

def nfa2dfa():
    """Determinize NFA."""
    raise NotImplementedError
    
def dfa2nfa():
    """Relax state addition constraint of determinism."""
    raise NotImplementedError

class OmegaAutomaton(FiniteStateAutomaton):
    def __init__(self, name='', states=[], initial_states=[], final_states=[],
                 input_alphabet=[], atomic_proposition_based=True):
        FiniteStateAutomaton.__init__(self,
            name=name, states=states, initial_states=initial_states,
            final_states=final_states, input_alphabet=input_alphabet,
            atomic_proposition_based=atomic_proposition_based
        )

class BuchiAutomaton(OmegaAutomaton):
    def __init__(self, name='', states=[], initial_states=[], final_states=[],
                 input_alphabet=[], atomic_proposition_based=True):
        OmegaAutomaton.__init__(
            self, name=name, states=states, initial_states=initial_states,
            final_states=final_states, input_alphabet=input_alphabet,
            atomic_proposition_based=atomic_proposition_based
        )
    
    def __add__(self, other):
        """Union of two automata, with equal states identified."""
        raise NotImplementedError
    
    def __mul__(self, ts_or_ba):
        return self.sync_prod(ts_or_ba)
    
    def __or__(self, ba):
        return self.async_prod(ba)
        
    def __ba_ba_sync_prod__(self, ba2):
        ba1 = self
        
        raise NotImplementedError
        #TODO BA x BA sync prod algorithm

    def sync_prod(self, ts_or_ba):
        """Synchronous product between (BA, TS), or (BA1, BA2).
        
        The result is always a Buchi Automaton.
        
        If C{ts_or_ba} is a Finite Transition System, then the result is the
        Buchi Automaton equal to the synchronous product of this Buchi Automaton
        with the given Transition System. Note that the accepting states of the
        product system are the preimage under projection of the set of accepting
        states of this Buchi Automaton.
        
        If C{ts_or_ba} is a Buchi Automaton, then the result is the Buchi Automaton
        equal to the synchronous product between this Buchi Automaton and the
        given Buchi Automaton. The set of accepting states of the resulting
        Buchi Automaton is equal to the intersection of the preimages under
        projection of the sets of accepting states of the individual Buchi Automata.
        
        This definition of accepting set extends Def.4.8, p.156 [Baier] to NBA.
        
        caution
        -------
        This method includes semantics for true\in\Sigma (p.916, [Baier]),
        so there is a slight overlap with logic grammar.
        In other words, this module is not completely isolated from logics.
        
        see also
        --------        
        ts_ba_sync_prod.
        """
        
        if isinstance(ts_or_ba, BuchiAutomaton):
            return self.__ba_ba_sync_prod__(ts_or_ba)
        elif isinstance(ts_or_ba, FiniteTransitionSystem):
            ts = ts_or_ba
            return __ba_ts_sync_prod__(self, ts)
        else:
            raise Exception('argument should be an FTS or a BA.')
    
    def async_prod(self, other):
        """Should it be defined in a superclass ?"""
        raise NotImplementedError
    
    def acceptance_condition(self, prefix, suffix):
        """Check if given infinite word over alphabet \Sigma is accepted."""
        

class BA(BuchiAutomaton):
    def __init__(self, name='', states=[], initial_states=[], final_states=[],
                 input_alphabet=[], atomic_proposition_based=True):
        BuchiAutomaton.__init__(
            self, name=name, states=states, initial_states=initial_states,
            final_states=final_states, input_alphabet=input_alphabet,
            atomic_proposition_based=atomic_proposition_based
        )

def __ba_ts_sync_prod__(buchi_automaton, transition_system):
    """Construct Buchi Automaton equal to synchronous product TS x NBA.
    
    returns
    -------
    C{prod_ba}, the product Buchi Automaton.
    
    see also
    --------
    __ts_ba_sync_prod__, BuchiAutomaton.sync_prod
    """
    (prod_ts, persistent) = __ts_ba_sync_prod__(transition_system, buchi_automaton)
    
    prod_name = buchi_automaton.name +'*' +transition_system.name
    prod_ba = BuchiAutomaton(name=prod_name)
    
    # copy S, S0, from prod_TS-> prod_BA
    prod_ba.states.add_from(prod_ts.states() )
    prod_ba.states.add_initial_from(prod_ts.states.initial)
    
    # final states = persistent set
    prod_ba.states.add_final_from(persistent)
    
    # copy edges, translating transitions, i.e., chaning transition labels
    if buchi_automaton.alphabet.atomic_proposition_based:
        # direct access, not the inefficient
        #   prod_ba.alphabet.add_from(buchi_automaton.alphabet() ),
        # which would generate a combinatorially large alphabet
        prod_ba.alphabet.atomic_propositions.add_from(
            buchi_automaton.alphabet.atomic_propositions()
        )
    else:
        msg ="""
            Buchi Automaton must be Atomic Proposition-based,
            otherwise the synchronous product is not well-defined.
            """
        raise Exception(msg)
    
    for (from_state, to_state) in prod_ts.edges_iter():
        # prject prod_TS state to TS state        
        ts_to_state = to_state[0]
        msg = 'prod_TS: to_state =\n\t' +str(to_state) +'\n'
        msg += 'TS: ts_to_state =\n\t' +str(ts_to_state)
        dprint(msg)
        
        transition_label = transition_system.atomic_propositions.of(ts_to_state)
        prod_ba.transitions.add_labeled(from_state, to_state, transition_label)   
    
    return prod_ba

def __ts_ba_sync_prod__(transition_system, buchi_automaton):
    """Construct transition system equal to synchronous product TS x NBA.
    
    returns
    -------
    C{(prod_ts, persist) }, where C{prod_ts} is a transition system representing
    the synchronous product between the transition system TS and the
    non-deterministic Buchi Automaton NBA. C{persist} is the subset of states of
    C{prod_ts} which is the preimage under projection of the set of accepting
    states of the Buchi Automaton BA.
    
    Def. 4.62, p.200 [Baier]
    
    erratum
    -------
    note the erratum: P_{pers}(A) is ^_{q\in F} !q, verified from:
        http://www-i2.informatik.rwth-aachen.de/~katoen/errata.pdf
    
    see also
    --------
    __ba_ts_sync_prod, FiniteTransitionSystem.sync_prod
    """
    if not buchi_automaton.alphabet.atomic_proposition_based:
        msg = """Buchi automaton not stored as Atomic Proposition-based.
                synchronous product with Finite Transition System
                is not well-defined."""
        raise Exception(msg)
    
    fts = transition_system
    ba = buchi_automaton
    
    prodts_name = fts.name +'*' +ba.name
    # using set() destroys order
    prodts = FiniteTransitionSystem(name=prodts_name, states=set() )
    prodts.atomic_propositions.add_from(ba.states() )
    prodts.actions.add_from(fts.actions() )

    # construct initial states of product automaton
    s0s = fts.states.initial.copy()
    q0s = ba.states.initial.copy()
    
    final_states_preimage = set()    
    
    for s0 in s0s:
        dprint('----\nChecking initial state:\n\t' +str(s0) )        
        
        Ls0 = fts.atomic_propositions.of(s0)
        Ls0_dict = {'in_alphabet': Ls0}
        
        for q0 in q0s:
            enabled_ba_trans = ba.transitions.with_label(q0, desired_label=Ls0_dict)
            
            # q0 blocked ?
            if enabled_ba_trans == set():
                continue
            
            # which q next ?     (note: curq0 = q0)
            for (curq0, q, edge_key) in enabled_ba_trans:
                new_sq0 = (s0, q)                
                prodts.states.add(new_sq0)
                prodts.states.add_initial(new_sq0)
                prodts.atomic_propositions.label_state(new_sq0, {q} )
                
                # final state ?
                if ba.states.is_final(q):
                    final_states_preimage.add(new_sq0)
    
    dprint(prodts)    
    
    # start visiting reachable in DFS or BFS way
    # (doesn't matter if we are going to store the result)    
    queue = prodts.states.initial.copy()
    visited = set()
    while queue:
        sq = queue.pop()
        visited.add(sq)
        (s, q) = sq
        
        dprint('Current product state:\n\t' +str(sq) )
        
        # get next states
        next_ss = fts.states.post_single(s)
        next_sqs = set()
        for next_s in next_ss:
            dprint('Next state:\n\t' +str(next_s) )
            
            Ls = fts.atomic_propositions.of(next_s)
            Ls_dict = {'in_alphabet': Ls}

            dprint("Next state's label:\n\t" +str(Ls_dict) )
            
            enabled_ba_trans = ba.transitions.with_label(q, desired_label=Ls_dict)
            dprint('Enabled BA transitions:\n\t' +str(enabled_ba_trans) )
            
            if enabled_ba_trans == set():
                continue
            
            for (q, next_q, edge_key) in enabled_ba_trans:
                new_sq = (next_s, next_q)
                next_sqs.add(new_sq)
                dprint('Adding state:\n\t' +str(new_sq) )
                
                prodts.states.add(new_sq)
                
                if ba.states.is_final(next_q):
                    final_states_preimage.add(new_sq)
                    dprint(str(new_sq) +' contains a final state.')
                
                prodts.atomic_propositions.label_state(new_sq, {next_q} )
                
                dprint('Adding transitions:\n\t' +str(sq) +'--->' +str(new_sq) )
                # is fts transition labeled with an action ?
                ts_enabled_trans = fts.transitions.with_label(
                    s, to_states={next_s}, desired_label='any'
                )
                for (from_s, to_s, edge_key) in ts_enabled_trans:
                    attr_dict = fts.get_edge_data(from_s, to_s, key=edge_key)
                    assert(from_s == s)
                    assert(to_s == next_s)
                    dprint(attr_dict)
                    
                    # labeled transition ?
                    if attr_dict == {}:
                        prodts.transitions.add(sq, new_sq)
                    else:
                        #TODO open FTS
                        prodts.transitions.add_labeled(sq, new_sq, attr_dict.values()[0] )
        
        # discard visited & push them to queue
        new_sqs = set()
        for next_sq in next_sqs:
            if next_sq not in visited:
                new_sqs.add(next_sq)
                queue.add(next_sq)
    
    return (prodts, final_states_preimage)

class RabinAutomaton(OmegaAutomaton):
    def acceptance_condition(self, prefix, suffix):
        raise NotImplementedError

class StreettAutomaton(OmegaAutomaton):
    def acceptance_condition(self, prefix, suffix):
        raise NotImplementedError

class MullerAutomaton(OmegaAutomaton):
    """Probably not very useful as a data structure for practical purposes."""
    
    def acceptance_condition(self, prefix, suffix):
        raise NotImplementedError

def ba2dra():
    """Buchi to Deterministic Rabin Automaton converter."""

def ba2ltl():
    """Buchi Automaton to Linear Temporal Logic formula convertr."""

class ParityAutomaton(OmegaAutomaton):
    
    def dump_gr1c():
        raise NotImplementedError

class ParityGameGraph():
    """Parity Games."""

class WeightedAutomaton():
    """."""

###########
# Finite-State Machines : I/O = Reactive = Open Systems
###########
def is_valuation(ports, valuations):
    for name, port_type in ports.items():
        curvaluation = valuations[name]     
        
        # functional set membership description ?
        if callable(port_type):
            ok = port_type(curvaluation)
        else:
            ok = curvaluation in port_type
        
        if not ok:
            raise TypeError('Not a valuation.')

class FiniteStateMachineSimulation(object):
    """Store, replay and export traces of runs."""
    
    def __init__(self):
        self.execution # execution_trace (Lee) (I, state, O)
        
        # derived from execution trace        
        self.path # state_trajectory (Lee)
        self.observable_trace # (I/O) = (inputs, actions)
        
        # separately inputed
        self.trace # state labels = variable names
        self.guard_valuations
        self.variables_valuations
    
    def __str__():
        """Output trace to terminal.
        
        For GUI output, use either wxpython or matlab.
        """
        raise NotImplementedError
        
    def save():
        """Dump to file."""
        raise NotImplementedError

class FiniteStateMachine(LabeledStateDiGraph):
    """Transducer, i.e., a system with inputs and outputs.
    
    Takes letters in the input alphabet one-by-one,
    follows transitions labeled by these letters, and
    returns letters in the output alphabet.
    
    Note that a transducer may operate on either finite or infinite words, i.e.,
    it is not equipped with interpretation semantics on the words,
    so it does not "care" about word length.
    It continues as long as its input is fed with letters.
    """
    def __init__(self):
        LabeledStateDiGraph.__init__(self)        
        
        self.input_ports = {'name':'type'}
        self.output_ports ={'name': 'type'}
        
        self.guards = {}
        self.output_actions = {}
        self.set_actions = {}        
        
        self.input_alphabets = {}
        self.output_alphabets = {}
        
        self.variables = {'name':'type'}
        
        self.default_export_fname = 'fsm'
    
    def is_deterministic(self):
        """Does there exist a transition for each state and each input letter ?"""
        raise NotImplementedError
    
# operations on single state machine
    def complement(self):
        raise NotImplementedError

    def determinize(self):
        raise NotImplementedError

# operations between state machines
    def sync_product(self):
        raise NotImplementedError
        
    def async_product(self):
        raise NotImplementedError
    
    def run(self, input_sequence):
        self.simulation = FiniteStateMachineSimulation()
        raise NotImplementedError

class FSM(FiniteStateMachine):
    """Alias for Finite-state Machine."""
    
    def __init__(self):
        FiniteStateMachine.__init__(self)

## transducers
class MooreMachine(FiniteStateMachine):
    """Moore machine."""
    #raise NotImplementedError

class MealyMachine(FiniteStateMachine):
    """Mealy machine."""
    #raise NotImplementedError

def moore2mealy(moore_machine, mealy_machine):
    """Convert Moore machine to equivalent Mealy machine"""
    raise NotImplementedError

####
# Program Graph (memo)
####


####
# Stochastic
####
class MarkovDecisionProcess():
    """what about
    https://code.google.com/p/pymdptoolbox/
    """
    #raise NotImplementedError

class MDP(MarkovDecisionProcess):
    """Alias."""

class PartiallyObservableMarkovDecisionProcess():
    """
    http://danielmescheder.wordpress.com/2011/12/05/training-a-pomdp-with-python/
    """
    #raise NotImplementedError

class POMDP(PartiallyObservableMarkovDecisionProcess):
    """Alias."""