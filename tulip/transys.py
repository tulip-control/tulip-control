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
import warnings
import copy
from pprint import pformat

try:
    import pydot
except ImportError:
    print "pydot package not found.\nHence dot export not unavailable."
    # python-graph package not found. Disable dependent methods.
    pydot = None

def dprint(s):
    """Debug mode print."""
    print(s)

class States(object):
    """Methods to manage states, initial states, current state.
        
    add, remove, count, test membership
    """
    def __init__(self, graph, initial_states=[], current_state=None):
        self.graph = graph
        self.initial = set(initial_states)
        self.current = current_state
    
    def __call__(self, data=False):
        """Return set of states.
        
        Default: state annotation not returned.
        To obtain that use argumet data=True.
        """
        return self.graph.nodes(data=data)
    
    def __str__(self):
        return 'States:\n\t' +pformat(self(data=False) )
    
    def __exist_labels__(self):
        """State labeling defined ?"""
        if not hasattr(self.graph, '__state_label_def__'):
            raise Exception('No state labeling defined for this class.')        
    
    def __dot_str__(self, to_pydot_graph):
        """Copy nodes to given graph, with attributes for dot export."""
        
        def if_initial_add_incoming_edge(g, state, initial_states):
            if state in initial_states:
                phantom_node = 'phantominit' +str(state)
                
                g.add_node(phantom_node, label='""', shape='none')
                g.add_edge(phantom_node, state)
        
        def form_node_label(state, state_data, label_def):
            node_dot_label = '"' +str(state) +'\\n'
            for (label_type, label_value) in state_data.iteritems():
                if label_type in label_def:
                    node_dot_label += label_type +':' +str(label_value) +'\\n'
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
        
        self.__exist_labels__()        
        
        # get labeling def
        label_def = self.graph.__state_label_def__
        
        for (state, state_data) in self.graph.nodes_iter(data=True):
            if_initial_add_incoming_edge(to_pydot_graph, state, self.initial)
            
            node_dot_label = form_node_label(state, state_data, label_def)
            node_shape = decide_node_shape(self.graph, state)
            
            # TODO replace with int to reduce size
            to_pydot_graph.add_node(state, label=node_dot_label, shape=node_shape,
                                    style='rounded')
        
    # states
    def add(self, state):
        """Create single state.
        
        C{state} can be any hashable object except None
        See networkx.MultiDiGraph.add_node.
        
        For annotating a state with a subset of atomic propositions,
        or other (custom) annotation, use the functions provided by
        AtomicPropositions, or directly the add_node function of networkx.
        """
        self.graph.add_node(state)
    
    def add_from(self, states):
        """Add multiple states from iterable container states.
        
        See networkx.MultiDiGraph.add_nodes_from.
        """
        self.graph.add_nodes_from(states)
    
    def number(self):
        """Total number of states."""
        return self.graph.number_of_nodes()
        
    def include(self, states):
        """Check if given_set_of_states \\in set_of_states."""
        for state in states:
            if not self.graph.has_node(state):
                return False
        return True
    
    def is_member(self, state):
        """Check if single state \\in set_of_states."""
        return self.graph.has_node(state)
    
    def remove(self, state):
        """Remove single state."""
        self.graph.remove_node(state)    
    
    def remove_from(self, states):
        """Remove a list of states."""
        self.graph.remove_nodes_from(states)
    
    def set_current(self, state):
        """Select current state.
        
        State membership is checked.
        If state \\notin states, exception raised.
        """
        
        if not self.is_member(state):
            raise Exception('State given is not in set of states.\n'+
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
    
    def post(self, states):
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
        if not self.include(states):
            raise Exception('Not all states given are in the set of states.')
        
        successors = set()
        for state in states:
            successors |= set(self.graph.successors(state) )
        return successors
    
    def pre_single(self, state):
        """Direct predecessors of single state.
        
        pre_single() exists to contrast with pre().
        See also post() vs post_single().
        """
        return self.pre({state} )
    
    def pre(self, states):
        """Predecessor set (1-hop) for given state.
        """
        if not self.include(states):
            raise Exception('Not all states given are in the set of states.')
        
        predecessors = set()
        for state in states:
            predecessors |= set(self.graph.predecessors(state) )
        return predecessors

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
        
        if not graph.states.is_member(from_state):
            raise Exception('from_state \\notin states.')
        
        if not graph.states.is_member(to_state):
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
        
        if not self.graph.states.include(from_states):
            raise Exception('from_states \\not\\subseteq states.')
        
        if not self.graph.states.include(to_states):
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
            self.graph.states.add_from({from_state, to_state} )
        
        if not self.graph.states.is_member(from_state):
            msg = str(from_state) +' = from_state \\notin state'
            raise Exception(msg)
        
        if not self.graph.states.is_member(to_state):
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
        if isinstance(labels, list):
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
            cur_label_def = label_def[typename]
            label_set = cur_label_def()
            
            if not check_label:
                cur_label_def.add(label)
            elif not label in label_set:
                raise Exception('Given label not in set of transition labels.')
                
        return edge_label
        
    def __dot_str__(self, to_pydot_graph):
        """Return label for dot export.
        """        
        def form_edge_label(edge_data, label_def):
            edge_dot_label = '"'
            for (label_type, label_value) in edge_data.iteritems():
                if label_type in label_def:
                    edge_dot_label += label_type +':' +str(label_value) +'\\n'
            edge_dot_label += '"'
            
            return edge_dot_label
        
        self.__exist_labels__()     
        
        # get labeling def
        label_def = self.graph.__transition_label_def__
        
        for (u, v, key, edge_data) in self.graph.edges_iter(data=True, keys=True):
            edge_dot_label = form_edge_label(edge_data, label_def)
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
        
        input:
            -C{labels} is single label, if single action set /alphabet defined,
            or if multiple action sets /alphabets, then either:
                list of labels in proper oder
                or dict of action_set_name : label pairs
        
        Checks states are already in set of states.
        Checks action is already in set of actions.
        If not, raises exception.
        
        To override, use check = False.
        Then given states are added to set of states,
        and given action is added to set of actions.
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
        for from_state in from_states:
            for to_state in to_states:
                self.add_labeled(from_state, to_state, labels, check=check)

class LabeledStateDiGraph(nx.MultiDiGraph):
    """Species: System & Automaton."""
    
    def __init__(self, name='', states=[], initial_states=[], current_state=None):
        nx.MultiDiGraph.__init__(self, name=name)
        
        self.states = States(self, initial_states, current_state)
        self.transitions = LabeledTransitions(self)

        self.dot_node_shape = {'normal':'circle'}
    
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
    # isomorphism, not implemented yet, but an Exception
    
    def __lt__(self, other):
        return self.__le__(other) and self.__ne__(other)
        
    def __ge__(self, other):
        return other.__le__(self)
        
    def __gt__(self, other):
        return other.__lt__(self)

	# operations on single transitions system
    def reachable(self):
        """Return reachable subautomaton."""
        
    def trim_dead(self):
        pass
    
    def trim_unreachable(self):
        pass
    
    # file i/o
    def load_xml(self):
        pass
        
    def dump_xml(self):
        pass
    
    def write_xml_file(self):
        pass
    
    def dump_dot(self):
        """Return dot string.
        
        Requires pydot.        
        """
        pydot_graph = self.__to_pydot__()
        
        return pydot_graph.to_string()
    
    def dot_str(self):
        """Alias to dump_dot()."""
        return self.dump_dot()
    
    def write_dot_file(self, path):
        """Save .dot file.
        
        Requires pydot.        
        """
        pydot_graph = self.__to_pydot__()
        pydot_graph.write_dot(path)
    
    def write_pdf_file(self, path):
        """Save .pdf file.
        
        Requires pydot.        
        """
        pydot_graph = self.__to_pydot__()
        pydot_graph.write_pdf(path)
    
    def dump_dot_color(self):
        pass
    
    def write_dot_color_file(self):
        pass
    
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
    
    def add(self, atomic_proposition):
        self.atomic_propositions.add(atomic_proposition)
    
    def add_from(self, atomic_propositions):
        self.atomic_propositions |= set(atomic_propositions)
        
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
        return len(self.atomic_propositions)
    
    def include(self, atomic_proposition):
        return len(self.atomic_propositions.intersect(set(atomic_proposition) ) )
    
    def add_labeled_state(self, state, ap_label):
        """Add single state with its label.
        
        input:
            state = defines element to be added to set of states S
                  = hashable object (int, str, etc)
            ap_label \in 2^AP
        """
        self.graph.states.add(state)
        self.add_from(ap_label)
        
        kw = {self.name: ap_label}
        self.graph.add_node(state, **kw)
    
    def label_state(self, state, ap_label, check=True):
        """Label state with subset of AP (Atomic Propositions).
        
        State and AP label checked, override with check = False.        
        """
        if not check:
            self.add_labeled_state(state, ap_label)
            return
        
        if not self.graph.states.is_member(state):
            raise Exception('State not in set of states.')
        
        if not set(ap_label) <= self.atomic_propositions:
            raise Exception('Label \\notsubset AP.')
        
        kw = {self.name: ap_label}
        self.graph.add_node(state, **kw)
    
    def label_states(self, states, ap_label, check=True):
        """Label multiple states with the same AP label."""
        for state in states:
            self.label_state(state, ap_label, check=True)
    
    def delabel_state(self, state):
        """Alias for remove_label_from_state()."""
    
    def remove_label_from_state(self, state):
        """."""
        
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
    
    def find_states_with_label(self, labels):
        """Return all states with label in given set."""
        pass
    
    def remove_labels_from_states(self, states):
        pass

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
    
    def add(self, action=[]):
        self.actions.add(action)
    
    def add_from(self, actions=[]):
        self.actions |= set(actions)
    
    def number(self):
        return len(self.actions)
    
    def is_member(self, action):
        return self.actions.issuperset({action} )
    
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
    """
    
    def __init__(self, name='', states=[], initial_states=[], current_state=None,
                 atomic_propositions=[], actions=[] ):
        """Note first sets of states in order of decreasing importance,
        then first state labeling, then transitin labeling (states more
        fundamentalthan transitions, because transitions need states in order to
        be defined).
        """
        LabeledStateDiGraph.__init__(
            self, name=name, states=[], initial_states=initial_states,
            current_state=current_state
        )
        
        self.atomic_propositions = AtomicPropositions(self, 'ap', atomic_propositions)
        self.actions = Actions(self, 'actions', actions)
        
        self.__state_label_def__ = {'ap': self.atomic_propositions}
        
        self.__transition_label_def__ = {'actions': self.actions}
        self.__transition_label_order__ = ['actions']

        self.dot_node_shape = {'normal':'box'}

    def __str__(self):
        s = str(self.states) +'\nState Labels:\n' +pformat(self.states(data=True) )
        s += '\n' +str(self.transitions) +'\n' +str(self.actions) +'\n'
        s += str(self.atomic_propositions) +'\n'
        
        return s
    
    def is_blocking(self):
        """Does each state have at least one outgoing transition ?
        
        Note that edge labels are NOT checked, i.e.,
        it is not checked whether for each state and each possible symbol/letter
        in the input alphabet, there exists at least one transition.
        
        The reason is that edge labels do not have any semantics at this level,
        so they are not yet regarded as guards.
        For more semantics, use a FiniteStateMachine.
        """
    
    def merge_states(self):
        pass

    # operations between transition systems
    def union(self):
        pass
    
    def intersection(self):
        pass
        
    def difference(self):
        pass

    def composition(self):
        pass
    
    def projection_on(self):
        pass
    
    def simulate(self, state_sequence="random"):
        """
            simulate automaton
                inputs="random" | given array
                mode="acceptor" | "transfucer"
        """
        # generating simulation
    
    def is_simulation(self, simulation=FTSSim() ):
        pass
    
    def loadSPINAut():
        pass

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
        
        self.__transition_label_def__ = {'sys_actions': self.sys_actions,
                                         'env_actions': self.env_actions}
        self.__transition_label_order__ = ['sys_actions', 'env_actions']
        
        self.dot_node_shape = {'normal':'box'}
        
    def __str__(self):
        s = str(self.states) +'\nState Labels:\n' +pformat(self.states(data=True) )
        s += '\n' +str(self.transitions) +'\n'
        s += str(self.sys_actions) +'\n' +str(self.env_actions) +'\n'
        s += str(self.atomic_propositions) +'\n'
        
        return s

###########
# Automata
###########
class Alphabet(object):
    """Stores input letters annotating transitions of an automaton."""
    
    def __init__(self, graph, name, letters=[]):
        self.name = name
        self.alphabet = set(letters)
        self.graph = graph
    
    def __str__(self):
        return 'Alphabet:\n' +pformat(self.alphabet)
    
    def __call__(self):
        return self.alphabet
    
    def add(self, new_input_letter):
        self.alphabet.add(new_input_letter)
    
    def add_from(self, new_input_letters):
        self.alphabet |= set(new_input_letters)
    
    def number(self):
        return len(self.alphabet)
    
    def is_member(self, letter):
        return letter in self.alphabet
    
    def remove(self, rm_input_letter):
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
    """
    
    def __init__(self, name='', states=[], initial_states=[], final_states=[],
                 input_alphabet=[]):
        LabeledStateDiGraph.__init__(
            self, name=name,
            states=states, initial_states=initial_states
        )
        
        self.final_states = set()
        self.add_final_states_from(final_states)
        
        self.alphabet = Alphabet(self, 'in_alphabet')
        self.alphabet.add_from(input_alphabet)
        
        self.__state_label_def__ = {'in_alphabet': self.alphabet}
        
        self.__transition_label_def__ = {'in_alphabet': self.alphabet}
        self.__transition_label_order__ = ['in_alphabet']
        
        self.dot_node_shape = {'normal':'circle', 'final':'doublecircle'}
        
    def __str__(self):
        s = str(self.states) +'\nState Labels:\n' +pformat(self.states(data=True) )
        s += '\n' +str(self.transitions) +'\n' +str(self.alphabet) +'\n'
        
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
        run = [initial_state]
        for letter in input_word:
            print(letter)
            
            # blocked
        
        return FSASim()

    # operations on two automata
    def add_subautomaton(self):
        pass
   
class StarAutomaton(FiniteStateAutomaton):
    """Finite-word finite-state automaton."""

class DeterninisticFiniteAutomaton(StarAutomaton):
    """Deterministic finite-word finite-state Automaton."""

    # check each initial state added
    # check each transitin added
    
    
    def is_deterministic(self):
        return True
    
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
    
def dfa2nfa():
    """Relax state addition constraint of determinism."""

class OmegaAutomaton(FiniteStateAutomaton):
    def __init__(self, states=[], initial_states=[], final_states=[],
                 input_alphabet=[]):
        FiniteStateAutomaton.__init__(self,
            states=states, initial_states=initial_states,
            final_states=final_states, input_alphabet=input_alphabet
        )

class BuchiAutomaton(OmegaAutomaton):
    def __init__(self, states=[], initial_states=[], final_states=[],
                 input_alphabet=[]):
        OmegaAutomaton.__init__(self,
            states=states, initial_states=initial_states,
            final_states=final_states, input_alphabet=input_alphabet
        )
        

    def acceptance_condition(self, prefix, suffix):
        pass

def ts_nba_synchronous_product(transition_system, buchi_automaton):
    """Construct transition system equal to synchronous product TS x NBA.
    
    Def. 4.62, p.200 [Baier]
    """
    
    prod_ts = FiniteTransitionSystem() 

class RabinAutomaton(OmegaAutomaton):
    def acceptance_condition(self, prefix, suffix):
        pass

class StreettAutomaton(OmegaAutomaton):
    def acceptance_condition(self, prefix, suffix):
        pass

class MullerAutomaton(OmegaAutomaton):
    """Probably not very useful as a data structure for practical purposes."""
    
    def acceptance_condition(self, prefix, suffix):
        pass

def ba2dra():
    """Buchi to Deterministic Rabin Automaton converter."""

def ba2ltl():
    """Buchi Automaton to Linear Temporal Logic formula convertr."""

class ParityAutomaton(OmegaAutomaton):
    
    def dump_gr1c():
        pass

class ParityGameGraph():
    """Parity Games."""

class WeightedAutomaton():
    pass

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
        
    def save():
        """Dump to file."""

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
        
        self.initial_states = {}        
        
        self.input_ports = {'name':'type'}
        self.output_ports ={'name': 'type'}
        
        self.guards = {}
        self.output_actions = {}
        self.set_actions = {}        
        
        self.input_alphabets = {}
        self.output_alphabets = {}
        
        self.variables = {'name':'type'}
    
    def set_initial_states(self):
        pass
        
    def get_initial_states(self):
        pass
    
    def is_deterministic(self):
        """Does there exist a transition for each state and each input letter ?"""
    
# operations on single state machine
    def complement(self):
        pass

    def determinize(self):
        pass

# operations between state machines
    def sync_product(self):
        pass
        
    def async_product(self):
        pass
    
    def run(self, input_sequence):
        self.simulation = FiniteStateMachineSimulation()

class FSM(FiniteStateMachine):
    """Alias for Finite-state Machine."""

## transducers
class MooreMachine(FiniteStateMachine):
    """Moore machine."""

class MealyMachine(FiniteStateMachine):
    """Mealy machine."""

def moore2mealy(moore_machine, mealy_machine):
    """Convert Moore machine to equivalent Mealy machine"""

####
# Program Graph (memo)
####


####
# Stochastic
####
class MarkovDecisionProcess():
    """
    Many implementations already available, to be adapted to networkx
    
    http://aima.cs.berkeley.edu/python/mdp.html
    https://code.google.com/p/pymdptoolbox/
    http://vrde.wordpress.com/2008/01/13/pythonic-markov-decision-process-mdp/
    http://nicky.vanforeest.com/probability/mdp/mdp.html
    https://github.com/stober/gridworld
    """
    pass

class MDP():
    """Alias."""

class PartiallyObservableMarkovDecisionProcess():
    """
    http://danielmescheder.wordpress.com/2011/12/05/training-a-pomdp-with-python/
    """

class POMDP(PartiallyObservableMarkovDecisionProcess):
    """Alias."""