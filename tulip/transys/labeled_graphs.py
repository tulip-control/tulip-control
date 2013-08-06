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
from pprint import pformat
from collections import Iterable
from cStringIO import StringIO
import warnings
import copy

import networkx as nx
#from scipy.sparse import lil_matrix # is this really needed ?

from mathset import PowerSet, is_subset

hl = 60 *'-'
debug = True

try:
    import pydot
except ImportError:
    warnings.warn('pydot package not found.\nHence dot export not unavailable.')
    # python-graph package not found. Disable dependent methods.
    pydot = None

try:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    matplotlib = True
except ImportError:
    warnings.warn('matplotlib package not found.\nSo no loading of dot plots.')
    matplotlib = None

try:
    from IPython.display import display, Image
    IPython = True
except ImportError:
    warnings.warn('IPython not found.\nSo loaded dot images not inline.')
    IPython = None

if debug:
    #import traceback
    
    def dprint(s):
        """Debug mode print."""
        print(s)
else:
    def dprint(s):
        pass

def vprint(string, verbose=True):
    if verbose:
        print(string)

class LabelConsistency(object):
    """Container of methods for checking sublabel consistency.
    
    Used by both LabeledStates and LabeledTransitions
    to verify that sublabels on states and edges are in their
    corresponding (math) sets.
    
    For example, if the 'actions' sublabel set has type: {'yes', 'no'},
    an attempt to label an FTS transition with 'not sure' will fail.
    """
    def __init__(self, label_def):
        """Link to the label definition from within the systems.
        
        @param label_def: dict defining labels:
                - for states: _state_label_def
                - for transitions: _transition_label_def
            from within each system (FTS, FSA, FSM etc)
        @type label_def: dict of the form {sublabel_name : sublabel_type}.
            For example: {'actions' : {'start', 'stop', 'turn'}, ...}
        """
        self.label_def = label_def
    
    def _attr_dict2sublabels(self, attr_dict, as_dict):
        """Extract sublabels representation from edge attribute dict.
        
        - If C{as_dict==True}, then return dict of:
            {sublabel_type : sublabel_value, ...}
        Otherwise return list of sublabel values:
            [sublabel_value, ...]
        ordered by _attr_dict2sublabels_list.
        
        see also
        --------
        _attr_dict2sublabels_list
        """
        if as_dict:
            sublabels_dict = self._attr_dict2sublabels_dict(attr_dict)
            annotation = sublabels_dict
        else:
            sublabel_values = self._attr_dict2sublabels_list(attr_dict)
            annotation = sublabel_values
        
        return annotation
    
    def _attr_dict2sublabels_list(self, attr_dict):
        """Convert attribute dict to tuple of sublabel values."""
        sublabels_dict = self._attr_dict2sublabels_dict(attr_dict)
        sublabel_values = self._sublabels_dict2list(sublabels_dict)
        return sublabel_values
    
    def _attr_dict2sublabels_dict(self, attr_dict):
        """Filter the edge attributes which are not labels.
        
        see also
        --------
        _attr_dict2sublabels_list
        
        @return: sublabel types with their values
        @rtype: {C{sublabel_type} : C{sublabel_value},...}
        """
        #self._exist_labels()
        
        sublabel_ordict = self.label_def
        sublabels_dict = {k:v for k,v in attr_dict.iteritems()
                              if k in sublabel_ordict}
        
        return sublabels_dict
            
    def _sublabels_dict2list(self, sublabels_dict):
        """Return ordered sulabel values.
        
        Sublabel values are ordered according to sublabel ordering
        defined in graph._transition_label_def, which is an OrderedDict.
        
        see also
        --------
        _sublabels_list2dict
        """
        #self._exist_labels()
        
        sublabel_ordict = self.label_def
        sublabel_values = [sublabels_dict[k] for k in sublabel_ordict
                                             if k in sublabels_dict]
        
        return sublabel_values
    
    def _sublabels_list2dict(self, sublabel_values, check_label=True):
        """Return sublabel values dict from tuple.
        
        see also
        --------
        _sublabels_dict2list
        
        @param sublabels_tuple: ordered sublabel values
        @type sublabels_tuple: tuple
        
        @param check_label: verify existence of label
        @type check_label: bool
        """
        # get labeling def
        label_def = self.label_def
        
        # single label ?
        if len(label_def) == 1:
            # hack strings for now, until deciding
            if label_def.has_key('ap'):
                sublabel_values = str2singleton(sublabel_values)
            
            dprint('Replaced sublabel value:\n\t' +str(sublabel_values) )
            sublabel_values = [sublabel_values]
            dprint('with the singleton:\n\t' +str(sublabel_values) )
        
        # constuct label dict
        edge_label = dict()
        if isinstance(sublabel_values, list) or \
        isinstance(sublabel_values, tuple):
            for i in range(len(sublabel_values) ):
                cur_name = label_def.keys()[i]
                cur_label = sublabel_values[i]
                
                edge_label[cur_name] = cur_label
        elif isinstance(sublabel_values, dict):
            edge_label = sublabel_values
        else:
            raise Exception('Bug')
        
        # check if dict is consistent with label defs
        for (typename, sublabel) in edge_label.iteritems():
            possible_labels = label_def[typename]
            
            # iterable sublabel descreption ? (i.e., discrete ?)
            if isinstance(possible_labels, Iterable):
                if not check_label:
                    if isinstance(possible_labels, PowerSet):
                        possible_labels.math_set |= sublabel
                    elif hasattr(possible_labels, 'add'):
                        possible_labels.add(sublabel)
                    elif hasattr(possible_labels, 'append'):
                        possible_labels.append(sublabel)
                    else:
                        msg = 'Possible labels described by Iterable of type:\n'
                        msg += str(type(possible_labels) ) +'\n'
                        msg += 'but it is not a PowerSet, nor does it have'
                        msg += 'an .add or .append method.\n'
                        msg += 'Failed to add new label_value.'
                        raise TypeError(msg)
                elif sublabel not in possible_labels:
                    msg = 'Given label:\n\t' +str(sublabel) +'\n'
                    msg += 'not in set of transition labels:\n\t'
                    msg += str(possible_labels) +'\n'
                    msg += 'If Atomic Propositions involved,\n'
                    msg += 'did you forget to pass an iterable of APs,\n'
                    msg += 'instead of a single AP ?\n'
                    msg += "(e.g., {'p'} instead of 'p')"
                    raise Exception(msg)
                
                continue
            
            # not iterable, check using convention:
            
            # sublabel type not defined ?
            if possible_labels == None:
                print('Undefined sublabel type')
                continue
            
            # check given ?
            #TODO change to is_valid_sublabel
            if not hasattr(possible_labels, 'is_valid_guard'):
                raise TypeError('SubLabel type V does not have method is_valid.')
            
            # check sublabel type
            if not possible_labels.is_valid_guard(sublabel):
                raise TypeError('Sublabel:\n\t' +str(sublabel) +'\n' +
                                'not valid for sublabel type:\n\t' +
                                str(possible_labels) )
            
        return edge_label
    
    def label_is_desired(self, attr_dict, desired_label):
        for (label_type, desired_val) in desired_label.iteritems():
            dprint('SubLabel type checked:\n\t' +str(label_type) )
            cur_val = attr_dict[label_type]
            dprint('possible label values:\n\t' +str(cur_val) )
            dprint('Desired label:\n\t' +str(desired_val) )
            
            if cur_val != desired_val and True not in cur_val:
                # common bug
                if isinstance(cur_val, (set,list) ) and \
                   isinstance(desired_val, (set, list) ) and \
                   cur_val.__class__ != desired_val.__class__:
                       msg = 'Set label:\n\t' +str(cur_val)
                       msg += 'compared to list label:\n\t' +str(desired_val)
                       msg += 'Did you mix sets & lists when setting AP labels ?'
                       raise Exception(msg)
                    
                return False
        return True

class States(object):
    """Methods to manage states, initial states, current state.
        
    add, remove, count, test membership
    
    mutable states
    --------------
    During language parsing, LTL->BA converion or partition refinement
    it is convenient to keep revisiting states and replacing them by others
    which refine them.
    
    For this it useful to store the objects that will be further processed
    directly as states. For example, suppose we want to store ['a', '||', 'b']
    as a state, then visit it, recognize the operator '||' and replace
    ['a', '||', 'b'] by two new states: 'a' and: 'b'.
    
    However, we cannot store ['a', '||', 'b'] as a NetworkX state, because
    a list is not hashable. There are two solutions:
    
    - recursively freeze everything
    - store actual states as labels of int states
    - maintain a bijection between states and ints,
      using the later as NetworkX states
    
    The first alternative is painful and requires that each user write their
    custom freezing code, depending on the particular data structure stored.
    The second alternative is even worse.
    
    The second approach is implemented if C{mutable==True}.
    From the user's viewpoint, everything remains the same.
    
    Using this flag can slow down comparisons, so it is appropriate for the
    special case of refinement. In many cases the resulting states after the
    refinement are hashable without special arrangements (e.g. strings).
    So the final result would then be storable in an ordinary NetworkX graph.
    
    see also
    --------
    LabeledStateDiGraph, LabeledTransitions, Transitions
    
    @param mutable: enable storage of unhashable states
    @type mutable: bool (default: False)
    """
    def __init__(self, graph, states=[], initial_states=[], current_state=None,
                 mutable=False, removed_state_callback=None):
        self.graph = graph
        self.list = list() # None when list disabled
        
        # biject mutable states <-> ints ?
        if mutable:
            self.mutants = dict()
            self.min_free_id = 0
            self._initial = list()
        else:
            self.mutants = None
            self.min_free_id = None
            self._initial = set()
        
        self.add_from(states)
        self.add_initial_from(initial_states)
        self.set_current(current_state)
        
        self._removed_state_callback = removed_state_callback
    
    def __get__(self):
        return self.__call__()
    
    def __call__(self, data=False, listed=False):
        """Return set of states.
        
        Default: state annotation not returned.
        To obtain that use argumet data=True.
        
        @param data: include annotation dict of each state
        @type data: bool
        
        @param listed:
            Return ordered states (instead of random list).
            Available only if order maintained.
            List is always returned anyway, to avoid issues with mutable states.
        
        @returns:
            If C{data==True},
                then return [(state, attr_dict),...]
            If C{data==False} and C{listed==True} and state order maintained,
                then return [state_i,...]
            If C{data==False} and C{listed==True} but no order maintained,
                then return [state_i,...] (RANDOM ORDER)
        """
        if (data != True) and (data != False):
            raise Exception('Functionality of States() changed.')
        
        if data == True:
            state_id_data_pairs = self.graph.nodes(data=True)
            
            # no replacement needed ?
            if not self._is_mutable():
                state_data_pairs = state_id_data_pairs
                return state_data_pairs
            
            # replace state_id-> state
            state_data_pairs = []
            for (state_id, attr_dict) in state_id_data_pairs:
                state = self._int2mutant(state_id)
                
                state_data_pairs += [(state, attr_dict) ]
                
            return state_data_pairs
        elif data == False:
            if listed:
                if self.list is None:
                    raise Exception('State ordering not maintained.')
                state_ids = self.list
            else:
                state_ids = self.graph.nodes(data=False)
            
            # return list, so avoid hashing issues when states are mutable
            # selection here avoids infinite recursion
            if self._is_mutable():
                return self._ints2mutants(state_ids)
            else:
                states = state_ids
                return states
        else:
            raise Exception("data must be bool\n")
    
    def __str__(self):
        return 'States:\n' +pformat(self(data=False) )    
    
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
    
    def __len__(self):
        """Total number of states."""
        return self.graph.number_of_nodes()
    
    def _is_mutable(self):
        if self.mutants is None:
            return False
        else:
            return True
    
    def _mutant2int(self, state):
        """Convert mutant to its integer ID.
        
        If C{state} \\in states, then return its int ID.
        Otherwise return the smallest available int ID, if mutable,
        or the given state, if immutable.
        
        note
        ----
        If not mutable, no check that given state is valid,
        because this direction (also) inputs to the data structure new states.
        
        see also
        --------
        _int2mutant
        
        @param state: state to check for
        
        @returns:
            If states not mutable,
                then return given C{state}.
            If C{state} does not exist and states mutable,
                then return min free int ID.
            If C{state} does exist and states mutable,
                then return its int ID.
        """
        
        # classic NetworkX ?
        if not self._is_mutable():
            #dprint('Immutable states (must be hashable): classic NetworkX.\n')
            return state
        dprint('Mutable states.')
        
        mutants = self.mutants
        state_id = [x for x in mutants if mutants[x] == state]
        
        dprint('Converted: state = ' +str(state) +' ---> ' +str(state_id) )
        
        # found state ?
        if len(state_id) == 0:
            dprint('No states matching. State is new.\n')
        elif len(state_id) == 1:
            return state_id[0]
        else:
            msg = 'Found multiple state_ids with the same state !\n'
            msg += 'This violates injectivity from IDs to states.\n'
            msg += 'In particular, state:\n\t' +str(state) +'\n'
            msg += 'is a common value for the keys:\n\t' +str(state_id)
            raise Exception(msg)
        
        # new, get next free id
        return self.min_free_id
    
    def _int2mutant(self, state_id):
        """Convert integer ID to its mutant.
        
        If C{state_id} \\in used IDs, then return corresponding state.
        Otherwise return None, or the given state, if not mutable.
        
        note
        ----
        If not mutable, given int checked to be valid state,
        because this direction outputs to the world.
        
        see also
        --------
        _mutant2int_
        
        @param state_id: ID number to check for
        @type state_id:
            int, if mutable
            valid state, if immutable
        
        @returns:
            If states not mutable,
                then return given argument, because it is the actual state.
            If states are mutable and C{state_id} is used,
                then return corresponding C{state}.
            If states are mutable but C{state_id} is free,
                then return None.
        """
        
        # classic NetworkX ?
        if not self._is_mutable():
            state = state_id
            
            if state not in self():
                msg = 'States are immutable.\n.'
                msg = 'Given integer ID is not a state.\n'
                raise Exception(msg)
            return state
        
        mutants = self.mutants
        
        # found ID ?
        if state_id in mutants:
            state = mutants[state_id]
            dprint('For ID:\n\t' +str(state_id) +'\n'
                   +'Found state:\n\t' +str(state) )
            return state
        
        # mutable, but ID unused
        dprint('Mutable states, but this ID is currently unused.')
        return None
    
    def _mutants2ints(self, states):
        return map(self._mutant2int, states)
    
    def _ints2mutants(self, ints):
        return map(self._int2mutant, ints)
    
    def __contains__(self, state):
        """Check if single state \\in set_of_states.
        
        @param state: Check if C{state} already in states.
        @type state: if mutable states enabled, then any type,
            otherwise must be hashable
        
        @return: C{True} if C{state} is in states.
        @rtype: bool
        """
        state_id = self._mutant2int(state)
        return self.graph.has_node(state_id)
    
    def _exist_final_states(self, msg=True):
        """Check if system has final states."""
        if not hasattr(self.graph, 'final_states'):
            if msg:
                warnings.warn('System of type: ' +str(type(self.graph) ) +
                              'does not have final states.')
            return False
        else:
            return True
    
    def _warn_if_state_exists(self, state):
        if state in self():
            if self.list is not None:
                raise Exception('State exists and ordering enabled: ambiguous.')
            else:
                warnings.warn('State already exists.')
                return
    
    # states
    def add(self, new_state):
        """Create single state.
        
        The new state must be hashable, unless mutable states are enabled.
        For details about mutable states see the docstring of transys.States.
        
        For annotating a state with a subset of atomic propositions,
        or other (custom) annotation, use the functions provided by
        AtomicPropositions, or directly the NetworkX.MultiDiGraph.add_node method.
        
        see also
        --------
        networkx.MultiDiGraph.add_node
        
        @param new_state:
            Single new state to add.
        @type new_state:
            If states immutable, then C{state} must be a hashable object.
                Any hashable allowed, except for None (see nx add_node below).
            If states mutable, then C{state} can be unhashable.
        """
        new_state_id = self._mutant2int(new_state)
        self._warn_if_state_exists(new_state)
        
        dprint('Adding new id: ' +str(new_state_id) )
        self.graph.add_node(new_state_id)
        
        # mutant ?
        if self._is_mutable():
            self.mutants[new_state_id] = new_state
            
            # find min free id
            found = False
            while not found:
                self.min_free_id = self.min_free_id +1
                
                if not self.mutants.has_key(self.min_free_id):
                    found = True
        
        # list maintained ?
        if self.list is not None:
            self.list.append(new_state)
    
    def add_from(self, new_states, destroy_order=False):
        """Add multiple states from iterable container.
        
        see also
        --------
        networkx.MultiDiGraph.add_nodes_from.
        """
        def check_order(new_states):
            # ordered ?
            if isinstance(new_states, list):
                return
            
            # interable at least ?
            if not isinstance(new_states, Iterable):
                raise Exception('New set of states must be iterable container.')
            
            # no order currently maintained ?
            if self.list is None:
                return
            
            # no states stored yet ?
            if len(self.list) == 0:
                warnings.warn("Will add non-list to empty system with ordering."+
                              "Won't remember state order from now on.")
                self.list = None
                return
            
            # cancel ordering of already stored states ?
            if destroy_order:
                warnings.warn('Added non-list of new states.'+
                              'Existing state order forgotten.')
                self.list = None
                return
            
            raise Exception('Ordered states maintained.'+
                            'Please add list of states instead.')
        
        check_order(new_states)
        
        # iteration used for comprehensible error message
        for new_state in new_states:
            self._warn_if_state_exists(new_state)
        
        # mutable ?
        if self._is_mutable():
            for new_state in new_states:
                self.add(new_state)
        else:
            self.graph.add_nodes_from(new_states)
        
            # list maintained ?
            if self.list is not None:
                self.list = self.list +list(new_states)
    
    def remove(self, rm_state):
        """Remove single state."""
        
        # not a state ?
        if rm_state not in self():
            warnings.warn('Attempting to remove inexistent state.')
            return
        
        state_id = self._mutant2int(rm_state)
        self.graph.remove_node(state_id)
        
        # are mutants ?
        if self._is_mutable():
            self.mutants.pop(state_id)
            self.min_free_id = min(self.min_free_id, state_id)
        
        # ordering maintained ?
        if self.list is not None:
            self.list.remove(rm_state)
        
        # rm if init
        if self.is_initial(rm_state):
            self.remove_initial(rm_state)
        
        # chain to parent (for final states etc)
        if self._removed_state_callback:
            self._removed_state_callback(rm_state)
    
    def remove_from(self, rm_states):
        """Remove a list of states."""
        for rm_state in rm_states:
            self.remove(rm_state)
    
    def set_current(self, states):
        """Select current state.
        
        State membership is checked.
        If state \\notin states, exception raised.
        
        None is possible.
        """
        if states is None:
            self.current = None
            return
        
        if not is_subset(states, self() ):
            raise Exception('Current state given is not in set of states.\n'+
                            'Cannot set current state to given state.')
        
        self.current = states
    
	# initial states
    def _get_initial(self):
        return self._initial
    
    initial = property(_get_initial)
    
    def add_initial(self, new_initial_state):
        """Add state to set of initial states.
        
        C{new_initial_state} should already be a state.
        First use states.add to include it in set of states,
        then states.add_initial.
        """
        if not new_initial_state in self():
            raise Exception(
                'New initial state \\notin States.\n'
                'Add it first to states using sys.states.add()\n'
                'FYI: new initial state:\n\t' +str(new_initial_state) +'\n'
                'and States:\n\t' +str(self() )
            )
        
        # ensure uniqueness for unhashable states
        if self.is_initial(new_initial_state):
            warnings.warn('Already an initial state.\n')
            return
        
        # use sets when possible for efficiency
        if self._is_mutable():
            self._initial.append(new_initial_state)
        else:
            self._initial.add(new_initial_state)

    def add_initial_from(self, new_initial_states):
        """Add multiple initial states.
        
        Should already be in set of states.
        """
        if len(new_initial_states) == 0:
            return
        
        if self._is_mutable():
            self._initial |= set(new_initial_states)
        else:
            for new_initial_state in new_initial_states:
                self.add_initial(new_initial_state)
    
    def remove_initial(self, rm_initial_state):
        """Delete single state from set of initial states."""
        if self.is_initial(rm_initial_state):
            self._initial.remove(rm_initial_state)
        else:
            warnings.warn('Attempting to remove inexistent initial state.'
                          +str(rm_initial_state) )
    
    def remove_initial_from(self, rm_initial_states):
        """Delete multiple states from set of initial states."""
        if len(rm_initial_states) == 0:
            return
        
        if self._is_mutable():
            self._initial = self._initial.difference(rm_initial_states)
        else:
            # mutable states
            for rm_initial_state in rm_initial_states:
                self.remove_initial(rm_initial_state)
    
    def is_initial(self, state):
        return is_subset([state], self._initial)
    
    def is_final(self, state):       
        """Check if state \\in final states.
        
        Convenience method, violates class independence,
        so might be removed in the future.
        """
        if not self._exist_final_states():
            return
        
        return is_subset([state], self.graph.final_states)
    
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
        if not is_subset(self._initial, self() ):
            warnings.warn('Ininital states \\not\\subseteq states.')
        
        if self.current is None:
            warnings.warn('Current state unset.')
            return
        
        if self.current not in self():
            warnings.warn('Current state \\notin states.')
        
        print('States and Initial States are ok.\n'
              +'For final states, refer to my parent.')
    
    def post_single(self, state):
        """Direct successors of a single state.
        
        post_single() exists to contrast with post().
        
        post() cannot guess when it is passed a single state, or multiple states.
        Reason is that a state may happen to be anything,
        so possibly something iterable.
        """
        state_id = self._mutant2int(state)
        return self.post([state_id] )
    
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
        if not is_subset(states, self() ):
            raise Exception('Not all states given are in the set of states.\n'+
                            'Did you mean to use port_single() instead ?')
        
        state_ids = self._mutants2ints(states)
        
        successors = set()
        for state_id in state_ids:
            successors |= set(self.graph.successors(state_id) )
        return successors
    
    def pre_single(self, state):
        """Direct predecessors of single state.
        
        pre_single() exists to contrast with pre().
        
        see also
        --------
        post() vs post_single().
        """
        state_id = self._mutant2int(state)
        return self.pre([state_id] )
    
    def pre(self, states):
        """Predecessor set (1-hop) for given state.
        """
        if not is_subset(states, self() ):
            raise Exception('Not all states given are in the set of states.')
        
        state_ids = self._mutants2ints(states)
        
        predecessors = set()
        for state_id in state_ids:
            predecessors |= set(self.graph.predecessors(state_id) )
        return predecessors
    
    def add_final(self, state):
        """Convenience for FSA.add_final_state().
        
        see also
        --------
        self.add_final_from  
        """
        if not self._exist_final_states():
            return
        
        self.graph.add_final_state(state)
    
    def add_final_from(self, states):
        """Convenience for FSA.add_final_states_from().
        
        see also
        --------
        self.add_final
        """
        if not self._exist_final_states():
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
        
class LabeledStates(States):
    """States with annotation.
    
    For FTS and OpenFTS each state label consists of a single sublabel,
    which a subset of AP, the set of atomic propositions.
    
    For Machines, each state label consists of (possibly multiple) sublabels,
    each of which is either a variable, or, only for Moore machines,
    may be an output.
    """
    """Store & print set of atomic propositions.

    Note that any transition system or automaton is just annotated by atomic
    propositions. They are either present or absent.
    Their interpretation is external to this module.
    That is, evaluating whether an AP is true or false, so present or absent as
    a member of a set of APs requires semantics defined and processed elsewhere.
    
    The simplest representation for APs stored here is a set of strings.
    """
    def __init__(self, graph, states=[], initial_states=[], current_state=None,
                 mutable=False, removed_state_callback=None):
        States.__init__(self, graph, states=states,
                        initial_states=initial_states,
                        current_state=current_state, mutable=mutable,
                        removed_state_callback=removed_state_callback)
        
        # labeling defined ?
        if hasattr(self.graph, '_state_label_def'):
            self._label_check = \
                LabelConsistency(self.graph._state_label_def)
        else:
            self._label_check = None
    
    def _exist_state_labels(self):
        if self._label_check is None:
            raise Exception('State labeling not defined for this system.\n' +
                            'The system type = ' +str(type(self.graph) ) )
    
    def _exist_labels(self):
        """State labeling defined ?"""
        if hasattr(self.graph, '_state_label_def'):
            return True
        else:
            # don't prevent dot export
            msg = 'No state labeling defined for class:\n\t'
            msg += str(type(self.graph) )
            dprint(msg)
            return False
    
    def _dot_str(self, to_pydot_graph):
        """Copy nodes to given Pydot graph, with attributes for dot export."""
        
        def add_incoming_edge(g, state):
            phantom_node = 'phantominit' +str(state)
            
            g.add_node(phantom_node, label='""', shape='none', width='0')
            g.add_edge(phantom_node, state)
        
        def form_node_label(state, state_data, label_def, label_format):
            # node itself
            node_dot_label = '"' +str(state) +'\\n'
            
            # add node annotations from action, AP sets etc
            # other key,values in state attr_dict ignored
            sep_label_sets = label_format['separator']
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
            if not self._exist_final_states(msg=False):
                return node_shape
            
            # check for final states
            if self.is_final(state):
                node_shape = graph.dot_node_shape['final']
                
            return node_shape
        
        # get labeling def        
        if self._exist_labels():
            label_def = self.graph._state_label_def
            label_format = self.graph._state_dot_label_format
        
        for (state_id, state_data) in self.graph.nodes_iter(data=True):
            state = self._int2mutant(state_id)
            
            if  self.is_initial(state):
                add_incoming_edge(to_pydot_graph, state_id)
            
            node_shape = decide_node_shape(self.graph, state)
            
            # state annotation
            if self._exist_labels():
                node_dot_label = form_node_label(state, state_data, label_def, label_format)
            else:
                node_dot_label = str(state)
            
            # state boundary color
            if state_data.has_key('color'):
                node_color = state_data['color']
            else:
                node_color = '"black"'
            
            # state interior color
            node_style = '"rounded'
            if state_data.has_key('fillcolor'):
                node_style += ',filled"'
                fill_color = state_data['fillcolor']
            else:
                node_style += '"'
                fill_color = "none"
            
            # TODO option to replace with int to reduce size,
            # TODO generate separate LaTeX legend table (PNG option ?)
            to_pydot_graph.add_node(
                state_id, label=node_dot_label, shape=node_shape,
                style=node_style, color=node_color, fillcolor=fill_color)
    
    def _check_state(self, state, check=True):
        if not check:
            # add only if absent, to avoid order-preserving exceptions
            if state not in self:
                self.graph.states.add(state)
        
        if state not in self:
            msg = 'State:\n\t' +str(state) +'\n'
            msg += 'is not in set of states:\n\t' +str(self)
            raise Exception(msg)
    
    def label(self, state, label, check=True):
        """Add single state with its label.
        
        see also
        --------
        labels, find
        
        @param state: added to set of states
        @type state: for mutable system states any type,
            otherwise must be hashable (if in classic NetworkX mode)
        
        @param label: sublabels used to annotate C{state}
        @type label: ordered Iterable of labels
        """
        self._exist_state_labels()
        self._check_state(state, check=check)
        
        sublabels = label
        
        state_label = self._label_check._sublabels_list2dict(
            sublabels, check_label=check
        )
        
        state_id = self._mutant2int(state)
        self.graph.add_node(state_id, **state_label)
        
        #if not is_subset(label, self.graph.atomic_propositions):
        #    msg = 'Label \\not\\subset AP.'
        #    msg += 'FYI Label:\n\t' +str(label) +'\n'
        #    msg += 'AP:\n\t' +str(self.graph.atomic_propositions) +'\n'
        #    msg += 'Note that APs must be in an iterable,\n'
        #    msg += "even single ones, e.g. {'p'}."
        #    raise Exception(msg)
    
    def labels(self, states, label_list='paired', check=True):
        """Label multiple states, each with a (possibly) different AP label.
        
        input formats
        -------------
        Two ways of passing the state labeling:
            - separately:
                - C{states = [s0, s1, ...] }
                - C{label_list = [L0, L1, ...] }
            where state: s0, is labeled with label: L0, etc.
            
            For labeling all states, this approach requires typing
            fewer parentheses. If C{label_list} is a singleton,
            then it is used for labeling all states.
            
            - paired:
                - C{states = [(s0, L0), (s1, L1), ...] }
                - C{label_list = 'paired'}
            
            Setting C{label_list} resolves ambiguity which can
            be caused otherwise, because states are allowed to
            be anything. So testing if states is an Iterable
            of pairs would not necessarily imply they are
            state-label pairs. They could be states which are
            themselves tuples, e.g. in case of product automata.
            
            To reduce the user's load, by default
            C{label_list} is C{'paired'}.
        
        creating states
        ---------------
        If no states currently exist and C{states='create'},
        then new states 0,...,N-1 are created,
        where: N = C{len(label_list) }.
        
        examples
        --------
        >>> from tulip import transys as trs
        >>> fts = trs.FTS()
        >>> fts.states.add_from(['s0', 's1'] )
        
        >>> fts.atomic_propositions.math_set |= ['p', '!p']
        
        >>> states = ['s0', 's1']
        >>> state_labels = [{'p'}, {'!p'} ]
        >>> fts.states.labels(states, state_labels)
        
        You can skip adding them and directly label:
        
        >>> fts.states.labels(states, state_labels, check=False)
        
        The following three are equivalent:
        
        >>> fts.states.labels([1, 2], [{'p'}, {'!p'}], check=False)
        >>> fts.states.labels(range(2), [{'p'}, {'!p'}], check=False)
        >>> fts.states.labels('create', [{'p'}, {'!p'}] )
        
        see also
        --------
        label, find, LabeledTransitions, States, FTS, BA, FSM
        
        @param states: existing states to be labeled with ap_label_list,
            or string 'create' to cause creation of new int ID states
        @type states: interable container of existing states
            | str 'create'
        
        @param ap_label_list: valid AP labels for annotating C{states}
        @type ap_label_list: list of valid labels
            | 'paired' (see "input formats" above)
        
        @param check: check if given states and given labels already exist.
            If C{check=False}, then each state passed is added to system,
            and each AP is added to the APs of the system.
        @type check: bool
        """
        if states == 'create':
            n = len(label_list)
            states = range(n)
            check = False
        
        # already paired ?
        if label_list == 'paired':
            state_label_pairs = states
        else:
            # single label for all states ?
            if len(label_list) == 1:
                print('Single state label applied to each state.')
                label_list = len(states) *label_list
            
            state_label_pairs = zip(states, label_list)
        
        for state, curlabel in state_label_pairs:
            self.label(state, curlabel, check=check)
    
    def find(self, states='any', desired_label='any', as_dict=True):
        """Filter by desired states and by desired state labels.
        
        see also
        --------
        label, labels, LabeledTransitions.find
        
        @param states: subset of states over which to search
        @type states: 'any' (default)
            | iterable of valid states
            | single valid state
        
        @param desired_label: label with which to filter the states
        @type desired_label: {sublabel_type : desired_sublabel_value, ...}
            | 'any', to allow any state label (default)
        
        @param as_dict:
            - If C{True}, then return sublabels as dict:
                {sublabel_type : sublabel_value}
            - Otherwise return sublabel values ordered in list,
              the list order based on graph._transition_label_def
        @type as_dict: bool
        
        @return: set of labeled states:
                (C{state}, label)
            such that:
                C{state} \\in C{states} given as first argument
                
        @rtype: list of labeled states
                = [(C{state}, C{label}),...]
            where:
                - C{state} \\in C{states}
                - C{label}: dict
                    | tuple of edge annotation,
                    determined by C{as_dict}.
        """
        #TODO support tuples as desired_labels, using available conversion
        if states is 'any':
            state_ids = 'any'
        else:
            # singleton check
            if states in self:
                state = states
                msg = 'LabeledStates.find got single state: ' +str(state) +'\n'
                msg += 'instead of Iterable of states.\n'
                states = [state]
                msg += 'Replaced given states = ' +str(state)
                msg += ' with states = ' +str(states)
                warnings.warn(msg)
                
            state_ids = self._mutants2ints(states)
            dprint(state_ids)
        
        found_state_label_pairs = []
        for state_id, attr_dict in self.graph.nodes_iter(data=True):
            dprint('Checking state_id = ' +str(state_id) +
                   ', with attr_dict = ' +str(attr_dict) )
            
            if state_id not in state_ids and state_ids is not 'any':
                dprint('state_id = ' +str(state_id) +', not desired.')
                continue
            
            dprint('Checking state label:\n\t attr_dict = ' +str(attr_dict) +
                   '\n vs:\n\t desired_label = ' +str(desired_label) )
            if desired_label is 'any':
                dprint('Any label acceptable.')
                ok = True
            else:
                ok = self._label_check.label_is_desired(attr_dict, desired_label)
            
            if ok:
                dprint('Label Matched:\n\t' +str(attr_dict) +
                      ' == ' +str(desired_label) )
                
                state = self._int2mutant(state_id)
                annotation = \
                    self._label_check._attr_dict2sublabels(attr_dict, as_dict)
                state_label_pair = (state, annotation)
                
                found_state_label_pairs.append(state_label_pair)
            else:
                dprint('No match for label---> state discarded.')
        
        return found_state_label_pairs
        
        #except KeyError:
        #warnings.warn("State: " +str(state) +", doesn't have AP label.")
        #return None
    
    def delabel(self, states):
        """Remove labels from states."""
        
        raise NotImplementedError
    
    def _check_sublabeling(self, state_sublabel_value):
        """Verify consistency: all sublabels used are defined.
        
        To be used after or before removing a sublabel value from
        its corresponding type definition.
        Previously these checks where automatically performed,
        using dedicated classes. But dedicating classes to
        checks unnecessarily duplicates data types as
        for example MathSet or PowerSet, which are otherwise
        between systems and can be reused.
        """
        raise NotImplementedError
        
        node_ap = nx.get_node_attributes(self.graph, self.name)
        
        nodes_using_ap = set()
        for (node, ap_subset) in node_ap.iteritems():
            if state_sublabel_value in ap_subset:
                nodes_using_ap.add(node)                
        
        if nodes_using_ap:
            msg = 'AP (=' +str(state_sublabel_value) +') still used '
            msg += 'in label of nodes: ' +str(nodes_using_ap)
            raise Exception(msg)
        
        self.atomic_propositions = \
            self.atomic_propositions.difference({state_sublabel_value} )

class Transitions(object):
    """Building block for managing unlabeled transitions = edges.
    
    Note that a directed edge is an ordered set of nodes.
    Unlike an edge, a transition is a labeled edge.
    However, labelings may vary, so they are defined separately and methods for
    working with labeled transitions are defined in the respective classes.
    """
    def __init__(self, graph):
        self.graph = graph
    
    def __call__(self):
        """Return list of transitions.
        
        The transitions are yet unlabeled, so they are graph edges,
        i.e., ordered pairs of states: (s1, s2).
        The edge direction is from s1 to s2, i.e., s1-> s2.
        
        LabeledTransitions overload this to return transitions,
        i.e., labeled edges = triples: (s1, s2, label).
        
        see also
        --------
        LabeledTransitions.__call__
        """
        return self.graph.edges(data=False)
    
    def __str__(self):
        return 'Transitions:\n' +pformat(self() )
    
    def _mutant2int(self, from_state, to_state):
        from_state_id = self.graph.states._mutant2int(from_state)
        to_state_id = self.graph.states._mutant2int(to_state)
        
        return (from_state_id, to_state_id)
    
    def add(self, from_state, to_state, check_states=True):
        """Add unlabeled transition, checking states \\in set of states.
        
        If either state not in set of states, raise exception.
        
        Argument check_states = False can override the check.
        If check_states = False, and states not already in set of states,
        then they are added.
        """
        if not isinstance(check_states, bool):
            raise TypeError('check_states must be bool.\n'
                            +'Maybe you intended to call add_labeled instead ?')
        
        if not check_states:
            self.graph.states.add_from({from_state, to_state} )
        
        if from_state not in self.graph.states():
            raise Exception('from_state:\n\t' +str(from_state) +
                            '\\notin states:\n\t' +str(self.graph.states() ) )
        
        if to_state not in self.graph.states():
            raise Exception('to_state:\n\t' +str(to_state) +
                            '\\notin states:\n\t' +str(self.graph.states() ) )
        
        (from_state_id, to_state_id) = self._mutant2int(from_state, to_state)
        
        dprint('Adding transition:\n\t'
               +str(from_state_id) +'--->' +str(to_state_id) )
        
        # if another un/labeled edge already exists between these nodes,
        # then avoid duplication of edges
        if not self.graph.has_edge(from_state_id, to_state_id):
            self.graph.add_edge(from_state_id, to_state_id)
    
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
        
        if not is_subset(from_states, self.graph.states() ):
            raise Exception('from_states \\not\\subseteq states.')
        
        if not is_subset(to_states, self.graph.states() ):
            raise Exception('to_states \\not\\subseteq states.')
        
        for from_state in from_states:
            for to_state in to_states:
                self.graph.add_edge(from_state, to_state)
    
    def add_adj(self, adj):
        """Add multiple transitions from adjacency matrix.
        
        These transitions are not labeled.
        To label then, use either LabeledTransitions.relabel(),
        or remove() and then LabeledTransitions.add_labeled_adj().
        
        @param adj: new transitions, represented by the
            non-zero elements of an adjacency matrix.
            Note that adjacency here is in the sense of nodes
            and not spatial.
        @type adj: scipy.sparse.lil (list of lists)
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
            dprint('Added ordered list of states: ' +str(self.graph.states.list) )
        
        # convert to format friendly for edge iteration
        nx_adj = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph())
        
        # add each edge using existing checks
        states_list = self.graph.states.list
        for edge in nx_adj.edges_iter():
            (from_idx, to_idx) = edge
            
            from_state = states_list[from_idx]
            to_state = states_list[to_idx]
            
            self.add(from_state, to_state)
    
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
        (from_state_id, to_state_id) = self._mutant2int(from_state, to_state)
        
        edge_set = copy.copy(self.graph.get_edge_data(from_state_id, to_state_id) )
        for (edge_key, label) in edge_set.iteritems():
            if label == {}:
                self.graph.remove_edge(from_state_id, to_state_id, key=edge_key)
    
    def remove_from(self, from_states, to_states):
        """Delete all unlabeled transitions between multiple state pairs.
        
        See also remove().        
        """
        for from_state in from_states:
            for to_state in to_states:
                self.remove(from_state, to_state)
    
    def between(self, from_states, to_states):
        """Return list of edges between given nodes.
        
        Filtering the edge set E is based on end-point states of edge,
        because edges are not yet labeled.
        To search over labeled edges = transitions, see LabeledTransitions.find
        
        note
        ----
        filter around NetworkX.MultiDiGraph.edges()
        
        see also
        --------
        LabeledTransitions.find
        
        @param start_states: from where transition should start
        @type start_states: valid states
        
        @param end_states: where transition should end
        @type end_states: valid states
        
        @return: Edges between given subsets of states
        @rtype: list of state pairs as tuples:
            [(C{from_state}, C{to_state}), ...]
        such that:
            - C{from_state} \\in C{from_states} and
            - C{to_state} \\in C{to_states}
        """
        edges = []
        for (from_state, to_state) in self.graph.edges_iter(
            from_states, data=False, keys=False
        ):
            if to_state in to_states:
                edges.append((from_state, to_state) )
        return edges

class LabeledTransitions(Transitions):
    """Manage labeled transitions (!= edges).
    
    Each transition is a graph edge (s1, s2) paired with a label.
    The label can consist of one or more pieces, called sub-labels.
    Each sub-label is an element from some pre-defined set.
    This set might e.g. be various actions, as {work, sleep}.
    
    This class is for defining and managing transitions, together with
    the set of elements from which sub-label are picked.
    Note that in case an edge label comprises of a single sub-label,
    then the notions of label and sub-label are identical.
    
    But for systems with more sub-labels,
        e.g., {system_actions, environment_actions}
    a label consists of two sub-labels, each of which can be selected
    from the set of available system actions and environment actions.
    Each of these sets is defined using this class.
    
    The purpose is to support labels with any number of sub-labels,
    without the need to re-write keyword-value management of
    NetworkX edge dictionaries every time this is needed.
    
    caution
    -------
    Before removal of a sublabel value from the sublabel type V,
    remember to check using sys.transitions.check_sublabeling()
    that the value is not currently used by any edges.
    
    example
    -------
    The action taken when traversing an edge.
    Each edge is annotated by a single action.
    If an edge (s1, s2) can be taken on two transitions,
    then 2 copies of that same edge are stored.
    Each copy is annotated using a different action,
    the actions must belong to the same action set.
    That action set is defined as a ser instance.
    This description is a (closed) FTS.
    
    The system and environment actions associated with an edge
    of a reactive system. To store these, 2 sub-labels are used
    and their sets are encapsulated within the same (open) FTS.
    
    In more detail, the following classes encapsulate this one:
        FiniteTransitionSystem (closed)
        OpenFiniteTransitionSystem
        FiniteStateAutomaton
        FiniteStateMachine
    
    see also
    --------
    Transitions
    """
    def __init__(self, graph):
        Transitions.__init__(self, graph)
        
        # labeling defined ?
        if hasattr(self.graph, '_transition_label_def'):
            self._label_check = \
                LabelConsistency(self.graph._transition_label_def)
        else:
            self._label_check = None
    
    def __call__(self, labeled=False, as_dict=True):
        """Return all edges, optionally paired with labels.
        
        note
        ----
        __call__(labeled=True, as_dict=True) is equivalent to find(),
        i.e., find without any restrictions on the desired
        from_state, to_state, nor sublabels.
        
        see also
        --------
        find
        
        @param labeled: If C{True}, then return labeled edges
        @type labeled: bool
        
        @param as_dict:
            - If C{True}, then return sublabel values keyed by sublabel type:
                {sublabel_type : sublabel_value, ...}
            - Otherwise return list of sublabel values ordered by
                _transition_label_def
        @type as_dict: bool
        
        @return: labeled or unlabeled edges, depending on args.
        @rtype: list of edges = unlabeled transitions = [(s1, s2), ...]
            | list of labeled edges = transitions = [(s1, s2, L), ...]
            where the label L = dict | list, depending on args.
        """
        if not labeled:
            return self.graph.edges(data=False)
        
        edges = [] # if labeled, should better be called "transitions"
        for (from_node, to_node, attr_dict) in self.graph.edges_iter(data=True):
            annotation = self._label_check._attr_dict2sublabels(attr_dict, as_dict)
            edge = (from_node, to_node, annotation)
            edges.append(edge)
        
        return edges
    
    def _exist_labels(self):
        if self._label_check is None:
            raise Exception('Transition labeling not defined for:\n\t.' +
                            str(self.graph) )
    
    def _check_states(self, from_state, to_state, check=True):
        """Are from_state, to_state \\in states.
        
        If check == False, then add them.
        """
        if not check:
            # attempt adding only if not already in set of states
            # to avoid ordering-related exceptions
            if from_state not in self.graph.states:
                self.graph.states.add(from_state)
            if to_state not in self.graph.states:
                self.graph.states.add(to_state)
        
        if from_state not in self.graph.states:
            msg = 'from_state:\n\t' +str(from_state)
            msg += '\n\\notin States:' +str(self.graph.states() )
            raise Exception(msg)
        
        if to_state not in self.graph.states:
            msg = str(to_state) +' = to_state \\notin state'
            raise Exception(msg)
        
    def _dot_str(self, to_pydot_graph):
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
                    elif isinstance(label_value, Iterable):
                        label_str = str(list(label_value) )
                    else:
                        label_str = str(label_value)
                    
                    edge_dot_label += type_name +sep_type_value
                    edge_dot_label += label_str +sep_label_sets
            edge_dot_label += '"'
            
            return edge_dot_label
        
        self._exist_labels()
        
        # get labeling def
        label_def = self.graph._transition_label_def
        label_format = self.graph._transition_dot_label_format
        
        for (u, v, key, edge_data) in self.graph.edges_iter(data=True, keys=True):
            edge_dot_label = form_edge_label(edge_data, label_def, label_format)
            to_pydot_graph.add_edge(u, v, key=key, label=edge_dot_label)
    
    def _mutable2ints(self, from_states, to_states):
        """Convert possibly unhashable states to internal ones.
        
        If states are hashable, the internal ones are the same.
        Otherwise the internal ones are ints maintained in bijection
        with the mutable states.
        """
        if from_states == 'any':
            from_state_ids = 'any'
        else:
            from_state_ids = self.graph.states._mutants2ints(from_states)
        
        if to_states == 'any':
            to_state_ids = 'any'
        else:
            to_state_ids = self.graph.states._mutants2ints(to_states)
        
        return (from_state_ids, to_state_ids)
    
    def remove_labeled(self, from_state, to_state, label):
        self._exist_labels()
        self._check_states(from_state, to_state, check=True)
        edge_label = \
            self._label_check._sublabels_list2dict(label, check_label=True)
        
        # get all transitions with given label
        (from_state_id, to_state_id) = self._mutant2int(from_state, to_state)
        edge_set = copy.copy(self.graph.get_edge_data(from_state_id, to_state_id,
                                                      default={} ) )
        
        found_one = 0
        for (edge_key, label) in edge_set.iteritems():
            dprint('Checking edge with:\n\t key = ' +str(edge_key) +'\n')
            dprint('\n\t label = ' +str(label) +'\n')
            dprint('\n against: ' +str(edge_label) )
            
            if label == edge_label:
                dprint('Matched. Removing...')
                self.graph.remove_edge(from_state_id, to_state_id, key=edge_key)
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
        self._exist_labels()
        self._check_states(from_state, to_state, check=True)
        
        # chek if same unlabeled transition exists
        from_state_id = self.graph.states._mutant2int(from_state)
        to_state_id  = self.graph.states._mutant2int(to_state)
        
        trans_from_to = self.graph.get_edge_data(
            from_state_id, to_state_id, default={} )
        
        if {} not in trans_from_to.values():
            msg = 'Unlabeled transition from_state-> to_state '
            msg += "doesn't exist,\n where:\t"
            msg += 'from_state = ' +str(from_state) +'\n'
            msg += 'and:\t to_state = ' +str(to_state) +'\n'
            msg += 'So it cannot be labeled.\n'
            msg += 'Either add it first using: transitions.add() '
            msg += 'and then label it,\n'
            msg += 'or use transitions.add_labeled(), '
            msg += 'same with a single call.\n'
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
        self._exist_labels()
        self._check_states(from_state, to_state, check=True)
        
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
        self._exist_labels()
        self._check_states(from_state, to_state, check=check)
        
        # chek if same unlabeled transition exists
        (from_state_id, to_state_id) = self._mutant2int(from_state, to_state)
        trans_from_to = self.graph.get_edge_data(from_state_id, to_state_id,
                                                 default={} )
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
        edge_label = \
            self._label_check._sublabels_list2dict(labels, check_label=check)
        
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
        self.graph.add_edge(from_state_id, to_state_id, **edge_label)
    
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
        
        @param adj: new transitions represented by adjacency matrix.
            Note that here adjacency is in the sense of nodes,
            not spatial.
        @type adj: scipy.sparse.lil (list of lists)
        
        @param labels: combination of labels with which to annotate each of
            the new transitions created from matrix adj.
            Each label value must be already in one of the
            transition labeling sets.
        @type labels: tuple of valid transition labels
        
        @param check_labels: check validity of labels, or just add them as new
        @type check_labels: bool
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
            dprint('Added ordered list of states: ' +
                   str(self.graph.states.list) )
        
        # convert to format friendly for edge iteration
        nx_adj = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph())
        
        # add each edge using existing checks
        states_list = self.graph.states.list
        for edge in nx_adj.edges_iter():
            (from_idx, to_idx) = edge
            
            from_state = states_list[from_idx]
            to_state = states_list[to_idx]
            
            self.add_labeled(from_state, to_state, labels,
                             check=check_labels)
        
        # in-place replace nodes, based on map
        # compose graphs (vs union, vs disjoint union)
        
        # TODO add overwriting (=delete_labeled +add once more) capability
    
    def find(self, from_states='any', to_states='any', desired_label='any',
             as_dict=True):
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
        intersected with given subset of edges:
            L^{-1}(desired_label) \\cap (from_states x to_states)
        
        TODO support partial labels
        
        note
        ----
        -  __call__
        
        - If called with C{from_states} = all states,
        then the labels annotating returned edges are those which
        appear at least once as edge annotations.
        This may not be the set of all possible
        labels, in case there valid but yet unused edge labels.
        
        - find could have been named ".from...", but it would elongate its
        name w/o adding information. Since you search for transitions, there
        are underlying states and this function naturally provides the option
        to restrict those states to a subset of the possible ones.
        
        see also
        --------
        label, relabel, add_labeled, add_labeled_adj, __call__
        
        @param from_states: subset of states from which transition must start
        @type from_states: 'any' (default)
            | iterable of valid states
            | single valid state
        
        @param to_states: set of states to which the transitions must lead
        @type to_states: 'any' (default)
            | iterable of valid states
            | single valid state
        
        @param desired_label: label with which to filter the transitions
        @type desired_label: {sublabel_type : desired_sublabel_value, ...}
            | 'any', to allow any label (default)
        
        @param as_dict:
            - If C{True}, then return sublabels as dict:
                {sublabel_type : sublabel_value}
            - Otherwise return sublabel values ordered in list,
              the list order based on graph._transition_label_def
        @type as_dict: bool
        
        @return: set of transitions = labeled edges:
                (C{from_state}, C{to_state}, label)
            such that:
                (C{from_state}, C{to_state} )
                \\in C{from_states} x C{to_states}
                
        @rtype: list of transitions = list of labeled edges
                = [(C{from_state}, C{to_state}, C{label}),...]
            where:
                - C{from_state} \\in C{from_states}
                - C{to_state} \\in C{to_states}
                - C{label}: dict
                    | tuple of edge annotation,
                    determined by C{as_dict}.
        """
        (from_state_ids, to_state_ids) = self._mutable2ints(from_states,
                                                            to_states)
        found_transitions = []        
        for from_state_id, to_state_id, attr_dict in self.graph.edges(
            from_state_ids, data=True, keys=False
        ):
            if to_state_id not in to_state_ids and to_states is not 'any':
                continue
            
            # any guard ok ?
            if desired_label is 'any':
                ok = True
            else:
                dprint('Checking guard.')
                ok = self._label_check.label_is_desired(attr_dict, desired_label)
            
            if ok:
                dprint('Transition label matched desired label.')
                
                from_state = self.graph.states._int2mutant(from_state_id)
                to_state = self.graph.states._int2mutant(to_state_id)
                
                annotation = \
                    self._label_check._attr_dict2sublabels(attr_dict, as_dict)
                transition = (from_state, to_state, annotation)
                
                found_transitions.append(transition)
            
        return found_transitions
    
    def _label_of(self, from_states, to_states='any', as_dict=True):
        """Depreceated: use find instead. This to be removed."""
        raise NotImplementedError
        if to_states == 'any':
            to_states = self.graph.states.post_single(from_state)
        
        if edge_key == 'any':
            attr_dict = self.graph.get_edge_data(from_state, to_state)
        else:
            attr_dict = self.graph.get_edge_data(from_state, to_state,
                                                 key=edge_key)
        
        if attr_dict is None:
            msg = 'No transition from state: ' +str(from_state)
            msg += ', to state: ' +str(to_state) +', with key: '
            msg += str(edge_key) +' exists.'
            warnings.warn(msg)
        
        label_def = self.graph._transition_label_def
        transition_label_values = list()
        for label_type in label_def:
            cur_label_value = attr_dict[label_type]
            transition_label_values.append(cur_label_value)
    
        return transition_label_values
    
    def _check_sublabeling(self, sublabel_name, sublabel_value):
        """Check which sublabels are still being used."""
        raise NotImplementedError
        edge_sublabels = nx.get_edge_attributes(self.graph, sublabel_name)
        
        edges_using_sublabel_value = set()
        for (edge, cur_sublabel_value) in edge_sublabels.iteritems():
            if cur_sublabel_value == sublabel_value:
                edges_using_sublabel_value.add(edge)                
        
        if edges_using_sublabel_value:
            msg = 'AP (=' +str(sublabel_name) +') still used '
            msg += 'in label of nodes: ' +str(edges_using_sublabel_value)
            raise Exception(msg)
        
        #self.actions.remove(action)

class LabeledStateDiGraph(nx.MultiDiGraph):
    """Species: System & Automaton."""
    
    def __init__(self, name='', states=[], initial_states=[],
                 current_state=None, mutable=False,
                 removed_state_callback=None,
                 from_networkx_graph=None):
        nx.MultiDiGraph.__init__(self, name=name)
        
        if from_networkx_graph is not None and len(states) > 0:
            raise ValueError('Give either states or Networkx graph, not both.')
        
        if from_networkx_graph is not None:
            states = from_networkx_graph.nodes()
            edges = from_networkx_graph.edges()
        
        self.states = LabeledStates(self, states=states, initial_states=initial_states,
                             current_state=current_state, mutable=mutable,
                             removed_state_callback=removed_state_callback)
        self.transitions = LabeledTransitions(self)

        self.dot_node_shape = {'normal':'circle'}
        self.default_export_path = './'
        self.default_export_fname = 'out'
        self.default_layout = 'dot'
        
        if from_networkx_graph is not None:
            for (from_state, to_state) in edges:
                self.transitions.add(from_state, to_state)
        
    def _add_missing_extension(self, path, file_type):
        import os
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
    
    def _pydot_missing(self):
        if pydot is None:
            msg = 'Attempted calling dump_dot.\n'
            msg += 'Unavailable due to pydot not installed.\n'
            warnings.warn(msg)
            return True
        
        return False
    
    def _to_pydot(self):
        """Convert to properly annotated pydot graph."""
        if self._pydot_missing():
            return
        
        dummy_nx_graph = nx.MultiDiGraph()
        
        self.states._dot_str(dummy_nx_graph)
        self.transitions._dot_str(dummy_nx_graph)
        
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
    
    def is_deterministic(self):
        """Does there exist a transition for each state and each input letter ?"""
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
        pydot_graph = self._to_pydot()
        
        return pydot_graph.to_string()
    
    def dot_str(self):
        """Alias to dump_dot()."""
        return self.dump_dot()
    
    def save(self, fileformat='pdf', path='default',
             add_missing_extension=True, rankdir='LR', prog=None):
        """Save image to file.
        
        Recommended: pdf, svg (can render LaTeX labels with inkscape export)
        
        caution
        -------
        rankdir experimental argument
        
        depends
        -------
        dot, pydot
        
        see also
        --------
        plot, pydot.Dot.write
        
        @param fileformat: type of image file
        @type fileformat: str = 'dot' | 'pdf'| 'png'| 'svg' | 'gif' | 'ps'
            (for more, see pydot.write)
        
        @param path: path to image
            (extension C{.fileformat} appened if missing and
             C{add_missing_extension==True} )
        @type path: str
        
        @param add_missing_extension: if extension C{.fileformat} missing,
            it is appended
        @type add_missing_extension: bool
        
        @param rankdir: direction for dot layout
        @type rankdir: str = 'TB' | 'LR'
            (i.e., Top->Bottom | Left->Right)
        
        @param prog: executable to call
        @type prog: dot | circo | ... see pydot.Dot.write
        """
        path = self._export_fname(path, fileformat, addext=add_missing_extension)
        
        if prog is None:
            prog = self.default_layout
        
        pydot_graph = self._to_pydot()
        pydot_graph.set_rankdir(rankdir)
        pydot_graph.set_splines('true')
        pydot_graph.write(path, format=fileformat, prog=prog)
    
    def dump_dot_color(self):
        raise NotImplementedError
    
    def write_dot_color_file(self):
        raise NotImplementedError
    
    def plot(self, rankdir='LR', prog=None):
        """Plot image using dot.
        
        No file I/O involved.
        Requires GraphViz dot and either Matplotlib or IPython.
        
        NetworkX does not yet support plotting multiple edges between 2 nodes.
        This method fixes that issue, so users don't need to look at files
        in a separate viewer during development.
        
        see also
        --------
        save
        
        depends
        -------
        dot and either of IPython or Matplotlib
        """
        # anything to plot ?
        if len(self.states) == 0:
            print(60*'!'+"\nThe system doesn't have any states to plot.\n"+60*'!')
            return
        
        if prog is None:
            prog = self.default_layout
        
        pydot_graph = self._to_pydot()
        pydot_graph.set_rankdir(rankdir)
        pydot_graph.set_splines('true')
        png_str = pydot_graph.create_png(prog=prog)
        
        # installed ?
        if IPython:
            dprint('IPython installed.')
            
            # called by IPython ?
            try:
                cfg = get_ipython().config
                dprint('Script called by IPython.')
                
                # Caution!!! : not ordinary dict, but IPython.config.loader.Config
                
                # qtconsole ?
                if cfg['IPKernelApp']:
                    dprint('Within IPython QtConsole.')
                    display(Image(data=png_str) )
                    return True
            except:
                print('IPython installed, but not called from it.')
        else:
            dprint('IPython not installed.')
        
        # not called from IPython QtConsole, try Matplotlib...
        
        # installed ?
        if matplotlib:
            dprint('Matplotlib installed.')
            
            sio = StringIO()
            sio.write(png_str)
            sio.seek(0)
            img = mpimg.imread(sio)
            imgplot = plt.imshow(img, aspect='equal')
            plt.show(block=False)
            return imgplot
        else:
            dprint('Matplotlib not installed.')
        
        warnings.warn('Neither IPython QtConsole nor Matplotlib available.')
        return None

def str2singleton(ap_label, verbose=False):
        """If string, convert to set(string).
        
        Convention: singleton str {'*'}
        can be passed as str '*' instead.
        """
        if isinstance(ap_label, str):
            vprint('Saw str state label:\n\t' +ap_label, verbose)
            ap_label = {ap_label}
            vprint('Replaced with singleton:\n\t' +str(ap_label) +'\n',
                   verbose)
        return ap_label

def prepend_with(states, prepend_str):
    """Prepend items with given string.
    
    example
    -------
    states = [0, 1]
    prepend_str = 's'
    states = prepend_with(states, prepend_str)
    assert(states == ['s0', 's1'] )
    
    see also
    --------
    tuple2ba, tuple2fts
    
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
