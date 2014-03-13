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

from .mathset import MathSet, SubSet, PowerSet, TypedDict, \
    is_subset, unique
from .export import save_d3, graph2dot

class LabelConsistency(object):
    """Container of methods for checking sublabel consistency.
    
    Used by both L{LabeledStates} and L{LabeledTransitions}
    to verify that sublabels on states and edges are in their
    corresponding (math) sets.
    
    For example, if the 'actions' sublabel set has type: {'yes', 'no'},
    an attempt to label an L{FTS} transition with 'not sure' will fail.
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
    
    def _attr_dict2sublabels(self, attr_dict, as_dict=True, typed_only=True):
        """Extract sublabels representation from edge attribute dict.
        
        If C{as_dict==True}, then return dict of::
            {sublabel_type : sublabel_value, ...}
        Otherwise return list of sublabel values::
            [sublabel_value, ...]
        ordered by L{_attr_dict2sublabels_list}.
        
        See Also
        ========
        L{_attr_dict2sublabels_list}
        """
        if as_dict:
            sublabels_dict = self._attr_dict2sublabels_dict(attr_dict, typed_only)
            annotation = sublabels_dict
        else:
            sublabel_values = self._attr_dict2sublabels_list(attr_dict, typed_only)
            annotation = sublabel_values
        
        return annotation
    
    def _attr_dict2sublabels_list(self, attr_dict, typed_only=False):
        """Convert attribute dict to tuple of sublabel values."""
        sublabels_dict = self._attr_dict2sublabels_dict(attr_dict, typed_only)
        sublabel_values = self._sublabels_dict2list(sublabels_dict, typed_only)
        return sublabel_values
    
    def _attr_dict2sublabels_dict(self, attr_dict, typed_only=False):
        """Filter the edge attributes which are not labels.
        
        See Also
        ========
        L{_attr_dict2sublabels_list}
        
        @return: sublabel types with their values
        @rtype: {C{sublabel_type} : C{sublabel_value},...}
        """
        #self._exist_labels()
        
        sublabel_ordict = self.label_def
        if typed_only:
            sublabels_dict = {k:v for k,v in attr_dict.iteritems()
                                  if k in sublabel_ordict}
        else:
            sublabels_dict = {k:v for k,v in attr_dict.iteritems()}
        
        return sublabels_dict
            
    def _sublabels_dict2list(self, sublabels_dict, typed_only=False):
        """Return ordered sulabel values.
        
        Sublabel values are ordered according to sublabel ordering
        defined in graph._transition_label_def, which is an OrderedDict.
        
        See Also
        ========
        L{_sublabels_list2dict}
        """
        #self._exist_labels()
        sublabel_ordict = self.label_def
        
        if typed_only:
            sublabel_values = [sublabels_dict[k] for k in sublabel_ordict
                                                 if k in sublabels_dict]
        else:
            sublabel_values = [sublabels_dict[k] for k in sublabel_ordict]
            
        return sublabel_values
    
    def _sublabels_list2dict(self, sublabel_values, check_label=True):
        """Return sublabel values dict from tuple.
        
        See Also
        ========
        L{_sublabels_dict2list}
        
        @param sublabel_values: ordered sublabel values
        @type sublabel_values: tuple
        
        @param check_label: verify existence of label
        @type check_label: bool
        """
        label_def = self.label_def
        
        # already a dict ?
        if not isinstance(sublabel_values, dict):
            # single label ?
            if len(label_def) == 1:
                # hack strings for now, until deciding
                if label_def.has_key('ap'):
                    sublabel_values = str2singleton(sublabel_values)
                
                logger.debug('Replaced sublabel value:\n\t' +
                       str(sublabel_values) )
                sublabel_values = [sublabel_values]
                logger.debug('with the singleton:\n\t' +str(sublabel_values) )
            
            # constuct label dict
            try:
                edge_label = dict(zip(label_def, sublabel_values) )
            except:
                raise Exception('Bug')
        else:
            edge_label = sublabel_values
        
        # check if dict is consistent with label defs
        for (typename, sublabel) in edge_label.iteritems():
            possible_labels = label_def[typename]
            
            if isinstance(possible_labels, PowerSet):
                # label with empty set
                if sublabel is None:
                    logger.debug('None given: label with empty set')
                    edge_label[typename] = set()
                    continue
            
            # discrete sublabel type ?
            if check_label and isinstance(possible_labels, Iterable):
                if sublabel in possible_labels:
                    continue
                
                msg = 'Given SubLabel:\n\t' +str(sublabel) +'\n'
                msg += 'not in possible SubLabels:\n\t'
                msg += str(possible_labels) +'\n'
                msg += 'Usual cause: when label comprised of\n'
                msg += 'single SubLabel, pass the value itself,\n'
                msg += 'instead of an Iterable like [value],\n'
                msg += 'because it gets converted to [value] anyway.'
                raise Exception(msg)
            
            if isinstance(possible_labels, PowerSet):
                possible_labels.math_set |= sublabel
                continue
            
            try:
                possible_labels.add(sublabel)
                continue
            except:
                logger.debug('no add method')
            
            try:
                possible_labels.append(sublabel)
                continue
            except:
                logger.debug('no append method')
            
            # iterable sublabel description ? (i.e., discrete ?)
            if isinstance(possible_labels, Iterable):
                msg = 'Possible labels described by Iterable of type:\n'
                msg += str(type(possible_labels) ) +'\n'
                msg += 'but it is not a PowerSet, nor does it have'
                msg += 'an .add or .append method.\n'
                msg += 'Failed to add new label_value.'
                raise TypeError(msg)
            
            # not iterable, check using convention:
            
            # sublabel type not defined ?
            if possible_labels is None:
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
        def test_common_bug(cur_val, desired_val):
            if isinstance(cur_val, (set, list) ) and \
            isinstance(desired_val, (set, list) ) and \
            cur_val.__class__ != desired_val.__class__:
               msg = 'Set SubLabel:\n\t' +str(cur_val)
               msg += 'compared to list SubLabel:\n\t' +str(desired_val)
               msg += 'Did you mix sets & lists when setting AP labels ?'
               raise Exception(msg)
        
        def evaluate_guard(eval_guard, guard, input_port_value):
            logger.debug('Found guard semantics:\n\t')
            logger.debug(eval_guard)
            
            guard_value = eval_guard(guard, input_port_value)
            
            return guard_value
        
        def match_singleton_guard(cur_val, desired_val):
            logger.debug('Actual SubLabel value:\n\t' +str(cur_val) )
            logger.debug('Desired SubLabel value:\n\t' +str(desired_val) )
            
            if cur_val == desired_val:
                return True
            else:
                try:
                    return True in cur_val
                except:
                    return False
        
        label_def = self.label_def
        for (type_name, desired_val) in desired_label.iteritems():
            cur_val = attr_dict[type_name]
            type_def = label_def[type_name]
            
            logger.debug('Checking SubLabel type:\n\t' +str(type_name) )
            
            # guard semantics ?
            if hasattr(type_def, 'eval_guard'):
                guard_value = evaluate_guard(
                    type_def.eval_guard, cur_val, desired_label
                )
                if not guard_value:
                    return False
                else:
                    continue
            
            # no guard semantics given,
            # then by convention:
            #   guard is singleton {cur_val},
            # so test for equality
            guard_value = match_singleton_guard(cur_val, desired_val)
            if not guard_value:
                test_common_bug(cur_val, desired_val)
                return False
        return True

class States(object):
    """Methods to manage states, initial states, current state.
        
    add, remove, count, test membership
    
    Mutable States
    ==============
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
    
    See Also
    ========
    L{LabeledStateDiGraph}, L{LabeledTransitions}, L{Transitions}
    
    @param mutable: enable storage of unhashable states
    @type mutable: bool (default: False)
    """
    def __init__(self, graph, mutable=False,
                 accepting_states_type=None):
        self.graph = graph
        self.list = None # None when list disabled
        
        # biject mutable states <-> ints ?
        if mutable:
            self.mutants = dict()
            self.min_free_id = 0
        else:
            self.mutants = None
            self.min_free_id = None
        
        self.initial = SubSet(self)
        
        self._accepting_type = accepting_states_type
        if accepting_states_type:
            self.accepting = accepting_states_type(self)
        
        self.select_current([], warn=False)
    
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
        
        @return:
            - If C{data==True},
              then return [(state, attr_dict),...]
            - If C{data==False} and C{listed==True} and state order maintained,
              then return [state_i,...]
            - If C{data==False} and C{listed==True} but no order maintained,
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
    
    def __iter__(self):
        return iter(self() )
        
    def __ior__(self, new_states):
        #TODO carefully test this
        self.add_from(new_states)
        return self
    
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
    
    def __setattr__(self, name, value):
        ok = True
        if name is 'initial':
            if not isinstance(value, SubSet):
                ok = False
            if ok:
                ok = value.has_superset(self)
        
        if name is 'accepting':
            if not isinstance(value, self._accepting_type):
                ok = False
            if ok:
                ok = value.has_superset(self)
        
        if not ok:
            msg = 'States.initial must be of class StateSubset.'
            msg += 'Got instead:\n\t' +str(value) +'\nof class:\n\t'
            msg += str(type(value) )
            raise Exception(msg)
        
        object.__setattr__(self, name, value)
    
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
        
        Note
        ====
        If not mutable, no check that given state is valid,
        because this direction (also) inputs to the data structure new states.
        
        See Also
        ========
        L{_int2mutant}
        
        @param state: state to check for
        
        @return:
            - If states not mutable,
              then return given C{state}.
            - If C{state} does not exist and states mutable,
              then return min free int ID.
            - If C{state} does exist and states mutable,
              then return its int ID.
        """
        
        # classic NetworkX ?
        if not self._is_mutable():
            #logger.debug('Immutable states (must be hashable): classic NetworkX.\n')
            return state
        logger.debug('Mutable states.')
        
        mutants = self.mutants
        state_id = [x for x in mutants if mutants[x] == state]
        
        logger.debug('Converted: state = ' +str(state) +' ---> ' +str(state_id) )
        
        # found state ?
        if not state_id:
            logger.debug('No states matching. State is new.\n')
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
        
        Note
        ====
        If not mutable, given int checked to be valid state,
        because this direction outputs to the world.
        
        See Also
        ========
        L{_mutant2int}
        
        @param state_id: ID number to check for
        @type state_id:
            int, if mutable
            valid state, if immutable
        
        @return:
            - If states not mutable,
              then return given argument, because it is the actual state.
            - If states are mutable and C{state_id} is used,
              then return corresponding C{state}.
            - If states are mutable but C{state_id} is free,
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
            logger.debug('For ID:\n\t' +str(state_id) +'\n'
                   +'Found state:\n\t' +str(state) )
            return state
        
        # mutable, but ID unused
        logger.debug('Mutable states, but this ID is currently unused.')
        return None
    
    def _mutants2ints(self, states):
        return map(self._mutant2int, states)
    
    def _ints2mutants(self, ints):
        return map(self._int2mutant, ints)
    
    def _exist_accepting_states(self, warn=True):
        """Check if system has accepting states."""
        if not hasattr(self, 'accepting'):
            if warn:
                msg = 'System of type: ' +str(type(self.graph) )
                msg += 'does not have accepting states.'
                warnings.warn(msg)
            return False
        else:
            return True
    
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
        """Create single state.
        
        The new state must be hashable, unless mutable states are enabled.
        For details about mutable states see the docstring of transys.States.
        
        For annotating a state with a subset of atomic propositions,
        or other (custom) annotation, use the functions provided by
        AtomicPropositions, or directly the NetworkX.MultiDiGraph.add_node method.
        
        See Also
        ========
        networkx.MultiDiGraph.add_node
        
        @param new_state:
            Single new state to add.
        @type new_state:
            - If states immutable, then C{state} must be a hashable object.
              Any hashable allowed, except for None (see nx add_node below).
            - If states mutable, then C{state} can be unhashable.
        """
        new_state_id = self._mutant2int(new_state)
        self._warn_if_state_exists(new_state)
        
        logger.debug('Adding new id: ' +str(new_state_id) )
        self.graph.add_node(new_state_id, attr_dict, check, **attr)
        
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
        
        See Also
        ========
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
            if not self.list:
                print(
                    'Adding non-list to empty system with ordering.\n'+
                    "Won't remember state order from now on."
                )
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
        if rm_state in self.initial:
            self.initial.remove(rm_state)
        
        # chain to parent (for accepting states etc)
        if self._exist_accepting_states(warn=False):
            if rm_state in self.accepting:
                self.accepting.remove(rm_state)
    
    def remove_from(self, rm_states):
        """Remove a list of states."""
        for rm_state in rm_states:
            self.remove(rm_state)
    
    def select_current(self, states, warn=True):
        """Select current state.
        
        State membership is checked.
        If state \\notin states, exception raised.
        
        None is possible.
        """
        self._current = MathSet()
        
        if not states:
            msg = 'System has no states, current set to None.\n'
            msg += 'You can add states using sys.states.add()'
            if warn:
                warnings.warn(msg)
            return
        
        # single state given instead of singleton ?
        if not isinstance(states, Iterable) and states in self:
            msg = 'Single state provided, setting current states to it.\n'
            msg += 'A subset of states expected as argument.'
            warnings.warn(msg)
            states = [states]
        
        if not is_subset(states, self() ):
            msg = 'Current state given is not in set of states.\n'
            msg += 'Cannot set current state to given state.'
            raise Exception(msg)
        
        self._current.add_from(states)
    
    @property
    def current(self):
        """Return list of current states.
        
        Multiple current states are meaningful in the context
        of non-deterministic automata.
        """
        return list(self._current)
    
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
        
        try:
            state_ids = self._mutants2ints(states)
        except:
            raise Exception('Not all states given are in the set of states.\n')
        
        successor_ids = list()
        for state_id in state_ids:
            successor_ids += self.graph.successors(state_id)
        
        successor_ids = unique(successor_ids)
        successors = self._ints2mutants(successor_ids)
        
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
        
        try:
            state_ids = self._mutants2ints(states)
        except:
            raise Exception('Some given items are not states.')
        
        predecessor_ids = list()
        for state_id in state_ids:
            predecessor_ids += self.graph.predecessors(state_id)
        
        predecessor_ids = unique(predecessor_ids)
        predecessors = self._ints2mutants(predecessor_ids)
        
        return predecessors
    
    def forward_reachable(self, state):
        """Return states reachable from given state.
        
        Iterated post(), a wrapper of networkx.descendants.
        """
        state_id = self._mutant2int(state)
        descendant_ids = nx.descendants(self, state_id)
        descendants = self._ints2mutants(descendant_ids)
        return descendants
    
    def backward_reachable(self, state):
        """Return states from which the given state can be reached.
        """
        state_id = self._mutant2int(state)
        descendant_ids = nx.ancestors(self, state_id)
        ancestors = self._ints2mutants(descendant_ids)
        return ancestors
    
    def rename(self, new_states_dict):
        """Map states in place, based on dict.

        See Also
        ========
        networkx.relabel_nodes
        
        @param new_states_dict: {old_state : new_state}
            (partial allowed, i.e., projection)
        """
        return nx.relabel_nodes(self.graph, new_states_dict, copy=False)
    
    def check(self):
        """Check sanity of various state sets.
        
        Checks if:
            - Initial states \\subseteq states
            - Current state is set
            - Current state \\subseteq states
        """
        if not is_subset(self.initial, self() ):
            warnings.warn('Initial states \\not\\subseteq states.')
        
        if self.current is None:
            warnings.warn('Current state unset.')
            return
        
        if self.current not in self():
            warnings.warn('Current state \\notin states.')
        
        print('States and Initial States are ok.\n'
              +'For accepting states, refer to my parent.')
    
    def paint(self, state, color):
        """Color the given state.
        
        The state is filled with given color,
        rendered with dot when plotting and saving.
        
        @param state: valid system state
        
        @param color: with which to paint C{state}
        @type color: str of valid dot color
        """
        state_id = self._mutant2int(state)
        self.graph.node[state_id]['style'] = 'filled'
        self.graph.node[state_id]['fillcolor'] = color
    
class LabeledStates(States):
    """States with annotation.
    
    For L{FTS} and L{OpenFTS} each state label consists of a single sublabel,
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
    def __init__(self, graph, mutable=False,
                 accepting_states_type=None):
        States.__init__(
            self, graph, mutable=mutable,
            accepting_states_type=accepting_states_type
        )
        
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
            logger.error(msg)
            return False
    
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
        
        See Also
        ========
        L{labels}, L{find}
        
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
        
        Input Formats
        =============
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
        
        Creating States
        ===============
        If no states currently exist and C{states='create'},
        then new states 0,...,N-1 are created,
        where: N = C{len(label_list) }.
        
        Examples
        ========
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
        
        See Also
        ========
        L{label}, L{find}, L{LabeledTransitions}, L{States}, L{FTS},
        L{BA}, L{FSM}
        
        @param states: existing states to be labeled with ap_label_list,
            or string 'create' to cause creation of new int ID states
        @type states: interable container of existing states
            | str 'create'
        
        @param label_list: valid AP labels for annotating C{states}
        @type label_list: list of valid labels
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
    
    def labeled_with(self, desired_label):
        """Return states with given label.
        
        Convenience method for calling find.

        See Also
        ========
        L{find}, L{label_of}
        
        @param desired_label: search for states with this label
        @type desired_label: dict of form:
            {sublabel_type : sublabel_value}
        
        @return: states with C{desired_label}
        @rtype: list
        """
        state_label_pairs = self.find(with_attr_dict=desired_label)
        states = [state for (state, label) in state_label_pairs]
        return states
        
    def label_of(self, state):
        """Return label of given state.
        
        Convenience method for calling find.
        
        See Also
        ========
        L{find}, L{labeled_with}
        
        @param state: single system state
        
        @return: label of C{state}
        @rtype: If C{as_dict is True}, then C{dict} of the form::
                {sublabel_type : sublabel_value}
        """
        state_label_pairs = self.find([state])
        (state_, label) = state_label_pairs[0]
        return label
    
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
                logger.debug(msg)
                
            state_ids = self._mutants2ints(states)
            logger.debug(state_ids)
        
        found_state_label_pairs = []
        for state_id, attr_dict in self.graph.nodes_iter(data=True):
            logger.debug('Checking state_id = ' +str(state_id) +
                          ', with attr_dict = ' +str(attr_dict) )
            
            if state_ids is not 'any':
                if state_id not in state_ids:
                    logger.debug('state_id = ' +str(state_id) +', not desired.')
                    continue
            
            msg = 'Checking state label:\n\t attr_dict = '
            msg += str(attr_dict)
            msg += '\n vs:\n\t desired_label = ' + str(with_attr_dict)
            logger.debug(msg)
            
            if not with_attr_dict:
                logger.debug('Any label acceptable.')
                ok = True
            else:
                ok = self._label_check.label_is_desired(attr_dict, with_attr_dict)
            
            if ok:
                logger.debug('Label Matched:\n\t' +str(attr_dict) +
                              ' == ' +str(with_attr_dict) )
                
                state = self._int2mutant(state_id)
                annotation = \
                    self._label_check._attr_dict2sublabels(attr_dict, as_dict=True)
                state_label_pair = (state, annotation)
                
                found_state_label_pairs.append(state_label_pair)
            else:
                logger.debug('No match for label---> state discarded.')
        
        return found_state_label_pairs
        
        #except KeyError:
        #warnings.warn("State: " +str(state) +", doesn't have AP label.")
        #return None

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
        
        See Also
        ========
        L{LabeledTransitions.__call__}
        """
        return self.graph.edges(data=False)
    
    def __str__(self):
        return 'Transitions:\n' +pformat(self() )
    
    def __len__(self):
        """Count transitions."""
        return self.graph.number_of_edges()
    
    def _mutant2int(self, from_state, to_state):
        from_state_id = self.graph.states._mutant2int(from_state)
        to_state_id = self.graph.states._mutant2int(to_state)
        
        return (from_state_id, to_state_id)
    
    def add(self, from_state, to_state, attr_dict=None, check=True, **attr):
        """Wrapper for L{LabeledDiGraph.add_edge}.
        """
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
        
        See also L{remove}.
        """
        for from_state in from_states:
            for to_state in to_states:
                self.remove(from_state, to_state)

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
    
    But for systems with more sub-labels, e.g.,::
        {system_actions, environment_actions}
    a label consists of two sub-labels, each of which can be selected
    from the set of available system actions and environment actions.
    Each of these sets is defined using this class.
    
    The purpose is to support labels with any number of sub-labels,
    without the need to re-write keyword-value management of
    NetworkX edge dictionaries every time this is needed.
    
    Caution
    =======
    Before removal of a sublabel value from the sublabel type V,
    remember to check using L{_check_sublabeling} that the value is
    not currently used by any edges.
    
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
    
    In more detail, the following classes encapsulate this one:
      - L{FiniteTransitionSystem} (closed)
      - L{OpenFiniteTransitionSystem}
      - L{FiniteStateAutomaton}
      - L{FiniteStateMachine}
    
    See Also
    ========
    L{Transitions}
    """
    def __init__(self, graph, deterministic=False):
        Transitions.__init__(self, graph)
        
        # labeling defined ?
        if hasattr(self.graph, '_transition_label_def'):
            self._label_check = \
                LabelConsistency(self.graph._transition_label_def)
        else:
            self._label_check = None
        
        self._deterministic = deterministic
    
    def __call__(self, labeled=False, as_dict=True):
        """Return all edges, optionally paired with labels.
        
        Note
        ====
        __call__(labeled=True, as_dict=True) is equivalent to find(),
        i.e., L{find} without any restrictions on the desired
        from_state, to_state, nor sublabels.
        
        See Also
        ========
        L{find}
        
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
    
    def _mutable2ints(self, from_states, to_states):
        """Convert possibly unhashable states to internal ones.
        
        If states are hashable, the internal ones are the same.
        Otherwise the internal ones are ints maintained in bijection
        with the mutable states.
        """
        if from_states == 'any':
            from_state_ids = 'any'
        else:
            if from_states in self.graph.states:
                raise TypeError('from_states is a single state,\n'
                                'should be iterable of states.')
            from_state_ids = self.graph.states._mutants2ints(from_states)
        
        if to_states == 'any':
            to_state_ids = 'any'
        else:
            if to_states in self.graph.states:
                raise TypeError('to_states is a single state,\n'
                                    'should be iterable of states.')
            to_state_ids = self.graph.states._mutants2ints(to_states)
        
        return (from_state_ids, to_state_ids)
    
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
            logger.debug('Checking edge with:\n\t key = ' +
                          str(edge_key) + '\n')
            logger.debug('\n\t label = ' +str(label) +'\n')
            logger.debug('\n against: ' +str(edge_label) )
            
            if label == edge_label:
                logger.debug('Matched. Removing...')
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
        
        A labeled transition is (uniquely) identified by the list::
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
        already exists, then:
            - if check, then raise error
            - otherwise produce warning
        
        Checks states are already in set of states.
        Checks action is already in set of actions.
        If not, raises exception or warning.
        
        To override, use check = False.
        Then given states are added to set of states,
        given action is added to set of actions,
        and if same edge with identical label already exists,
        a warning is issued.
        
        See Also
        ========
        L{add}, L{label}, L{relabel}, L{add_labeled_adj}
        
        @param from_state: start state of the transition
        @type from_state: valid system state
        
        @param to_state: end state of the transition
        @type to_state: valid system state
        
        @param labels: annotate this edge with these labels
        @type labels:
            - if single action set /alphabet defined,
                then single label
            - if multiple action sets /alphabets,
                then either:
                    - list of labels in proper oder
                    - dict of action_set_name : label pairs
        
        @param check: if same edge with identical annotation
            (=sublabels) already exists, then:
                - if C{False}, then raise exception
                - otherwise produce warning
        @type check: bool
        """
        self._exist_labels()
        self._check_states(from_state, to_state, check=check)
        
        # chek if same unlabeled transition exists
        (from_state_id, to_state_id) = self._mutant2int(
            from_state, to_state
        )
        trans_from_to = self.graph.get_edge_data(
            from_state_id, to_state_id, default={}
        )
        
        if {} in trans_from_to.values():
            msg = 'Unlabeled transition: '
            msg += 'from_state-> to_state already exists,\n'
            msg += 'where:\t from_state = ' +str(from_state) +'\n'
            msg += 'and:\t to_state = ' +str(to_state) +'\n'
            raise Exception(msg)
        
        # note that first we add states, labels, if check =False,
        # then we check to see if same transition already exists
        #
        # if states were not previously in set of states,
        # then transition is certainly new,
        # so we won't abort due to finding an existing transition,
        # in the middle, having added states, but not the transition,
        # because that is impossible.
        #
        # if labels were not previously in label set,
        # then a similar issue can arise only
        # if an unlabeled transition already exists.
        #
        # Avoided by first checking for an unlabeled transition.        
        edge_label = self._label_check._sublabels_list2dict(
            labels, check_label=check
        )
        
        msg = 'Same labeled transition:\n'
        msg += 'from_state---[label]---> to_state\n'
        msg += 'already exists, where:\n'
        msg += '\t from_state = ' +str(from_state) +'\n'
        msg += '\t to_state = ' +str(to_state) +'\n'
        msg += '\t label = ' +str(edge_label) +'\n'
        
        # check if same labeled transition exists
        if edge_label in trans_from_to.values():
            if check:
                raise Exception(msg)
            else:
                warnings.warn(msg)
        
        self._breaks_determinism(from_state, labels)
        
        # states, labels checked, no same unlabeled nor labeled,
        # so add it
        self.graph.add_edge(from_state_id, to_state_id, **edge_label)
    
    def add_labeled_from(self, from_states, to_states, labels, check=True):
        """Add multiple labeled transitions.
        
        Adds transitions between all states in set from_states,
        to all states in set to_states, annotating them with the same labels.
        For more details, see L{add_labeled}.
        """
        for from_state in from_states:
            for to_state in to_states:
                self.add_labeled(from_state, to_state, labels, check=check)
    
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
        
        TODO support partial labels
        
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
        
        (from_state_ids, to_state_ids) = self._mutable2ints(from_states,
                                                            to_states)
        
        found_transitions = []
        
        if from_state_ids is 'any':
            from_state_ids = self.graph.nodes()
        
        for from_state_id, to_state_id, attr_dict in self.graph.edges_iter(
            from_state_ids, data=True, keys=False
        ):
            if to_states is not 'any':
                if to_state_id not in to_state_ids:
                    continue
            
            if not with_attr_dict:
                logger.debug('Any label is allowed.')
                ok = True
            elif not attr_dict:
                logger.debug('No guard defined.')
                ok = True
            else:
                logger.debug('Checking guard.')
                ok = self._label_check.label_is_desired(
                    attr_dict, with_attr_dict
                )
            
            if ok:
                logger.debug('Transition label matched desired label.')
                
                from_state = self.graph.states._int2mutant(from_state_id)
                to_state = self.graph.states._int2mutant(to_state_id)
                
                annotation = \
                    self._label_check._attr_dict2sublabels(
                        attr_dict, as_dict=True, typed_only=typed_only
                    )
                transition = (from_state, to_state, annotation)
                
                found_transitions.append(transition)
            
        return found_transitions
    
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

class LabeledDiGraph(nx.MultiDiGraph):
    """Directed multi-graph with constrained labeling.
    
    Provided facilities to define labeling functions on
    vertices and edges, with given co-domains,
    so that labels are type-checked, not arbitrary.
    
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
        max_outdegree_per_label=None,
        mutable=False,
        **kwargs
    ):
        """Initialize the types of labelings on states and edges.
        
        @param state_label_types: defines the state labeling functions:
            
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
        
        self.states = LabeledStates(self, mutable=False)
        
        #todo: handle accepting states separately
        if max_outdegree == 1:
            deterministic = True
        else:
            deterministic = False
        self.transitions = LabeledTransitions(self, deterministic)
        
        # export properties
        self.dot_node_shape = {'normal':'circle'}
        self.default_export_path = './'
        self.default_export_fname = 'out'
        self.default_layout = 'dot'
        
    def _init_labeling(self, label_types):
        """
        @type domain: 'state' | 'transition'
        
        @param label_types: see L{__init__}.
        
        Note
        ====
        'state' will be renamed to 'node' in the future
        'transition' will be renamed to 'edge' in the future
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
    
    def add_node(self, n, attr_dict=None, check=True, **attr):
        """Use a ConstrainedDict as attribute dict.
        
        Log warning if node already exists.
        
        All other functionality remains the same.
        
        @param check: if True and untyped keys are passed,
            then raise C{AttributeError}.
        """
        # avoid multiple additions
        if n in self:
            logger.warn('Graph alreay has node: ' + str(n))
        
        if attr_dict is None:
            attr_dict=attr
        else:
            try:
                attr_dict.update(attr)
            except AttributeError:
                raise nx.NetworkXError(\
                    "The attr_dict argument must be a dictionary.")
        
        typed_attr = TypedDict()
        typed_attr.set_types(self._node_label_types)
        typed_attr.update(attr_dict) # type checking happens here
        
        logger.debug('node typed_attr: ' + str(typed_attr))
        
        self._check_for_untyped_keys(typed_attr, self._node_label_types, check)
        
        nx.MultiDiGraph.add_node(self, n, attr_dict=typed_attr)
    
    def add_nodes_from(self, nodes, **attr):
        for n in nodes:
            try:
                n not in self.succ
                node = n
                attr_dict = attr
            except TypeError:
                node, ndict = n
                attr_dict = attr.copy()
                attr_dict.update(ndict)
            
            self.add_node(node, attr_dict=attr_dict)
    
    def add_edge(self, u, v, attr_dict=None, check=True, **attr):
        """Use a ConstrainedDict as attribute dict.
        
        Raise exception if C{u} or C{v} are not already graph nodes.
        Warn if edge with same attr_dict already exists.
        Each label defines a different labeled edge.
        So to "change" the label, either:
        
            - remove the edge with this label, then add a new one, or
            - find the edge key, then use subscript notation:
                
                G[i][j][key]['attr_name'] = attr_value
        
        Argument C{key} has been removed compared to
        L{networkx.MutliDigraph.add_edge}, because edges are defined
        by their labeling, i.e., multiple edges with same labeling
        are not allowed.
        
        @param check: control how untyped attributes are handled:
            
            - if C{True} and C{attr_dict} has untyped keys,
              then raise C{AttributeError},
            
            - otherwise warn
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
        
        if attr_dict is None:
            attr_dict=attr
        else:
            try:
                attr_dict.update(attr)
            except AttributeError:
                raise nx.NetworkXError(\
                    "The attr_dict argument must be a dictionary.")
        
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
        
        self._check_for_untyped_keys(typed_attr, self._edge_label_types, check)
        
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
        
        self.states = LabeledStates(
            self, mutable=mutable,
            accepting_states_type=accepting_states_type
        )
        self.transitions = LabeledTransitions(self, deterministic)

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
