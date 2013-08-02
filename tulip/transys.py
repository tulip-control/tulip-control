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
based on code from:
    Automaton class and supporting methods (Scott Livingston)
and
    Automaton Module (TuLiP distribution v0.3c)

@author: Ioannis Filippidis
"""

"""
TODO
    Baier bisimulation algorithm
    Moore to Mealy
    timed automata (with counters, dense time semantics ?)

 import from
   string/text file
   promela
   xml

 simulation
   random
   via matlab
   transducer mode

 conversions between automata types
   either internally or
   by calling external converters (e.g. ltl2dstar)
 operations between trasition systms and automata or game graphs

 dependent on other modules
   ltl2ba: uses also spec classs
"""

import networkx as nx
#from scipy.sparse import lil_matrix # is this really needed ?
import warnings
import copy
from pprint import pformat
from itertools import chain, combinations
from collections import Iterable, Hashable, OrderedDict
from cStringIO import StringIO
from time import strftime

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

class MathSet(object):
    """Mathematical set, allows unhashable elements."""
    
    def __init__(self, iterable=[]):
        if not isinstance(iterable, Iterable):
            raise TypeError('iterable must be an iterable container.')
        
        self.set = set(filter(lambda x: isinstance(x, Hashable), iterable) )
        self.list = filter(lambda x: not isinstance(x, Hashable), iterable)
    
    def __str__(self):
        return str(self.set) +' U ' +str(self.list)
    
    def __call__(self):
        return list(self.set) +self.list
    
    def add(self, item):
        if isinstance(item, Hashable):
            self.set.add(item)
        else:
            if item not in self.list:
                self.list.append(item)
            else:
                warnings.warn('item already in MathSet.')
    
    def remove(self, item):
        if isinstance(item, Hashable):
            self.set.remove(item)
        else:
            self.list.remove(item)

def unique(iterable):
    """Return unique elements.
    
    If all items in iterable are hashable, then returns set.
    If iterable contains unhashable item, then returns list of unique elements.
    
    note
    ----
    Always returning a list for consistency was tempting,
    however this defeats the purpose of creating this function
    to achieve brevity elsewhere in the code.
    """
    # hashable items ?
    try:
        unique_items = set(iterable)
    except:
        unique_items = []
        for item in iterable:
            if item not in unique_items:
                unique_items.append(item)
    
    return unique_items

def contains_multiple(iterable):
    """Does iterable contain any item multiple times ?"""    
    return len(iterable) != len(unique(iterable) )

def is_subset(small_iterable, big_iterable):
    """Comparison for handling list <= set, and lists with unhashable items.
    """
    # asserts removed when compiling with optimization on...
    # it would have been elegant to use instead:
    #   assert(isinstance(big_iterable, Iterable))
    # since the error msg is succintly stated by the assert itself
    if not isinstance(big_iterable, Iterable):
        raise TypeError('big_iterable must be Iterable, '
                        'otherwise subset relation undefined.\n'
                        'Got:\n\t' +str(big_iterable) +'\ninstead.')
        
    if not isinstance(small_iterable, Iterable):
        raise TypeError('small_iterable must be Iterable, '
                        'otherwise subset relation undefined.\n'
                        'Got:\n\t' +str(big_iterable) +'\ninstead.')
    
    # nxor
    if isinstance(small_iterable, str) != isinstance(big_iterable, str):
        raise TypeError('Either both or none of small_iterable, '
                        'big_iterable should be strings.\n'
                        'Otherwise subset relation between string '
                        'and non-string may introduce bugs.\nGot:\n\t' +
                        str(big_iterable) +',\t' +str(small_iterable) +
                        '\ninstead.')
    
    try:
        # first, avoid object duplication
        if not isinstance(small_iterable, set):
            small_iterable = set(small_iterable)
        
        if not isinstance(big_iterable, set):
            big_iterable = set(big_iterable)
        
        return small_iterable <= big_iterable
    except TypeError:
        # not all items hashable...
    
        # list to avoid: unhashable \in set ? => error
        if not isinstance(big_iterable, list):
            # avoid object duplication
            big_iterable = list(big_iterable)
        
        for item in small_iterable:
            if item not in big_iterable:
                return False
        return True
    except:
        raise Exception('Failed to compare iterables.')

def powerset(iterable):
        """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
        
        From:
            http://docs.python.org/2/library/itertools.html,
        also in:
            https://pypi.python.org/pypi/more-itertools
        """
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1) )

class PowerSet(object):
    """Efficiently store power set of a mathematical set.
    
    Set here isn't necessarily a Python set,
    i.e., it may comprise of unhashable elements.
    
    Upon initialization the iterable defining the (math) set is checked.
    If all elements are hashable, then a set is used internally.
    Otherwise a list is used, filtering items to be unique.
    
    usage
    -----
    s = [[1, 2], '3', {'a':1}, 1]
    p = PowerSet(iterable=s)
    
    q = Powerset()
    q.define_set(s)
    
    p.add_set_element({3: 'a'} )
    p.remove_set_element([1,2] )
    
    see also
    --------
    is_subset
    
    @param iterable: mathematical set S of elements, on which this 2^S defined.
    @type iterable: iterable container
    """
    def __init__(self, iterable=[]):
        self.define_set(iterable)
    
    def __get__(self, instance, value):
        return self()
    
    def __str__(self):
        return str(self.__call__() )
    
    def __contains__(self, item):
        """Is item \\in 2^iterable = this powerset(iterable)."""
        if not isinstance(item, Iterable):
            raise Exception('Not iterable:\n\t' +str(item) +',\n'
                            'this is a powerset, so it contains (math) sets.')
        
        return is_subset(item, self.math_set)
    
    def __call__(self):
        """Return the powerset as list of subsets, each subset as tuple."""
        return list(powerset(self.math_set) )
    
    def __iter__(self):
        return iter(self() )
    
    def __len__(self):
        return 2**len(self.math_set)
    
    def define_set(self, iterable):
        self.math_set = unique(iterable)
    
    def add_set_element(self, element):
        """Add new element to underlying set S.
        
        This powerset is 2^S.
        """
        if isinstance(self.math_set, list):
            if element not in self.math_set:
                self.math_set.append(element)
            return
        
        # set
        if isinstance(element, Hashable):
            if element not in self.math_set:
                self.math_set.add(element)
            return
            
        # element not Hashable, so cannot be \in set, hence new
        # switch to list storage
        self.math_set = unique(list(self.math_set) +[element] )
    
    def add_set_elements(self, elements):
        """Add multiple new elements to underlying set S.
        
        This powerset is 2^S.
        """
        for element in elements:
            self.add_set_element(element)
    
    def remove_set_element(self, element):
        """Remove existing element from set S.
        
        This powerset is 2^S.
        """
        try:
            self.math_set.remove(element)
        except (ValueError, KeyError) as e:
            warnings.warn('Set element not in set S.\n'+
                          'Maybe you targeted another element for removal ?')
        
        # already efficient ?
        if isinstance(self.math_set, set):
            return
        
        # maybe all hashable after removal ?
        try:
            self.math_set = set(self.math_set)
        except:
            return

if debug:
    import traceback
    
    def dprint(s):
        """Debug mode print."""
        print(s)
else:
    def dprint(s):
        pass

def vprint(string, verbose=True):
    if verbose:
        print(string)

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
            dprint('Immutable states (must be hashable): classic NetworkX.\n')
            return state
        
        mutants = self.mutants
        state_id = [x for x in mutants if mutants[x] == state]
        
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
        """Check if single state \\in set_of_states."""
        state_id = self._mutant2int(state)
        return self.graph.has_node(state_id)    
    
    def _exist_labels(self):
        """State labeling defined ?"""
        if hasattr(self.graph, '_state_label_def'):
            return True
        else:
            msg = 'No state labeling defined for class:\n\t'
            msg += str(type(self.graph) )
            dprint(msg)
            return False
    
    def _exist_final_states(self, msg=True):
        """Check if system has final states."""
        if not hasattr(self.graph, 'final_states'):
            if msg:
                warnings.warn('System does not have final states.')
            return False
        else:
            return True
    
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
        
    def number_of_initial(self):
        """Count initial states."""
        return len(self._initial)
    
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
        return 'Transitions:\n\t' +pformat(self() )
    
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
            print('Added ordered list of states: ' +str(self.graph.states.list) )
        
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
        for (from_node, to_node, attr_dict) in self.graph.edge_iter(data=True):
            annotation = self._attr_dict2sublabels(attr_dict, as_dict)
            edge = (from_node, to_node, annotation)
            edges.append(edge)
        
        return edges
    
    def _exist_labels(self):
        """Labeling defined ?"""
        if not hasattr(self.graph, '_transition_label_def'):
            raise Exception('No transition labeling defined for this class.')
    
    def _check_states(self, from_state, to_state, check=True):
        """Are from_state, to_state \\in states.
        
        If check == False, then add them.
        """
        if not check:
            # attempt adding only if not already in set of states
            # to avoid ordering-related exceptions
            if from_state not in self.graph.states():
                self.graph.states.add(from_state)
            if to_state not in self.graph.states():
                self.graph.states.add(to_state)
        
        if from_state not in self.graph.states():
            msg = 'from_state:\n\t' +str(from_state)
            msg += '\n\\notin States:' +str(self.graph.states() )
            raise Exception(msg)
        
        if to_state not in self.graph.states():
            msg = str(to_state) +' = to_state \\notin state'
            raise Exception(msg)
    
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
        self._exist_labels()
        
        sublabel_ordict = self.graph._transition_label_def        
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
        self._exist_labels()
        
        sublabel_ordict = self.graph._transition_label_def        
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
        self._exist_labels()
        
        # get labeling def
        label_def = self.graph._transition_label_def
        
        # single label ?
        if len(label_def) == 1:
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
                    possible_labels.add(sublabel)
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
            if not hasattr(possible_labels, 'is_valid_guard'):
                raise TypeError('SubLabel type V does not have method is_valid.')
            
            # check sublabel type
            if not possible_labels.is_valid_guard(sublabel):
                raise TypeError('Sublabel:\n\t' +str(sublabel) +'\n' +
                                'not valid for sublabel type:\n\t' +
                                str(possible_labels) )
            
        return edge_label
        
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
        edge_label = self._sublabels_list2dict(label, check_label=True)
        
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
        (from_state_id, to_state_id) = self._mutant2int(from_state, to_state)
        trans_from_to = self.graph.get_edge_data(from_state_id, to_state_id,
                                                 default={} )
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
        edge_label = self._sublabels_list2dict(labels, check_label=check)
        
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
            print('Added ordered list of states: ' +str(self.graph.states.list) )
        
        # convert to format friendly for edge iteration
        nx_adj = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph())
        
        # add each edge using existing checks
        states_list = self.graph.states.list
        for edge in nx_adj.edges_iter():
            (from_idx, to_idx) = edge
            
            from_state = states_list[from_idx]
            to_state = states_list[to_idx]
            
            self.add_labeled(from_state, to_state, labels, check=check_labels)
        
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
            | 'any', to search over all transitions (default)
        
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
        def label_is_desired(attr_dict, desired_label):
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
                           warnings.warn('Set label compared to list label,\n'
                                         'did you mixed sets and lists when '
                                         'initializing AP labels ?')
                    
                    return False
            return True
        
        # interface
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
                ok = label_is_desired(attr_dict, desired_label)
            
            if ok:
                dprint('Transition label matched desired label.')
                
                from_state = self.graph.states._int2mutant(from_state_id)
                to_state = self.graph.states._int2mutant(to_state_id)
                
                annotation = self._attr_dict2sublabels(attr_dict, as_dict)
                transition = (from_state, to_state, annotation)
                
                found_transitions.append(transition)
            
        return found_transitions
    
    def _label_of(self, from_states, to_states='any', as_dict=True):
        """Depreceated: use find instead. This to be removed."""
        
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
    
    def check_sublabeling(self, sublabel_name, sublabel_value):
        """Check which sublabels are still being used."""
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
        
        self.states = States(self, states=states, initial_states=initial_states,
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
            plt.show()
            return imgplot
        else:
            dprint('Matplotlib not installed.')
        
        warnings.warn('Neither IPython QtConsole nor Matplotlib available.')
        return None

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
        
        msg += self._print(path, trace, action_trace)
        
        msg += "\n\t Simulation Suffix:\n\t"
        
        path = self.path.suffix
        trace = self.trace.suffix
        action_trace = self.action_trace.suffix
        
        msg += self._print(path, trace, action_trace)
        
        return msg
        
    def _print(self, path, trace, action_trace):
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
    
    def __init__(self, **args):
        FiniteTransitionSystemSimulation.__init__(self, **args)

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
    
    def _check_state(self, state):
        if state not in self.graph.states():
            msg = 'State:\n\t' +str(state) +'\n'
            msg += 'is not in set of states:\n\t' +str(self.graph.states() )
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
        
        self._check_state(state)
        
        # note: after moving, this will change to \in PowerSet(AP)
        if not is_subset(ap_label, self.atomic_propositions):
            raise Exception('Label \\not\\subset AP.'
                            'FYI Label:\n\t' +str(ap_label) +'\n'
                            'AP:\n\t' +str(self.atomic_propositions) +'\n' +
                            'Note that APs must be in an iterable,\n' +
                            "even single ones, e.g. {'p'}.")
        
        kw = {self.name: ap_label}
        self.graph.add_node(state, **kw)
    
    def label_states(self, states, ap_label, check=True):
        """Label multiple states with the same AP label."""
        
        for state in states:
            self.label_state(state, ap_label, check=check)
    
    def label_per_state(self, states, ap_label_list, check=True):
        """Label multiple states, each with a (possibly) different AP label.
        
        If no states currently exist and C{states=[]} passed,
        then new states 0,...,N-1 are created,
        where N = C{len(ap_label_list) } the number of AP labels in the list.
        Note that these AP labels are not necessarily different with each other.
        
        examples
        --------
        fts.states.add_from(['s0', 's1'] )
        fts.atomic_propositions.add_from(['p', '!p'] )
        fts.atomic_propositions.label_per_state(['s0', 's1'], [{'p'}, {'!p'}] )
        
        or to skip adding them first:
        fts.atomic_propositions.label_per_state(['s0', 's1'], [{'p'}, {'!p'}], check=False)
        
        The following 3 are equivalent:
        fts.atomic_propositions.label_per_state([1, 2], [{'p'}, {'!p'}], check=False)
        fts.atomic_propositions.label_per_state(range(2), [{'p'}, {'!p'}], check=False)
        fts.atomic_propositions.label_per_state('create', [{'p'}, {'!p'}] )
        
        @param states: existing states to be labeled with ap_label_list,
            or string 'create' to cause creation of new int ID states
        @type states: interable container of existing states |
            str 'create'
        
        @param ap_label_list: valid AP labels for annotating C{states}
        @type ap_label_list: list of valid labels
        
        @param check: check if given states and given labels already exist.
            If C{check=False}, then each state passed is added to system,
            and each AP is added to the APs of the system.
        @type check: bool
        """
        if states == 'create':
            states = range(len(ap_label_list) )
            check = False
        
        for state, curlabel in zip(states, ap_label_list):
            self.label_state(state, curlabel, check=check)
    
    def delabel_state(self, state):
        """Alias for remove_label_from_state()."""
        
        raise NotImplementedError
    
    def of(self, state):
        """Get AP set labeling given state.
        
        If state does I{not} have AP label, return None.
        """
        
        self._check_state(state)
        try:
            state_sublabel_name = self.name
            return self.graph.node[state][state_sublabel_name]
        except KeyError:
            warnings.warn("State: " +str(state) +", doesn't have AP label.")
            return None
        
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

class Actions(set):
    """Store set of system or environment actions."""
    
    def add_from(self, actions=[]):
        """Add multiple actions.
        
        @type actions: iterable
        """
        self |= set(actions)
        
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
    Format transition labels using C{_transition_dot_label_format} which is a
    dict with values:
        - 'actions' (=name of transitions attribute): type before separator
        - 'type?label': separator between label type and value
        - 'separator': between labels for different sets of actions
            (e.g. sys, env). Not used for closed FTS, because it has single set
            of actions.
    """
    
    def __init__(self, atomic_propositions=[], actions=[], **args):
        """Note first sets of states in order of decreasing importance,
        then first state labeling, then transitin labeling (states more
        fundamentalthan transitions, because transitions need states in order to
        be defined).
        """
        LabeledStateDiGraph.__init__(self, **args)
        
        # state labels
        self._state_label_def = OrderedDict(
            [['ap', AtomicPropositions(self, 'ap', atomic_propositions) ]]
        )
        self.atomic_propositions = self._state_label_def['ap']
        self._state_dot_label_format = {'ap':'',
                                           'type?label':'',
                                           'separator':'\\n'}
        
        # edge labels comprised of sublabels (here single sublabel)
        self._transition_label_def = OrderedDict(
            [['actions', Actions(actions)]]
        )
        self.actions = self._transition_label_def['actions']
        self._transition_dot_label_format = {'actions':'',
                                                'type?label':'',
                                                'separator':'\\n'}

        self.dot_node_shape = {'normal':'box'}
        self.default_export_fname = 'fts'

    def __str__(self):        
        s = hl +'\nFinite Transition System (closed)\n' +hl
        s += str(self.states) +'\n' +str(self.atomic_propositions) +'\n'
        s += 'State Labels:\n' +pformat(self.states(data=True) ) +'\n'
        s += 'Actions:\n\t' +str(self.actions) +'\n'
        s += str(self.transitions) +'\n' +hl +'\n'
        
        return s
    
    def __repr__(self):
        return self.__str__()
    
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
            return self._ts_ts_sync_prod(ts_or_ba)
        elif isinstance(ts_or_ba, BuchiAutomaton):
            ba = ts_or_ba
            return _ts_ba_sync_prod(self, ba)
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
    
    def dump_promela(self, procname=None):
        """Convert an automaton to Promela source code.
        
        Creats a process which can be simulated as an independent
        thread in the SPIN model checker.
        
        see also
        --------
        save, plot
        
        @param fname: file name
        @type fname: str
        
        @param procname: Promela process name, i.e., proctype procname())
        @type procname: str (default: system's name)
        
        @param add_missing_extension: add file extension 'pml', if missing
        @type add_missing_extension: bool
        """
        def state2promela(state, ap_label, ap_alphabet):
            s = str(state) +':\n'
            s += '\t printf("State: ' +str(state) +'\\n");\n\t atomic{'
            
            # convention ! means negation
            
            missing_props = filter(lambda x: x[0] == '!', ap_label)
            present_props = ap_label.difference(missing_props)
            
            assign_props = lambda x: str(x) + ' = 1;'
            if len(present_props) > 0:
                s += ' '.join(map(assign_props, present_props) )
            
            # rm "!"
            assign_props = lambda x: str(x[1:] ) + ' = 0;'
            if len(missing_props) > 0:
                s += ' '.join(map(assign_props, missing_props) )
            
            s += '}\n'
            return s
        
        def outgoing_trans2promela(transitions):
            s = '\t if\n'
            for (from_state, to_state, sublabels_dict) in transitions:
                s += '\t :: printf("' +str(sublabels_dict) +'\\n");\n'
                s += '\t\t goto ' +str(to_state) +'\n'
            s += '\t fi;\n\n'
            return s
        
        if procname is None:
            procname = self.name
        
        s = ''
        for ap in self.atomic_propositions():
            # convention "!" means negation
            if ap[0] != '!':
                s += 'bool ' +str(ap) +';\n'
        
        s += '\nactive proctype ' +procname +'(){\n'
        
        s += '\t if\n'
        for initial_state in self.states.initial:
            s += '\t :: goto ' +str(initial_state) +'\n'
        s += '\t fi;\n'
        
        for state in self.states():
            ap_alphabet = self.atomic_propositions()
            ap_label = self.atomic_propositions.of(state)
            s += state2promela(state, ap_label, ap_alphabet)
            
            outgoing_transitions = self.transitions.find({state}, as_dict=True)
            s += outgoing_trans2promela(outgoing_transitions)
        
        s += '}\n'
        return s
    
    def save_promela(self, fname=None, add_missing_extension=True):
        if fname is None:
            fname = self.name
            fname = self._export_fname(fname, 'pml', add_missing_extension)
        
        s = '/*\n * Promela file generated with TuLiP\n'
        s += ' * Data: '+str(strftime('%x %X %z') ) +'\n */\n\n'
        
        s += self.dump_promela()
        
        # dump to file
        f = open(fname, 'w')
        f.write(s)
        f.close()

class FTS(FiniteTransitionSystem):
    """Alias to FiniteTransitionSystem."""
    
    def __init__(self, **args):
        FiniteTransitionSystem.__init__(self, **args)

class OpenFiniteTransitionSystem(LabeledStateDiGraph):
    """Analogous to FTS, but for open systems, with system and environment."""
    def __init__(self, atomic_propositions=[], sys_actions=[],
                 env_actions=[], **args):
        LabeledStateDiGraph.__init__(self, **args)
        
        # state labeling
        self._state_label_def = OrderedDict(
            [['ap', AtomicPropositions(self, 'ap', atomic_propositions) ]]
        )
        self.atomic_propositions = self._state_label_def['ap']
        self._state_dot_label_format = {'ap':'',
                                           'type?label':'',
                                           'separator':'\\n'}
        
        # edge labeling (here 2 sublabels)
        self._transition_label_def = OrderedDict([
            ['sys_actions', Actions(sys_actions) ],
            ['env_actions', Actions(env_actions) ]
        ])
        self.sys_actions = self._transition_label_def['sys_actions']
        self.env_actions = self._transition_label_def['env_actions']
        self._transition_dot_label_format = {'sys_actions':'sys',
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

class OpenFTS(OpenFiniteTransitionSystem):
    """Alias to transys.OpenFiniteTransitionSystem."""
    def __init__(self, **args):
        OpenFiniteTransitionSystem.__init__(self, **args)

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
        
        msg += self._print(word, run)
        
        msg += "\n\t Simulation Suffix:\n\t"
        
        word = self.input_word.suffix
        run = self.run.suffix
        
        msg += self._print(word, run)
        
        return msg
        
    def _print(self, word, run):
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
    
    def __init__(self, **args):
        FiniteStateAutomatonSimulation.__init__(self, **args)

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
    
    alphabet
    --------
    Add single letter to alphabet.
        
    If C{atomic_proposition_based=False},
    then the alphabet is represented by a set.
    
    If C{atomic_proposition_based=True},
    the the alphabet is represented by a powerset 2^AP
    and you manage the set of Atomic Propositions AP within the powerset.
    
    see also
    --------    
    LabeledStateDiGraph._dot_str
        
    """
    
    def __init__(
        self, final_states=[], input_alphabet_or_atomic_propositions=[],
        atomic_proposition_based=True, mutable=False, **args
    ):
        LabeledStateDiGraph.__init__(
            self, mutable=mutable,
            removed_state_callback=self._removed_state_callback, **args
        )
        
        if mutable:
            self.final_states = list()
        else:
            self.final_states = set()
        self.add_final_states_from(final_states)
        
        # edge labeling
        if atomic_proposition_based:
            self.atomic_proposition_based = True
            alphabet = PowerSet(input_alphabet_or_atomic_propositions)
        else:
            self.atomic_proposition_based = False
            alphabet = set(input_alphabet_or_atomic_propositions)
        
        self._transition_label_def = OrderedDict([
            ['in_alphabet', alphabet]
        ])
        self.alphabet = self._transition_label_def['in_alphabet']
        
        # used before label value
        self._transition_dot_label_format = {'in_alphabet':'',
                                                'type?label':'',
                                                'separator':'\\n'}
        
        self.dot_node_shape = {'normal':'circle', 'final':'doublecircle'}
        self.default_export_fname = 'fsa'
        
    def __str__(self):
        s = str(self.states)
        s += '\nState Labels:\n' +pformat(self.states(data=True) ) +'\n'
        s += str(self.transitions) +'\n'
        s += 'Alphabet:\n' +str(self.alphabet) +'\n'
        s += 'Final States:\n\t' +str(self.final_states)
        
        return s
    
    def _removed_state_callback(self, rm_state):
        self.remove_final_state(rm_state)
    
    def _is_final(self, state):
        return is_subset([state], self.final_states)

    # final states
    def add_final_state(self, new_final_state):
        if not new_final_state in self.states():
            raise Exception('Given final state:\n\t' +str(new_final_state) +
                            '\n\\notin States:\n\t' +str(self.states() ) )
        
        # already final ?
        if self._is_final(new_final_state):
            warnings.warn('Attempting to add existing final state.\n')
            return
        
        # mutable states ?
        if self.states.mutants == None:
            self.final_states.add(new_final_state)
        else:
            self.final_states.append(new_final_state)

    def add_final_states_from(self, new_final_states):
        if not is_subset(new_final_states, self.states() ):
            raise Exception('Given Final States \\notsubset States.')
        
        # mutable states ?
        if self.states.mutants == None:
            self.final_states |= set(new_final_states)
        else:
            for new_final_state in new_final_states:
                self.add_final_state(new_final_state)
    
    def number_of_final_states(self):
        return len(self.final_states)
    
    def remove_final_state(self, rm_final_state):
        self.final_states.remove(rm_final_state)
    
    def remove_final_states_from(self, rm_final_states):
        if self.states.mutants == None:
            self.final_states = self.final_states.difference(rm_final_states)
        else:
            for rm_final_state in rm_final_states:
                self.remove_final_state(rm_final_state)

    # checks
    def is_deterministic(self):
        """overloaded method."""
        raise NotImplementedError
        
    def is_blocking(self):
        """overloaded method."""
        raise NotImplementedError
    
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
    def __init__(self, **args):
        FiniteStateAutomaton.__init__(self, **args)

class BuchiAutomaton(OmegaAutomaton):
    def __init__(self, **args):
        OmegaAutomaton.__init__(self, **args)
    
    def __str__(self):
        s = hl +'\nB' +u'\xfc' +'chi Automaton\n' +hl
        s += str(self.states) +'\n'
        s += 'Input Alphabet Letters (\in 2^AP):\n\t' +str(self.alphabet)
        s += '\n' +str(self.transitions) +'\n' +hl +'\n'
        
        return s
    
    def __repr__(self):
        return self.__str__()
    
    def __add__(self, other):
        """Union of two automata, with equal states identified."""
        raise NotImplementedError
    
    def __mul__(self, ts_or_ba):
        return self.sync_prod(ts_or_ba)
    
    def __or__(self, ba):
        return self.async_prod(ba)
        
    def _ba_ba_sync_prod(self, ba2):
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
            return self._ba_ba_sync_prod(ts_or_ba)
        elif isinstance(ts_or_ba, FiniteTransitionSystem):
            ts = ts_or_ba
            return _ba_ts_sync_prod(self, ts)
        else:
            raise Exception('ts_or_ba should be an FTS or a BA.\n'+
                            'Got type: ' +str(ts_or_ba) )
    
    def async_prod(self, other):
        """Should it be defined in a superclass ?"""
        raise NotImplementedError
    
    def acceptance_condition(self, prefix, suffix):
        """Check if given infinite word over alphabet \Sigma is accepted."""
    
    def determinize(self):
        raise NotImplementedError
    
    def complement(self):
        raise NotImplementedError

class BA(BuchiAutomaton):
    def __init__(self, **args):
        BuchiAutomaton.__init__(self, **args)

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

def negation_closure(atomic_propositions):
    """Given: ['p', ...], return: [True, 'p', '!p', ...].
    
    @param atomic_propositions: AP set
    @type atomic_propositions: iterable container of strings
    """
    def negate(x):
        # starts with ! ?
        if x.find('!') == 0:
            x = x[1:]
        else:
            x = '!'+x
        return x
    
    if not isinstance(atomic_propositions, Iterable):
        raise TypeError('atomic_propositions must be Iterable.'
                        'Got:\n\t' +str(atomic_propositions) +'\ninstead.')
    
    ap = [f(x)
          for x in atomic_propositions
          for f in (lambda x: x, negate) ] +[True]
    return unique(ap)

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

def tuple2ba(S, S0, Sa, Sigma_or_AP, trans, name='ba', prepend_str=None,
             atomic_proposition_based=True, verbose=False):
    """Create a Buchi Automaton from a tuple of fields.
    
    note
    ====
    "final states" in the context of \\omega-automata is a misnomer,
    because the system never reaches a "final" state, as in non-transitioning.

    So "accepting states" allows for an evolving behavior,
    and is a better description.

    "final states" is appropriate for NFAs.
    
    see also
    ========
    L{tuple2fts}

    @type ba_tuple: tuple
    @param ba_tuple: defines Buchi Automaton by a tuple (Q, Q_0, Q_F,
        \\Sigma, trans) (maybe replacing \\Sigma by AP since it is an
        AP-based BA ?)  where:

            - Q = set of states
            - Q_0 = set of initial states, must be \\subset S
            - Q_F = set of final states
            - \\Sigma = alphabet
            - trans = transition relation, represented by list of triples:
              [(from_state, to_state, guard), ...]
              where guard \\in \\Sigma.

    @param name: used for file export
    @type name: str
    """
    # args
    if not isinstance(S, Iterable):
        raise TypeError('States S must be iterable, even for single state.')
    
    if not isinstance(S0, Iterable) or isinstance(S0, str):
        S0 = [S0]
    
    if not isinstance(Sa, Iterable) or isinstance(Sa, str):
        Sa = [Sa]
    
    # comprehensive names
    states = S
    initial_states = S0
    accepting_states = Sa
    alphabet_or_ap = Sigma_or_AP
    transitions = trans
    
    # prepending states with given str
    if prepend_str:
        vprint('Given string:\n\t' +str(prepend_str) +'\n' +
               'will be prepended to all states.', verbose)
    states = prepend_with(states, prepend_str)
    initial_states = prepend_with(initial_states, prepend_str)
    accepting_states = prepend_with(accepting_states, prepend_str)
    
    ba = BA(name=name, atomic_proposition_based=atomic_proposition_based)
    
    ba.states.add_from(states)
    ba.states.add_initial_from(initial_states)
    ba.states.add_final_from(accepting_states)
    
    if atomic_proposition_based:
        ba.alphabet.add_set_elements(alphabet_or_ap)
    else:
        ba.alphabet.add(alphabet_or_ap)
    
    for transition in transitions:
        (from_state, to_state, guard) = transition
        [from_state, to_state] = prepend_with([from_state, to_state],
                                              prepend_str)
        # convention
        if atomic_proposition_based:
            guard = str2singleton(guard)
        ba.transitions.add_labeled(from_state, to_state, guard)
    
    return ba

def tuple2fts(S, S0, AP, L, Act, trans, name='fts',
              prepend_str=None, verbose=False):
    """Create a Finite Transition System from a tuple of fields.

    hint
    ====
    To rememeber the arg order:

    1) it starts with states (S0 requires S before it is defined)

    2) continues with the pair (AP, L), because states are more fundamental
    than transitions (transitions require states to be defined)
    and because the state labeling L requires AP to be defined.

    3) ends with the pair (Act, trans), because transitions in trans require
    actions in Act to be defined.

    see also
    ========
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
    @type trans: list of triples: [(from_state, to_state, act), ...] where act \\in Act
    
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
        
        try:
            (state, ap_label) = state_labeling[0]
        except ValueError:
            state_label_pairs = False
        
        if state_label_pairs:
            return
        
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
    ts.states.add_initial_from(initial_states)
    
    ts.atomic_propositions.add_from(ap)
    
    # note: verbosity before actions below
    # to avoid screening by possible error caused by action
    
    # state labeling assigned ?
    if state_labeling is not None:
        for (state, ap_label) in state_labeling:
            ap_label = str2singleton(ap_label, verbose=verbose)
            (state,) = prepend_with([state], prepend_str)
            
            vprint('Labeling state:\n\t' +str(state) +'\n' +
                  'with label:\n\t' +str(ap_label) +'\n', verbose)
            ts.atomic_propositions.label_state(state, ap_label)
    
    # any transition labeling ?
    if actions is None:
        for (from_state, to_state) in transitions:
            (from_state, to_state) = prepend_with([from_state, to_state],
                                                  prepend_str)
            vprint('Added unlabeled edge:\n\t' +str(from_state) +
                   '--->' +str(to_state) +'\n', verbose)
            ts.transitions.add(from_state, to_state)
    else:
        ts.actions.add_from(actions)
        for (from_state, to_state, act) in transitions:
            (from_state, to_state) = prepend_with([from_state, to_state],
                                                  prepend_str)
            vprint('Added labeled edge (=transition):\n\t' +
                   str(from_state) +'---[' +str(act) +']--->' +
                   str(to_state) +'\n', verbose)
            ts.transitions.add_labeled(from_state, to_state, act)
    
    return ts

def cycle_labeled_with(L):
    """Return cycle FTS with given labeling.
    
    @param L: state labeling
    @type L: iterable of state labels, e.g., [{'p', 'q'}, ...]
        Single strings are identified with singleton Atomic Propositions,
        so [..., 'p',...] and [...,{'p'},...] are equivalent.
    
    @returns: FTS with states ['s0', ..., 'sN'], where N=len(L)
        and state labels defined by L, i.e., ('s0', L[0]),...
    """
    n = len(L)
    S = range(n)
    S0 = [] # user will define them
    AP = negation_closure(L)
    Act = None
    from_states = range(0, n-1)
    to_states = range(1, n)
    trans = zip(from_states, to_states)
    trans += [(n-1, 0)] # close cycle
    
    ts = tuple2fts(S, S0, AP, L, Act, trans, prepend_str='s')
    return ts

def _ba_ts_sync_prod(buchi_automaton, transition_system):
    """Construct Buchi Automaton equal to synchronous product TS x NBA.
    
    returns
    -------
    C{prod_ba}, the product Buchi Automaton.
    
    see also
    --------
    _ts_ba_sync_prod, BuchiAutomaton.sync_prod
    """
    (prod_ts, persistent) = _ts_ba_sync_prod(transition_system, buchi_automaton)
    
    prod_name = buchi_automaton.name +'*' +transition_system.name
    prod_ba = BuchiAutomaton(name=prod_name)
    
    # copy S, S0, from prod_TS-> prod_BA
    prod_ba.states.add_from(prod_ts.states() )
    prod_ba.states.add_initial_from(prod_ts.states.initial)
    
    # final states = persistent set
    prod_ba.states.add_final_from(persistent)
    
    # copy edges, translating transitions, i.e., chaning transition labels
    if buchi_automaton.atomic_proposition_based:
        # direct access, not the inefficient
        #   prod_ba.alphabet.add_from(buchi_automaton.alphabet() ),
        # which would generate a combinatorially large alphabet
        prod_ba.alphabet.add_set_elements(buchi_automaton.alphabet.math_set)
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

def _ts_ba_sync_prod(transition_system, buchi_automaton):
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
    _ba_ts_sync_prod, FiniteTransitionSystem.sync_prod
    """
    if not isinstance(transition_system, FiniteTransitionSystem):
        msg = 'transition_system not transys.FiniteTransitionSystem.\n'
        msg += 'Actual type passed: ' +str(type(transition_system) )
        raise TypeError(msg)
    
    if not isinstance(buchi_automaton, BuchiAutomaton):
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
    # using set() destroys order
    prodts = FiniteTransitionSystem(name=prodts_name, states=set() )
    prodts.atomic_propositions.add_from(ba.states() )
    prodts.actions.add_from(fts.actions)

    # construct initial states of product automaton
    s0s = fts.states.initial.copy()
    q0s = ba.states.initial.copy()
    
    final_states_preimage = set()    
    
    for s0 in s0s:
        dprint('----\nChecking initial state:\n\t' +str(s0) )        
        
        Ls0 = fts.atomic_propositions.of(s0)
        Ls0_dict = {'in_alphabet': Ls0}
        
        for q0 in q0s:
            enabled_ba_trans = ba.transitions.find({q0}, desired_label=Ls0_dict)
            
            # q0 blocked ?
            if len(enabled_ba_trans) == 0:
                dprint('blocked q0 = ' +str(q0) )
                continue
            
            # which q next ?     (note: curq0 = q0)
            for (curq0, q, sublabels) in enabled_ba_trans:
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
            if Ls is None:
                raise Exception('No AP label for FTS state: ' +str(next_s) +
                                '\n Did you forget labeing it ?')
            Ls_dict = {'in_alphabet': Ls}

            dprint("Next state's label:\n\t" +str(Ls_dict) )
            
            enabled_ba_trans = ba.transitions.find({q}, desired_label=Ls_dict)
            dprint('Enabled BA transitions:\n\t' +str(enabled_ba_trans) )
            
            if len(enabled_ba_trans) == 0:
                continue
            
            for (q, next_q, sublabels) in enabled_ba_trans:
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
                ts_enabled_trans = fts.transitions.find(
                    {s}, to_states={next_s}, desired_label='any', as_dict=False
                )
                for (from_s, to_s, sublabel_values) in ts_enabled_trans:
                    #attr_dict = fts.get_edge_data(from_s, to_s, key=edge_key)
                    assert(from_s == s)
                    assert(to_s == next_s)
                    dprint('Sublabel value:\n\t' +str(sublabel_values) )
                    
                    # labeled transition ?
                    if len(sublabel_values) == 0:
                        prodts.transitions.add(sq, new_sq)
                    else:
                        #TODO open FTS
                        prodts.transitions.add_labeled(sq, new_sq,
                                                       sublabel_values[0] )
        
        # discard visited & push them to queue
        new_sqs = set()
        for next_sq in next_sqs:
            if next_sq not in visited:
                new_sqs.add(next_sq)
                queue.add(next_sq)
    
    return (prodts, final_states_preimage)

class RabinAutomaton(OmegaAutomaton):
    """Remember to override the final set management."""
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

#############################################################################
# Finite-State Machines : I/O = Reactive = Open Systems
#############################################################################
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
    
    inputs
    ------
    P = {p1, p2,...} is the set of input ports.
    An input port p takes values in a set Vp.
    Set Vp is called the "type" of input port p.
    A "valuation" is an assignment of values to the input ports in P.
    
    We call "inputs" the set of pairs:
    
        {(p_i, Vp_i),...}
    
    of input ports p_i and their corresponding types Vp_i.
    
    A guard is a predicate (bool-valued) used as sub-label for a transition.
    A guard is defined by a set and evaluated using set membership.
    So given an input port value p=x, then if:
    
        x \in guard_set
    
    then the guard is True, otherwise it is False.
    
    The "inputs" are defined by an OrderedDict:
    
        {'p1':explicit, 'p2':check, 'p3':None, ...}
    
    where:
        - C{explicit}:
            is an iterable representation of Vp,
            possible only for discrete Vp.
            If 'p1' is explicitly typed, then guards are evaluated directly:
            
                input_port_value == guard_value ?
        
        - C{check}:
            is a class with methods:
            
                - C{.is_valid(x) }:
                    check if value given to input port 'p1' is
                    in the set of possible values Vp.
                
                - C{.contains(guard_set, input_port_value) }:
                    check if C{input_port_value} \\in C{guard_set}
                    This allows flexible type definitions.
                    
                    For example, C{input_port_value} might be assigned
                    int values, but the C{guard_set} be defined by
                    a symbolic expression as the str: 'x<=5'.
                    
                    Then the user is responsible for providing
                    the appropriate method to the Mealy Machine,
                    using the custom C{check} class described here.
                    
                    Note that we could provide a rudimentary library
                    for the basic types of checks, e.g., for
                    the above simple symbolic case, where using
                    function eval() is sufficient.
            
        - C{None}:
            signifies that no type is currently defined for
            this input port, so input type checking and guard
            evaluation are disabled.
            
            This can be used to skip type definitions when
            they are not needed by the user.
            
            However, since Machines are in general the output
            of synthesis, it follows that they are constructed
            by code, so the benefits of typedefs will be
            considerable compared to the required coding effort.
    
    An OrderedDict is used to allow setting guards using tuples
    (so order of inputs) or dicts, to avoid writing dicts for each
    guard definition (which would be quite cumbersome).
    
    Guards annotate transitions:
        
        Guards: States x States ---> Input_Predicates
    
    outputs
    -------
    Similarly defined to inputs, but:
    
        - for Mealy Machines they annotate transitions
        - for Moore Machines they annotate states
    
    state variables
    ---------------
    Similarly defined to inputs, they annotate states,
    for both Mealy and Moore machines:
    
        States ---> State_Variables
    
    update function
    ---------------
    The transition relation:
    
        - for Mealy Machines:
        
                States x Input_Valuations ---> Output_Valuations x States
                
            Note that in the range Output_Valuations are ordered before States
            to emphasize that an output_valuation is produced
            during the transition, NOT at the next state.
            
            The data structure representation of the update function is
            by storage of the Guards function and definition of Guard
            evaluation for each input port via the OrderedDict discussed above.
        
        - for Moore Machines:
        
            States x Input_Valuations ---> States
            States ---> Output_valuations
    
    note
    ----
    A transducer may operate on either finite or infinite words, i.e.,
    it is not equipped with interpretation semantics on the words,
    so it does not "care" about word length.
    It continues as long as its input is fed with letters.
    
    see also
    --------
    FMS, MealyMachine, MooreMachine
    """
    def __init__(self, **args):
        LabeledStateDiGraph.__init__(
            self, removed_state_callback=self._removed_state_callback, **args
        )
        
        # values will point to values of _*_label_def below
        self.state_vars = OrderedDict()
        self.inputs = OrderedDict()
        self.outputs = OrderedDict()
        #self.set_actions = {}
        
        # state labeling
        self._state_label_def = OrderedDict()
        self._state_dot_label_format = {'type?label':':',
                                        'separator':'\\n'}
        
        # edge labeling
        self._transition_label_def = OrderedDict()
        self._transition_dot_label_format = {'type?label':':',
                                             'separator':'\\n'}
        
        self.default_export_fname = 'fsm'
    
    def _removed_state_callback(self):
        """Remove it also from anywhere within this class, besides the states."""
    
    def add_inputs(self, new_inputs_ordered_dict):
        for (in_port_name, in_port_type) in new_inputs_ordered_dict.iteritems():
            # append
            self._transition_label_def[in_port_name] = in_port_type
            
            # inform inputs
            self.inputs[in_port_name] = self._transition_label_def[in_port_name]
            
            # printing format
            self._transition_dot_label_format[in_port_name] = str(in_port_name)
    
    def add_state_vars(self, new_vars_ordered_dict):
        for (var_name, var_type) in new_vars_ordered_dict.iteritems():
            # append
            self._state_label_def[var_name] = var_type
            
            # inform state vars
            self.state_vars[var_name] = self._state_label_def[var_name]
            
            # printing format
            self._state_dot_label_format[var_name] = str(var_name)
    
    def is_blocking(self, state):
        """From this state, for each input valuation, there exists a transition.
        
        @param state: state to be checked as blocking
        @type state: single state to be checked
        """
        raise NotImplementedError
    
    def is_receptive(self, states='all'):
        """For each state, for each input valuation, there exists a transition.
        
        @param states: states to be checked whether blocking
        @type states: iterable container of states
        """
        for state in states:
            if self.is_blocking(state):
                return False
                
        return True

    # operations between state machines
    def sync_product(self):
        raise NotImplementedError
        
    def async_product(self):
        raise NotImplementedError
    
    def simulate(self, input_sequence):
        self.simulation = FiniteStateMachineSimulation()
        raise NotImplementedError

class FSM(FiniteStateMachine):
    """Alias for Finite-state Machine."""
    
    def __init__(self, **args):
        FiniteStateMachine.__init__(self, **args)

class MooreMachine(FiniteStateMachine):
    """Moore machine.
    
    A Moore machine implements the discrete dynamics:
        x[k+1] = f(x[k], u[k] )
        y[k] = g(x[k] )
    where:
        -k: discrete time = sequence index
        -x: state = valuation of state variables
        -X: set of states = S
        -u: inputs = valuation of input ports
        -y: output actions = valuation of output ports
        -f: X-> 2^X, transition function
        -g: X-> Out, output function
    Observe that the output depends only on the state.
    
    note
    ----
    valuation: assignment of values to each port
    """
    def __init__(self, **args):
        FiniteStateMachine.__init__(self, **args)
        self.default_export_fname = 'moore'
        
        raise NotImplementedError
    
    def add_outputs(self, new_outputs_ordered_dict):
        for (out_port_name, out_port_type) in \
        new_outputs_ordered_dict.iteritems():
            # append
            self._state_label_def[out_port_name] = out_port_type
            
            # inform state vars
            self.outputs[out_port_name] = \
                self._state_label_def[out_port_name]
            
            # printing format
            self._state_dot_label_format[out_port_name] = \
                '/out:' +str(out_port_name)

class MealyMachine(FiniteStateMachine):
    """Mealy machine.
    
    A Mealy machine implements the discrete dynamics:
        x[k+1] = f(x[k], u[k] )
        y[k] = g(x[k], u[k] )
    where:
        -k: discrete time = sequence index
        -x: state = valuation of state variables
        -X: set of states = S
        -u: inputs = valuation of input ports
        -y: output actions = valuation of output ports
        -f: X-> 2^X, transition function
        -g: X-> Out, output function
    Observe that the output is defined when a reaction occurs to an input.
    
    note
    ----
    valuation: assignment of values to each port
    """
    def __init__(self, **args):
        FiniteStateMachine.__init__(self, **args)
        
        # will point to selected values of self._transition_label_def
        self.default_export_fname = 'mealy'
    
    def add_outputs(self, new_outputs_ordered_dict):
        for (out_port_name, out_port_type) in \
        new_outputs_ordered_dict.iteritems():
            # append
            self._transition_label_def[out_port_name] = out_port_type
            
            # inform state vars
            self.outputs[out_port_name] = \
                self._transition_label_def[out_port_name]
            
            # printing format
            self._transition_dot_label_format[out_port_name] = \
                '/out:' +str(out_port_name)
    
    def get_outputs(self, from_state, next_state):
        #labels = 
        
        output_valuations = dict()
        for output_port in self.outputs:
            output_valuations[output_port]
    
    def update(self, input_valuations, from_state='current'):
        if from_state != 'current':
            if self.states.current != None:
                warnings.warn('from_state != current state,\n'+
                              'will set current = from_state')
            self.current = from_state
        
        transitions = self.transitions.find({from_state},
                                            desired_label=input_valuations)
        next_states = [v for u,v,l in transitions]
        outputs = self.get_outputs(from_state, next_states,
                                   desired_label=input_valuations)
        self.states.set_current(next_states)
        
        return zip(outputs, next_states)

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
