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
Finite State Machines Module
"""
from collections import OrderedDict
from pprint import pformat
from random import choice

from .labeled_graphs import LabeledDiGraph
from . import executions
from .export import machine2scxml

_hl = 40 *'-'

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

def create_machine_ports(spc_vars):
    """Create proper port domains of valuations, given port types.

    @param spc_vars: port names and types inside tulip.
        For arbitrary finite types the type can be a list of strings,
        instead of a range of integers.
        These are as originally defined by the user or synth.
    """
    ports = OrderedDict()
    for env_var, var_type in spc_vars.items():
        if var_type == 'boolean':
            domain = {0,1}
        elif isinstance(var_type, tuple):
            # integer domain
            start, end = var_type
            domain = set(range(start, end+1))
        elif isinstance(var_type, list):
            # arbitrary finite domain defined by list var_type
            domain = set(var_type)

        ports[env_var] = domain
    return ports


class FiniteStateMachine(LabeledDiGraph):
    """Transducer, i.e., a system with inputs and outputs.
    
    Inputs
    ======
    P = {p1, p2,...} is the set of input ports.
    An input port p takes values in a set Vp.
    Set Vp is called the "type" of input port p.
    A "valuation" is an assignment of values to the input ports in P.
    
    We call "inputs" the set of pairs::
    
        {(p_i, Vp_i),...}
    
    of input ports p_i and their corresponding types Vp_i.
    
    A guard is a predicate (bool-valued) used as sub-label for a transition.
    A guard is defined by a set and evaluated using set membership.
    So given an input port value p=x, then if::
    
        x \in guard_set
    
    then the guard is True, otherwise it is False.
    
    The "inputs" are defined by an OrderedDict::
    
        {'p1':explicit, 'p2':check, 'p3':None, ...}
    
    where:
        - C{explicit}:
          is an iterable representation of Vp,
          possible only for discrete Vp.
          If 'p1' is explicitly typed, then guards are evaluated directly::
            
                input_port_value == guard_value ?
        
        - C{check}:
          is a class with methods:
            
                - C{__contains__(x) }:
                  check if guard value given to input port 'p1' is
                  in the set of possible values Vp.
                
                - C{__call__(guard_set, input_port_value) }:
                  check if C{input_port_value} \\in C{guard_set}
                  This allows symbolic type definitions.
                    
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
    
    Guards annotate transitions::
        
        Guards: States x States ---> Input_Predicates
    
    Outputs
    =======
    Similarly defined to inputs, but:
    
        - for Mealy Machines they annotate transitions
        - for Moore Machines they annotate states
    
    State Variables
    ===============
    Similarly defined to inputs, they annotate states,
    for both Mealy and Moore machines::
    
        States ---> State_Variables
    
    Update Function
    ===============
    The transition relation:
    
        - for Mealy Machines::
        
                States x Input_Valuations ---> Output_Valuations x States
                
          Note that in the range Output_Valuations are ordered before States
          to emphasize that an output_valuation is produced
          during the transition, NOT at the next state.
            
          The data structure representation of the update function is
          by storage of the Guards function and definition of Guard
          evaluation for each input port via the OrderedDict discussed above.
        
        - for Moore Machines::
        
            States x Input_Valuations ---> States
            States ---> Output_valuations
    
    Note
    ====
    A transducer may operate on either finite or infinite words, i.e.,
    it is not equipped with interpretation semantics on the words,
    so it does not "care" about word length.
    It continues as long as its input is fed with letters.
    
    For Machines, each state label consists of (possibly multiple) sublabels,
    each of which is either a variable, or, only for Moore machines,
    may be an output.
    
    See Also
    ========
    FSM, MealyMachine, MooreMachine
    """
    def __init__(self, **args):
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
        self._transition_dot_mask = dict()
        
        self.default_export_fname = 'fsm'
        
        LabeledDiGraph.__init__(self, **args)
        
        self.dot_node_shape = {'normal':'ellipse'}
        self.default_export_fname = 'fsm'
    
    def _to_ordered_dict(self, x):
        if not isinstance(x, OrderedDict):
            try:
                x = OrderedDict(x)
            except:
                raise TypeError('Argument must be an OrderedDict, ' +
                                'or be directly convertible to an OrderedDict.')
        return x
    
    def add_inputs(self, new_inputs, masks={}):
        """Create new inputs.
        
        @param new_inputs: ordered pairs of port_name : port_type
        @type new_inputs: OrderedDict | list, of::
                (port_name, port_type)
            where:
                - port_name: str
                - port_type: Iterable | check class
        
        @param masks: custom mask functions, for each sublabel
            based on its current value
            each such function returns:
                - True, if the sublabel should be shown
                - False, otherwise (to hide it)
        @type masks: dict of functions
            keys are port_names (see arg: new_outputs)
            each function returns bool
        """
        new_inputs = self._to_ordered_dict(new_inputs)
        
        for (in_port_name, in_port_type) in new_inputs.iteritems():
            # append
            self._transition_label_def[in_port_name] = in_port_type
            
            # inform inputs
            self.inputs[in_port_name] = self._transition_label_def[in_port_name]
            
            # printing format
            self._transition_dot_label_format[in_port_name] = str(in_port_name)
            
            if in_port_name in masks:
                mask_func = masks[in_port_name]
                self._transition_dot_mask[in_port_name] = mask_func
    
    def add_state_vars(self, new_state_vars):
        new_state_vars = self._to_ordered_dict(new_state_vars)
        
        for (var_name, var_type) in new_state_vars.iteritems():
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

        UNDER DEVELOPMENT; function signature may change without
        notice.  Calling will result in NotImplementedError.
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
        """

        UNDER DEVELOPMENT; function signature may change without
        notice.  Calling will result in NotImplementedError.
        """
        raise NotImplementedError
        
    def async_product(self):
        """

        UNDER DEVELOPMENT; function signature may change without
        notice.  Calling will result in NotImplementedError.
        """
        raise NotImplementedError

class FSM(FiniteStateMachine):
    """Alias for Finite-state Machine."""
    
    def __init__(self, **args):
        FiniteStateMachine.__init__(self, **args)

class MooreMachine(FiniteStateMachine):
    """Moore machine.
    
    A Moore machine implements the discrete dynamics::
        x[k+1] = f(x[k], u[k] )
        y[k] = g(x[k] )
    where:
      - k: discrete time = sequence index
      - x: state = valuation of state variables
      - X: set of states = S
      - u: inputs = valuation of input ports
      - y: output actions = valuation of output ports
      - f: X-> 2^X, transition function
      - g: X-> Out, output function
    Observe that the output depends only on the state.
    
    Note
    ====
    valuation: assignment of values to each port
    """
    def __init__(self, **args):
        """

        UNDER DEVELOPMENT; function signature may change without
        notice.  Calling will result in NotImplementedError.
        """
        FiniteStateMachine.__init__(self, **args)
        
        self.dot_node_shape = {'normal':'ellipse'}
        self.default_export_fname = 'moore'
        
        raise NotImplementedError
    
    def __str__(self):
        """Get informal string representation."""
        #TODO: improve port formatting
        s = _hl +'\nMoore Machine: ' +self.name +'\n' +_hl +'\n'
        s += 'State Variables:\n\t' +pformat(self.state_vars) +'\n'
        s += 'Output ports:\n\t' +pformat(self.outputs) +'\n'
        s += 'States & labeling w/ State Vars & Output Ports:\n\t'
        s += str(self.states(data=True) ) +'\n'
        s += 'Initial States:\n'
        s += pformat(self.states.initial, indent=3) +2*'\n'
        s += 'Input ports:\n\t' +pformat(self.inputs) +'\n'
        s += 'Transitions & labeling w/ Input Port guards:\n\t'
        s += str(self.transitions(data=True) ) +'\n' +_hl +'\n'
        
        return s
    
    def add_outputs(self, new_outputs):
        new_outputs = self._to_ordered_dict(new_outputs)
        
        for (out_port_name, out_port_type) in new_outputs.iteritems():
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
    
    Examples
    ========
    Traffic Light: Fig. 3.14, p.72 U{[LS11]
    <http://tulip-control.sourceforge.net/doc/bibliography.html#ls11>}
    
    >>> m = MealyMachine()
    >>> pure_signal = {'present', 'absent'}
    >>> m.add_inputs([('tick', pure_signal) ])
    >>> m.add_outputs([('go', pure_signal), ('stop', pure_signal) ])
    >>> m.states.add_from(['red', 'green', 'yellow'])
    >>> m.states.initial.add('red')
    
    For brevity:
    
    >>> p = 'present'
    >>> a = 'absent'
    
    The transitions can equivalently be defined with dict().
    So instead of the previous C{m.transitions.add}, we can use:
    
    >>> label = {'tick':p, 'go':p, 'stop':a}
    >>> m.transitions.add('red', 'green', **label)
    >>> label = {'tick':p, 'go':a, 'stop':p}
    >>> m.transitions.add('green', 'yellow', **label)
    >>> label = {'tick':p, 'go':a, 'stop':p}
    >>> m.transitions.add('yellow', 'red', **label)
    
    This avoids any ordering issues, i.e., changing the
    order of the sublabels does not matter:
    
    >>> label = {'go':p, 'tick':p, 'stop':a}
    >>> m.transitions.add('red', 'green', **label)
    
    Theory
    ======
    A Mealy machine implements the discrete dynamics::
        x[k+1] = f(x[k], u[k] )
        y[k] = g(x[k], u[k] )
    where:
      - k: discrete time = sequence index
      - x: state = valuation of state variables
      - X: set of states = S
      - u: inputs = valuation of input ports
      - y: output actions = valuation of output ports
      - f: X-> 2^X, transition function
      - g: X-> Out, output function
    Observe that the output is defined when a reaction occurs to an input.
    
    Note
    ====
    valuation: assignment of values to each port
    """
    def __init__(self, **args):
        FiniteStateMachine.__init__(self, **args)
        
        # will point to selected values of self._transition_label_def
        self.dot_node_shape = {'normal':'ellipse'}
        self.default_export_fname = 'mealy'
    
    def __str__(self):
        """Get informal string representation."""
        def print_ports(port_dict):
            s = ''
            for (port_name, port_type) in port_dict.iteritems():
                s += '\t' +str(port_name) +' : '
                s += pformat(port_type) +'\n'
            s += '\n'
            return s
        
        def print_label(label_dict):
            s = ''
            for (name, value) in label_dict.iteritems():
                s += '\t\t' +str(name) +' : ' +str(value) +'\n'
            s += '\n'
            return s
        
        s = _hl +'\nMealy Machine: ' +self.name +'\n' +_hl +'\n'
        s += 'State Variables:\n\t(name : type)\n'
        s += print_ports(self.state_vars)
        
        s += 'States & State Var Values:\n'
        for (state, label_dict) in self.states(data=True):
            s += '\t' +str(state) +' :\n'
            s += print_label(label_dict)
        
        s += 'Initial States:\n'
        s += pformat(self.states.initial, indent=3) +2*'\n'
        
        s += 'Input Ports:\n\t(name : type)\n'
        s += print_ports(self.inputs)
        
        s += 'Output Ports:\n\t(name : type)\n'
        s += print_ports(self.outputs)
        
        s += 'Transitions & Labels: (from --> to : label)\n'
        for (from_state, to_state, label_dict) in \
        self.transitions(data=True):
            s += '\t' +str(from_state) +' ---> '
            s += str(to_state) +' :\n'
            s += print_label(label_dict)
        s += _hl +'\n'
        
        return s
    
    def _save(self, path, fileformat):
        """Export options available only for Mealy machines.
        
        @type fileformat: 'scxml'
        """
        if fileformat != 'scxml':
            return False
        
        s = machine2scxml.mealy2scxml(self)
        
        # dump to file
        f = open(path, 'w')
        f.write(s)
        f.close()
        return True
    
    def add_outputs(self, new_outputs, masks={}):
        """Add new outputs.
        
        @param new_outputs: ordered pairs of port_name : port_type
        @type new_outputs: OrderedDict | list, of::
                (port_name, port_type)
        where:
            - port_name: str
            - port_type: Iterable | check class
        
        @param masks: custom mask functions, for each sublabel
            based on its current value
            each such function returns:
                - True, if the sublabel should be shown
                - False, otherwise (to hide it)
        @type masks: dict of functions
            keys are port_names (see arg: new_outputs)
            each function returns bool
        """
        new_outputs = self._to_ordered_dict(new_outputs)
        
        for (out_port_name, out_port_type) in new_outputs.iteritems():
            # append
            self._transition_label_def[out_port_name] = out_port_type
            
            # inform state vars
            self.outputs[out_port_name] = \
                self._transition_label_def[out_port_name]
            
            # printing format
            self._transition_dot_label_format[out_port_name] = \
                '/out:' +str(out_port_name)
            
            if out_port_name in masks:
                mask_func = masks[out_port_name]
                self._transition_dot_mask[out_port_name] = mask_func
    
    def simulate(
            self, inputs_sequence='manual', iterations=100,
            current_state=None
        ):
        """Manual, random or guided machine run.
        
        If the argument current_state is passed,
        then simulation starts from there.
        
        Otherwise if MealyMachine.states.current is non-empty,
        then simulation starts from there.
        
        If current states are empty,
        then if MealyMachine.states.initial is non-empty,
        then simulation starts from there.
        
        Otherwise an exception is raised.
        
        @param inputs_sequence: inputs for guided simulation
        @type inputs_sequence: 'manual' | list of input valuations
        
        @param iterations: number of steps for manual or random simulation
        @type iterations: int
        
        @param current_state: state from where to start the simulation
        @type current_state: element in MealyMachine.states
            Note that this allows simulating from any desired state,
            irrespective of whether it is reachable from the subset of
            initial states.
        """
        max_count = iterations
        if inputs_sequence not in ['manual', 'random'] and \
        not isinstance(inputs_sequence, executions.MachineInputSequence):
            raise Exception(
                'Available simulation modes:\n' +
                'manual, random, or guided by given MachineInputSequence.'
            )
        
        if current_state:
            self.states.select_current([current_state] )
        elif not self.states.current:
            print('Current state unset.')
            if not self.states.initial:
                msg = 'Initial state(s) unset.\n'
                msg += 'Set either states.current, or states.initial\n'
                msg += 'before calling .simulate.'
                print(msg)
            else:
                self.states.current = set(self.states.initial)
        
        if isinstance(inputs_sequence, executions.MachineInputSequence):
            self._guided_simulation(inputs_sequence)
            return
        
        count = 1
        stop = False
        mode = inputs_sequence
        while not stop:
            print(60 *'-' +'\n Current States:\t' +
                  str(self.states.current) +2*'\n')
            
            count = self._step_simulation(mode, count)
            
            if mode == 'manual':
                stop = count is None
            elif mode == 'random':
                stop = count is None or count > max_count
            else:
                raise Exception('Bug: mode has unkown value.')
    
    def _guided_simulation(self, inputs_sequence):
        raise NotImplementedError
    
    def _step_simulation(self, mode, count):
        def select_state(cur_states, mode):
            if mode == 'random':
                return choice(cur_states)
            
            cur_states = list(cur_states) # order them
            while True:
                msg = 'Found more than 1 current state.\n'
                msg += 'Select from which state to step forward.\n'
                msg += 'Available choices:' +2*'\n'
                for (num, state) in enumerate(cur_states):
                    msg += '\t' +str(num) +' : '
                    msg += str(state) +'\n'
                msg += '\n'
                msg += 'Select from the above states by giving\n'
                msg += 'the integer corresponding to your selection.\n'
                msg += 'Press "Enter" to terminate the simulation:\n'
                msg += '\t int = '
                
                id_selected = raw_input(msg)
                if not id_selected:
                    return None
                try:
                    return cur_states[int(id_selected) ]
                except:
                    print('Could not convert your input to integer.\n' +
                          'Please try again.\n')
        
        def select_transition(transitions, mode):
            if mode == 'random':
                return choice(transitions)
            
            while True:
                msg = 'Found more than 1 outgoing transitions:' +2*'\n'
                for (num, transition) in enumerate(transitions):
                    (from_state, to_state, guard) = transition
                    msg += '\t' +str(num) +' : '
                    msg += str(from_state) +' ---> ' +str(to_state) +'\n'
                    msg += '\t\t' +str(guard) +'\n'
                msg += '\n'
                msg += 'Select from the available transitions above\n'
                msg += 'by giving its integer,\n'
                msg += 'Press "Enter" to stop the simulation:\n'
                msg += '\t int = '
                
                id_selected = raw_input(msg)
                if not id_selected:
                    return None
                try:
                    return transitions[int(id_selected) ]
                except:
                    print('Could not convert your input to integer.\n' +
                          'Please try again.\n')
        
        count += 1
        cur_states = self.states.current
        
        if not cur_states:
            print('No current states: Stopping simulation.')
            return None
        
        if len(cur_states) > 1:
            state_selected = select_state(cur_states, mode)
            if state_selected is None:
                return None
        elif cur_states:
            state_selected = choice(list(cur_states) )
        else:
            raise Exception('Bug: "if not" above must have caught this.')
            
        print('State to step from:\t' +str(state_selected) )
        from_state = state_selected
        
        transitions = self.transitions.find([from_state] )
        
        if not transitions:
            print('Did not find any outgoing transitions:\n' +
                  'Stopping simulation.')
            return None
        
        if len(transitions) > 1:
            transition_selected = select_transition(transitions, mode)
            if transition_selected is None:
                return None
        elif transitions:
            transition_selected = transitions[0]
        else:
            raise Exception('Bug: must have been caught by "if not" above.')
        
        (from_, to_state, guard) = transition_selected
        
        msg = 'Moving (from state ---> to state):\n\t'
        msg += str(from_state) +' ---> ' +str(to_state) +'\n'
        msg += 'via transition with guard:\n\t' +str(guard) +'\n'
        print(msg)
        
        self.states._current.remove(from_state)
        self.states._current.add(to_state)
        
        return count

class Mealy(MealyMachine):
    """Alias to Mealy machine.
    """
    def __init__(self, *args, **kwargs):
        MealyMachine.__init__(self, *args, **kwargs)

pure = {'present', 'absent'}

def moore2mealy(moore_machine, mealy_machine):
    """Convert Moore machine to equivalent Mealy machine

    UNDER DEVELOPMENT; function signature may change without notice.
    Calling will result in NotImplementedError.
    """
    raise NotImplementedError

####
# Program Graph (memo)
####

