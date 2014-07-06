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
import copy
from pprint import pformat
from random import choice

from .labeled_graphs import LabeledDiGraph
from . import executions
from .export import machine2scxml

_hl = 40 *'-'

# port type
pure = {'present', 'absent'}

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
    ports = dict()
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
        self.state_vars = dict()
        self.inputs = dict()
        self.outputs = dict()
        #self.set_actions = {}
        
        # state labeling
        self._state_label_def = dict()
        self._state_dot_label_format = {'type?label':':',
                                        'separator':'\n'}
        
        # edge labeling
        self._transition_label_def = dict()
        self._transition_dot_label_format = {'type?label':':',
                                             'separator':'\n'}
        self._transition_dot_mask = dict()
        self._state_dot_mask = dict()
        
        self.default_export_fname = 'fsm'
        
        LabeledDiGraph.__init__(self, **args)
        
        self.dot_node_shape = {'normal':'ellipse'}
        self.default_export_fname = 'fsm'
    
    def add_inputs(self, new_inputs, masks=None):
        """Create new inputs.
        
        @param new_inputs: C{dict} of pairs {port_name : port_type}
            where:
                - port_name: str
                - port_type: Iterable | check class
        @type new_inputs: dict
        
        @param masks: custom mask functions, for each sublabel
            based on its current value
            each such function returns:
                - True, if the sublabel should be shown
                - False, otherwise (to hide it)
        @type masks: C{dict} of functions C{{port_name : mask_function}}
            each C{mask_function} returns bool
        """
        for (port_name, port_type) in new_inputs.iteritems():
            # append
            self._transition_label_def[port_name] = port_type
            
            # inform inputs
            self.inputs[port_name] = port_type
            
            # printing format
            self._transition_dot_label_format[port_name] = str(port_name)
            
            if masks is None:
                continue
            
            if port_name in masks:
                mask_func = masks[port_name]
                self._transition_dot_mask[port_name] = mask_func
    
    def add_state_vars(self, new_state_vars):
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
    
    def is_receptive(self, states=None):
        """For each state, for each input valuation, there exists a transition.
        
        @param states: states to be checked whether blocking
        @type states: iterable container of states
        """
        if states is None:
            states = self
        
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
    
    Reference
    =========
    U{[M56]
    <http://tulip-control.sourceforge.net/doc/bibliography.html#m56>}
    """
    def __init__(self, **args):
        """Instantiate a Moore state machine.
        """
        FiniteStateMachine.__init__(self, **args)
        
        self.dot_node_shape = {'normal':'ellipse'}
        self.default_export_fname = 'moore'
    
    def __str__(self):
        """Get informal string representation.
        """
        s = _hl + '\nMoore Machine: ' + self.name + '\n' +_hl + '\n'
        s += 'State Variables:\n\t(name : type)\n'
        s += _print_ports(self.state_vars)
        
        s += 'Input Ports:\n\t(name : type)\n'
        s += _print_ports(self.inputs)
        
        s += 'Output Ports:\n\t(name : type)\n'
        s += _print_ports(self.outputs)
        
        s += 'States & State Var Values: (state : outputs : vars)\n'
        for (state, label_dict) in self.states(data=True):
            s += '\t' +str(state) +' :\n'
            
            # split into vars and outputs
            var_values = {k:v for k, v in label_dict.iteritems()
                          if k in self.state_vars}
            output_values = {k:v for k, v in label_dict.iteritems()
                             if k in self.outputs}
            
            s += _print_label(var_values) + ' : '
            s += _print_label(output_values)
        
        s += 'Initial States:\n'
        s += pformat(self.states.initial, indent=3) +2*'\n'
        
        s += 'Transitions & Labels: (from --> to : label)\n'
        for (from_state, to_state, label_dict) in \
        self.transitions(data=True):
            s += '\t' +str(from_state) +' ---> '
            s += str(to_state) +' :\n'
            s += _print_label(label_dict)
        s += _hl +'\n'
        
        return s
    
    def add_outputs(self, new_outputs, masks=None):
        for (port_name, port_type) in new_outputs.iteritems():
            # append
            self._state_label_def[port_name] = port_type
            
            # inform state vars
            self.outputs[port_name] = port_type
            
            # printing format
            self._state_dot_label_format[port_name] = \
                '/out:' +str(port_name)
            
            if masks is None:
                continue
            
            if port_name in masks:
                mask_func = masks[port_name]
                self._state_dot_mask[port_name] = mask_func

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
    
    Reference
    =========
    U{[M55]
    <http://tulip-control.sourceforge.net/doc/bibliography.html#m55>}
    """
    def __init__(self, **args):
        FiniteStateMachine.__init__(self, **args)
        
        # will point to selected values of self._transition_label_def
        self.dot_node_shape = {'normal':'ellipse'}
        self.default_export_fname = 'mealy'
    
    def __str__(self):
        """Get informal string representation.
        """
        s = _hl +'\nMealy Machine: ' +self.name +'\n' +_hl +'\n'
        s += 'State Variables:\n\t(name : type)\n'
        s += _print_ports(self.state_vars)
        
        s += 'States & State Var Values:\n'
        for (state, label_dict) in self.states(data=True):
            s += '\t' +str(state) +' :\n'
            s += _print_label(label_dict)
        
        s += 'Initial States:\n'
        s += pformat(self.states.initial, indent=3) +2*'\n'
        
        s += 'Input Ports:\n\t(name : type)\n'
        s += _print_ports(self.inputs)
        
        s += 'Output Ports:\n\t(name : type)\n'
        s += _print_ports(self.outputs)
        
        s += 'Transitions & Labels: (from --> to : label)\n'
        for (from_state, to_state, label_dict) in \
        self.transitions(data=True):
            s += '\t' +str(from_state) +' ---> '
            s += str(to_state) +' :\n'
            s += _print_label(label_dict)
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
    
    def add_outputs(self, new_outputs, masks=None):
        """Add new outputs.
        
        @param new_outputs: dict of pairs {port_name : port_type}
          where:
            - port_name: str
            - port_type: Iterable | check class
        @type new_outputs: dict
        
        @param masks: custom mask functions, for each sublabel
            based on its current value
            each such function returns:
              - True, if the sublabel should be shown
              - False, otherwise (to hide it)
        @type masks: dict of functions
            keys are port_names (see arg: new_outputs)
            each function returns bool
        """
        for (port_name, port_type) in new_outputs.iteritems():
            # append
            self._transition_label_def[port_name] = port_type
            
            # inform state vars
            self.outputs[port_name] = \
                self._transition_label_def[port_name]
            
            # printing format
            self._transition_dot_label_format[port_name] = \
                '/out:' +str(port_name)
            
            if masks is None:
                continue
            
            if port_name in masks:
                mask_func = masks[port_name]
                self._transition_dot_mask[port_name] = mask_func
    
    def simulate(self, inputs=None, from_state=None):
        """Guided simulation run (programmatic or interactive).
        
        Simulation starts from (with decreasing precedence):
        
          - C{from_state}
          - C{self.current_states} (deterministic, so unique)
          - C{self.states.initial}
        
        @param inputs: sequence of input port valuations
            If absent, then start an interactive simulation.
        @type inputs: dict of lists
            {'in_port_name':values_history}
            The lists must be of equal length.
        
        @type from_state: in C{self.states}
        """
        if from_state:
            self.states.select_current([from_state] )
        elif not self.states.current:
            print('Current state unset.')
            if not self.states.initial:
                msg = 'Initial state(s) unset.\n'
                msg += 'Set either states.current, or states.initial\n'
                msg += 'before calling simulate().'
                print(msg)
            else:
                self.states.current = set(self.states.initial)
        
        if inputs is not None:
            self._guided_simulation(inputs)
            return
        
        while True:
            print(60 *'-' + '\n Current State:\t' +
                  str(self.states.current) + 2*'\n')
            
            if self._interactive_simulation() is None:
                break
    
    def _guided_simulation(self, inputs):
        missing_ports = set(self.inputs).difference(inputs)
        if missing_ports:
            raise ValueError('missing input port(s): ' + missing_ports)
        
        # uniform lengths (if lists) ?
        try:
            len_lists = set(len(x) for x in inputs.itervalues() )
            if len(len_lists) > 1:
                raise ValueError('all input ports should have same length lists')
        except:
            pass
        
        # dict given (1 time step) ?
        for k, v in inputs.iteritems():
            if not isinstance(v, list):
                inputs[k] = list(v)
        
        # now it is dict of lists
        n = len(inputs.itervalues()[0])
        for i in range(n):
            input_values = {k:v[i] for k, v in inputs}
            
            (current_state,) = self.states.current
            trans = self.transitions.find(current_state, attr_dict=input_values)
            
            # must be deterministic
            assert(len(trans) <= 1)
            
            _, next_state, _ = trans[0]
            
            self.states.current.remove(current_state)
            self.states.current.add(next_state)
    
    def _interactive_simulation(self):
        """Single transition of interactive simulation.
        """
        def select_state(cur_states):
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
        
        def select_transition(transitions):
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
        
        cur_state = self.states.current
        
        if not cur_state:
            print('No current state: Stopping simulation.')
            return None
        
        if len(cur_state) > 1:
            state_selected = select_state(cur_state)
            if state_selected is None:
                return None
        elif cur_state:
            state_selected = choice(list(cur_state) )
        else:
            raise Exception('Bug: "if not" above must have caught this.')
            
        print(':\t' +str(state_selected) )
        from_state = state_selected
        
        transitions = self.transitions.find([from_state] )
        
        if not transitions:
            print('Did not find any outgoing transitions:\n' +
                  'Stopping simulation.')
            return None
        
        if len(transitions) > 1:
            transition_selected = select_transition(transitions)
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
        
        return True

def moore2mealy(moore):
    """Convert Moore machine to equivalent Mealy machine.
    
    Reference
    =========
    U{[LS11]
    <http://tulip-control.sourceforge.net/doc/bibliography.html#ls11>}
    
    @type moore: L{MooreMachine}
    
    @rtype: L{MealyMachine}
    """
    if not isinstance(moore, MooreMachine):
        raise TypeError('moore must be a MooreMachine')
    
    mealy = MealyMachine()
    
    # cp inputs
    for port_name, port_type in moore.inputs.iteritems():
        mask_func = moore._transition_dot_mask.get(port_name)
        if mask_func is None:
            masks = None
        else:
            masks = {port_name:mask_func}
        
        mealy.add_inputs({port_name:port_type}, masks=masks)
    
    # cp outputs
    for port_name, port_type in moore.outputs.iteritems():
        mask_func = moore._state_dot_mask.get(port_name)
        if mask_func is None:
            masks = None
        else:
            masks = {port_name:mask_func}
        
        mealy.add_outputs({port_name:port_type}, masks=masks)
    
    # cp states
    mealy.states.add_from(moore.states() )
    mealy.states.initial.add_from(moore.states.initial)
    
    # cp transitions
    for si in moore:
        output_values = {
            k:v for k, v in moore.states[si].iteritems()
            if k in moore.outputs
        }
        output_values = copy.deepcopy(output_values)
        
        for si_, sj, attr_dict in moore.transitions.find(si):
            # note that we don't filter only input ports,
            # so other edge annotation is preserved
            attr_dict = copy.deepcopy(attr_dict)
            attr_dict.update(output_values)
            
            mealy.transitions.add(si, sj, attr_dict)
    
    return mealy

def mealy2moore(mealy):
    """Convert Mealy machine to almost equivalent Moore machine.
    
    A Mealy machine cannot be transformed to an equivalent Moore machine.
    It can be converted to a Moore machine with an arbitrary initial output,
    which outputs the Mealy output at its next reaction.
    
    Reference
    =========
    U{[LS11]
    <http://tulip-control.sourceforge.net/doc/bibliography.html#ls11>}
    
    @type mealy: L{MealyMachine}
    
    @rtype: L{MooreMachine}
    """
    # TODO: check for when Mealy is exactly convertible to Moore
    
    if not isinstance(mealy, MealyMachine):
        raise TypeError('moore must be a MealyMachine')
    
    moore = MooreMachine()
    
    # cp inputs
    for port_name, port_type in mealy.inputs.iteritems():
        mask_func = mealy._transition_dot_mask.get(port_name)
        if mask_func is None:
            masks = None
        else:
            masks = {port_name:mask_func}
        
        moore.add_inputs({port_name:port_type}, masks=masks)
    
    # cp outputs
    for port_name, port_type in mealy.outputs.iteritems():
        mask_func = mealy._transition_dot_mask.get(port_name)
        if mask_func is None:
            masks = None
        else:
            masks = {port_name:mask_func}
        
        moore.add_outputs({port_name:port_type}, masks=masks)
    
    # initial state with arbitrary label
    out = {k:list(v)[0] for k, v in mealy.outputs.iteritems()}
    s0 = list(mealy.states.initial)[0]
    
    moore2mealy_states = dict() # {qj : si} (function)
    mealy2moore_states = dict() # {si : {qj, qk, ...} } (relation)
    
    new_s0 = _create_state_str(
        s0, out, moore, moore2mealy_states,
        mealy2moore_states
    )
    
    moore.states.add(new_s0, out)
    moore.states.initial.add(new_s0)
    
    # cp transitions and create appropriate states
    Q = set()
    S = set()
    
    Q.add(new_s0)
    S.add(new_s0)
    while Q:
        new_si = Q.pop()
        si = moore2mealy_states[new_si]
        
        for si_, sj, attr_dict in mealy.transitions.find(si):
            in_values, out_values = _split_io(attr_dict, mealy)
            
            new_sj = _create_state_str(
                sj, out_values, moore, moore2mealy_states,
                mealy2moore_states
            )
            
            moore.transitions.add(new_si, new_sj, in_values)
            
            if new_sj not in S:
                Q.add(new_sj)
                S.add(new_sj)
    
    return moore

def _print_ports(port_dict):
    s = ''
    for (port_name, port_type) in port_dict.iteritems():
        s += '\t' +str(port_name) +' : '
        s += pformat(port_type) +'\n'
    s += '\n'
    return s

def _print_label(label_dict):
    s = ''
    for (name, value) in label_dict.iteritems():
        s += '\t\t' +str(name) +' : ' +str(value) +'\n'
    s += '\n'
    return s

def _create_state_str(mealy_state, output, moore,
                      moore2mealy_states,
                      mealy2moore_states):
    """Used to create Moore states when converting Mealy -> Moore.
    """
    for s in mealy2moore_states.setdefault(mealy_state, set() ):
        # check output values
        if moore.states[s] == output:
            return s
    
    # create new
    n = len(moore)
    s = 's' + str(n)
    
    moore.states.add(s, output)
    
    moore2mealy_states[s] = mealy_state
    mealy2moore_states[mealy_state].add(s)
    
    return s

def _split_io(attr_dict, machine):
    """Split into inputs and outputs.
    """
    input_values = {k:v for k, v in attr_dict.iteritems()
                    if k in machine.inputs}
    output_values = {k:v for k,v in attr_dict.iteritems()
                     if k in machine.outputs}
    return input_values, output_values

####
# Program Graph (memo)
####

