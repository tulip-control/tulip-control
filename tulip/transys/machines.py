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

