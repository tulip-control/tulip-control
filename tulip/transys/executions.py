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
Classes for executions, traces, words, input port valuations, etc.
"""
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

    Attributes::
    
        execution = s0, a1, s1, a1, ..., aN, sN (Prefix)
                    sN, a(N+1), ..., aM, sN (Suffix)
        path = s0, s1, ..., sN (Prefix)
               sN, s(N+1), ..., sN (Suffix)
        trace = L(s0), L(s1), ..., L(sN) (Prefix)
                L(sN), L(s(N+1) ), ..., L(sN) (Suffix)
    
    where::
        sI \in States
        aI \in Actions (=Transition_Labels =Edge_Labels)
        L(sI) \in State_Labels
    
    Note
    ====
    trace computation avoided because it requires definition of
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

        UNDER DEVELOPMENT; function signature may change without
        notice.  Calling will result in NotImplementedError.
        """
        raise NotImplementedError

class FTSSim(FiniteTransitionSystemSimulation):
    """Alias for L{FiniteTransitionSystemSimulation}."""
    
    def __init__(self, **args):
        FiniteTransitionSystemSimulation.__init__(self, **args)

class InfiniteWord(InfiniteSequence):
    """Store word.
    
    Caution that first symbol corresponds to w1, not w0.
    
    word = w1, w2, ..., wN
    """
    
    def __init__(self, prefix=[], suffix=[]):
        InfiniteSequence.__init__(self, prefix, suffix)

class FiniteStateAutomatonSimulation(object):
    """Store automaton input word and run.

    Attributes::

        input_word = w1, w2, ...wN (Prefix)
                     wN, ..., wM (Suffix)
        run = s0, s1, ..., sN (Prefix)
              sN, ..., sM (Suffix)
    
    These are interpreted as occurring in alternation::
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
        """Dump to file.
        
        We need to decide a format.

        UNDER DEVELOPMENT; function signature may change without
        notice.  Calling will result in NotImplementedError.
        """
        raise NotImplementedError
        
class FSASim(FiniteStateAutomatonSimulation):
    """Alias."""
    
    def __init__(self, **args):
        FiniteStateAutomatonSimulation.__init__(self, **args)

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

        UNDER DEVELOPMENT; function signature may change without
        notice.  Calling will result in NotImplementedError.
        """
        raise NotImplementedError
        
    def save():
        """Dump to file.

        UNDER DEVELOPMENT; function signature may change without
        notice.  Calling will result in NotImplementedError.
        """
        raise NotImplementedError

class MachineInputSequence(object):
    """Stores sequence of input port valuations.
    
    An input port valuation is an assignment of values to input ports.
    So it can be viewed as a list of dictionaries.
    
    However, storing a list of dictionaries is more expensive than
    storing a dictionary of lists.
    
    So this is a dictionary of lists, keyed by input port name.
    In addition, it performs type checking for the input port values.
    """
    def __init__(self, machine):
        """Initialize by defining inputs.
        
        @param machine: from where to copy the input port definitions
        @type machine: FiniteStateMachine
        """
        if not hasattr(machine, 'inputs'):
            raise TypeError(
                'machine has no inputs field\n.' +
                'Got type:\n\t' +str(type(machine) ) )
        
        self.inputs = machine.inputs
        self._input_valuations = dict()
    
    def __str__(self):
        s = ''
        for i in range(len(self) ):
            cur_port_values = [values[i]
                               for values in self._input_valuations.values() ]
            s += str(zip(self.inputs, cur_port_values) )
        return s
    
    def __getitem__(self, num):
        return [self._input_valuations[port_name][num]
                for port_name in self.inputs]
    
    def __len__(self):
        return min(self._input_valuations.values() )
    
    def set_input_sequence(self, input_port_name, values_sequence):
        """Define sequence of input values for single port.
        
        @param input_port_name: name of input port
        @type input_port_name: str in C{self.input_ports}
        
        @param values_sequence: history of input values for C{input_port}
        @type values_sequence: Iterable of values for this port.
            Values must be valid with respect to the C{input_port} type.
            The type is defined by self.inputs[input_port].
        """
        input_port_type = self.inputs[input_port_name]
        
        # check given values
        for value in values_sequence:
            input_port_type.is_valid_value(value)
        
        self._input_valuations[input_port_name]
