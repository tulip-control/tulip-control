# Copyright (c) 2013-2015 by California Institute of Technology
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
"""Finite State Machines Module"""
from __future__ import absolute_import
import copy
from pprint import pformat
from random import choice
from tulip.transys.labeled_graphs import LabeledDiGraph
# inline imports:
#
# from tulip.transys.export import machine2scxml


_hl = 40 * '-'
# port type
pure = {'present', 'absent'}


def is_valuation(ports, valuations):
    for name, port_type in ports.iteritems():
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
    for env_var, var_type in spc_vars.iteritems():
        if var_type == 'boolean':
            domain = {0, 1}
        elif isinstance(var_type, tuple):
            # integer domain
            start, end = var_type
            domain = set(range(start, end + 1))
        elif isinstance(var_type, list):
            # arbitrary finite domain defined by list var_type
            domain = set(var_type)
        ports[env_var] = domain
    return ports


class Transducer(LabeledDiGraph):
    """Sequential Transducer, i.e., a letter-to-letter function.

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

    def __init__(self):
        # values will point to values of _*_label_def below
        self.state_vars = dict()
        self.inputs = dict()
        self.outputs = dict()
        # self.set_actions = {}

        # state labeling
        self._state_label_def = dict()
        self._state_dot_label_format = {'type?label': ':',
                                        'separator': r'\\n'}

        # edge labeling
        self._transition_label_def = dict()
        self._transition_dot_label_format = {'type?label': ':',
                                             'separator': r'\\n'}
        self._transition_dot_mask = dict()
        self._state_dot_mask = dict()

        self.default_export_fname = 'fsm'

        LabeledDiGraph.__init__(self)

        self.dot_node_shape = {'normal': 'ellipse'}
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
        for port_name, port_type in new_inputs.iteritems():
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
        for var_name, var_type in new_state_vars.iteritems():
            # append
            self._state_label_def[var_name] = var_type
            # inform state vars
            self.state_vars[var_name] = self._state_label_def[var_name]
            # printing format
            self._state_dot_label_format[var_name] = str(var_name)


class MooreMachine(Transducer):
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

    def __init__(self):
        """Instantiate a Moore state machine."""
        Transducer.__init__(self)
        self.dot_node_shape = {'normal': 'ellipse'}
        self.default_export_fname = 'moore'

    def __str__(self):
        """Get informal string representation."""
        s = (
            _hl + '\nMoore Machine: ' + self.name + '\n' + _hl + '\n' +
            'State Variables:\n\t(name : type)\n' +
            _print_ports(self.state_vars) +
            'Input Ports:\n\t(name : type)\n' +
            _print_ports(self.inputs) +
            'Output Ports:\n\t(name : type)\n' +
            _print_ports(self.outputs) +
            'States & State Var Values: (state : outputs : vars)\n')
        for state, label_dict in self.states(data=True):
            s += '\t' + str(state) + ' :\n'
            # split into vars and outputs
            var_values = {k: v for k, v in label_dict.iteritems()
                          if k in self.state_vars}
            output_values = {k: v for k, v in label_dict.iteritems()
                             if k in self.outputs}
            s += (_print_label(var_values) + ' : ' +
                  _print_label(output_values))
        s += (
            'Initial States:\n' +
            pformat(self.states.initial, indent=3) + 2 * '\n')
        s += 'Transitions & Labels: (from --> to : label)\n'
        for from_state, to_state, label_dict in self.transitions(data=True):
            s += (
                '\t' + str(from_state) + ' ---> ' +
                str(to_state) + ' :\n' +
                _print_label(label_dict))
        s += _hl + '\n'
        return s

    def add_outputs(self, new_outputs, masks=None):
        for port_name, port_type in new_outputs.iteritems():
            # append
            self._state_label_def[port_name] = port_type
            # inform state vars
            self.outputs[port_name] = port_type
            # printing format
            self._state_dot_label_format[port_name] = \
                '/' + str(port_name)
            if masks is None:
                continue
            if port_name in masks:
                mask_func = masks[port_name]
                self._state_dot_mask[port_name] = mask_func


class MealyMachine(Transducer):
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

    def __init__(self):
        Transducer.__init__(self)
        # will point to selected values of self._transition_label_def
        self.dot_node_shape = {'normal': 'ellipse'}
        self.default_export_fname = 'mealy'

    def __str__(self):
        """Get informal string representation."""
        s = (
            _hl + '\nMealy Machine: ' + self.name + '\n' + _hl + '\n' +
            'State Variables:\n\t(name : type)\n' +
            _print_ports(self.state_vars))
        s += 'States & State Var Values:\n'
        for state, label_dict in self.states(data=True):
            s += ('\t' + str(state) + ' :\n' +
                  _print_label(label_dict))
        s += (
            'Initial States:\n' +
            pformat(self.states.initial, indent=3) + 2 * '\n' +
            'Input Ports:\n\t(name : type)\n' +
            _print_ports(self.inputs) +
            'Output Ports:\n\t(name : type)\n' +
            _print_ports(self.outputs) +
            'Transitions & Labels: (from --> to : label)\n')
        for from_state, to_state, label_dict in self.transitions(data=True):
            s += (
                '\t' + str(from_state) + ' ---> ' +
                str(to_state) + ' :\n' +
                _print_label(label_dict))
        s += _hl + '\n'
        return s

    def _save(self, path, fileformat):
        """Export options available only for Mealy machines.

        @type fileformat: 'scxml'
        """
        if fileformat != 'scxml':
            return False
        from tulip.transys.export import machine2scxml
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
        for port_name, port_type in new_outputs.iteritems():
            # append
            self._transition_label_def[port_name] = port_type
            # inform state vars
            self.outputs[port_name] = \
                self._transition_label_def[port_name]
            # printing format
            self._transition_dot_label_format[port_name] = \
                '/' + str(port_name)
            if masks is None:
                continue
            if port_name in masks:
                mask_func = masks[port_name]
                self._transition_dot_mask[port_name] = mask_func

    def reaction(self, from_state, inputs, lazy=False):
        """Return next state and output, when reacting to given inputs.

        The machine must be deterministic.
        (for each state and input at most a single transition enabled,
        this notion does not coincide with output-determinism)

        Not exactly a wrapper of L{Transitions.find},
        because it matches only that part of an edge label
        that corresponds to the inputs.

        @param from_state: transition starts from this state.
        @type from_state: element of C{self.states}

        @param inputs: C{dict} assigning a valid value to each input port.
        @type inputs: {'port_name':port_value, ...}

        @param lazy: Lazy evaluation of inputs? If lazy=True, then
            allow an incomplete specification of input if there is
            precisely one enabled transition.
        @type lazy: bool

        @return: output values and next state.
        @rtype: (outputs, next_state)
          where C{outputs}: C{{'port_name':port_value, ...}}
        """
        if lazy:
            restricted_inputs = set(self.inputs).intersection(inputs.keys())
        else:
            restricted_inputs = self.inputs
        # match only inputs (explicit valuations, not symbolic)
        enabled_trans = [
            (i, j, d)
            for i, j, d in self.edges_iter([from_state], data=True)
            if project_dict(d, restricted_inputs) == inputs]

        if len(enabled_trans) == 0:
            some_possibilities = []
            for i, j, d in self.edges_iter([from_state], data=True):
                # The number of possible inputs to suggest here is
                # arbitrary. Consider making it a function parameter.
                if len(some_possibilities) >= 5:
                    break
                possible_inputs = project_dict(d, restricted_inputs)
                if possible_inputs not in some_possibilities:
                    some_possibilities.append(possible_inputs)

        # must be deterministic
        try:
            ((_, next_state, attr_dict), ) = enabled_trans
        except ValueError:
            if len(enabled_trans) == 0:
                if len(some_possibilities) == 0:
                    raise Exception(
                        'state {from_state} is a dead-end. '
                        'There are no possible inputs from '
                        'it.'.format(from_state=from_state))
                else:
                    raise Exception(
                        'not a valid input, '
                        'some possible inputs include: '
                        '{t}'.format(t=some_possibilities))
            else:
                raise Exception(
                    'must be input-deterministic, '
                    'found enabled transitions: '
                    '{t}'.format(t=enabled_trans))
        outputs = project_dict(attr_dict, self.outputs)
        return (next_state, outputs)

    def reactionpart(self, from_state, inputs):
        """Wraps reaction() with lazy=True
        """
        return self.reaction(from_state, inputs, lazy=True)

    def run(self, from_state=None, input_sequences=None):
        """Guided or interactive run.

        @param input_sequences: if C{None}, then call L{interactive_run},
            otherwise call L{guided_run}.

        @return: output of L{guided_run}, otherwise C{None}.
        """
        if input_sequences is None:
            interactive_run(self, from_state=from_state)
        else:
            return guided_run(self, from_state=from_state,
                              input_sequences=input_sequences)


def guided_run(mealy, from_state=None, input_sequences=None):
    """Run deterministic machine reacting to given inputs.

    @param from_state: start simulation

    @param mealy: input-deterministic Mealy machine
    @type mealy: L{MealyMachine}

    @param from_state: start simulation at this state.
        If C{None}, then use the unique initial state C{Sinit}.

    @param input_sequences: one sequence of values for each input port
    @type input_sequences: C{dict} of C{lists}

    @return: sequence of states and sequence of output valuations
    @rtype: (states, output_sequences)
      where:
        - C{states} is a C{list} of states excluding C{from_state}
        - C{output_sequences} is a C{dict} of C{lists}
    """
    seqs = input_sequences  # abbrv
    missing_ports = set(mealy.inputs).difference(seqs)
    if missing_ports:
        raise ValueError('missing input port(s): ' + missing_ports)
    # dict of lists ?
    non_lists = {k: v for k, v in seqs.iteritems() if not isinstance(v, list)}
    if non_lists:
        raise TypeError('Values must be lists, for: ' + str(non_lists))
    # uniform list len ?
    if len(set(len(x) for x in seqs.itervalues())) > 1:
        raise ValueError('All input sequences must be of equal length.')
    # note: initial sys state non-determinism not checked
    # initial sys edge non-determinism checked instead (more restrictive)
    if from_state is None:
        state = next(iter(mealy.states.initial))
    else:
        state = from_state
    n = len(next(seqs.itervalues()))
    states_seq = []
    output_seqs = {k: list() for k in mealy.outputs}
    for i in range(n):
        inputs = {k: v[i] for k, v in seqs.iteritems()}
        state, outputs = mealy.reaction(state, inputs)
        states_seq.append(state)
        for k in output_seqs:
            output_seqs[k].append(outputs[k])
    return (states_seq, output_seqs)


def random_run(mealy, from_state=None, N=10):
    """Return run from given state for N random inputs.

    Inputs selected randomly in a way that does not block the machine
    So they are not arbitrarily random.
    If the machine is a valid synthesis solution,
    then all safe environment inputs can be generated this way.

    Randomly generated inputs may violate liveness assumption on environment.

    @param mealy: input-deterministic Mealy machine
    @type mealy: C{MealyMachine}

    @param N: number of reactions (inputs)
    @type N: int

    @return: same as L{guided_run}
    """
    if from_state is None:
        state = next(iter(mealy.states.initial))
    else:
        state = from_state
    states_seq = []
    output_seqs = {k: list() for k in mealy.outputs}
    for i in xrange(N):
        trans = mealy.transitions.find([state])
        # choose next transition
        selected_trans = choice(list(trans))
        _, new_state, attr_dict = selected_trans
        # extend execution trace
        states_seq.append(new_state)
        # extend output traces
        outputs = project_dict(attr_dict, mealy.outputs)
        for k in output_seqs:
            output_seqs[k].append(outputs[k])
        # updates
        old_state = state
        state = new_state
        # printing
        inputs = project_dict(attr_dict, mealy.inputs)
        print(
            'move from\n\t state: ' + str(old_state) +
            '\n\t with input:' + str(inputs) +
            '\n\t to state: ' + str(new_state) +
            '\n\t reacting by producing output: ' + str(outputs))
    return (states_seq, output_seqs)


def interactive_run(mealy, from_state=None):
    """Run input-deterministic Mealy machine using user input.

    @param mealy: input-deterministic Mealy machine
    @type mealy: L{MealyMachine}
    """
    if from_state is None:
        state = next(iter(mealy.states.initial))
    else:
        state = from_state
    while True:
        print('\n Current state: ' + str(state))
        if _interactive_run_step(mealy, state) is None:
            break


def _interactive_run_step(mealy, state):
    if state is None:
        raise Exception('Current state is None')
    # note: the spaghettiness of previous version was caused
    #   by interactive simulation allowing both output-non-determinism
    #   and implementing spawning (which makes sense only for generators,
    #   *not* for transducers)
    trans = mealy.transitions.find([state])
    if not trans:
        print('Stop: no outgoing transitions.')
        return None
    while True:
        try:
            selected_trans = _select_transition(mealy, trans)
        except:
            print('Selection not recognized. Please try again.')
    if selected_trans is None:
        return None
    (from_, to_state, attr_dict) = selected_trans
    inputs = project_dict(attr_dict, mealy.inputs)
    outputs = project_dict(attr_dict, mealy.outputs)
    print(
        'Moving from state: ' + str(state) +
        ', to state: ' + str(to_state) + '\n' +
        'given inputs: ' + str(inputs) + '\n' +
        'reacting with outputs: ' + str(outputs))
    return True


def _select_transition(mealy, trans):
    msg = 'Found more than 1 outgoing transitions:' + 2 * '\n'
    for i, t in enumerate(trans):
        (from_state, to_state, attr_dict) = t
        inputs = project_dict(attr_dict, mealy.inputs)
        outputs = project_dict(attr_dict, mealy.outputs)
        msg += (
            '\t' + str(i) + ' : ' +
            str(from_state) + ' ---> ' + str(to_state) + '\n' +
            '\t inputs:' + str(inputs) +
            '\t outputs:' + str(outputs) +
            '\n\n')
    msg += (
        '\n' +
        'Select from the available transitions above\n' +
        'by giving its integer,\n' +
        'Press "Enter" to stop the simulation:\n' +
        '\t int = ')
    id_selected = raw_input(msg)
    if not id_selected:
        return None
    return trans[int(id_selected)]


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
            masks = {port_name: mask_func}
        mealy.add_inputs({port_name: port_type}, masks=masks)
    # cp outputs
    for port_name, port_type in moore.outputs.iteritems():
        mask_func = moore._state_dot_mask.get(port_name)
        if mask_func is None:
            masks = None
        else:
            masks = {port_name: mask_func}
        mealy.add_outputs({port_name: port_type}, masks=masks)
    # cp states
    mealy.states.add_from(moore.states())
    mealy.states.initial.add_from(moore.states.initial)
    # cp transitions
    for si in moore:
        output_values = {
            k: v for k, v in moore.states[si].iteritems()
            if k in moore.outputs}
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
            masks = {port_name: mask_func}
        moore.add_inputs({port_name: port_type}, masks=masks)
    # cp outputs
    for port_name, port_type in mealy.outputs.iteritems():
        mask_func = mealy._transition_dot_mask.get(port_name)
        if mask_func is None:
            masks = None
        else:
            masks = {port_name: mask_func}
        moore.add_outputs({port_name: port_type}, masks=masks)
    # initial state with arbitrary label
    out = {k: list(v)[0] for k, v in mealy.outputs.iteritems()}
    s0 = list(mealy.states.initial)[0]
    # create maps between Moore and Mealy states
    moore2mealy_states = dict()  # {qj : si} (function)
    mealy2moore_states = dict()  # {si : {qj, qk, ...} } (relation)
    new_s0 = _create_state_str(
        s0, out, moore, moore2mealy_states,
        mealy2moore_states)
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
                mealy2moore_states)
            moore.transitions.add(new_si, new_sj, in_values)
            if new_sj not in S:
                Q.add(new_sj)
                S.add(new_sj)
    return moore


def _print_ports(port_dict):
    s = ''
    for port_name, port_type in port_dict.iteritems():
        s += '\t' + str(port_name) + ' : '
        s += pformat(port_type) + '\n'
    s += '\n'
    return s


def _print_label(label_dict):
    s = ''
    for name, value in label_dict.iteritems():
        s += '\t\t' + str(name) + ' : ' + str(value) + '\n'
    s += '\n'
    return s


def _create_state_str(mealy_state, output, moore,
                      moore2mealy_states,
                      mealy2moore_states):
    """Used to create Moore states when converting Mealy -> Moore."""
    for s in mealy2moore_states.setdefault(mealy_state, set()):
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
    """Split into inputs and outputs."""
    input_values = {k: v for k, v in attr_dict.iteritems()
                    if k in machine.inputs}
    output_values = {k: v for k, v in attr_dict.iteritems()
                     if k in machine.outputs}
    return input_values, output_values


project_dict = lambda x, y: {k: x[k] for k in x if k in y}
trim_dict = lambda x, y: {k: x[k] for k in x if k not in y}


def strip_ports(mealy, names):
    """Remove ports in C{names}.

    For example, to remove the atomic propositions
    labeling the transition system C{ts} used
    (so they are dependent variables), call it as:

      >>> strip_ports(mealy, ts.atomic_propositions)

    @type mealy: L{MealyMachine}

    @type names: iterable container of C{str}
    """
    new = MealyMachine()

    new.add_inputs(trim_dict(mealy.inputs, names))
    new.add_outputs(trim_dict(mealy.outputs, names))

    new.add_nodes_from(mealy)
    new.states.initial.add_from(mealy.states.initial)

    for u, v, d in mealy.edges_iter(data=True):
        d = trim_dict(d, names)
        new.add_edge(u, v, **d)
    return new
