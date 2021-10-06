# Copyright (c) 2020 by California Institute of Technology
# and Iowa State University
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
# 3. Neither the name of the copyright holder(s) nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
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

"""Interface to `stormpy`.

<https://moves-rwth.github.io/stormpy/>
"""
import copy

import stormpy

from tulip.transys import MarkovChain as MC
from tulip.transys import MarkovDecisionProcess as MDP
from tulip.transys.mathfunc import FunctionOnLabeledState


def build_stormpy_model(path):
    """Return the `stormpy` model created from the `prism` file.

    @type path: a string indicating the path to the `prism` file
    """
    prism_program = stormpy.parse_prism_program(path)
    return stormpy.build_model(prism_program)


def print_stormpy_model(model):
    """Print the `stormpy` model"""
    print(
        "Model type {}, number of states {}, number of transitins {}".format(
            model.model_type, model.nr_states, model.nr_transitions
        )
    )
    for state in model.states:
        for action in state.actions:
            for transition in action.transitions:
                print(
                    "  From state {}, labels {}, action {}, with probability {:.4f}, go to state {}".format(
                        state,
                        state.labels,
                        action,
                        transition.value(),
                        transition.column,
                    )
                )


def get_action_map(stormpy_model, tulip_transys):
    """Get a map of action of `stormpy_model` and that of `tulip_transys`.

    @return a dictionary whose key is
        the string representation of an action of `stormpy_model`
        and value is the corresponding action of `tulip_transys`
    """
    action_map = {}
    for from_state_stormpy in stormpy_model.states:
        from_state_tulip = to_tulip_state(from_state_stormpy, tulip_transys)
        for stormpy_action in from_state_stormpy.actions:
            possible_actions = action_map.get(str(stormpy_action), None)
            if possible_actions is None:
                action_map[str(stormpy_action)] = list(tulip_transys.actions)
                possible_actions = action_map[str(stormpy_action)]
            _update_possible_actions_with_transitions(
                possible_actions,
                stormpy_model,
                tulip_transys,
                from_state_tulip,
                stormpy_action.transitions,
                prob_tol=1e-6,
            )
    return action_map


def to_tulip_action(
        stormpy_action,
        stormpy_model,
        tulip_transys,
        action_map=None):
    """Return action of `tulip_transys` on model.

    Get an action of `tulip_transys` that
    corresponds to `stormpy_action` on `stormpy_model`.

    The computation is based on the transition probability,
    i.e., it returns an action such that, from each pair of
    source and target state, the transition probability
    matches that of `stormpy_action`.
    """
    if action_map is None:
        action_map = get_action_map(stormpy_model, tulip_transys)

    possible_tulip_action = action_map[str(stormpy_action)]
    if len(possible_tulip_action) == 0:
        raise ValueError(
            "Cannot find an action on tulip_transys corresponding to {}".format(
                stormpy_action
            )
        )

    return possible_tulip_action[0]


def to_tulip_labels(stormpy_state, tulip_transys):
    """Get set of atomic propositions at `stormpy_state` for `tulip_transys`.

    This typically involves getting rid of `stormpy` internal labels such as
    `"init"`, `"deadlock"`, etc.

    @return a `MathSet` object corresponding to
        the set of atomicic propositions
        of `stormpy_state` in `tulip_transys`
    """
    return tulip_transys.atomic_propositions.intersection(stormpy_state.labels)


def to_tulip_state(stormpy_state, tulip_transys):
    """Get unique state of `tulip_transys` that corresponds to `stormpy_state`.

    The computation is based on the state labels,
    i.e., it returns the unique state
    in `tulip_transys` with the same labels as `stormpy_state`.

    If such a state does not exist or is not unique,
    `ValueError` will be raised.
    """
    possible_states = [
        s for s in tulip_transys.states
        if set(tulip_transys.states[s]["ap"])
        == set(to_tulip_labels(stormpy_state, tulip_transys))
    ]
    if len(possible_states) != 1:
        raise ValueError(
            "Cannot find a unique state corresponding to label {}".format(
                stormpy_state.labels
            )
        )
    return possible_states[0]


def to_tulip_transys(path):
    """Return Markov chain or decisiom process from `prism` file.

    Return either a `MarkovChain` or
    `MarkovDecisionProcess` object,
    created from the `prism` file.

    @type path: a string indicating the path to the `prism` file
    """
    # Convert a state of prism file
    # (typically represented by an integer) to
    # state on the tulip transition system.
    def get_ts_state(in_model_state):
        return "s" + str(in_model_state)

    # Only allow DTMC and MDP models
    in_model = build_stormpy_model(path)
    assert (
        in_model.model_type == stormpy.storage.ModelType.DTMC
        or in_model.model_type == stormpy.storage.ModelType.MDP
    )

    # The list of states and initial states
    state_list = [get_ts_state(s) for s in in_model.states]
    initial_state_list = [
        get_ts_state(s)
        for s in in_model.initial_states]
    ts = MC() if in_model.model_type == stormpy.storage.ModelType.DTMC else MDP()
    ts.states.add_from(state_list)
    ts.states.initial.add_from(initial_state_list)
    # Neglect stormpy internal state labels
    neglect_labels = ['init', 'deadlock']
    # Populate the set of atomic propositinos and
    # compute the labels and transitions at each state
    for state in in_model.states:
        state_ap = copy.deepcopy(state.labels)
        for label in neglect_labels:
            state_ap.discard(label)
        ts.atomic_propositions.add_from(state_ap)
        ts.states[get_ts_state(state)]['ap'] = state_ap
        for action in state.actions:
            if in_model.model_type == stormpy.storage.ModelType.MDP:
                ts.actions.add(str(action))
            for transition in action.transitions:
                transition_attr = {MC.probability_label: transition.value()}
                if in_model.model_type == stormpy.storage.ModelType.MDP:
                    transition_attr[MDP.action_label] = str(action)
                ts.transitions.add(
                    get_ts_state(state),
                    get_ts_state(transition.column),
                    transition_attr,
                )
    return ts


def to_prism_file(ts, path):
    """Write a prism file corresponding to ts

    @type ts: `MarkovChain` or `MarkovDecisionProcess`
    @param path: path of output `prism` file
    @type path: `str`
    """
    # Only deal with MDP and MC with a unique initial state
    assert type(ts) == MDP or type(ts) == MC
    assert len(ts.states) > 0
    assert len(ts.states.initial) == 1
    # The state and action of prism file will be
    # of the form si and acti
    # where i is the index in state_list and
    # action_list of the state and action
    # of the tulip model.
    # This is to deal with restriction on
    # naming of prism model.
    # For example, it doesn't allow action
    # that is just an integer.
    state_var = 's'
    state_list = list(ts.states)
    action_list = list(ts.actions) if type(ts) == MDP else []
    # Given all the transitions from
    # a state s of ts, return a dictionary whose
    # key is an action a and the value is
    # a tuple (probability, to_state)
    # indicating the transition probability
    # from state s to to_state under action a.
    def get_transition_dict(state_transitions):
        transition_dict = {}
        for transition in state_transitions:
            action = transition[2].get(MDP.action_label, None)
            if action is None:
                action = 0
            if action not in transition_dict:
                transition_dict[action] = []
            transition_dict[action].append(
                (transition[2].get(MC.probability_label), transition[1])
            )

        return transition_dict
    # Return a properly formatted string for the given probability
    def get_prob_str(prob):
        return "{:.4f}".format(prob)
    # Return a string representing the
    # state with the given index and whether
    # it is primed (primed in prism file
    # indicates the next state).
    def get_state_str(state_index, is_prime):
        state_str = state_var
        if is_prime:
            state_str += "'"
        state_str += '=' + str(state_index)
        return state_str
    # Return the description of transitions in prism file
    def get_transition_str(transitions):
        str_list = [
            get_prob_str(transition[0])
            + ' : ('
            + get_state_str(state_list.index(transition[1]), True)
            + ')'
            for transition in transitions
        ]
        return ' + '.join(str_list)
    # Return a dictionary whose key is an atomic proposition
    # and whose value is a list of state such that the atomic proposition]
    # is in its state labels.
    def get_label_dict():
        label_dict = {}
        for label in ts.atomic_propositions:
            label_dict[label] = [
                state for state in ts.states
                if label in ts.states[state]['ap']
            ]
        return label_dict
    # Use the above functions to describe the model in prism format.
    with open(path, 'w') as f:
        # Type of model
        if type(ts) == MDP:
            f.write('mdp')
        elif type(ts) == MC:
            f.write('dtmc')
        # The set of states and initial state
        f.write('\n\nmodule sys_model\n')
        f.write(
            "    {} : [0..{}] init {};\n".format(
                state_var,
                len(ts.states) - 1,
                state_list.index(list(ts.states.initial)[0]),
            )
        )
        f.write('\n')
        # Transitions
        for idx, state in enumerate(state_list):
            transition_dict = get_transition_dict(ts.transitions.find(state))
            for action, transitions in transition_dict.items():
                action_str = ''
                if type(ts) == MDP:
                    action_str = "act" + str(action_list.index(action))
                f.write(
                    "    [{}] {} -> {};\n".format(
                        action_str,
                        get_state_str(idx, False),
                        get_transition_str(transitions),
                    )
                )
        f.write('\nendmodule\n\n')
        # Labels
        for label, states in get_label_dict().items():
            state_str = '|'.join(
                ['(' + get_state_str(state_list.index(s), False) + ')'
                 for s in states]
            )
            f.write('label "{}" = {};\n'.format(label, state_str))


def model_checking(
        tulip_transys,
        formula,
        prism_file_path,
        extract_policy=False):
    """Model check `tulip_transys` against `formula`.

    @type tulip_transys: `MarkovChain` or
        `MarkovDecisionProcess`
    @param formula: a string describing PCTL formula
        according to Prism format
    @type formula: `str`
    @param prism_file_path: a string indicating the
        path to export the intermediate
        `prism` file (mostly for debugging purposes)
    @type prism_file_path: `str`
    @param extract_policy: boolean that indicates
        whether to extract policy
    @type extract_policy: `bool`

    @return result
        - If `extract_policy == False`,
          then for each `state` in `model.states`,
          `result[state]` is the probability of
          satisfying the formula starting at `state`.
        - If `extract_policy == True`, then
          `result = (prob, policy)`, where for each
          `state` in `model.states`, `prob[state]` is
          the probability of satisfying the
          formula starting at `state`, and
          `policy[state]` is the action to be
          applied at `state`.
    """
    assert type(tulip_transys) == MDP or type(tulip_transys) == MC
    to_prism_file(tulip_transys, prism_file_path)
    prism_program = stormpy.parse_prism_program(prism_file_path)
    stormpy_model = stormpy.build_model(prism_program)
    properties = stormpy.parse_properties(formula, prism_program)
    result = stormpy.model_checking(
        stormpy_model, properties[0], extract_scheduler=extract_policy
    )
    prob = _extract_probability(result, stormpy_model, tulip_transys)
    if not extract_policy:
        return prob
    policy = _extract_policy(result, stormpy_model, tulip_transys)
    return (prob, policy)


def _extract_policy(stormpy_result, stormpy_model, tulip_transys):
    """Extract policy from `stormpy_result`."""
    assert stormpy_result.has_scheduler
    stormpy_policy = stormpy_result.scheduler
    assert stormpy_policy is not None
    tulip_policy = FunctionOnLabeledState("state", MDP.action_label)
    action_map = get_action_map(stormpy_model, tulip_transys)
    for state in stormpy_model.states:
        tulip_state = to_tulip_state(state, tulip_transys)
        choice = stormpy_policy.get_choice(state)
        action = choice.get_deterministic_choice()
        tulip_policy.add(
            tulip_state,
            to_tulip_action(action, stormpy_model, tulip_transys, action_map),
            to_tulip_labels(state, tulip_transys),
        )

    return tulip_policy


def _extract_probability(stormpy_result, stormpy_model, tulip_transys):
    """Extract probability of satisfying specification at each state.

    Extracts the probability of satisfying
    a specification at each state
    from `stormpy_result`.
    """
    probability = FunctionOnLabeledState('state', 'probability')
    for state in stormpy_model.states:
        tulip_state = to_tulip_state(state, tulip_transys)
        probability.add(tulip_state, stormpy_result.at(state))
    return probability


def _update_possible_actions_with_transitions(
        possible_actions,
        stormpy_model,
        tulip_transys,
        from_state_tulip,
        stormpy_transitions,
        prob_tol=1e-6):
    """Return subset of `possible_actions` according to conditions.

    Return a subset of `possible_actions` from
    `from_state_tulip`, such that the probability of
    transition to each state matches `stormpy_transitions`
    with the given `prob_tol`.
    """
    for stormpy_transition in stormpy_transitions:
        to_state_tulip = to_tulip_state(
            stormpy_model.states[stormpy_transition.column], tulip_transys
        )
        probability = stormpy_transition.value()
        for tulip_transition in tulip_transys.transitions.find(
            from_state_tulip, [to_state_tulip]
        ):
            if abs(tulip_transition[2][MC.probability_label] - probability) > prob_tol:
                try:
                    idx = possible_actions.index(tulip_transition[2][MDP.action_label])
                    possible_actions.pop(idx)
                except ValueError:
                    pass
