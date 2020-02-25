import os, json
import stormpy
import numpy
from itertools import product


class StateTransition:
    def __init__(self, to_state, probability, action=0):
        self.to_state = to_state
        self.action = action
        self.probability = probability


class Model:
    def __init__(self, model_type, domain, initial_state):
        self.model_type = model_type
        self.domain = domain
        self.initial_state = initial_state
        self.state_var = "s"
        self.states_transitions = {}
        self.states_labels = {}

    def __str__(self):
        ret = (
            "Model type: "
            + str(self.model_type)
            + ", number of states "
            + str(len(self.states_transitions))
            + "\n"
        )
        for state, state_transitions in self.states_transitions.items():
            for transition in state_transitions:
                ret += (
                    "  From state "
                    + str(state)
                    + ", label "
                    + str(self.states_labels[state])
                    + ", action "
                    + str(transition.action)
                    + ", with probability "
                    + self._get_prob_str(transition.probability)
                    + ", go to state "
                    + str(transition.to_state)
                    + "\n"
                )
        return ret

    def _get_state_str(self, state, is_prime):
        state_str = []
        for coord in range(len(state)):
            this_str = "(" + self.state_var + str(coord)
            if is_prime:
                this_str += "'"
            this_str += "=" + str(state[coord]) + ")"
            state_str.append(this_str)
        return " & ".join(state_str)

    @staticmethod
    def _get_prob_str(prob):
        return "{:.4f}".format(prob)

    @staticmethod
    def _get_transition_dict(state_transitions):
        transition_dict = {}

        for transition in state_transitions:
            if transition.action not in transition_dict:
                transition_dict[transition.action] = []
            transition_dict[transition.action].append(
                (transition.probability, transition.to_state)
            )

        return transition_dict

    def _get_label_dict(self):
        all_labels = set([l for labels in self.states_labels.values() for l in labels])
        label_dict = {}
        for label in all_labels:
            label_dict[label] = [
                state
                for state in self.states_labels.keys()
                if label in self.states_labels[state]
            ]
        return label_dict

    def _get_transition_str(self, transitions):
        str_list = [
            self._get_prob_str(transition[0])
            + " : "
            + self._get_state_str(transition[1], True)
            for transition in transitions
        ]
        return " + ".join(str_list)

    def add_state(self, state, labels):
        if state not in self.states_transitions:
            self.states_transitions[state] = []
            self.states_labels[state] = labels

    def add_transition(self, from_state, to_state, probability, action=None):
        assert from_state in self.states_transitions
        assert to_state in self.states_transitions
        transition = StateTransition(to_state, probability, action)
        self.states_transitions[from_state].append(transition)

    def validate(self, eps=1e-3):
        for state, state_transitions in self.states_transitions.items():
            transition_dict = self._get_transition_dict(state_transitions)
            for action, transitions in transition_dict.items():
                total_probability = sum([t[0] for t in transitions])
                if abs(1 - total_probability) > eps:
                    print(
                        "State {}, action {}, total probability: {}".format(
                            state, action, total_probability
                        )
                    )
                    return False
        return True

    def write(self, fname):
        with open(fname, "w") as f:
            f.write(self.model_type)
            f.write("\n\nmodule composition\n")
            for i in range(len(self.domain)):
                f.write(
                    "    {}{} : [{}..{}] init {};\n".format(
                        self.state_var,
                        i,
                        self.domain[i][0],
                        self.domain[i][-1],
                        self.initial_state[i],
                    )
                )
            f.write("\n")

            for state, state_transitions in self.states_transitions.items():
                transition_dict = self._get_transition_dict(state_transitions)
                for action, transitions in transition_dict.items():
                    action_str = ""
                    if self.model_type == "mdp":
                        action_str = "act{}".format(action)
                    f.write(
                        "    [{}] {} -> {};\n".format(
                            action_str,
                            self._get_state_str(state, False),
                            self._get_transition_str(transitions),
                        )
                    )

            f.write("\nendmodule\n\n")

            for label, states in self._get_label_dict().items():
                if label == "init":
                    continue
                state_str = "|".join(
                    ["(" + self._get_state_str(state, False) + ")" for state in states]
                )
                f.write('label "{}" = {};\n'.format(label, state_str))


def _get_labels(models, state):
    return [
        l
        for coord in range(len(state))
        for l in models[coord].states[state[coord]].labels
        if l != "init"
    ]


def _is_prism_program(candidate):
    return isinstance(candidate, stormpy.storage.PrismProgram)


def print_model(model, name=""):
    model = get_model(model)
    print(
        "Model {}, type {}, number of states {}, number of transitins {}".format(
            name, model.model_type, model.nr_states, model.nr_transitions
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


def get_model(model):
    if _is_prism_program(model):
        return stormpy.build_model(model)
    return model


def build_prism_program(path):
    return stormpy.parse_prism_program(path)


def compose_models(models, path, debug=False):
    domain = []
    initial_state = []
    model_type = "dtmc"
    mdp_indices = []

    all_models = []
    for idx, model in enumerate(models):
        this_model = get_model(model)
        all_models.append(this_model)

        assert (
            this_model.model_type == stormpy.storage.ModelType.DTMC
            or this_model.model_type == stormpy.storage.ModelType.MDP
        )
        if this_model.model_type == stormpy.storage.ModelType.MDP:
            model_type = "mdp"
            mdp_indices.append(idx)
        domain.append(range(0, this_model.nr_states))
        assert len(this_model.initial_states) == 1
        initial_state.append(this_model.initial_states[0])

    model = Model(model_type, domain, initial_state)
    for state in product(*domain):
        model.add_state(state, _get_labels(all_models, state))
        states_transitions = []
        for coord in range(len(state)):
            states_transitions.append(
                [
                    StateTransition(transition.column, transition.value(), action)
                    for action in all_models[coord].states[state[coord]].actions
                    for transition in action.transitions
                ]
            )

        for state_transition in product(*states_transitions):
            to_state = tuple(t.to_state for t in state_transition)
            action = 0
            if len(mdp_indices) == 1:
                action = state_transition[mdp_indices[0]].action
            elif len(mdp_indices) > 1:
                action = tuple(state_transition[idx].action for idx in mdp_indices)
            trans_prob = numpy.prod([t.probability for t in state_transition])
            model.add_state(to_state, _get_labels(all_models, to_state))
            model.add_transition(state, to_state, trans_prob, action)

    assert model.validate()
    if debug:
        print(model)
    model.write(path)

    return stormpy.parse_prism_program(path)


def model_checking(prism_program, formula):
    assert _is_prism_program(prism_program)
    model = stormpy.build_model(prism_program)
    properties = stormpy.parse_properties(formula, prism_program)
    result = stormpy.model_checking(model, properties[0], extract_scheduler=True)
    return (result, model)


def export_policy(model, result, path):
    model = get_model(model)

    assert result.has_scheduler
    policy = result.scheduler
    policy_dict = {}
    for state in model.states:
        action = 0
        if policy is not None:
            choice = policy.get_choice(state)
            action = choice.get_deterministic_choice()
        policy_dict[int(state)] = {
            "labels": list(state.labels),
            "action": action
        }

    with open(path, 'w') as outfile:
        json.dump(policy_dict, outfile, indent=4)


def apply_policy(model, policy_path, path, debug=False):
    model = get_model(model)
    assert model.model_type == stormpy.storage.ModelType.MDP
    assert len(model.initial_states) == 1

    with open(policy_path) as policy_file:
        policy = json.load(policy_file)

    domain = [range(0, model.nr_states)]
    initial_state = [model.initial_states[0]]
    model_type = "dtmc"

    mc_model = Model(model_type, domain, initial_state)
    for state in model.states:
        from_state = (int(state),)
        mc_model.add_state(from_state, _get_labels([model], from_state))
        policy_action = policy[str(state)]["action"]
        for action in state.actions:
            if str(action) != str(policy_action):
                continue
            for transition in action.transitions:
                to_state = (int(transition.column),)
                mc_model.add_state(to_state, _get_labels([model], to_state))
                mc_model.add_transition(from_state, to_state, transition.value(), action)

    assert mc_model.validate()
    if debug:
        print(mc_model)
    mc_model.write(path)

    return stormpy.parse_prism_program(path)
