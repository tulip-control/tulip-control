# This test includes unit tests for the tulip.transys.mathfunc module
#
# * single_input_test(): a unit test for the case where there is a single input
# * multiple_inputs_test(): a unit test for the case where there are multiple inputs

import os
from tulip.transys.mathfunc import FunctionOnLabeledState


def single_input_test():
    func = FunctionOnLabeledState("state", "action")
    num = 10
    _fill(num, func)
    _common_tests(num, func)

    (state, input_dict) = func.get_state_and_input_dict((0, 0))
    assert state == (0, 0)
    assert len(input_dict) == 0

    _get_state_input_output_pair(
        func, (0, 0), {}, {"action": str(1)}, [])
    _get_state_input_output_pair(
        func, (1, 1), {}, {"action": str(2)}, ["odd"])


def multiple_inputs_test():
    func = FunctionOnLabeledState(["state", "mode"], "action")
    num = 10
    _fill(num, func)
    _common_tests(num, func)

    (state, input_dict) = func.get_state_and_input_dict((0, 0))
    assert state == 0
    assert input_dict == {"mode": 0}

    _get_state_input_output_pair(
        func, 0, {"mode": 0}, {"action": str(1)}, [])
    _get_state_input_output_pair(
        func, 1, {"mode": 1}, {"action": str(2)}, ["odd"])


def _fill(num, func):
    for i in range(num):
        if i % 2:
            func.add((i, i), str(i + 1), ["odd"])
        else:
            func.add((i, i), str(i + 1))


def _common_tests(num, func):
    assert len(func) == num
    for i in range(num):
        assert func[(i, i)] == str(i + 1)

    output_dict = func.get_output_dict(str(1))
    assert len(output_dict) == 1
    assert output_dict["action"] == str(1)

    assert func.get_output_tuple({"action": str(2)}) == str(2)

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmpfunc.json")
    func.save(path)
    os.remove(path)


def _get_state_input_output_pair(func, state, input_dict, output_dict, labels):
    pair = func.get_state_input_output_pair(state, input_dict)
    assert pair.state == state
    assert pair.input_dict == input_dict
    assert pair.output_dict == output_dict
    assert pair.labels == labels
