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

"""A module for defining functions"""
from collections.abc import Iterable
import json


class LabeledStateInputOutputPair:
    """Stores state, inputs, outputs, and state labels.

    A class for storing state and other inputs, outputs and
    labels on the state.
    """

    def __init__(self, state, input_dict, output_dict, labels=[]):
        self.state = state
        self.input_dict = input_dict
        self.output_dict = output_dict
        self.labels = labels

    def __str__(self):
        ret = f'(state={self.state},label={self.labels}'
        for key, val in self.input_dict.items():
            ret += f',{key}={val}'
        ret += ") -> ("
        output_str_list = []
        for key, val in self.output_dict.items():
            output_str_list.append(f'{key}={val}')
        ret += ",".join(output_str_list) + ")"
        return ret

    def is_at(self, state, input_dict):
        """Whether this is at the given state and input"""
        return self.state == state and self.input_dict == input_dict

    def to_json(self, transys=None):
        """Convert this object to a jsonable object, i.e., a dictionary"""
        labels = (
            transys.states[self.state]["ap"]
            if transys is not None else self.labels)
        ret = dict(
            state=self.state,
            labels=labels)
        for key, val in self.input_dict.items():
            ret[str(key)] = val
        for key, val in self.output_dict.items():
            ret[str(key)] = val
        return ret


class FunctionOnLabeledState:
    r"""Represents a function of a labeled state.

    A class that represents a function of a labeled state,
    with possibly additional inputs.

    This class represents a function

    ```
    f: S \times I_1 \times \ldots \times I_n
       \to O_1 \times \ldots \times O_m
    ```

    where:

    - `I_1`, \ldots, `I_n` are the sets of inputs and
    - `O_1`, \ldots, `O_m` are the sets of outputs.

    Examples
    ========

    ```python
    f = FunctionOnLabeledState("state", "action")
    ```

    Then `f` is a memoryless policy,
    i.e., `f[s] = a` where `s` is a state and `a` is an action.

    ```python
    f = FunctionOnLabeledState(["state", "mode"], "action")
    ```

    Then `f` is a finite memory policy, i.e., `f[(s,q)] = a`
    where `s` is a state, `q` is a mode and `a` is an action.

    Note that "state" has to be part of the input.
    It has a special meaning here since it will be used to compute labels.
    """

    def __init__(self, input_keys, output_keys):
        self._state_input_output_list = []
        if (
                isinstance(input_keys, Iterable) and
                not isinstance(input_keys, str)):
            self._input_keys = input_keys
        else:
            self._input_keys = [input_keys]
        if (
                isinstance(output_keys, Iterable) and
                not isinstance(output_keys, str)):
            self._output_keys = output_keys
        else:
            self._output_keys = [output_keys]
        self._state_index = input_keys.index("state")

    def __str__(self):
        ret = [str(pair) for pair in self._state_input_output_list]
        return ", ".join(ret)

    def __len__(self):
        return len(self._state_input_output_list)

    def __getitem__(self, input_tuple):
        (state, input_dict) = self.get_state_and_input_dict(input_tuple)
        pair = self.get_state_input_output_pair(state, input_dict)
        if pair is None:
            raise KeyError(
                f"Input tuple {input_tuple} does not exist")
        return self.get_output_tuple(pair.output_dict)

    def get_state_and_input_dict(self, input_tuple):
        """Separate a given tuple into state and other input.

        @return: a `tuple` `(state, input_dict)` where
            - `state` is the state
            - `input_dict` is a dictionary of input keys and values
        """
        if len(self._input_keys) == 1:
            input_tuple = (input_tuple,)
        assert len(self._input_keys) == len(input_tuple)
        state = input_tuple[self._state_index]
        input_dict = {}
        for ind, input_key in enumerate(self._input_keys):
            if ind == self._state_index:
                continue
            input_dict[input_key] = input_tuple[ind]
        return (state, input_dict)

    def get_output_dict(self, output_tuple):
        """Convert a tuple of outputs to a dictionary.

        @return: `dict` whose keys are the output keys and
            values are their corresponding values
            obtained from the given tuple
        """
        if len(self._output_keys) == 1:
            output_tuple = (output_tuple,)
        assert len(self._output_keys) == len(output_tuple)
        output_dict = {}
        for ind, output_key in enumerate(self._output_keys):
            output_dict[output_key] = output_tuple[ind]
        return output_dict

    def get_output_tuple(self, output_dict):
        """Convert an output dictionary to a tuple.

        @return: a tuple of output
        """
        if len(output_dict) == 0:
            return None
        if len(output_dict) == 1:
            return next(iter(output_dict.values()))
        return (output_dict[key] for key in self._output_keys)

    def get_state_input_output_pair(self, state, input_dict):
        """Find the first element with the given state and additional input.

        @return: a `LabeledStateInputOutputPair` object `obj`
            in `self._state_input_output_list`
            such that `obj.is_at(state, input_dict)`
        """
        return next(
            (
                pair
                for pair in self._state_input_output_list
                if pair.is_at(state, input_dict)
            ),
            None)

    def add(self, input_tuple, output_tuple, labels=[]):
        """Add a map `input_tuple` -> `output_tuple` to this function.

        An optional label of the state may be provided.
        """
        (state, input_dict) = self.get_state_and_input_dict(input_tuple)
        output_dict = self.get_output_dict(output_tuple)
        pair = self.get_state_input_output_pair(
            state, input_dict)
        if pair is not None:
            print(
                'Warning: replacing output at '
                f'state {state} with {output_tuple}')
            pair.output_dict = output_dict
            pair.labels = labels
            return
        self._state_input_output_list.append(
            LabeledStateInputOutputPair(
                state,
                input_dict,
                output_dict,
                labels))

    def save(self, path, transys=None):
        """Export this to a JSON file.

        If transys is provided, the label at each state will be
        computed based on the labeling function of `transys`.
        """
        with open(path, "w") as outfile:
            json.dump(
                [pair.to_json(transys)
                 for pair in self._state_input_output_list],
                outfile,
                indent=4)
