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
#
"""
Classes representing trajectories of dynamical systems.
"""


class DiscreteTimeFiniteTrajectory(object):
    """A class that represents a discrete time finite trajectory with finite timed word

    Examples
    ========
    >>> trajectory = DiscreteTimeFiniteTrajectory(tstep, states)

    * tstep is a positive float that represents the time duration between consecutive states.
    * states is a nonempty list of states.

    >>> trajectory[t]

    returns a state at time t. t must be a float such that 0 <= t <= trajectory.get_final_time()

    >>> len(trajectory)

    returns the number of states.

    >>> trajectory.get_states()

    returns the list of states.

    >>> trajectory.get_final_time()

    returns the time at the last state.

    >>> trajectory.get_final_state()

    returns the last state.
    Note that trajectory.get_final_state() == trajectory[trajectory.get_final_time()].

    >>> trajectory.get_finite_timed_word(labeling_function)

    returns the finite timed word of trajectory.
    Here, labeling_function is a function that takes a state and returns a set of labels.
    The finite timed word is a list of tuple (labels, duration)
    where labels is a set of labels and duration is the amount of time.

    """

    def __init__(self, tstep, states, teps=1e-3):
        if not isinstance(states, list):
            raise TypeError("State must of a list")
        if len(states) == 0:
            raise ValueError("Initial state must be included")
        if tstep <= 0:
            raise ValueError("tstep needs to be positive")

        self._tstep = tstep
        self._states = states
        self._teps = teps
        self._final_time = self._get_time_at(len(states) - 1)

    def __str__(self):
        ret = [
            self._to_str(ind * self._tstep, state)
            for ind, state in enumerate(self._states)
        ]
        return ", ".join(ret)

    def __len__(self):
        """Return the number of states"""
        return len(self._states)

    def __getitem__(self, time):
        """Return the state at the given time"""
        if time > self._final_time + self._teps or time < 0:
            raise KeyError("time {} is more than final time".format(time))

        if time > self._final_time - self._teps:
            return self._states[-1]

        index = round(time / self._tstep)
        excess = (index * self._tstep) - time

        if abs(excess) < self._teps:
            return self._states[index]

        other_index = index + round(excess / abs(excess))
        return self._states[index] + (
            (self._states[other_index] - self._states[index])
            * (((time / self._tstep) - index) / (other_index - index))
        )

    def __add__(self, other):
        """Returns the concatenation of this trajectory and other

        The first state of other is removed since it should be the same
        or close to the last state of this trajectory.
        """
        if self._tstep != other._tstep:
            raise ValueError("Cannot concatenate trajectories with different tstep")

        return DiscreteTimeFiniteTrajectory(
            self._tstep, self.get_states() + other.get_states()[1:], self._teps
        )

    def __iadd__(self, other):
        """Concatenate other to this

        The first state of other is removed since it should be the same
        or close to the last state of this trajectory.
        """
        if self._tstep != other._tstep:
            raise ValueError("Cannot concatenate trajectories with different tstep")

        self._states.extend(other._states[1:])
        self._final_time += other._final_time
        return self

    def get_states(self):
        """Returns the list of states"""
        return self._states

    def get_final_time(self):
        """Return the time at the last state"""
        return self._final_time

    def get_final_state(self):
        """Return the last state"""
        return self._states[-1]

    def get_finite_timed_word(self, labeling_function, sampling_step_size=1):
        """Returns the finite timed word of trajectory.

        See L.I.R. Castro, P. Chaudhari, J. Tumova, S. Karaman, E. Frazzoli, D. Rus.
        Incremental Sampling-based Algorithm for Minimum-violation Motion Planning,
        CDC, 2013.
        for definition of finite timed word of a finite trajectory.
        """
        finite_timed_word = []
        prev_label_time = None
        for ind in range(0, len(self._states), sampling_step_size):
            state = self._states[ind]
            if prev_label_time is None:
                prev_label_time = (labeling_function(state), self._get_time_at(ind))
                continue
            current_label = labeling_function(state)
            if current_label != prev_label_time[0]:
                current_time = self._get_time_at(ind)
                finite_timed_word.append(
                    (prev_label_time[0], current_time - prev_label_time[1])
                )
                prev_label_time = (current_label, current_time)

        finite_timed_word.append(
            (prev_label_time[0], self._final_time - prev_label_time[1])
        )
        return finite_timed_word

    def _to_str(self, time, state):
        return "t={:.3f},s={}".format(time, state)

    def _get_time_at(self, state_index):
        return self._tstep * state_index
