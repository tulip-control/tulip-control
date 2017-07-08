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
Finite State Machine examples

Bibliography:

[LS11] E.A. Lee and S.A. Seshia. *Introduction to Embedded Systems - A Cyber-
Physical Systems Approach*. `LeeSeshia.org <http://LeeSeshia.org>`_, 2011.
"""
import tulip.transys as trs
import tulip.transys.machines as mc

hl = 60*'='
save_fig = False

def mealy_machine_example():
    import numpy as np

    class check_diaphragm(object):
        """camera f-number."""
        def is_valid_value(x):
            if x <= 0.7 or x > 256:
                raise TypeError('This f-# is outside allowable range.')

        def __contains__(self, guard):
            # when properly implemented, do appropriate syntactic check
            if isinstance(guard, float):
                return True

            return False

        def __call__(self, guard_set, input_port_value):
            """This method "knows" that we are using x to denote the input
            within guards."""
            self.is_valid_value(input_port_value)

            guard_var_def = {'x': input_port_value}
            guard_value = eval(guard_set, guard_var_def)

            if not isinstance(guard_value, bool):
                raise TypeError('Guard value is non-boolean.\n'
                                'A guard is a predicate, '
                                'so it can take only boolean values.')

    class check_camera(object):
        """is it looking upwards ?"""
        def is_valid_value(self, x):
            if x.shape != (3,):
                raise Exception('Not a 3d vector!')

        def __contains__(self, guard):
            # when properly implemented, do appropriate syntactic check
            if isinstance(guard, np.ndarray) and guard.shape == (3,):
                return True

            return False

        def __call__(self, guard_set, input_port_value):
            self.is_valid_value(input_port_value)

            v1 = guard_set # guard_halfspace_normal_vector
            v2 = input_port_value # camera_direction_vector

            if np.inner(v1, v2) > 0.8:
                return True

            return False

    # note: guards are conjunctions,
    # any disjunction is represented by 2 edges

    # input defs
    inputs = {
        'speed': {'zero', 'low', 'high', 'crazy'},
        'seats': trs.PowerSet(range(5) ),
        'aperture': check_diaphragm(),
        'camera': check_camera()}

    # outputs def
    outputs = {'photo': {'capture', 'wait'}}

    # state variables def
    state_vars = {'light': {'on', 'off'}}

    # define the machine itself
    m = trs.MealyMachine()

    m.add_state_vars(state_vars)
    m.add_inputs(inputs)
    m.add_outputs(outputs)

    m.states.add('s0')
    m.states.add_from(['s1', 's2'])
    m.states.initial.add('s0')

    m.transitions.add(
        's0', 's1',
        speed='low',
        seats=(0, 1),
        aperture=0.3,
        camera=np.array([0,0,1]),
        photo='capture'
    )

    guard = {'camera':np.array([1,1,1]),
             'speed':'high',
             'aperture':0.3,
             'seats':(2, 3),
             'photo':'wait'}
    m.transitions.add('s1', 's2', **guard)

    m.plot(rankdir='TB')

    return m

def garage_counter(ploting=True):
    """Example 3.4, p.49 [LS11], for M=2

    no state variables in this Finite-State Machine
    """
    m = trs.MealyMachine()

    m.add_inputs({
        'up': {'present', 'absent'},
        'down': {'present', 'absent'}})

    m.add_outputs({'count': list(range(3))})

    m.states.add_from(list(range(3)) )
    m.states.initial.add(0)

    m.transitions.add(0, 1, up='present', down='absent', count=1)
    m.transitions.add(1, 0, up='absent', down='present', count=0)

    m.transitions.add(1, 2, up='present', down='absent', count=2)
    m.transitions.add(2, 1, up='absent', down='present', count=1)

    if ploting:
        m.plot()

    return m

def garage_counter_with_state_vars():
    """Example 3.8, p.57 [LS11], for M=2
    unfolded with respect to state variable c
    """
    m = garage_counter(ploting=False)

    m.add_state_vars({'c': range(3)})
    m.states.add(0, c=0)
    m.states.add(1, c=1)
    m.states.add(2, c=2)

    m.save()

    return m

def thermostat_with_hysteresis():
    """Example 3.5, p.50 [LS11]
    """
    class temperature_type(object):
        def is_valid_value(x):
            if not isinstance(x, [float, int]):
                raise TypeError('Input temperature must be float.')

        def __contains__(self, guard):
            # when properly implemented, do appropriate syntactic check
            if isinstance(guard, float):
                return True

            return False

    m = trs.MealyMachine()

    m.add_inputs({'temperature': set()})

def traffic_light():
    m = trs.MealyMachine()
    pure_signal = {'present', 'absent'}

    m.add_inputs({'tick': pure_signal})
    m.add_outputs({'go': pure_signal, 'stop': pure_signal})

    m.states.add_from(['red', 'green', 'yellow'])
    m.states.initial.add('red')

    p = 'present'
    a = 'absent'

    m.transitions.add('red', 'green', tick=p, go=p, stop=a)
    m.transitions.add('green', 'yellow', tick=p, go=a, stop=p)
    m.transitions.add('yellow', 'red', tick=p, go=a, stop=p)

    m.save()
    return m

def pedestrians():
    """Example 2.14, p.63 [LS11]
    """
    m = trs.MealyMachine()

    m.add_inputs({
        'sigR': mc.pure,
        'sigG': mc.pure,
        'sigY': mc.pure})

    m.add_outputs({'pedestrian': mc.pure})

    m.states.add_from(['none', 'waiting', 'crossing'] )
    m.states.initial.add('crossing')

    for sigR in mc.pure:
        for sigG in mc.pure:
            for sigY in mc.pure:
                m.transitions.add(
                    'none', 'none',
                    sigR=sigR, sigG=sigG, sigY=sigY,
                    pedestrian='absent'
                )

                m.transitions.add(
                    'none', 'waiting',
                    sigR=sigR, sigG=sigG, sigY=sigY,
                    pedestrian='present'
                )

    m.transitions.add(
        'waiting', 'crossing',
        sigR='present', sigG='absent', sigY='absent',
        pedestrian='absent'
    )
    m.transitions.add(
        'crossing', 'none',
        sigR='absent', sigG='present', sigY='absent',
        pedestrian='absent'
    )
    m.save()
    return m

if __name__ == '__main__':
    saving = False

    m1 = mealy_machine_example()
    m2 = garage_counter()
    m3 = garage_counter_with_state_vars()
    m4 = pedestrians()
    m5 = traffic_light()

    m5.run(4)

    # save animated javascript
    if saving:
        m4.save('index.html', 'html')
