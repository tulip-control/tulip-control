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
"""
import networkx as nx
from collections import OrderedDict
import tulip.transys as trs
import tulip.transys.machines as mc
import warnings

hl = 60*'='
save_fig = False

def mealy_machine_example():
    import numpy as np   
    
    class check_diaphragm():
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
    
    class check_camera():
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
    inputs = [
        ('speed', {'zero', 'low', 'high', 'crazy'} ),
        ('seats', trs.PowerSet(range(5) ) ),
        ('aperture', check_diaphragm() ),
        ('camera', check_camera() )
    ]
    
    # outputs def
    outputs = [('photo', {'capture', 'wait'} ) ]
    
    # state variables def
    state_vars = [('light', {'on', 'off'} ) ]
    
    # define the machine itself
    m = trs.MealyMachine()
    
    m.add_state_vars(state_vars)
    m.add_inputs(inputs)
    m.add_outputs(outputs)
    
    m.states.add('s0')
    m.states.add_from(['s1', 's2'])
    m.states.initial.add('s0')
    
    m.states.label('s0', {'light':'on'} )
    m.states.label('s1', ('off') )
    
    # guard defined using input sub-label ordering
    guard = ('low', (0, 1), 0.3, np.array([0,0,1]), 'capture')
    m.transitions.add_labeled('s0', 's1', guard)
    
    # guard defined using input sub-label names
    guard = {'camera':np.array([1,1,1]),
             'speed':'high',
             'aperture':0.3,
             'seats':(2, 3),
             'photo':'wait'}
    m.transitions.add_labeled('s1', 's2', guard)
    
    m.plot(rankdir='TB')
    
    return m

def garage_counter(ploting=True):
    """Example 3.4, p.49 [Lee-Seshia], for M=2
    
    no state variables in this Finite-State Machine
    """
    m = trs.Mealy()
    
    m.add_inputs([
        ['up', {'present', 'absent'}],
        ['down', {'present', 'absent'}]
    ])
    
    m.add_outputs([
        ('count', range(3) )
    ])
    
    m.states.add_from(range(3) )
    m.states.initial.add(0)
    
    m.transitions.add_labeled(0, 1, ('present', 'absent', 1) )
    m.transitions.add_labeled(1, 0, ('absent', 'present', 0) )
    
    m.transitions.add_labeled(1, 2, ('present', 'absent', 2) )
    m.transitions.add_labeled(2, 1, ('absent', 'present', 1) )
    
    if ploting:
        m.plot()
    
    return m

def garage_counter_with_state_vars():
    """Example 3.8, p.57 [Lee-Seshia], for M=2
    unfolded with respect to state variable c
    """
    m = garage_counter(ploting=False)
    
    m.add_state_vars([('c', range(3)) ])
    m.states.label(0, 0)
    m.states.label(1, 1)
    m.states.label(2, 2)
    
    m.plot()
    
    return m

def thermostat_with_hysteresis():
    """Example 3.5, p.50 [Lee-Seshia]
    """
    class temperature_type():
        def is_valid_value(x):
            if not isinstance(x, [float, int]):
                raise TypeError('Input temperature must be float.')
        
        def __contains__(self, guard):
            # when properly implemented, do appropriate syntactic check
            if isinstance(guard, float):
                return True
            
            return False
    
    m = trs.Mealy()
    
    m.add_inputs([('temperature', ) ])

def traffic_light_1():
    m = trs.Mealy()
    pure_signal = {'present', 'absent'}
    
    m.add_inputs([('tick', pure_signal) ])
    m.add_outputs([('go', pure_signal), ('stop', pure_signal) ])
    
    m.states.add_from(['red', 'green', 'yellow'])
    m.states.initial.add('red')
    
    p = 'present'
    a = 'absent'
    
    m.transitions.add_labeled('red', 'green', (p, p, a) )
    m.transitions.add_labeled('green', 'yellow', (p, a, p) )
    m.transitions.add_labeled('yellow', 'red', (p, a, p) )
    
    m.plot()
    return m

def traffic_light_2():
    m = trs.Mealy()
    pure_signal = {'present', 'absent'}
    
    m.add_inputs([('tick', pure_signal) ])
    m.add_outputs([('go', pure_signal), ('stop', pure_signal) ])
    
    m.states.add_from(['red', 'green', 'yellow'])
    m.states.initial.add('red')
    
    p = 'present'
    a = 'absent'
    
    m.transitions.add_labeled('red', 'green',
                              {'tick':p, 'go':p, 'stop':a} )
    m.transitions.add_labeled('green', 'yellow',
                              {'tick':p, 'go':a, 'stop':p} )
    m.transitions.add_labeled('yellow', 'red',
                              {'tick':p, 'go':a, 'stop':p} )
    
    m.plot()
    return m

def pedestrians():
    """Example 2.14, p.63 [Lee-Seshia]
    """
    m = trs.Mealy()
    
    m.add_inputs([
        ('sigR', mc.pure),
        ('sigG', mc.pure),
        ('sigY', mc.pure)
    ])
    
    m.add_outputs([
        ('pedestrian', mc.pure)
    ])
    
    m.states.add_from(['none', 'waiting', 'crossing'] )
    m.states.initial.add('crossing')
    
    for sigR in mc.pure:
        for sigG in mc.pure:
            for sigY in mc.pure:
                m.transitions.add_labeled(
                    'none', 'none',
                    (sigR, sigG, sigY, 'absent')
                )
                
                m.transitions.add_labeled(
                    'none', 'waiting',
                    (sigR, sigG, sigY, 'present')
                )
    
    m.transitions.add_labeled('waiting', 'crossing',
                              ('present', 'absent', 'absent', 'absent') )
    m.transitions.add_labeled('crossing', 'none',
                              ('absent', 'present', 'absent', 'absent') )
    m.plot()
    return m

if __name__ == '__main__':
    saving = False
    
    m1 = mealy_machine_example()
    m2 = garage_counter()
    m3 = garage_counter_with_state_vars()
    m4 = pedestrians()
    m5 = traffic_light_1()
    m6 = traffic_light_2()
    
    m6.simulate('random', 4)
    #m6.simulate() for manual simulation
    
    # save animated javascript
    if saving:
        m4.save('index.html', 'html')
