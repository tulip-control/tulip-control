#!/usr/bin/env python

""" 
-----------------
Simulation Module
-----------------

Nok Wongpiromsarn (nok@cds.caltech.edu)

:Date: August 20, 2010
:Version: 0.1.0
"""

import random
from automaton import *
from errorprint import printWarning, printError

def grsim(aut, init_state, env_states=[], num_it=20, deterministic_env=True, verbose=0):
    """
    Simulate an execution of the given automaton and return a sequence of states.

    Input:

    - `aut`: an Automaon object or the name of the text file containing the automaton generated from
      jtlvint.synthesize or jtlv.computeStrategy function.
    - `init_state`: a dictionary that (partially) specifies the initial state. 
    - `env_states`: a list of dictionary of environment state, specifying the sequence of
      environment states after the initial state. If the length of this sequence is
      less than `num_it`, then this function will automatically pick the environment states
      for the rest of the execution.
    - `num_it`: the number of iterations.
    - `deterministic_env`: If len(env_states) < num_it, then `deterministic_env` specifies
      whether this function will choose the environment state deterministically.
    """
    if (isinstance(aut, str)):
        aut = Automaton(states_or_file=aut,verbose=verbose)
    aut_state = aut.findNextAutState(current_aut_state=None, env_state=init_state)
    states = [aut_state.state]

    for i in xrange(0, num_it):
        if (len(env_states) >= i+1):
            aut_state = aut.findNextAutState(current_aut_state=aut_state, \
                                                 env_state=env_states[i])
            if (not isinstance(aut_state, AutomatonState)):
                printError('The specified sequence of environment states ' + \
                               'does not satisfy the environment assumption.')
                return states
        else:
            transition = aut_state.transition[:]
            for trans in aut_state.transition:
                tmp_aut_state = aut.getAutState(trans)
                if (len(tmp_aut_state.transition) == 0):
                    transition.remove(trans)
            if (len(transition) == 0):
                printWarning('Environment cannot satisfy its assumption')
                return states
            elif (deterministic_env):
                aut_state = aut.getAutState(transition[0])
            else:
                aut_state = aut.getAutState(random.choice(transition))
        states.append(aut_state.state)
    return states
