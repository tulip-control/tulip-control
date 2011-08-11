# !/usr/bin/env python
#
# Copyright (c) 2011 by California Institute of Technology
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
# 
# $Id$

""" 
-----------------
Simulation Module
-----------------

Nok Wongpiromsarn (nok@cds.caltech.edu)

:Date: August 20, 2010
:Version: 0.1.0

minor refactoring by SCL <slivingston@caltech.edu>
3 May 2011.

graph visualization added by Yuchen Lin
23 June 2011
"""

import time
import random
from multiprocessing import Process
from subprocess import call

from automaton import Automaton, AutomatonState
from errorprint import printWarning, printError
from congexf import dumpGexf, changeGexfAttvalue
from gephistream import GephiStream

def grsim(aut, init_state, env_states=[], num_it=20, deterministic_env=True, verbose=0):
    """
    Simulate an execution of the given automaton and return a sequence of automaton
    states.

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
    aut_states = [aut_state]

    for i in xrange(0, num_it):
        if (len(env_states) > i):
            aut_state = aut.findNextAutState(current_aut_state=aut_state, \
                                                 env_state=env_states[i])
            if (not isinstance(aut_state, AutomatonState)):
                printError('The specified sequence of environment states ' + \
                               'does not satisfy the environment assumption.')
                return aut_states
        else:
            transition = aut_state.transition[:]
            for trans in aut_state.transition:
                tmp_aut_state = aut.getAutState(trans)
                if (len(tmp_aut_state.transition) == 0):
                    transition.remove(trans)
            if (len(transition) == 0):
                printWarning('Environment cannot satisfy its assumption')
                return aut_states
            elif (deterministic_env):
                aut_state = aut.getAutState(transition[0])
            else:
                aut_state = aut.getAutState(random.choice(transition))
        aut_states.append(aut_state)
    return aut_states

###################################################################
def writeStatesToFile(aut_list, aut_states_list, destfile, label_vars=None):
    output = dumpGexf(aut_list, label_vars=label_vars)
    
    # 'aut_states_list' is a list of lists of automaton states. Transitioning
    # from one 'aut_states' to the next should correspond to changing
    # automata in the receding horizon case.
    iteration = 1
    for (i, aut_states) in enumerate(aut_states_list):
        for state in aut_states:
            output = changeGexfAttvalue(output, "is_active", iteration,
                                        node_id=str(i) + '.' + str(state.id))
            iteration += 1
    print "Writing simulation result to " + destfile
    f = open(destfile, 'w')
    f.write(output)
    f.close()

###################################################################
def simulateGraph(aut_states_list, destfile):
    # Number of seconds between each graph update.
    delay = 2
    long_delay = 8
    
    # Changes to the graph will be streamed from here.
    gs = GephiStream('server')
    
    # Open Gephi in a separate thread.
    print "Opening " + destfile + " in Gephi."
    gephi = Process(target=lambda: call(["gephi", destfile]))
    gephi.start()
    
    # 'aut_states_list' is a list of lists of automaton states. Transitioning
    # from one 'aut_states' to the next should correspond to changing
    # automata in the receding horizon case.
    active_nodes = {}
    iteration = 1
    for (i, aut_states) in enumerate(aut_states_list):
        for state in aut_states:
            gs.changeNode(i, state, {'is_active': iteration})
            active_nodes[state] = iteration  # useful for future extension?
            iteration += 1
            time.sleep(delay)
    
    # Close the graph streaming server and the Gephi thread.
    gs.close()
    print 'Close Gephi to exit.'
    gephi.join()
