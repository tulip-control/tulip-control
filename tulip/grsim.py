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
"""

import time
import random
from multiprocessing import Process
from subprocess import call

from automaton import Automaton, AutomatonState
from errorprint import printWarning, printError
from congexf import dumpGexf, changeGexfAttvalue
from gephistream import GephiStream

activeID = 'is_active'

def grsim(aut_list, aut_trans_dict={}, env_states=[{}], num_it=20,
          deterministic_env=True, graph_vis=False, destfile="sim_graph.gexf",
          label_vars=None, delay=2, vis_depth=3):
    """
    Simulate an execution of the given list of automata and return a sequence
    of automaton states. If graph_vis is True, simulate the automata live in
    Gephi.
    
    For the simplest use case, try something like:
        aut_states_list = grsim([aut], env_states=[{state: var...}])
    
    Note that 'aut_states_list' is composed of (autID, autState) tuples,
    'aut' is enclosed in a list, and 'env_states' is a list of state
    dictionaries.
    
    Arguments:

    - `aut_list` -- a list of Automaton objects containing the automata
        generated from jtlvint.synthesize or jtlv.computeStrategy function.
    - `aut_trans_dict` -- a dictionary in which the keys correspond to exit states
        of automata and the values are the entry states of the next automata.
        Both keys and values are tuples in the following format:
            (AutomatonID, AutomatonState)
        where 'AutomatonID' is an integer corresponding to the
        index of the current automaton and 'AutomatonState'
        is the current automaton state.
    - `env_states` -- a list of dictionaries of environment state, specifying
        the sequence of environment states. If the length of this sequence
        is less than `num_it`, then this function will automatically pick
        the environment states for the rest of the execution.
    - `num_it` -- the number of iterations.
    - `deterministic_env` -- specify whether this function will choose
        the environment state deterministically.
    - `graph_vis` -- specify whether to visualize the simulation in Gephi.
    - `destfile` -- for graph visualization, the string name of the desired
        destination '.gexf' file.
    - `label_vars` -- for graph visualization, a list of the names of the system
        or environment variables to be encoded as labels.
    - `delay` -- for graph visualization, the time between simulation steps.
    - `vis_depth` -- for graph visualization, set the number of previous states to
        continue displaying.
        
    Return:
    List of automaton states, corresponding to a sequence of simulated
    transitions. Each entry will be formatted as follows:
        (AutomatonID, AutomatonState)
    as described above under 'aut_trans_dict'.
    """
    if not (isinstance(aut_list, list) and len(aut_list) > 0 and
            isinstance(aut_trans_dict, dict) and
            isinstance(env_states, list) and len(env_states) > 0 and
            isinstance(num_it, int) and isinstance(deterministic_env, bool) and
            isinstance(graph_vis, bool) and isinstance(destfile, str) and
            (label_vars == None or isinstance(label_vars, list)) and
            isinstance(delay, int) and isinstance(vis_depth, int)):
        raise TypeError("Invalid arguments to grsim")
    
    # aut_states will hold the list of automaton states traversed.
    aut_states = []
    # Start at the first automaton.
    aut_id = 0
    aut = aut_list[aut_id]
    # Set 'aut_state' to 'None' to find a valid initial state in
    # 'findNextAutState'.
    aut_state = None
    
    if graph_vis:
        visualizer = visualizeGraph(aut_list, destfile, label_vars=label_vars)
        active_nodes = []
        
    
    # Simulate automata by stepping through the sequence of environment
    # states. If there are no more available, randomly choose a valid
    # transition.
    for i in range(num_it):
        # Set current env_state, if available.
        if i < len(env_states):
            env_state = env_states[i]
        else:
            env_state = {}
        
        # Find an automaton state that satisfies 'env_state'. Note that
        # 'findNextAutState' returns -1 if no state is possible, and that
        # a 'current_aut_state' of None means to choose an initial state.
        aut_state = aut.findNextAutState(current_aut_state=aut_state,
                                         env_state=env_state,
                                         deterministic_env=deterministic_env)
        if aut_state == -1:
            printError('The specified sequence of environment states ' + \
                       'does not satisfy the environment assumption.')
            return aut_states
        
        if graph_vis:
            # Update active_nodes list.
            active_nodes.append((aut_id, aut_state))
            # Stream updated active_nodes.
            visualizer.update(active_nodes)
            if len(active_nodes) > vis_depth:
                del active_nodes[0]
            time.sleep(delay)
        aut_states.append((aut_id, aut_state))
        
        # Transition to next automaton? If yes, change 'aut' and 'aut_state'
        # to the corresponding ones in the next automaton.
        if (aut_id, aut_state) in aut_trans_dict.keys():
            new_state_tuple = aut_trans_dict[(aut_id, aut_state)]
            aut_id = new_state_tuple[0]
            aut = aut_list[aut_id]
            aut_state = new_state_tuple[1]
            
            if graph_vis:
                # Update active_nodes list.
                active_nodes.append((aut_id, aut_state))
                # Stream updated active_nodes.
                visualizer.update(active_nodes)
                if len(active_nodes) > vis_depth:
                    del active_nodes[0]
                time.sleep(delay)
            aut_states.append((aut_id, aut_state))
    if graph_vis:
        visualizer.close()
    return aut_states


###################################################################
def writeSimStatesToFile(states, file, verbose=0):
    """
    Write a simulation trace (sequence of states) to a text file. 

    Arguments:

    - `states` -- a list of tuples of automaton states, formatted as:
            (AutomatonID, AutomatonState)
        where 'AutomatonID' is an integer corresponding to the
        index of the current automaton and 'AutomatonState'
        is the current automaton state.
    - `file` -- the string name of the desired destination file.
    
    Return:
    (nothing)
    """
    f = open(file, 'w')
    if (verbose > 0):
        print 'Writing simulation result to ' + file
    for state in states:
        for var in state[1].state.keys():
            f.write(var + ':')
            f.write(str(state[1].state[var]) + ', ')
        f.write('\n')
    f.close()


###################################################################
def writeStatesToFile(aut_list, destfile, aut_states_list=[], label_vars=None):
    """
    Write the states and transitions from a list of automata to a '.gexf'
    graph file. If a list of simulated states is given, record the
    sequence of traversed states.

    Arguments:

    - `aut_list` -- a list of Automaton objects.
    - `destfile` -- the string name of the desired destination file.
    - `aut_states_list` -- a list of tuples of automaton states, formatted as:
            (AutomatonID, AutomatonState)
        where 'AutomatonID' is an integer corresponding to the
        index of the current automaton and 'AutomatonState'
        is the current automaton state.
    - `label_vars` -- a list of the names of the system or environment
        variables to be encoded as labels.
    
    Return:
    (nothing)
    """
    if not (isinstance(aut_list, list) and isinstance(destfile, str) and
            isinstance(aut_states_list, list) and
            (label_vars == None or isinstance(label_vars, list))):
        raise TypeError("Invalid arguments to writeStatesToFile")

    # Generate a Gexf-formatted string of automata.
    output = dumpGexf(aut_list, label_vars=label_vars)
    
    # 'aut_states_list' is a list of automaton/automaton state tuples.
    # Transitioning from one tuple to the next should correspond to changing
    # automaton states in the receding horizon case.
    iteration = 1
    for state_tuple in aut_states_list:
        output = changeGexfAttvalue(output, activeID, iteration,
                                    node_id=str(state_tuple[0]) + '.' +
                                            str(state_tuple[1]))
        iteration += 1
    print "Writing graph states to " + destfile
    f = open(destfile, 'w')
    f.write(output)
    f.close()



class visualizeGraph:
    """
    Open Gephi (a graph visualization application) and stream
    a live automaton simulation to it.

    Arguments:

    - `aut_list` -- a list of Automaton objects.
    - `destfile` -- the string name of a '.gexf' graph file to be opened
        in Gephi.
    - `label_vars` -- a list of the names of the system or environment
        variables to be encoded as labels. 
    
    Fields:

    - `gs` -- a Gephi streaming server.
    - `gephi` -- an asynchronous process running Gephi.
    """
    def __init__(self, aut_list, destfile, label_vars=None):
        # First write the automata to file and open them in Gephi.
        writeStatesToFile(aut_list, destfile, label_vars=label_vars)
        
        # Changes to the graph will be streamed from this server.
        self.gs = GephiStream('server')
        
        # Open Gephi in a separate thread.
        print "Opening " + destfile + " in Gephi."
        self.gephi = Process(target=lambda: call(["gephi", destfile]))
        self.gephi.start()
        
        # Wait for user before streaming simulation.
        raw_input("When Gephi has loaded, press 'return' or 'enter'\n" + \
                  "to start streaming the automaton simulation.\n")
    
    def close(self):
        # Close the graph streaming server and the Gephi thread.
        self.gs.close()
        print 'Close Gephi to exit.'
        self.gephi.join()
    
    def update(self, active_nodes):
        """Update the graph by streaming the changed active nodes.
        
        Arguments:

        - `active_nodes` -- an ordered list of nodes that should be active.
        
        Return:
        (nothing)
        """
        for (i, active_node) in enumerate(active_nodes):
            self.gs.changeNode(active_node[0], active_node[1],
                               {activeID: i})
