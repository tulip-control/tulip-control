#!/usr/bin/env python
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
----------------
Automaton Module
----------------

Nok Wongpiromsarn (nok@cds.caltech.edu)

:Date: August 3, 2010
:Version: 0.1.0
"""

import re, copy, os
from errorprint import printWarning, printError
    

###################################################################

class AutomatonState:
    """
    AutomatonState class for representing a state in a finite state automaton.
    An AutomatonState object contains the following fields:

    - `id`: an integer specifying the id of this AutomatonState object.
    - `state`: a dictionary whose keys are the names of the variables
      and whose values are the values of the variables.
    - `transition`: a list of id's of the AutomatonState objects to which this 
      AutomatonState object can transition.
    """
    def __init__(self, id=-1, state={}, transition=[]):
        self.id = id
        self.state = copy.copy(state)
        self.transition = transition[:]


###################################################################

class Automaton:
    """
    Automaton class for representing a finite state automaton.
    An Automaton object contains the following field:

    - `states`: a list of AutomatonState objects.

    Automaton([states_or_file, varname, verbose]) constructs an Automaton object based
    on the following input:

    - `states_or_file`: a string containing the name of the aut file to be loaded or
      a list of AutomatonState objects to be assigned to the `states` of this 
      Automaton object.
    - `varname`: a list of all the variable names. If it is not empty and 
      states_or_file is a string representing the name of the aut file to be loaded, 
      then this function will also check whether the variables in aut_file are in 
      varnames.
    """
    def __init__(self, states_or_file=[], varnames=[], verbose=0):
        # Construct this automaton from a list of AutomatonState objects
        if (isinstance(states_or_file, list)): 
            self.states = copy.deepcopy(states_or_file)
        # Construct this automaton from file
        elif (isinstance(states_or_file, str)):
            if (len(states_or_file) == 0):
                self.states = []
            else:
                self.loadFile(states_or_file, varnames=varnames, verbose=verbose)
    
    def loadFile(self, aut_file, varnames=[], verbose=0):
        """
        Construct an automation from aut_file.

        Input:

        - `aut_file`: the name of the text file containing the automaton.
        - `varnames`: a list of all the variable names. If it is not empty, then this 
          function will also check whether the variables in aut_file are in varnames.
        """
        self.states = []
        f = open(aut_file, 'r')
        stateID = -1
        for line in f:
            # parse states
            if (line.find('State ') >= 0):
                stateID = re.search('State (\d+)', line)
                stateID = int(stateID.group(1))
                state = dict(re.findall('(\w+):([-+]?\d+)', line))
                for var, val in state.iteritems():
                    state[var] = int(val)
                    if (len(varnames) > 0):
                        var_found = False
                        for var2 in varnames:
                            if (var == var2):
                                var_found = True
                        if (not var_found):
                            printWarning('Unknown variable ' + var, obj=self)
                if (len(state.keys()) < len(varnames)):
                    for var in varnames:
                        var_found = False
                        for var2 in state.keys():
                            if (var == var2):
                                var_found = True
                        if (not var_found):
                            printWarning('Variable ' + var + ' not assigned', obj=self)
                self.setAutStateState(stateID, state, verbose)

            # parse transitions
            if (line.find('successors') >= 0):
                transition = re.findall(' (\d+)', line)
                for i in xrange(0,len(transition)):
                    transition[i] = int(transition[i])
                self.setAutStateTransition(stateID, list(set(transition)), verbose)

    def size(self):
        """
        Return the number of states in this Automaton object.
        """
        return len(self.states)

    def addAutState(self, aut_state):
        """
        Add an AutomatonState object to this automaton.
        
        Input:

        - `aut_state`: an AutomatonState object to be added to this Automaton object.
        """
        if (isinstance(aut_state, AutomatonState)):
            self.states.append(aut_state)
        else:
            printError("Input to addAutState must be of type AutomatonState", obj=self)

    def getAutState(self, aut_state_id):
        """
        Return an AutomatonState object stored in this Automaton object 
        whose id is `aut_state_id`. 
        Return -1 if such AutomatonState object does not exist.

        Input:

        - `aut_state_id`: an integer specifying the id of the AutomatonState object
          to be returned by this function.
        """
        if (aut_state_id < self.size() and self.states[aut_state_id].id == aut_state_id):
            return self.states[aut_state_id]
        else:
            aut_state_index = self.size() - 1
            while (aut_state_index >= 0 and aut_state_id != self.states[aut_state_index].id):
                aut_state_index -= 1
            if (aut_state_index >= 0):
                return self.states[aut_state_index]
            else:
                return -1
    
    def setAutStateState(self, aut_state_id, aut_state_state, verbose=0):
        """ 
        Set the state of the AutomatonState object whose id is `aut_state_id` to 
        `aut_state_state`.
        If such an AutomatoSstate object does not exist, an AutomatonState object 
        whose id is `aut_state_id` and whose state is `aut_state_state` will be added.

        Input:

        - `aut_state_id`: an integer that specifies the id of the AutomatonState 
          object to be set.
        - `aut_state_state`: a dictionary that represents the new state of the 
          AutomatonState object.
        """
        aut_state = self.getAutState(aut_state_id)
        if (isinstance(aut_state, AutomatonState)):
            aut_state.state = aut_state_state
            if (verbose > 0):
                print 'Setting state of AutomatonState ' + str(aut_state_id) + \
                    ': ' + str(aut_state_state)
        else:
            self.addAutState(AutomatonState(id=aut_state_id, state=aut_state_state, \
                                                transition=[]))
            if (verbose > 0):
                print 'Adding state ' + str(aut_state_id) + ': ' + str(aut_state_state)
    
    def setAutStateTransition(self, aut_state_id, aut_state_transition, verbose=0):
        """
        Set the transition of the AutomatonState object whose id is `aut_state_id` to 
        `aut_state_transition`. If such automaton state does not exist, an 
        AutomatonState whose id is aut_state_id and whose transition is 
        aut_state_transition will be added.

        Input:

        - `aut_state_id`: an integer that specifies the id of the AutomatonState 
          object to be set.
        - `aut_state_transition`: a list of id's of the AutomatonState objects to which 
          the AutomatonState object with id `aut_state_id` can transition.
        """
        aut_state = self.getAutState(aut_state_id)
        if (isinstance(aut_state, AutomatonState)):
            aut_state.transition = aut_state_transition
            if (verbose > 0):
                print 'Setting transition of AutomatonState ' + str(aut_state_id) + \
                    ': ' + str(aut_state_transition)
        else:
            self.addAutState(AutomatonState(id=aut_state_id, state={}, \
                                                transition=aut_state_transition))
            if (verbose > 0):
                print 'Adding AutomatonState ' + str(aut_state_id) + \
                    ' with transition ' + str(aut_state_transition)

    def findAllAutState(self, state):
        """
        Return all the AutomatonState objects stored in this automaton whose state 
        matches `state`.
        Return -1 if such an AutomatonState objects is not found.

        Input:

        - `state`: a dictionary whose keys are the names of the variables
          and whose values are the values of the variables.
        """
        all_aut_states = []
        for aut_state in self.states:
            if (aut_state.state == state):
                all_aut_states.append(aut_state)
        return all_aut_states

    def findAutState(self, state):
        """
        Return the first AutomatonState object stored in this automaton whose state 
        matches `state`.
        Return -1 if such an AutomatonState objects is not found.

        Input:

        - `state`: a dictionary whose keys are the names of the variables
          and whose values are the values of the variables.
          """
        for aut_state in self.states:
            if (aut_state.state == state):
                return aut_state
        return -1

    def findNextAutState(self, current_aut_state, env_state):
        """
        Return the next AutomatonState object based on `env_state`.
        Return -1 if such an AutomatonState object is not found.

        Input:

        - `current_aut_state`: the current AutomatonState. Use current_aut_state = None
          for unknown current or initial automaton state.
        - `env_state`: a dictionary whose keys are the names of the environment 
          variables and whose values are the values of the variables.
        """
        transition = []
        if (current_aut_state is None):
            transition = range(0, self.size())
        else:
            transition = current_aut_state.transition
        for next_aut_state_id in transition:
            is_env = True
            for var in env_state.keys():
                if (self.states[next_aut_state_id].state[var] != env_state[var]):
                    is_env = False
            if (is_env):
                return self.states[next_aut_state_id]
        return -1


###################################################################

def createAut(aut_file, varnames=[], verbose=0):
    """
    Construct an automation from aut_file.

    Input:

    - `aut_file`: the name of the text file containing the automaton.
    - `varnames`: a list of all the variable names. If it is not empty, then this 
      function will also check whether the variables in aut_file are in varnames.
    """
    automaton = Automaton(states_or_file=aut_file, varnames=[], verbose=verbose)
    return automaton



###################################################################

# Test case
if __name__ == "__main__":
    print('Testing createAut')
    env_vars = {'park' : 'boolean', 'cellID' : '{0,1,2,3}'}
    disc_sys_vars = {'gear' : '{-1,0,1}'}
    newvarname = 'ccellID'
    varnames = env_vars.keys() + disc_sys_vars.keys() + [newvarname]
    aut = Automaton(states_or_file=os.path.join('tmpspec', 'testjtlvint.aut'), varnames=varnames, verbose=1)
    print('DONE')
    print('================================\n')

    ####################################

    print('Testing findAllAutState')
    state = {'park':0, 'cellID':1, 'ccellID':0, 'gear':1}
    all_aut_states = aut.findAllAutState(state)
    for aut_state in all_aut_states:
        print aut_state.id
    print('DONE')
    print('================================\n')

    ####################################

    print('Testing findAutState')
    aut_state = aut.findAutState(state)
    print aut_state.id
    print('DONE')
    print('================================\n')

    ####################################

    print('Testing findNextAutState')
    next_aut_state = aut.findNextAutState(aut.states[7], {'park':0, 'cellID':0})
    print next_aut_state.id
    print('DONE')
    print('================================\n')

