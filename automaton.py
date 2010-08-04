#!/usr/bin/env python

""" automaton.py --- Automaton module

Nok Wongpiromsarn (nok@cds.caltech.edu)
August 3, 2010
"""

import re, copy
from errorprint import printWarning, printError
    

###################################################################

class AutomatonState:
    """AutomatonState class with following fields:
        - id: an integer specifying the id of this state
        - state: a dictionary whose keys are the names of the variables
          and whose values are the values of the variables
        - transition: a list of id of the automaton state to which this state can transition
    """
    def __init__(self, id=-1, state={}, transition=[]):
        self.id = id
        self.state = copy.copy(state)
        self.transition = transition[:]


###################################################################

class Automaton:
    """Automaton class with the following fields
        - states: a list of AutomatonState
    """
    def __init__(self, states=[]):
        self.states = copy.deepcopy(states)

    def size(self):
        return len(self.states)

    def addAutState(self, aut_state):
        """Add an AutomatonState to this automaton

        - state is an AutomatonState to be added to this automaton
        Note that only a copy of state will be added so the automaton state stored in this
        automaton will not changed even if state is changed later after calling this function
        """
        if (isinstance(aut_state, AutomatonState)):
            self.states.append(copy.deepcopy(aut_state))
        else:
            printError("Input to addAutState must be of type AutomatonState")

    def getAutState(self, aut_state_id):
        """ Return an AutomatonState whose id is aut_state_id (int). Return -1 if such AutomatonState does not exist.
        """
        aut_state_index = self.size() - 1
        while (aut_state_index >= 0 and aut_state_id != self.states[aut_state_index].id):
            aut_state_index -= 1
        if (aut_state_index >= 0):
            return self.states[aut_state_index]
        else:
            return -1
    
    def setAutStateState(self, aut_state_id, aut_state_state, verbose=0):
        """ Set the state of the automaton state whose id is aut_state_id to aut_state_state.
            If such automaton state does not exist, an AutomatonState whose id is aut_state_id
            and whose state is aut_state_state will be added.

        - aut_state_id is an integer that specifies the id of AutomatonState to be set
        - aut_state_state is a dictionary that represents the new state of AutomatonState
        """
        aut_state = self.getAutState(aut_state_id)
        if (isinstance(aut_state, AutomatonState)):
            aut_state.state = aut_state_state
            if (verbose > 0):
                print 'Setting state of AutomatonState ' + str(aut_state_id) + ': ' + str(aut_state_state)
        else:
            self.addAutState(AutomatonState(id=aut_state_id, state=aut_state_state, transition=[]))
            if (verbose > 0):
                print 'Adding state ' + str(aut_state_id) + ': ' + str(aut_state_state)
    
    def setAutStateTransition(self, aut_state_id, aut_state_transition, verbose=0):
        """ Set the transition of the automaton state whose id is aut_state_id to aut_state_transition.
            If such automaton state does not exist, an AutomatonState whose id is aut_state_id
            and whose transition is aut_state_transition will be added.

        - aut_state_id is an integer that specifies the id of AutomatonState to be set
        - aut_state_transition is a list of id of the automaton state to which this state can transition
        """
        aut_state = self.getAutState(aut_state_id)
        if (isinstance(aut_state, AutomatonState)):
            aut_state.transition = aut_state_transition
            if (verbose > 0):
                print 'Setting transition of AutomatonState ' + str(aut_state_id) + ': ' + str(aut_state_transition)
        else:
            self.addAutState(AutomatonState(id=aut_state_id, state={}, transition=aut_state_transition))
            if (verbose > 0):
                print 'Adding AutomatonState ' + str(aut_state_id) + ' with transition ' + str(aut_state_transition)

    def findAllAutState(self, state):
        """Return all the AutomatonState in this automaton whose state matches state, return -1 if state is not found.

        - state is a dictionary whose keys are the names of the variables
          and whose values are the values of the variables 
        """
        all_aut_states = []
        for aut_state in self.states:
            if (aut_state.state == state):
                all_aut_states.append(aut_state)
        return all_aut_states

    def findAutState(self, state):
        """Return the first AutomatonState in this automaton whose state matches state, return -1 if state is not found.

        - state is a dictionary whose keys are the names of the variables
          and whose values are the values of the variables 
          """
        for aut_state in self.states:
            if (aut_state.state == state):
                return aut_state
        return -1

    def findNextAutState(self, current_aut_state, env_state):
        """Return the next AutomatonState based on env_state, return -1 if such AutomatonState is not found

        - current_aut_state is the current AutomatonState
        - env_state is a dictionary whose keys are the names of the environment variables
          and whose values are the values of the variables
          """
        for next_aut_state_id in current_aut_state.transition:
            is_env = True
            for var in env_state.keys():
                if (self.states[next_aut_state_id].state[var] != env_state[var]):
                    is_env = False
            if (is_env):
                return self.states[next_aut_state_id]
        return -1


###################################################################

def createAut(aut_file, varnames=[], verbose=0):
    """Construct an automation from aut_file.

    - aut_file is the name of the text file containing the automaton.
    - varnames is a list of all the variable names. If it is not empty, then this function will
      also check whether the variables in aut_file are in varnames.
    """
    automaton = Automaton(states=[])
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
                        printWarning('WARNING: Unknown variable ' + var)
            if (len(state.keys()) < len(varnames)):
                for var in varnames:
                    var_found = False
                    for var2 in state.keys():
                        if (var == var2):
                            var_found = True
                    if (not var_found):
                        printWarning('WARNING: Variable ' + var + ' not assigned')
            automaton.setAutStateState(stateID, state, verbose)

        # parse transitions
        if (line.find('successors') >= 0):
            transition = re.findall(' (\d+)', line)
            for i in range(0,len(transition)):
                transition[i] = int(transition[i])
            automaton.setAutStateTransition(stateID, list(set(transition)), verbose)

    return automaton



###################################################################

# Test case
if __name__ == "__main__":
    print('Testing createAut')
    env_vars = {'park' : 'boolean', 'cellID' : '{0,1,2,3}'}
    disc_sys_vars = {'gear' : '{-1,0,1}'}
    newvarname = 'ccellID'
    varnames = env_vars.keys() + disc_sys_vars.keys() + [newvarname]
    aut = createAut(aut_file='specs/test.aut', varnames=varnames, verbose=1)
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

