'''
Created on Jul 22, 2013

@author: ewolff
'''

def set_costs(TS):
    #Define costs
    costs = dict()
    for s in TS.states.itervalues():
        for act in s.actions.iterkeys():
            for t in s.actions[act]:
                costs[(s,act,t)] = 1#random.randint(0,2)
    costs = None