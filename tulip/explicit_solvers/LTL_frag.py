'''
Created on Aug 13, 2012

@author: ewolff
'''
import numpy as np
import time
import ex_graph as gr
import random
import itertools
import networkx as nx

######################################################################
## Start of main function
######################################################################
ti = time.time()

## INPUT:  TS, spec, prop2states, costs



TS_orig = TS.copy()
print(spec)

print("Original TS " +str(len(TS) ) ) #, TS

## Remove all actions that don't satisfy [](p --> X q) specs.
t0 = time.time()
TS = gr.remove_unsafe_actions(TS,spec['G_act'],prop2states)
t1 = time.time()
print("Safe actions " +str(len(TS) ) )#, TS
print(str(t1-t0) +"(sec)")

## Determine all nodes that are blocking, i.e., don't have any valid actions left
t0 = time.time()
blockingNodes = gr.blocking_nodes(TS)
t1 = time.time()
print("blockingNodes: " +str(len(blockingNodes) ) )
print(str(t1-t0) +" (sec)")


## Remove all states that don't satisfy safety [] specs (or must visit a state that doesn't)
if spec['G'] != '':
    t0 = time.time()
    unsafeNodes = gr.must_reach_target(TS, set(TS.states.values()) - prop2states[spec['G']])
    t1 = time.time()
    print("must_reachTarget " +str(t1-t0) +" (sec)")
    print("unsafeNodes: " +str(len(unsafeNodes) ) )
    unsafeNodes.update(blockingNodes)   # add the blocking nodes to those that need to be removed
    t0 = time.time()
    TS = gr.subgraph(TS,unsafeNodes)
    t1 = time.time()
    print("subgraph: " +str(t1-t0) +" (sec)")
    print("Safe state " +str(len(TS) ) )#, TS
    
## At this point, I have incorporated the [] and [](a -- b) safety properties.  Check to see if subgraph is non-empty.
if len(TS) == 0:
    print("The specification is invalid." +
        "The [] and/or [](p --> X q) specs are too constraining.")
    quit()


## Further restrict to the subgraph where the <>[] persistence property holds
TS_per = TS.copy()
if spec['FG'] != '':
    TS_per = gr.subgraph(TS_per, set(TS.states.values()) - prop2states[spec['FG']])
    print("Persistence " +str(len(TS_per) ) )#, TS_per


if spec['GF'] != '':
    ## Create paths between liveness states on TS_per subgraph
    reachF = dict()     #forced reach sets for accepting state sets F
    F = dict()          #sets of accepting states
    
    # Determine the states associated with each acceptance condition
    taskInd = range(len(spec['GF']))
    for i in taskInd:
        f = spec['GF'][i]
        F[i] = prop2states[f].copy()
        assert len(F[i]) != 0
        assert isinstance(F[i],set)
    
    # Compute generalized Buchi winning sets
    while True:
        print("working...")
        for i in taskInd[:-1]:
            t0 = time.time()
            if costs is None:
                reachF[i] = gr.reach_value(TS_per, F[i]) 
            else:
                reachF[i] = gr.opt_reach_value(TS_per, F[i], costs)
            t1 = time.time()
            print("reachF[" +str(i) +"] computed in " +
                str(t1-t0) +" sec.")
            F[i+1] = F[i+1] & set(reachF[i].keys())
            assert isinstance(F[i+1],set)
            if len(F[i+1]) == 0:
                print("The specification is invalid." +
                    "The <>[] and/or []<> specs are too constraining.")
                quit()

        # Compute reachValue for last task
        first = taskInd[0]
        last = taskInd[-1]
        t0 = time.time()
        if costs is None:
            reachF[last] = gr.reach_value(TS_per, F[last])
        else:
            reachF[last] = gr.opt_reach_value(TS_per, F[i], costs)
        t1 = time.time()
        print("reachF[" +str(last) +"] computed in " +str(t1-t0) +"sec.")
        
        F_new = F[first] & set(reachF[last].keys())
        
        if F[first] == F_new:
            break   # We have found a feasible control policy
        F[first] = F_new
        
    # Compute the task graph between generalized winning sets F[j] from above
    F_dict = dict()
    for i in taskInd:
        F_dict[i] = []
        F_dict[i].append(F[i])
        
    G_task, node2states, node2task, task2node = gr.task_graph(TS_per, F_dict, costs)
    print(task2node)
    
#    # Determine an optimal ordering and subsets of task sets F*[j]
#     1. Average cost per task cycle.  Here, I output the task graph and attempt to solve 
#     it using a TSP solver.  Branch and bound techniques work nicely here.
#    tcVal, tcOrder, tcNodeOrder = opt.task_cycle(G_task,taskInd,task2node,weight='weight')
#    print("Task cycle optimal value: " +str(tcVal) +
#        ", order: " +str(tcOrder) +", and node order: " +str(tcNodeOrder) )

#    # 2. Bottleneck cost between tasks
#    botVal,botStates = opt.bottleneck(G_task,taskInd,task2node)
#    print("Bottleneck optimal value: " +str(botVal) +
#        "and states:" +str(botStates) )

else:
    print("TODO")


tf = time.time()
print("Total time (sec) = " +str(tf-ti) )
print("NEED TO UPDATE THE PARENTS DIRECTORY WHEN DELETING ACTIONS " +
    "WHILE CREATING SUBGRAPHS AND SUCH.")
