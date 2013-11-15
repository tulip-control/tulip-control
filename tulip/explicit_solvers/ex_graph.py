"""
Eric Wolff
"""

import numpy as np
import time
import heapq
import itertools
import networkx as nx

class state:
    """state class for representing a state in a finite state
    automaton.  A gameState object contains the following
    fields:

    - `id`: an integer specifying the id of this state object.
    - `locs`: a dictionary whose keys are the names of the robot 'rob' and obstacles
        and whose values are the (x,y) coordinates.
    - `actions`: a dictionary whose keys are available actions and values are sets
        of states that can be transitioned to.
    - 'labels': a set of atomic propositions that are true
    - 'parents': the set of states that have a transition to the current state
    """
    def __init__(self, id=-1, locs = dict(), actions = dict(), labels = list(), parents = set()):
        self.id = id
        self.locs = locs    #(x,y) coordinates
        self.actions = actions
        self.labels = labels
        self.parents = parents
        
    def __copy__(self):
        return state(self.id, self.locs, self.actions, self.labels, self.parents)
        
    def copy(self):
        """Return copy of this game graph state."""
        return self.__copy__()
    
    def __str__(self):        
        tmp = dict()
        for a in self.actions:
            S = []
            for s in self.actions[a]:
                S.append(s.id)
            tmp[a] = S

        return 'id:'+ str(self.id) + '; act:' + str(tmp)
    
    def addAct(self,act,val):
        currAct = self.actions.copy()
        if act in currAct:
            currAct[act].add(val)
        else:
            currAct[act] = {val}
        self.actions = currAct
        
    def removeAct(self,act):
        currAct = self.actions.copy()
        if act in currAct:
            del currAct[act]
        self.actions = currAct

    def addLoc(self,type,pos):
        currLocs = self.locs.copy()
        currLocs[type] = pos
        self.locs = currLocs
        
    def newParents(self,parents):
        self.parents = parents
        
    def newActs(self,actions):
        self.actions = actions


class gameGraph:
    """gameGraph class for representing a game graph composed of states.
    A gameGraph object contains the following field:

    - `states`: a dictionary of gameState objects keyed by id.
    """
    def __init__(self, states = [], init = [], shape = []):
        stateDict = dict()
        for s in states:
            stateDict[s.id] = s
        self.states = stateDict
        self.init = init
        self.shape = shape
    
    def __len__(self):
        """Return number of states."""
        return len(self.states)
    
    def __str__(self):
        """Prints out the states."""
        tmp = dict()
        for s_id,s in self.states.iteritems():
            tmp[s_id] = str(s)

        return str(tmp.values())
    
    def __copy__(self):
        return gameGraph(self.states.values(), self.init, self.shape)
    
    def copy(self):
        """Return copy of gameGraph."""
        return self.__copy__()
    
    # TODO : MODIFY THIS FUNCTION TO UPDATE THE PARENTS
    def removeState(self,state):
        # Update the parents list for states that are children of state
        for s_set in state.actions.itervalues():
            for s in s_set:
                parents = s.parents.copy()
                parents.remove(state)
                s.newParents(parents)
                
        # Update the actions of the parents of current state
        
        # TODO
        
        states = self.states.copy()
        del states[state.id]
        self.states = states


def productGraph(T_rob,loc2state_rob, T_obs,loc2state_obs):
    """Return the product graph of T_rob and T_obs"""
    assert isinstance(T_rob,gameGraph)
    assert isinstance(T_obs,gameGraph)
    
    trans2product = dict()    # Maps (rob,obs) locations to all corresponding product states
    
    S = []          # list of product states
    s_init = []     # initial product state
    num = -1
    t0 = time.time()
    for s_rob in T_rob.states.itervalues():
        for s_obs in T_obs.states.itervalues():
            num += 1
            
            # Product state includes locations from each individual state
            prod_locs = dict()
            prod_locs['rob'] = s_rob.locs['rob']
            prod_locs['obs'] = s_obs.locs['obs']
            
            # Create new product state
            s =state(id=num, locs = prod_locs, actions = {}, labels = [], parents = {})
            S.append(s)
            
            # Determine if new product state is an initial state
            if s_rob == T_rob.init and s_obs == T_obs.init:
                s_init = s
            
            # Update map from individual gameGraphs to the product
            rob_loc = s_rob.locs['rob']
            obs_loc = s_obs.locs['obs']
            trans_loc = (rob_loc,obs_loc)
            
            if trans_loc not in trans2product:
                trans2product[trans_loc] = {s}
            else:
                trans2product[trans_loc].add(s)
                
    t1 = time.time()
    print("prod states " +str(t1-t0) +" sec")
                
    # Create new actions between product states
    t0 = time.time()
    for s in S:
        prod_act = dict()
        
        # Determine robot state and all neighboring robot and obstacle states.  
        # Then determine all product states that are consistent with the neighboring states
        s_rob = loc2state_rob[s.locs['rob']]
        s_obs = loc2state_obs[s.locs['obs']]
        act_num = -1
        for t_rob_set in s_rob.actions.itervalues():
            act_num += 1
            nextStates = set()
            
            for t_rob in t_rob_set:
                for t_obs_set in s_obs.actions.itervalues():
                    for t_obs in t_obs_set:
                        nextStates.update(trans2product[(t_rob.locs['rob'],t_obs.locs['obs'])])
                
            prod_act[act_num] = nextStates
        s.newActs(prod_act)
    t1 = time.time()
    print("prod act " +str(t1-t0) +" sec")
        
    # Compute the parents of each node
    t0 = time.time()
    parents = dict()
    for s in S:
        parents[s] = set()
        
    for s in S:
        for s_set in s.actions.itervalues():
            for t in s_set:
                parents[t].add(s)

    t1 = time.time()
    print("prod parents " +str(t1-t0) +" sec")
    
    # Update the parents of each node
    for s in S:
        s.newParents(parents[s])
        
    return gameGraph(states = S, init = s_init, shape = T_rob.shape)


def reach_value(G,targetSet):
    # For each state, compute minimum number of steps to reach the targetSet no matter what the environment does.
    # reachVal is 0 for states in targetSet and finite for all states that can reach targetSet.
    assert isinstance(G,gameGraph)
    assert isinstance(targetSet,set)
    for s in targetSet:
        assert isinstance(s,state)
    
    activeStates = targetSet.copy()
    reachedStates = targetSet.copy()
    
    val = 0
    dist = dict()
    for t in targetSet:
        dist[t] = val
            
    while True:   #Expand backwards from the targetSet
        val += 1
        newStates = set()   # set of states that are found reachable at this iteration
        
        for s in activeStates:
            for par in s.parents:
                if par not in dist:
                    # Does there exist an action whose successors are all in targetSet?
                    for s_set in par.actions.itervalues():
                        if s_set <= reachedStates:  #if s_set.issubset(reachedStates):
                            dist[par] = val
                            newStates.add(par)
                            break
        
        if len(newStates) == 0:     # all reachable states have been found
            break
        
        reachedStates.update(newStates)   # update the set of reachable states
        activeStates = newStates

    for s in dist.iterkeys():
        assert np.isfinite(dist[s])
    return dist



def opt_reach_value(G,targetSet,costs):
    # For each state, compute minimum cost to reach the targetSet no matter what the environment does.
    # reachVal is 0 for states in targetSet and finite for all states that can reach targetSet.
    assert isinstance(G,gameGraph)
    assert isinstance(targetSet,set)
    for s in targetSet:
        assert isinstance(s,state)
    assert isinstance(costs,dict)
    
    dist = {}  # dictionary of final distances
    fringe=[] # use heapq with (distance,label) tuples
    seen = dict()
    for s in targetSet:
        seen[s] = 0
        heapq.heappush(fringe,(0,s))
    
#    count = 0 
    while fringe:
#        t_iter = time.time()
        
        (d,u) = heapq.heappop(fringe)
        if u in dist:
            continue # already searched this state
        dist[u] = d
        
        for s in u.parents:
            if s not in dist:
                # Compute minimum worst-case value of parents of u
                tmpSys = np.Inf
                for act in s.actions:
                    # Maximize over environment for each (state,action)
                    tmpEnv = -np.Inf
                    for t in s.actions[act]:
                        try:
                            tmp = dist[t] + costs[(s,act,t)]
                        except:
                            tmpEnv = np.Inf
                            break              
                        if tmp > tmpEnv:
                            tmpEnv = tmp
                    if tmpEnv < tmpSys:
                        tmpSys = tmpEnv
                
                if np.isfinite(tmpSys):
                    if s not in seen or tmpSys < seen[s]:
                        seen[s] = tmpSys
                        heapq.heappush(fringe,(tmpSys,s))
                    
#        count += 1
#        if np.mod(count,1000) == 0:
#            print(str(count) +str(len(G.states) ) )
#            print("iteration: " +str(time.time() - t_iter) +"\n")

    for s in dist.iterkeys():
        assert np.isfinite(dist[s])
    return dist


def task_graph(G,F_dict,costs):
    G_task = nx.DiGraph()
    # Let each F_vec[i] map to subsets of states that satisfy the ith task
    # For each subset, create a new state in G_task.
    node2states = dict()
    node2task = dict()
    task2node = dict()
    
    nodeID = 0
    for task in F_dict:
        task2node[task] = []
        for F_feas in F_dict[task]: # F_feas is a feasible subset of states that satisfies each task
#            print("len(F_feas): " +str(len(F_feas) ) )
            for r in xrange(len(F_feas),len(F_feas)+1):#xrange(len(F_feas)-1,len(F_feas)+1):
#                print("  r: " +str(r) )
                for s_set in itertools.combinations(F_feas,r):
                    G_task.add_node(nodeID)
                    node2states[nodeID] = set(s_set)
                    node2task[nodeID] = task
                    task2node[task].append(nodeID)
                    nodeID += 1

    # For each state in G_task, compute edges based on reachability
    for u in G_task.nodes_iter():
        print("Computing transitions into state " +str(u) +
            "of" +str(len(G_task.nodes()) ) )
        u_states = node2states[u]
        if costs is None:
            dist = reach_value(G,u_states) 
        else:
            dist = opt_reach_value(G,u_states,costs)
        
        for v in G_task.nodes_iter():
            if node2task[v] != node2task[u]:      # Do not add edges between states in same task!
                v_states = node2states[v]
                
                # Take minimum and maximum over all states
                minWeight = np.Inf
                maxWeight = -np.Inf
                for s in v_states:
                    if s not in dist:
                        maxWeight = np.Inf
                    elif maxWeight < dist[s]:
                        maxWeight = dist[s]

                    if s in dist and minWeight > dist[s]:
                        minWeight = dist[s]
                if np.isfinite(maxWeight):
                    G_task.add_edge(v, u, weight=maxWeight, minWeight=minWeight)
        
#    print("Task graph: " +str(G_task.nodes() ) )
#    print(str(G_task.edges(None,True) ) +'\n')
    return G_task, node2states, node2task, task2node


def can_pre(G,targetSet):
    # Compute all states s in G that CAN be forced by system reach a state in targetSet in one step.
    
    assert isinstance(G,gameGraph)
    assert isinstance(targetSet,set)
    for s in targetSet:
        assert isinstance(s,state)
    
    pre = set()
    for t in targetSet:
        assert isinstance(t,state)
        
        for s in t.parents:
            assert isinstance(s,state)
            # Does there exist an action whose successors are all in targetSet?
            for s_set in s.actions.itervalues():
                if s_set <= targetSet:  #if s_set.issubset(targetSet):
                    pre.add(s)
                    break
    return pre


def must_pre(G,targetSet):
    # Compute all states s in G where the system MUST reach a state in targetSet in one step.
    
    assert isinstance(G,gameGraph)
    assert isinstance(targetSet,set)
    for s in targetSet:
        assert isinstance(s,state)
        
    pre = set()
    for t in targetSet:
        assert isinstance(t,state)
        for s in t.parents:
            assert isinstance(s,state)
            
            mustReach = True
            for s_set in s.actions.itervalues():   # Are do all actions result in successors inside targetSet?
                if not (s_set <= targetSet):
                    mustReach = False
                    break
            if mustReach:
                pre.add(s)
    return pre


def must_reach_target(G,targetSet):
    # Compute the states that MUST reach the targetSet if the environment wants
    
    assert isinstance(G,gameGraph)
    assert isinstance(targetSet,set)
    for s in targetSet:
        assert isinstance(s,state)
    
    reachTarget = targetSet.copy()  #states that can reach the target set
    
    visited = set()
    queue = targetSet.copy()
    while len(queue) > 0:   #Expand backwards from the targetSet
        curr = queue.pop()
        assert isinstance(curr,state)
        reachTarget.add(curr)
        visited.add(curr)   #mark as visited so don't loop indefinitely
        
        for s in must_pre(G,reachTarget):
            assert isinstance(s,state)
            if s not in visited:   #don't add already visited nodes to queue
                reachTarget.add(s)
                queue.add(s)
                
    return reachTarget


def remove_unsafe_actions(G_old,safeActions,prop2states):
    ## Locate all actions that don't satisfy [](p --> X q) specs
    ## safeActions is a list of tuples [(p,q),(r,s),...]
    
    assert isinstance(G_old,gameGraph)
    
    G = G_old.copy()
    unsafeActions = set()
    
    for s in G.states.itervalues():
        assert isinstance(s,state)

        for a in s.actions.iterkeys():
            for p,q in safeActions:
                if (s in prop2states[p]) and not (s.actions[a] <= prop2states[q]): 
                    unsafeActions.add(a)    # if s satisfies p and there exists a next state that doesn't satisfy q
                    
        for a in unsafeActions:     # Remove the unsafe actions at current state
            del s.actions[a]
            # TODO  is this correctly deleting the action?
        unsafeActions = set()
        
    return G


def subgraph(G_old,removeStates):
    # Create the subgraph T of graph G by deleting removeStates
    assert isinstance(G_old,gameGraph)
    assert isinstance(removeStates,set)
    for s in removeStates:
        assert isinstance(s,state)
    
    G = G_old.copy()
    Q = removeStates.copy()
    visited = set()
    
    while len(Q) > 0:
        curr = Q.pop()
        visited.add(curr)
        
        # Remove actions that lead to the removed state from its parents
        if curr.parents != None:
            for par in curr.parents:
                assert isinstance(par,state)
    
                for a in par.actions.iterkeys():
                    if curr in par.actions[a]:
                        par.removeAct(a)
                
                # Add par to the queue if it doesn't have any actions left.
                if len(par.actions.keys()) == 0 and (par not in visited):
                    Q.add(par)
                    
        G.removeState(curr)
    return G


def blocking_nodes(G):
    # Determine nodes in G that have no outgoing transitions, i.e., blocking.
    blocking = set()
    
    for s in G.states.itervalues():
        assert isinstance(s,state)
        
        if len(s.actions.keys()) == 0:
            blocking.add(s)
    return blocking
