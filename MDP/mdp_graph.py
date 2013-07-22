#!/usr/bin/env python
"""
Helper functions for CDC 2012 example code.
Eric Wolff
"""
import time

def no_reach(MDP,targetSet):
    # Compute the no-reachability set, i.e., the states that cannot reach the target set.    
    allStates = set(MDP.keys())     #all states
    targetSet = set(targetSet)      #target set
    reach = targetSet.copy()        #states that can reach the target set
    
    visited = set()
    queue = set()
    queue.update(reach)    
    while len(queue) > 0:   #Expand backwards from the targetSet and add all reachable nodes
        curr = queue.pop()
        reach.add(curr)
        visited.add(curr)   #mark as visited so don't loop indefinitely
        
        for nbhr in pre(curr,MDP):
            if nbhr not in visited:   #don't add already visited nodes to queue
                reach.add(nbhr)
                queue.add(nbhr)

    return allStates - reach


def pre(s,MDP):
    # Compute all states s1 that can reach state s in one step.
    pre = set()
    for s1 in MDP.keys():
        for a in MDP[s1].keys():
            if s in MDP[s1][a].keys():
                pre.add(s1)
    return pre

#REWRITE
def get_graph(T,A_T,MDP):
    # Create the induced digraph G_(T,A|T) from a given sub-MDP (T,A|T).
    #T is the set of states of the sub-MDP and A_T is the set of associated actions.
    #Output: A directed graph representation of (T,A_T)
    
    G = {}
    for s in T:
        neighbors = set()
        for a in A_T[s]:
            neighbors.update(MDP[s][a].keys())
        G[s] = neighbors
    return G


def get_sccs(G):
    # Compute strongly connected components (SCCs) of a given directed graph, G.
    preorder={}
    lowlink={}    
    scc_found={}
    scc_queue = []
    scc_list=[]
    i=0     # Preorder counter
    for source in G:
        if source not in scc_found:
            queue=[source]
            while queue:
                v=queue[-1]
                if v not in preorder:
                    i=i+1
                    preorder[v]=i
                done=1
                v_nbrs=G[v]
                for w in v_nbrs:
                    if w not in preorder:
                        queue.append(w)
                        done=0
                        break
                if done==1:
                    lowlink[v]=preorder[v]
                    for w in v_nbrs:
                        if w not in scc_found:
                            if preorder[w]>preorder[v]:
                                lowlink[v]=min([lowlink[v],lowlink[w]])
                            else:
                                lowlink[v]=min([lowlink[v],preorder[w]])
                    queue.pop()
                    if lowlink[v]==preorder[v]:
                        scc_found[v]=True
                        scc=[v]
                        while scc_queue and preorder[scc_queue[-1]]>preorder[v]:
                            k=scc_queue.pop()
                            scc_found[k]=True
                            scc.append(k)
                        scc_list.append(scc)
                    else:
                        scc_queue.append(v)
    scc_list.sort(key=len,reverse=True)            
    return scc_list

#REWRITE
def max_end_components(MDP):
    # Compute all maximal end components in MDP

    #Initialize
    A = {}
    for s in MDP.keys():
        A[s] = MDP[s].keys()[:]
    
    MEC = set([])
    MEC_new = [set(MDP.keys()[:])]    #all states
    
    # Loop until fixed-point
    #print '\nComputing maximal end components...'
    while MEC != MEC_new:
        #print 'New iteration...'
        MEC = MEC_new[:]
        MEC_new = []
        
        for T in MEC:
            A_T = {t: A[t] for t in T}
            R = set()                   #set of states to be removed
            G = get_graph(T,A_T,MDP)     #get digraph G_(T,A_T)
            
            SCCs = get_sccs(G)
            for i in SCCs:      
                if len(i) == 1 and len(A[i[0]]) == 0:  #remove trivial SCCs
                    #print 'removing a trivial SCC'
                    SCCs.remove(i)
    
            for T_i in SCCs:
                for s in T_i:       
                    # Remove actions that lead out of T_i
                    keep = []
                    for a in A[s]:
                        if set(MDP[s][a].keys()).issubset(set(T_i)):
                            keep.append(a)
                        else:
                            pass
                            #print 'removing action a'
                    A[s] = keep
                    if len(A[s]) == 0:
                        #print 'removing state s =',s
                        R.add(s)
            #print 'remove list: ',R

            while len(R) > 0:
                raise "Error!"
                s = R.pop()
                T.remove(s)
                print pre(s,MDP)
                for (t,b) in pre(s,MDP):
                    print t,b
                    if t not in T:
                        break
                    if b in A[t]:
                        A[t].remove(b)
                    if len(A[t]) == 0:
                        R.add(t)

            for T_i in SCCs:
                if len(set(T) & set(T_i)) > 0:  #if intersection is not empty
                    MEC_new.append( (set(T) & set(T_i)) )
            
    output = []
    for T in MEC:
        A_T = {t: A[t] for t in T}
        output.append([T,A_T])
    return output


#REWRITE
def to_ssp(prodMDP,targetSet,noReach):
    # Convert MDP model with S=1, S=0, and end component information into a MDP that satisfies the 
    #stochastic shortest path (SSP) assumptions
    #Input: An MDP along with S=1, S=0, and end component information
    #Output: A data structure that represents the transformed MDP
        
    # Initialize the new MDP
    S = set(prodMDP.keys())
    S_r = S - (noReach | targetSet)
    
    # Construct sub-MDP for states in the reachability set (S=?) only
    reachMDP = {}
    for s in S_r:
        reachMDP[s] = prodMDP[s].copy()
        for a in reachMDP[s].keys():
            #make sure actions don't lead to undefined states outside of reachability set
            if not set(reachMDP[s][a].keys()).issubset(S_r):
                del reachMDP[s][a]
    reachMECs = max_end_components(reachMDP)
    #print '\nMECs in reachable set', [row[0] for row in reachMECs]    

  
    # Update the state sets
    s_new = set()
    s_MEC = set()
    for i in xrange(len(reachMECs)):
        B = reachMECs[i][0]
        newState = 's'+str(i)
        s_new.add(newState)

        for s in B:
            s_MEC.add(s)
            
    S_hat = (S | s_new) - s_MEC
    S_r_hat = (S_r | s_new) - s_MEC
    
    # Create (state,action) pairs
    A_hat = {}
    for s in (S - s_MEC):
        stateAct = []
        for a in prodMDP[s].keys():
            stateAct.append((s,a))
        A_hat[s] = stateAct
        
    for sNew in s_new:
        B = reachMECs[int(sNew[1:])][0]
        D = reachMECs[int(sNew[1:])][1]
        stateAct = []
        
        for s in B:
            A_s = prodMDP[s].keys()
            for a in (set(A_s) - set(D[s])):
                stateAct.append((s,a))
        A_hat[sNew] = stateAct
    assert len(A_hat) == len(S) - len(s_MEC) + len(reachMECs)
    
    
    # Create new MDP with modified transition probabilities
    sspMDP = {}
    for s in S_hat:
        sspMDP[s] = {}
        
        for (u,a) in A_hat[s]:
            sspMDP[s][(u,a)] = {}
            
            prod_keys = set(prodMDP.keys())
            prod_u_keys = set(prodMDP[u].keys())
            prod_ua_keys = set(prodMDP[u][a].keys())
            
            for t in (S-s_MEC):
                #If non-zero transition probability from 'u' to 't' using 'a'
                if (u in prod_keys) and (a in prod_u_keys) and (t in prod_ua_keys):
                    sspMDP[s][(u,a)][t] = float(prodMDP[u][a][t])
        
            for sNew in s_new:
                prob = 0
                B = reachMECs[int(sNew[1:])][0]
                for t in B:
                    #If non-zero transition probability from 's' to 't' using 'a'
                    if (u in prod_keys) and (a in prod_u_keys) and (t in prod_ua_keys):
                        prob += float(prodMDP[u][a][t])
                if prob > 0:
                    sspMDP[s][(u,a)][sNew] = prob

    # Add the terminal state (doesn't affect anything in S_r, so OK to leave until now).
    sspMDP['t'] = {('t',0):{'t':1}}     #Add the absorbing terminal state 't'
    for s in (targetSet | noReach):     #Link states in targetSet and noReach directly to terminal state
        sspMDP[s] = {(s,0):{'t':1}}    
    noReach.add('t')

    # Create mapping from ssp states to original states (ignoring terminal state)
    ssp2orig = {}
    for s in sspMDP.keys():
        if s in s_new:
            B = reachMECs[int(s[1:])][0]
            ssp2orig[s] = B
        elif s == 't':
            pass
        else:
            ssp2orig[s] = set([s])
    
    return sspMDP, ssp2orig, S_r, s_MEC, A_hat

