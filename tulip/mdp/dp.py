#!/usr/bin/env python
"""
Helper functions for CDC 2012 example code.
Eric Wolff
"""

import sys, os
import numpy as np
from subprocess import call
import time
import mdp_graph, mdp_uncert, c_opt


def get_policy(sspMDP,prodMDP,ssp2prod,S_r,S_MEC,noReach,targetSet,rewards,Jprev,eps,type,typeData,disc):
    
    t0 = time.time()
    Jssp = value_iteration(sspMDP,noReach,targetSet,rewards,Jprev,eps,type,typeData,disc)
    Jprod = reward_map(prodMDP,sspMDP,ssp2prod,Jssp)
    t1= time.time()
    
    for s in noReach:   assert Jprod[s] == 0.0
    for s in targetSet: assert Jprod[s] == 1.0
    
    pi = value2policy(prodMDP,S_r,S_MEC,Jprod,eps,type,typeData,disc)    # Create an optimal policy for the prodMDP

    return pi,Jssp,Jprod


def value_iteration(MDP,noReach,targetSet,rewards,Jprev,epsilon,type,typeData,disc):
    # Input: An MDP and its associated transition uncertainty sets.
    # Output: The value function, J.
    
    # Load appropriate uncertainty set representations
    if type == 'likelihood':
        B_dict = typeData[0]
        Bmax_dict = typeData[1]
    if type == 'interval':
        interval = typeData[0]
    
    # Initialize rewards
    S = set(MDP.keys())
    for s in (noReach | targetSet):
        Jprev[s] = rewards[s]
    
    while True:
        J = Jprev.copy()
        for s in (S-(noReach | targetSet)):
            maxReward = -np.Inf
    
            for a in MDP[s]:
                probDist = MDP[s][a]
                
                if type == 'nominal':
                    envOpt = sum([p*Jprev[s1] for (s1,p) in probDist.iteritems()])
                
                elif type == 'likelihood':
                    B = B_dict[s][a]
                    Bmax = Bmax_dict[s][a]
                    assert B <= Bmax
                    f,v = nonzero2array(probDist,Jprev)
                    envOpt = c_opt.biEnvLikelihood(f,v,B,Bmax)  #min reward that environment can create

                elif type == 'interval':
                    f,v,pl,pu = nonzero2array_int(probDist,Jprev,interval[s][a])
                    envOpt = c_opt.biEnvInterval(f,v,pl,pu)

                tmp = rewards[s] + disc*envOpt
                if tmp > maxReward:
                    maxReward = tmp
            J[s] = maxReward
        
        #print 'Max error =',np.max([abs(J[s]-Jprev[s]) for s in MDP.keys()])
        if np.max([abs(J[s]-Jprev[s]) for s in MDP.keys()]) < epsilon:
            return J
        Jprev = J.copy()
        

#REWRITE
def value2policy(MDP, S_r, S_MEC, J, eps, type, typeData, disc):
    # Returns optimal control policy corresponding to MDP with reward vector J
    # Don't pick self-loops if have another choice, as they are never optimal action.
    
    # Load appropriate uncertainty set representations
    if type == 'likelihood':
        B_dict = typeData[2]
        Bmax_dict = typeData[3]
    if type == 'interval':
        interval = typeData[1]
    
    pi = {}
    S = set(MDP.keys())     #set of all states in MDP
    S_r = set(S_r)          #non-zero reachability states outside target set
    S_MEC = set(S_MEC)      #states in S_r in a maximal end component (MEC)
    visitedStates = S-S_r     # Initialize with all states not in S_r (this includes noReach and targetSet)

    # Determine set of valid actions for states in S_r (these are the ones where using act 'a' gives J(s))
    validAct = {}
    for s in S_r:
        validAct[s] = set()
        for a in MDP[s]:
            if s not in MDP[s][a] or abs(MDP[s][a][s]-1) > eps: #avoid self loops
                probDist = MDP[s][a]
                
                if type == 'nominal':
                    tmp = sum([p * J[s1] for (s1, p) in probDist.iteritems()])
                    
                elif type == 'likelihood':
                    B = B_dict[s][a]
                    Bmax = Bmax_dict[s][a]
                    f,v = nonzero2array(probDist,J)
                    tmp = c_opt.biEnvLikelihood(f,v,B,Bmax)
                    
                elif type == 'interval':
                    f,v,pl,pu = nonzero2array_int(probDist,J,interval[s][a])
                    tmp = c_opt.biEnvInterval(f,v,pl,pu)
                
                if abs(disc*tmp - J[s]) < eps:
                    validAct[s].add(a)
        if len(validAct[s]) == 1:
            pi[s] = int(validAct[s].pop())
            visitedStates.add(s)

    while S != visitedStates:
        assert (S_r - visitedStates) == (S - visitedStates)
        #print type,len(S),len(visitedStates)

        # Determine which validActions are most likely to leave MEC
        leaveRank = {}
        for s in (S_r - visitedStates):
            leaveRank[s] = ()
            maxProbLeave = 0.0
            for a in validAct[s]:
                probLeave = 0.0     #compute the probability of transition to visitedStates
                for (s1,p) in MDP[s][a].iteritems():
                    if (s1 in visitedStates) or (s1 in (S-S_MEC)):
                        probLeave += p
                if probLeave > maxProbLeave:
                    maxProbLeave = probLeave
                    leaveRank[s] = (probLeave,a)

        # Select action for state with highest probability of leaving MEC
        maxProbLeave = 0.0
        for s in (S_r - visitedStates):
            if len(leaveRank[s]) > 0:
                if leaveRank[s][0] > maxProbLeave:
                    maxProbLeave = leaveRank[s][0]
                    optState = s
                    optAct = int(leaveRank[s][1])
        pi[optState] = optAct
        visitedStates.add(optState)
        
    return pi




def get_adversary(advfile,prodMDP,targetSet):
    # Generate control policy from PRISM adversary file
    #Input: A PRISM .tra adversary file and corresponding .sta state file.
    #Output: A control policy for the MDP
    
    fAdv = file(advfile,'r')    # Open adversary file for reading
    rAdv = fAdv.readlines()     # Read information from file
    fAdv.close()                # Close file to clean-up
    
    # Convert adversary information to numeric tuple (curr,next,prob)
    adv = []
    for i in xrange(1,len(rAdv)):
        tmp = rAdv[i].split(' ')
        adv.append([int(tmp[0]),int(tmp[1]),float(tmp[2])])
    
    # Determine all states in adversary
    states = []
    for row in adv:
        if row[0] in states:
            pass
        else:
            if row[0] < len(prodMDP):
                states.append(row[0])
    
    # Determine optimal action for each state in adversary file (policy to reach target set)
    policy = {}
    for curr in states:
        transProb = {}
        
        for row in adv:
            if row[0] == curr:
                nextState = row[1]
                nextProb = row[2]
                transProb[nextState] = nextProb
        
        # Compare with MDP to find optimal action number
        for act in prodMDP[curr].keys():
            sortMDP = sorted(prodMDP[curr][act].items())
            sortTrans = sorted(transProb.items())
            same = True
            if len(sortMDP) != len(sortTrans):
                same = False
            else:
                for i in xrange(len(sortMDP)):
                    if (sortMDP[i][0] != sortTrans[i][0]) or (abs(sortMDP[i][1]-sortTrans[i][1]) > 1e-5):
                        same = False
                        break
            if same:
                policy[curr] = act
    return policy


def reward_map(MDP,sspMDP,ssp2prod,Jssp):
    # Map reward function from sspMDP to prodMDP.
    J = dict([(s, 0.0) for s in MDP.keys()])
    
    for s in sspMDP.keys():
        if s != 't':
            for s1 in ssp2prod[s]:
                J[s1] = Jssp[s]
    return J


# REWRITE
def expected_cost_likelihood(MDP,policy,noReach,targetSet,Bmax_dict,B_dict,Jguess,epsilon,disc):
    # Input: An MDP and its associated transition uncertainty sets.
    # Output: The worst-case reward function, J.
    epsilon = 1.5*epsilon
    rewards = dict([(s, 0.0) for s in MDP.keys()])
    for t in targetSet:
        rewards[t] = 1.0
    S = set(MDP.keys())
    Jprev = Jguess
        
    while True:
        J = Jprev.copy()
        for s in (S-(noReach | targetSet)):
            a = policy[s]
            probDist = MDP[s][a]
            B = B_dict[s][a]
            Bmax = Bmax_dict[s][a]
            f,v = nonzero2array(probDist,Jprev)
            envOpt = c_opt.biEnvLikelihood(f,v,B,Bmax)
            J[s] = rewards[s] + disc*envOpt
        
        #print 'Max error =',np.max([abs(J[s]-Jprev[s]) for s in MDP.keys()])
        if np.max([abs(J[s]-Jprev[s]) for s in MDP.keys()]) < epsilon:
            return J
        Jprev = J.copy()
    

def nonzero2array(f,v):
    f_nz = []
    v_nz = []
    for s in f.keys():
        if f[s] != 0.0:
            f_nz.append(f[s])
            v_nz.append(v[s])

    return np.array(f_nz,dtype=np.double),np.array(v_nz,dtype=np.double)


def nonzero2array_int(f,v,interval):
    f_nz = []
    v_nz = []
    pl_nz = []
    pu_nz = []
    for s in f.keys():
        if f[s] != 0.0:
            f_nz.append(f[s])
            v_nz.append(v[s])
            pl_nz.append(interval[s][0])
            pu_nz.append(interval[s][1])

    return np.array(f_nz,dtype=np.double),np.array(v_nz,dtype=np.double),\
            np.array(pl_nz,dtype=np.double),np.array(pu_nz,dtype=np.double)