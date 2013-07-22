#!/usr/bin/env python
"""
Create uncertainty representations for MDP transition probabilities.
Eric Wolff
"""

import numpy as np
import scipy.stats as stat
from math import exp, sqrt


def get_Bmax(MDP):
    # Compute Bmax from measured transition probabilities
    Bmax_dict = {}
    for s in MDP.keys():
        Bmax_dict[s] ={}
        
        for a in MDP[s].keys():
            Bmax = 0.0
            for s1 in MDP[s][a].keys():
                f = MDP[s][a][s1]
                if f != 0.0:     #define 0*log(0) := 0
                    Bmax += f*np.log(f)
            Bmax_dict[s][a] = Bmax
    return Bmax_dict


def create_likelihood(sysMDP,UL,N,obstacleIDs,UL_obs):
    # Create uncertainty sets (likelihood) for the system MDP
    Bmax_dict_sys = get_Bmax(sysMDP)
    B_dict_sys = {}
    for s in sysMDP.keys():
        B_dict_sys[s] = {}
        
        for a in Bmax_dict_sys[s].keys():
            Bmax = Bmax_dict_sys[s][a]
            dof = len(np.nonzero(sysMDP[s][a].values())[0]) - 1
            if dof > 0:
                if len( set(sysMDP[s][a].keys()) & obstacleIDs ) == 0:
                    B_dict_sys[s][a] = Bmax - stat.chi2.ppf(UL,dof)/(2.0*N)
                else:   #Increase uncertainty around obstacles
                    B_dict_sys[s][a] = Bmax - stat.chi2.ppf(UL_obs,dof)/(2.0*N)
            else:   #only single non-zero transition
                B_dict_sys[s][a] = Bmax
    return Bmax_dict_sys, B_dict_sys


def map_likelihood(prodMDP,sspMDP,prod2sys,ssp2prod,Bmax_dict_sys,B_dict_sys):
    # Map uncertainty sets from system MDP to the product MDP
    Bmax_dict_prod = {}
    B_dict_prod = {}
    for s in prodMDP.keys():
        Bmax_dict_prod[s] = {}
        B_dict_prod[s] = {}
        
        for a in prodMDP[s].keys():
            s_sys = prod2sys[s]
            Bmax_dict_prod[s][a] = Bmax_dict_sys[s_sys][a]
            B_dict_prod[s][a] = B_dict_sys[s_sys][a]    
    
    # Map uncertainty sets from product MDP to the stochastic shortest path MDP
    Bmax_dict_ssp = {}
    B_dict_ssp = {}
    for s in sspMDP.keys():
        Bmax_dict_ssp[s] = {}
        B_dict_ssp[s] = {}
        
        for a in sspMDP[s].keys():
            if s != 't':
                s_p = list(ssp2prod[s])[0]
                a_p = a[1]

                Bmax_dict_ssp[s][a] = Bmax_dict_prod[s_p][a_p]
                B_dict_ssp[s][a] = B_dict_prod[s_p][a_p]
            else:
                Bmax_dict_ssp[s][a] = 0.0
                B_dict_ssp[s][a] = 0.0

    return Bmax_dict_ssp, B_dict_ssp, Bmax_dict_prod, B_dict_prod



def create_interval(sysMDP,UL,N,obstacleIDs,UL_obs):
    tol = 1e-9      #how close to (0,1) can the interval come?
    Bmax_dict_sys, B_dict_sys = create_likelihood(sysMDP,UL,N,obstacleIDs,UL_obs)

    # Create uncertainty sets (interval) for the system MDP
    interval_sys = {}
    for s in sysMDP.keys():
        interval_sys[s] ={}
        
        for a in sysMDP[s].keys():
            interval_sys[s][a] ={}
            
            for s1 in sysMDP[s][a].keys():
                f = sysMDP[s][a][s1]        #nominal probability
                
                if f > 0.0 and f < 1.0:     #transitions cannot change from on to off and vice versa
                    Bmax = Bmax_dict_sys[s][a]
                    B = B_dict_sys[s][a]
                    
                    #Use Ellipsoidal approximation and then projections onto components
                    pl, pu = quadraticFormula(1, -2*f, f**2 - 2*f*(Bmax-B))
                    
                    pl = max(pl,tol)
                    pu = min(pu,1.0-tol)
                    assert pl <= pu
                    interval_sys[s][a][s1] = (pl,pu)
                else:       #transition probability is either 0 or 1 and must remain that way.
                    assert f == 0.0 or f == 1.0
                    interval_sys[s][a][s1] = (f,f)
                    
    return interval_sys

def quadraticFormula(a,b,c):
    q = sqrt(b**2 - 4*a*c)
    return (-b - q)/2.0, (-b + q)/2.0


def map_interval(prodMDP,sspMDP,prod2sys,ssp2prod,interval_sys):
    # Map (interval) uncertainty sets from system MDP to the product MDP
    interval_prod = {}
    for s in prodMDP.keys():
        interval_prod[s] = {}
        
        for a in prodMDP[s].keys():            
            interval_prod[s][a] ={}

            for s1 in prodMDP[s][a].keys():
                s_sys = prod2sys[s]
                s1_sys = prod2sys[s1]
                interval_prod[s][a][s1] = interval_sys[s_sys][a][s1_sys]    
    
    # Map interval uncertainty sets from product MDP to the stochastic shortest path MDP
    interval_ssp = {}
    for s in sspMDP.keys():
        interval_ssp[s] = {}
        
        for a in sspMDP[s].keys():
            interval_ssp[s][a] = {}
            
            for s1 in sspMDP[s][a].keys():
                if s == 't' or s1 == 't':
                    interval_ssp[s][a][s1] = (1.0,1.0)
                else:
                    s_p = list(ssp2prod[s])[0]
                    s1_p = list(ssp2prod[s1])[0]
                    a_p = a[1]
                    interval_ssp[s][a][s1] = interval_prod[s_p][a_p][s1_p]

    return interval_ssp, interval_prod



def cvx_env_likelihood(f,v,B):
    from cvxmod import matrix, problem, optvar, minimize, tp, value, printval
    from cvxmod.atoms import log as cvx_log, sum as cvx_sum
    
    f,v = nonzero2array(f,v)
    f = matrix(f)
    v = matrix(v)
   
    p = optvar('p', len(f))
    prob = problem( minimize(tp(v)*p), [p >= 0, cvx_sum(p) == 1, tp(f)*cvx_log(p) >= B])
    prob.solve(quiet=True)
    return prob.value


def cvx_env_interval(f,v,pl,pu):
    from cvxopt import matrix, solvers
    from cvxopt.modeling import variable, op, dot, sum
    
    # Convert to form for CVXOPT  
    f = matrix(f)
    v = matrix(v)
    pl = matrix(pl)
    pu = matrix(pu)
    
    solvers.options['show_progress'] = False
    
    p = variable(len(f))
    lp = op(dot(v,p), [p >= 0, sum(p) == 1, p >= pl, p <= pu])
    lp.solve()
    
    return lp.objective.value()[0]
