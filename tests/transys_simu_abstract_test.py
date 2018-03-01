#!/usr/bin/env python
"""Tests of `transys.transys.simu_abstract`."""
import sys, os

sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../tulip'))

import logging
from tulip.transys.transys import FTS, simu_abstract
import numpy as np
from polytope import box2poly
from tulip.abstract import prop2part, discretize
from tulip import hybrid

logging.basicConfig()
logger = logging.getLogger(__name__)


def build_FTS(index):
    # build test FTS
    # simple test
    if index == 0: 
        ts = FTS()
        ts.atomic_propositions.add_from({'a', 'b', 'c', 'd'})
        ts.states.add_from([('q1', {'ap': {'a'}}), ('q2', {'ap': {'a'}}),
                            ('q3', {'ap': {'b'}}), ('q4', {'ap': {'b'}}),
                            ('q5', {'ap': {'b'}}), ('q6', {'ap': {'c'}}),
                            ('q7', {'ap': {'d'}})])
        ts.transitions.add('q1', 'q3')
        ts.transitions.add('q1', 'q4')
        ts.transitions.add('q3', 'q6')
        ts.transitions.add('q4', 'q6')
        ts.transitions.add('q2', 'q4')
        ts.transitions.add('q2', 'q5')
        ts.transitions.add('q5', 'q7')
        ts.transitions.add('q6', 'q6')
        ts.transitions.add('q7', 'q7')
    # FTS generated by continuous system
    elif index == 1:
        # Problem parameters
        input_bound = 1.0
        # Continuous state space
        cont_state_space = box2poly([[-1, 1], [-1, 1]])
        # Continuous dynamics
        # (continuous-state, discrete-time)
        A = np.array([[0.5, 1.0], [ 0.75, -1.0]])
        B = np.array([[1, 0.], [ 0., 1]])
        # Available control, possible disturbances
        U = input_bound *np.array([[-1., 1.], [-1., 1.]])
        # Convert to polyhedral representation
        U = box2poly(U)
        # Construct the LTI system describing the dynamics
        sys_dyn = hybrid.LtiSysDyn(A, B, None, None, U, None, cont_state_space)
        # @dynamics_section_end@
        # @partition_section@
        # Define atomic propositions for relevant regions of state space
        cont_props = {}
        cont_props['a'] = box2poly([[-0.5, 0.5], [-0.5, 0.5]])
        cont_props['b'] = box2poly([[-1, -0.5], [-1, 1]])
        cont_props['c'] = box2poly([[-0.5, 1], [0.5, 1]])
        cont_props['d'] = box2poly([[0.5, 1], [-1, 0.5]])
        cont_props['e'] = box2poly([[-0.5, 0.5], [-1, -0.5]])
        # Compute the proposition preserving partition of the continuous state space
        cont_partition = prop2part(cont_state_space, cont_props)
        # @partition_section_end@
        # @discretize_section@
        # Given dynamics & proposition-preserving partition, find feasible transitions
        sys = discretize(
            cont_partition, sys_dyn, closed_loop=False,conservative=True,
            trans_length=1,N=1, use_all_horizon=False, min_cell_volume=0.0, 
            abs_tol=1e-7
        )
        ts = sys.ts
    return ts


def check_simulation(ts1, ts2, L12, L21):
    # check if ts1 is simulated by ts2
    # L12 is a mapping for nodes from ts1 to ts2
    # L21 is a mapping for nodes from ts2 to ts1
    #
    # check condition a: for each s1 in ts1, exist s2 in ts2, (s1,s2) in L12
    for i in ts1:
        assert(len(L12[i]) > 0)
    # check condition b: for all (s1,s2) in L12, s1|=a <==> s2|=a
    for i in L12:
        list_s2 = L12[i]
        for j in list_s2:
            assert(ts1.states[i]['ap'] ==
                   ts2.states[j]['ap'])
    # check condition c
    for i in L12:
        list_s2 = L12[i]
        succ1 = set(ts1.succ[i].keys())
        for j in list_s2:
            succ_j = ts2.succ[j].keys()
            succ2 = set()
            for k in succ_j:
                succ2 = succ2.union(L21[k])
            assert succ1.issubset(succ2) or succ1 == succ2
    return True


def simu_abstract_test():
    for index in range(2):
        ts = build_FTS(index)
        [bi_simu, bi_part] = simu_abstract(ts, 'bi')
        [dual_simu, dual_part] = simu_abstract(ts, 'dual')
        # check dual-simulation of bi_simu
        # pick the smallest cell in dual_simu for each state in ts
        K12 = dual_part['ts2simu'].copy()
        for i in K12:
            list_s2 = K12[i]
            point = 0
            curr_len = 0
            best_len = 1e10
            for j in list_s2:
                curr_len = len(dual_part['simu2ts'][j])
                if curr_len < best_len:
                    best_len = curr_len
                    point = j
            K12[i] = set([point])
        assert check_simulation(ts, dual_simu, K12,
                                dual_part['simu2ts'])
        assert check_simulation(dual_simu, ts, dual_part['simu2ts'],
                                dual_part['ts2simu'])
        # check bisimulation of bi_simu
        assert check_simulation(ts, bi_simu, bi_part['ts2simu'],
                                bi_part['simu2ts'])
        assert check_simulation(bi_simu, ts, bi_part['simu2ts'],
                                bi_part['ts2simu'])
        
if __name__ == "__main__":
    bi_simu = simu_abstract_test()
