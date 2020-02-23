"""
Tests for the abstraction from continuous dynamics to logic
"""
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# logging.getLogger('tulip').setLevel(logging.ERROR)
logger.setLevel(logging.DEBUG)

from nose.tools import assert_raises

import matplotlib
# to avoid the need for using: ssh -X when running tests remotely
matplotlib.use('Agg')

import networkx as nx
import numpy as np

from tulip import abstract
from tulip.abstract import feasible
from tulip import hybrid
import polytope as pc

input_bound = 0.4


def subsys0():
    dom = pc.box2poly([[0., 3.], [0., 2.]])

    A = np.eye(2)
    B = np.eye(2)

    U = pc.box2poly([[0., 1.],
                     [0., 1.]])
    U.scale(input_bound)

    sys_dyn = hybrid.LtiSysDyn(A, B, Uset=U, domain=dom)

    return sys_dyn


def subsys1():
    dom = pc.box2poly([[0., 3.], [0., 2.]])

    A = np.eye(2)
    B = np.eye(2)

    U = pc.box2poly([[0., 0.],
                     [-1., 0.]])
    U.scale(input_bound)

    sys_dyn = hybrid.LtiSysDyn(A, B, Uset=U, domain=dom)

    return sys_dyn


def transition_directions_test():
    """Unit test for correctness of abstracted transition directions, with:

      - uni-directional control authority
      - no disturbance
    """
    modes = list()
    modes.append(('normal', 'fly'))
    modes.append(('refuel', 'fly'))
    env_modes, sys_modes = zip(*modes)
    # dynamics
    cont_state_space = pc.box2poly([[0.0, 3.0], [0.0, 2.0]])
    pwa_sys = dict()
    pwa_sys[('normal', 'fly')] = hybrid.PwaSysDyn(
        [subsys0()], cont_state_space)
    pwa_sys[('refuel', 'fly')] = hybrid.PwaSysDyn(
        [subsys1()], cont_state_space)
    switched_dynamics = hybrid.SwitchedSysDyn(
        disc_domain_size=(len(env_modes), len(sys_modes)),
        dynamics=pwa_sys,
        env_labels=env_modes,
        disc_sys_labels=sys_modes,
        cts_ss=cont_state_space)
    # propositions
    cont_props = dict()
    cont_props['home'] = pc.box2poly([[0.0, 1.0], [0.0, 1.0]])
    cont_props['lot'] = pc.box2poly([[2.0, 3.0], [1.0, 2.0]])
    # partition
    ppp = abstract.prop2part(cont_state_space, cont_props)
    ppp, new2old = abstract.part2convex(ppp)
    # configure discretization
    N = 8
    trans_len = 1
    disc_params = dict()
    for mode in modes:
        disc_params[mode] = dict(N=N, trans_length=trans_len)
    # discretize
    swab = abstract.discretize_switched(
        ppp, switched_dynamics, disc_params,
        plot=True, show_ts=True, only_adjacent=False)
    # assertions
    ts = swab.modes[('normal', 'fly')].ts
    edges = {(0, 0), (1, 1), (2, 2), (3, 3),
             (4, 4), (5, 5),
             (1, 2), (1, 4), (1, 5),
             (2, 3), (2, 5), (2, 0),
             (3, 0),
             (4, 5),
             (5, 0)}
    h = nx.MultiDiGraph()
    h.add_edges_from(edges)
    assert nx.is_isomorphic(ts, h)
    ts = swab.ts
    assert nx.is_isomorphic(ts, h)
    for _, _, d in ts.edges(data=True):
        assert d['env_actions'] == 'normal'
        assert d['sys_actions'] == 'fly'

transition_directions_test.slow = True


def test_transient_regions():
    """drift is too strong, so no self-loop must exist

    This bug caused when running union is taken between Presets
    during solve_feasible, as happened with old use_all_horizon,
    cf:
        - 5b1e9681918739b276a221fcc1fd6eebfd058ce3
        - f5f4934ab9d21062f633eef3861ad935c3d3b54b
    """
    dom = pc.box2poly([[0.0, 4.0], [0.0, 3.0]])

    def cont_predicates():
        p = dict()
        p['safe'] = pc.box2poly([[0.5, 3.5], [0.5, 2.5]])

        ppp = abstract.prop2part(dom, p)
        ppp, new2old_reg = abstract.part2convex(ppp)
        ppp.plot()
        return ppp

    def drifting_dynamics():
        A = np.array([[1.0, 0.0],
                      [0.0, 1.0]])

        B = np.array([[1.0],
                      [0.0]])

        U = pc.box2poly([[0.0, 1.0]])

        K = np.array([[-100.0],
                      [0.0]])

        sys = hybrid.LtiSysDyn(A, B, None, K, U, None, dom)
        return sys

    ppp = cont_predicates()
    sys = drifting_dynamics()
    logger.info(sys)
    ab = abstract.discretize(ppp, sys, N=1, use_all_horizon=True,
                             trans_length=1)
    logger.debug(ab.ts)
    self_loops = {i for i, j in ab.ts.transitions() if i == j}
    logger.debug('self loops at states: ' + str(self_loops))

    assert(not self_loops)

    # ax = ab.plot(show_ts=True)
    # ax.figure.savefig('./very_simple.pdf')


def define_partition(dom):
    p = dict()
    p['a'] = pc.box2poly([[0.0, 10.0], [15.0, 18.0]])
    p['b'] = pc.box2poly([[0.0, 1.0], [0.0, 20.0]])

    ppp = abstract.prop2part(dom, p)
    ppp, new2old_reg = abstract.part2convex(ppp)
    return ppp


def define_dynamics(dom):
    A = np.eye(2)

    B = np.array([[1.0, -1.0],
                  [0.0, +1.0]])

    U = pc.box2poly([[0.0, 3.0],
                     [-3.0, 3.0]])

    E = np.array([[0.0],
                  [-1.0]])

    W = pc.box2poly([[-1.0, 1.0]])
    W.scale(0.4)

    K = np.array([[0.0],
                  [-0.4]])

    sys = hybrid.LtiSysDyn(A, B, E, K, U, W, dom)
    return sys


def define_dynamics_dual():
    # Continuous state space
    cont_state_space = pc.box2poly([[-1.5, 1.5]])

    # Continuous dynamics
    # (continuous-state, discrete-time)
    A = np.array([[2]])
    B = np.array([[1]])

    # Available control, possible disturbances
    U = np.array([[-2.0, 2.0]])

    # Convert to polyhedral representation
    U = pc.box2poly(U)

    # Construct the LTI system describing the dynamics
    sys_dyn = hybrid.LtiSysDyn(A, B, None, None, U, None, cont_state_space)
    # @dynamics_section_end@

    # @partition_section@
    # Define atomic propositions for relevant regions of state space
    cont_props = {}
    cont_props['a'] = pc.box2poly([[-1.5, -1]])
    cont_props['b'] = pc.box2poly([[-1, 1]])
    cont_props['c'] = pc.box2poly([[1, 1.5]])

    part = []
    part.append(pc.box2poly([[-1.5, -1]]))
    part.append(pc.box2poly([[-1, 1]]))
    part.append(pc.box2poly([[1, 1.5]]))
    part.append(pc.box2poly([[-1, 0.5]]))
    part.append(pc.box2poly([[-0.5, 1]]))
    part.append(pc.box2poly([[-0.5, 0.5]]))
    part.append(pc.box2poly([[-1.25, -1]]))
    part.append(pc.box2poly([[1, 1.25]]))
    # Compute the proposition preserving partition of the continuous state
    # space
    cont_partition = abstract.prop2part(cont_state_space, cont_props)
    return sys_dyn, cont_partition, part


def test_abstract_the_dynamics():
    """test_abstract_the_dynamics (known to fail without GLPK)"""
    dom = pc.box2poly([[0.0, 10.0], [0.0, 20.0]])
    ppp = define_partition(dom)
    sys = define_dynamics(dom)
    logger.info(sys)

    disc_options = {'N': 3, 'trans_length': 2, 'min_cell_volume': 1.5}

    ab = abstract.discretize(ppp, sys, plotit=False,
                             save_img=False, **disc_options)
    assert ab.ppp.compute_adj()
    assert ab.ppp.is_partition()
    # ax = ab.plot(show_ts=True, color_seed=0)
    # sys.plot(ax, show_domain=False)
    # print(ab.ts)

    # self_loops = {i for i,j in ab.ts.transitions() if i==j}
    # print('self loops at states: ' + str(self_loops))

test_abstract_the_dynamics.slow = True


def test_abstract_the_dynamics_dual():
    """test_abstract_the_dynamics using dual-simulation algorithm"""
    dom = pc.box2poly([[0.0, 10.0], [0.0, 20.0]])
    ppp = define_partition(dom)
    sys = define_dynamics(dom)
    logger.info(sys)

    disc_options = {'N': 3, 'trans_length': 2, 'min_cell_volume': 1.5}

    ab = abstract.discretize(ppp, sys, plotit=False,
                             save_img=False, simu_type='dual', **disc_options)
    assert ab.ppp.compute_adj()

    [sys_dyn, cont_partition, part] = define_dynamics_dual()
    disc_options = {'N': 1, 'trans_length': 1000, 'min_cell_volume': 0.0}
    ab_2 = abstract.discretize(cont_partition, sys_dyn,
                               simu_type='dual', **disc_options)

    table = np.zeros([len(part), 1])
    for i in ab_2.ppp.regions:
        for j in range(len(part)):
            if i == part[j]:
                table[j] = 1
    assert np.sum(table) == len(part)

test_abstract_the_dynamics_dual.slow = True


def test_is_feasible():
    """Difference between attractor and fixed horizon."""
    dom = pc.box2poly([[0.0, 4.0], [0.0, 3.0]])
    sys = drifting_dynamics(dom)
    p1 = pc.box2poly([[0.0, 1.0], [0.0, 1.0]])
    p2 = pc.box2poly([[2.0, 3.0], [0.0, 1.0]])
    n = 10
    r = feasible.is_feasible(p1, p2, sys, n, use_all_horizon=False)
    assert r is False, r
    r = feasible.is_feasible(p1, p2, sys, n, use_all_horizon=True)
    assert r is True, r


def drifting_dynamics(dom):
    A = np.array([[1.0, 0.0],
                  [0.0, 1.0]])
    B = np.array([[1.0],
                  [0.0]])
    U = pc.box2poly([[0.0, 1.0]])
    K = np.array([[1.0],
                  [0.0]])
    sys = hybrid.LtiSysDyn(A, B, None, K, U, None, dom)
    return sys


if __name__ == '__main__':
    test_abstract_the_dynamics()
    test_abstract_the_dynamics_dual()
