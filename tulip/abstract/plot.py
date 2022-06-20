# Copyright (c) 2011-2014 by California Institute of Technology
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
"""
Functions for plotting Partitions.
"""
from __future__ import division

import logging
logger = logging.getLogger(__name__)

import numpy as np
from scipy import sparse as sp
import networkx as nx
import polytope as pc
from polytope.plot import plot_partition, plot_transition_arrow

from tulip.abstract import find_controller

# inline imports:
#
# import matplotlib as mpl
# from tulip.graphics import newax


__all__ = [
    'plot_partition', 'plot_transition_arrow',
    'plot_abstraction_scc', 'plot_ts_on_partition',
    'project_strategy_on_partition', 'plot_strategy',
    'plot_trajectory']


def plot_abstraction_scc(ab, ax=None):
    """Plot Regions colored by strongly connected component.

    Handy to develop new examples or debug existing ones.
    """
    try:
        import matplotlib as mpl
    except:
        logger.error('failed to load matplotlib')
        return

    ppp = ab.ppp
    ts = ab.ts
    ppp2ts = ab.ppp2ts

    # each connected component of filtered graph is a symbol
    components = nx.strongly_connected_components(ts)

    if ax is None:
        ax = mpl.pyplot.subplot()

    l, u = ab.ppp.domain.bounding_box
    ax.set_xlim(l[0,0], u[0,0])
    ax.set_ylim(l[1,0], u[1,0])

    for component in components:
        # map to random colors
        red = np.random.rand()
        green = np.random.rand()
        blue = np.random.rand()

        color = (red, green, blue)

        for state in component:
            i = ppp2ts.index(state)
            ppp[i].plot(ax=ax, color=color)
    return ax

def plot_ts_on_partition(ppp, ts, ppp2ts, edge_label, only_adjacent, ax):
    """Plot partition and arrows from labeled digraph.

    Edges can be filtered by selecting an edge_label.
    So it can plot transitions of a single mode for a switched system.

    @param edge_label: desired label
    @type edge_label: dict
    """
    l,u = ppp.domain.bounding_box
    arr_size = (u[0,0]-l[0,0])/50.0

    ts2ppp = {v:k for k,v in enumerate(ppp2ts)}
    for from_state, to_state, label in ts.transitions.find(with_attr_dict=edge_label):
        i = ts2ppp[from_state]
        j = ts2ppp[to_state]

        if only_adjacent:
            if ppp.adj[i, j] == 0:
                continue

        plot_transition_arrow(ppp.regions[i], ppp.regions[j], ax, arr_size)

def project_strategy_on_partition(ppp, mealy):
    """Return an FTS with the PPP (spatial) transitions used by Mealy strategy.

    @type ppp: L{PropPreservingPartition}

    @type mealy: L{transys.MealyMachine}
    """
    n = len(ppp)
    proj_adj = sp.lil_matrix((n, n))

    for (from_state, to_state, label) in mealy.transitions.find():
        from_label = mealy.states[from_state]
        to_label = mealy.states[to_state]

        if 'loc' not in from_label or 'loc' not in to_label:
            continue

        from_loc = from_label['loc']
        to_loc = to_label['loc']

        proj_adj[from_loc, to_loc] = 1

    return proj_adj

def plot_strategy(ab, mealy):
    """Plot strategic transitions on PPP.

    Assumes that mealy is feasible for ab.

    @type ab: L{AbstractPwa} or L{AbstractSwitched}

    @type mealy: L{transys.MealyMachine}
    """
    proj_mealy = project_strategy_on_partition(ab.ppp, mealy)
    ax = plot_partition(ab.ppp, proj_mealy, color_seed=0)
    return ax

def plot_trajectory(ppp, x0, u_seq, ssys,
                    ax=None, color_seed=None):
    """Plots a PropPreservingPartition and the trajectory generated by x0
    input sequence u_seq.

    See Also
    ========
    C{plot_partition}, plot

    @type ppp: L{PropPreservingPartition}
    @param x0: initial state
    @param u_seq: matrix where each row contains an input
    @param ssys: system dynamics
    @param color_seed: see C{plot_partition}
    @return: axis object
    """
    try:
        from tulip.graphics import newax
    except:
        logger.error('failed to import graphics.newax')
        return

    if ax is None:
        ax, fig = newax()

    plot_partition(plot_numbers=False, ax=ax, show=False)

    A = ssys.A
    B = ssys.B

    if ssys.K is not None:
        K = ssys.K
    else:
        K = np.zeros(x0.shape)

    x = x0.flatten()
    x_arr = x0
    for i in range(u_seq.shape[0]):
        x = (
            np.dot(A, x).flatten() +
            np.dot(B, u_seq[i, :] ).flatten() +
            K.flatten())
        x_arr = np.vstack([x_arr, x.flatten()])

    ax.plot(x_arr[:,0], x_arr[:,1], 'o-')

    return ax


def simulate2d(
        env_inputs, sys_dyn, ctrl, disc_dynamics, T,
        qinit='\E \A',
        d_init=None,
        x_init=None,
        show_traj=True):
    r"""Simulation for systems with two-dimensional continuous dynamics.

    The first item in `env_inputs` is used as the initial environment
    discrete state, if `qinit == '\E \A' or qinit == \A \E'`.
    This item is used to find the initial transition in `ctrl`.

    The initial continuous state is selected within the initial polytope
    of the partition, if `qinit \in ('\E \E', '\E \A', '\A \E')`.

    The initial continuous state is `x_init` if `qinit == '\A \A'`,
    and is asserted to correspond to `d_init['loc']`.
    The initial discrete state is `d_init` if `qinit == '\A \A'`,
    and is used to find the initial transition in `ctrl`.

    @param env_inputs: `list` of `dict`, with length `T + 1`
    @param sys_dyn: system dynamics
    @type sys_dyn: `tulip.hybrid.LtiSysDyn`
    @type ctrl: `tulip.transys.machines.MealyMachine`
    @type disc_dynamics: `tulip.abstract.discretization.AbstractPwa`
    @param T: number of simulation steps
    @param qinit: quantification of initial conditions
    @param d_init: initial discrete state,
        given when `qinit == '\A \A'`
    @param x_init: initial continuous state,
        given when `qinit == '\A \A'`

    @param show_traj: plot trajectory
    """
    N = disc_dynamics.disc_params['N']
    # initialization:
    #     pick initial continuous state consistent with
    #     initial controller state (discrete)
    if qinit == '\E \A' or qinit == '\A \E':
        # pick an initial discrete system state given the#
        # initial discrete environment state
        (nd, out) = ctrl.reaction('Sinit', env_inputs[0])
        init_edges = ctrl.edges('Sinit', data=True)
        for u, v, edge_data in init_edges:
            assert u == 'Sinit', u
            if v == nd:
                break
        assert v == nd, (v, nd)
        s0_part = edge_data['loc']
        assert s0_part == out['loc'], (s0_part, out)
    elif qinit == '\E \E':
        # pick an initial discrete state
        init_edges = ctrl.edges('Sinit', data=True)
        u, v, edge_data = next(iter(init_edges))
        assert u == 'Sinit', u
        s0_part = edge_data['loc']
        nd = v
    elif qinit == '\A \A':
        assert d_init is not None
        assert x_init is not None
        s0_part = find_controller.find_discrete_state(
            x_init, disc_dynamics.ppp)
        assert s0_part == d_init['loc'], (s0_part, d_init)
        # find the machine node with `d_init` as discrete state
        init_edges = ctrl.edges('Sinit', data=True)
        for u, v, edge_data in init_edges:
            assert u == 'Sinit', u
            if edge_data == d_init:
                break
        assert edge_data == d_init, (edge_data, d_init)
        nd = v
    # initialize continuous state for the cases that the initial
    # continuous state is not given
    if qinit in ('\E \E', '\E \A', '\A \E'):
        # find initial polytope
        init_poly = disc_dynamics.ppp.regions[s0_part].list_poly[0]
        # disc_dynamics.ppp[s0_part][0]
        #
        # pick an initial continuous state
        x_init = pick_point_in_polytope(init_poly)
        # assert equal
        s0_part_ = find_controller.find_discrete_state(
            x_init, disc_dynamics.ppp)
        assert s0_part == s0_part_, (s0_part, s0_part_)
    x = [x_init[0]]
    y = [x_init[1]]
    for i in range(T):
        (nd, out) = ctrl.reaction(nd, env_inputs[i + 1])
        x0 = np.array([x[i * N], y[i * N]])
        start = s0_part
        end = disc_dynamics.ppp2ts.index(out['loc'])
        u = find_controller.get_input(
            x0=x0,
            ssys=sys_dyn,
            abstraction=disc_dynamics,
            start=start,
            end=end,
            ord=1,
            mid_weight=5)
        # continuous trajectory for interval `i`
        for ind in range(N):
            s_now = np.dot(
                sys_dyn.A, [x[-1], y[-1]]
            ) + np.dot(sys_dyn.B, u[ind])
            x.append(s_now[0])
            y.append(s_now[1])
        point = [x[-1], y[-1]]
        s0_part = find_controller.find_discrete_state(
            point, disc_dynamics.ppp)
        s0_loc = disc_dynamics.ppp2ts[s0_part]
        assert s0_loc == out['loc'], (s0_loc, out['loc'])
        print('outputs:\n    {out}\n'.format(out=out))
    if show_traj:
        from matplotlib import pyplot as plt
        plt.plot(x, label='x')
        plt.plot(y, '--', label='y')
        plt.xlabel('time')
        plt.legend()
        plt.grid()
        plt.show()


def pick_point_in_polytope(poly):
    """Return a point in polytope `poly`."""
    poly_vertices = pc.extreme(poly)
    n_extreme = poly_vertices.shape[0]
    x = sum(poly_vertices) / n_extreme
    return x
