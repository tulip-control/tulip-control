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
"""Functions for plotting Partitions."""
import functools as _ft
import logging

import networkx as nx
import numpy as np
import scipy.sparse as sp

import polytope as pc
import polytope.plot as _pplot

import tulip.abstract as _abs
import tulip.graphics as _graphics
_plt = _graphics._plt


__all__ = [
    'plot_partition',
    'plot_transition_arrow',
    'plot_abstraction_scc',
    'plot_ts_on_partition',
    'project_strategy_on_partition',
    'plot_strategy',
    'plot_trajectory']


logger = logging.getLogger(__name__)
plot_partition = _pplot.plot_partition
plot_transition_arrow = _pplot.plot_transition_arrow


def plot_abstraction_scc(ab, ax=None):
    """Plot regions colored by strongly-connected component.

    Handy to develop new examples or debug existing ones.
    """
    _assert_pyplot()
    if ax is None:
        ax = _plt.subplot()
    ppp = ab.ppp
    ts = ab.ts
    ppp2ts = ab.ppp2ts
    # each connected component of
    # filtered graph is a symbol
    components = nx.strongly_connected_components(ts)
    l, u = ab.ppp.domain.bounding_box
    ax.set_xlim(l[0, 0], u[0, 0])
    ax.set_ylim(l[1, 0], u[1, 0])
    def plot_state(state, color):
        i = ppp2ts.index(state)
        ppp[i].plot(
            ax=ax,
            color=color)
    for component in components:
        # map to random colors
        red = np.random.rand()
        green = np.random.rand()
        blue = np.random.rand()
        color = (red, green, blue)
        plot = _ft.partial(color=color)
        any(map(plot, component))
    return ax


def plot_ts_on_partition(
        ppp,
        ts,
        ppp2ts:
            list,
        edge_label:
            dict,
        only_adjacent,
        ax):
    """Plot partition, superimposing graph.

    The graph `ts` edges are drawn as arrows.

    Edges can be filtered by
    selecting an `edge_label`.
    So it can plot transitions of
    a single mode for a switched system.

    @param ppp2ts:
        states
    @param edge_label:
        desired label
    """
    l, u = ppp.domain.bounding_box
    arr_size = (u[0, 0] - l[0, 0]) / 50.0
    ts2ppp = {v: k for k, v in enumerate(ppp2ts)}
    transitions = ts.transitions.find(
        with_attr_dict=edge_label)
    for from_state, to_state, label in transitions:
        i = ts2ppp[from_state]
        j = ts2ppp[to_state]
        skip = (
            only_adjacent and
            ppp.adj[i, j] == 0)
        if skip:
            continue
        _pplot.plot_transition_arrow(
            ppp.regions[i],
            ppp.regions[j],
            ax,
            arr_size)


def project_strategy_on_partition(
        ppp:
            'PropPreservingPartition',
        mealy:
            'MealyMachine'):
    """Project transitions of `ppp` on `mealy`.

    @return:
        a matrix with the `ppp` (spatial)
        transitions that are used by
        the Mealy strategy `mealy`.
    """
    n = len(ppp)
    proj_adj = sp.lil_matrix((n, n))
    edges = mealy.transitions.find()
    for from_state, to_state, label in edges:
        from_label = mealy.states[from_state]
        to_label = mealy.states[to_state]
        intersects = (
            'loc' in from_label and
            'loc' in to_label)
        if not intersects:
            continue
        from_loc = from_label['loc']
        to_loc = to_label['loc']
        proj_adj[from_loc, to_loc] = 1
    return proj_adj


def plot_strategy(
        ab:
            'AbstractPwa | AbstractSwitched',
        mealy:
            'MealyMachine'):
    """Plot strategic transitions on PPP.

    Assumes that `mealy` is feasible for
    the system `ab`, i.e., concretizable.
    """
    proj_mealy = project_strategy_on_partition(
        ab.ppp, mealy)
    ax = _pplot.plot_partition(
        ab.ppp,
        proj_mealy,
        color_seed=0)
    return ax


def plot_trajectory(
        ppp:
            'PropPreservingPartition',
        x0,
        u_seq,
        ssys,
        ax=None,
        color_seed=None):
        color_seed:
            int |
            None=None
        ) -> '_mpl.axes.Axes':
    """Plot trajectory, starting from `x0`.

    The trajectory is drawn on the partition.

    Plots a `PropPreservingPartition` and
    the trajectory that is generated by the
    input sequence `u_seq`, starting from point `x0`.


    Relevant
    ========
    `plot_partition`, plot

    @param x0:
        initial state
    @param u_seq:
        matrix where each row contains an input
    @param ssys:
        system dynamics
    @param color_seed:
        read:
        - `polytope.plot_partition()`
        - `numpy.random.RandomState`
    """
    if ax is None:
        ax, fig = _graphics.newax()
    _pplot.plot_partition(
        plot_numbers=False,
        ax=ax,
        show=False)
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
            np.dot(B, u_seq[i, :]).flatten() +
            K.flatten())
        x_arr = np.vstack([x_arr, x.flatten()])
    ax.plot(
        x_arr[:, 0],
        x_arr[:, 1],
        'o-')
    return ax


def simulate2d(
        env_inputs:
            list[dict],
        sys_dyn:
            'LtiSysDyn',
        ctrl:
            'MealyMachine',
        disc_dynamics:
            'AbstractPwa',
        T,
        qinit=r'\E \A',
        d_init=None,
        x_init=None,
        show_traj=True):
    r"""Simulation for systems with two-dimensional continuous dynamics.

    The first item in `env_inputs` is used as the initial environment
    discrete state, if `qinit == r'\E \A' or qinit == r'\A \E'`.
    This item is used to find the initial transition in `ctrl`.

    The initial continuous state is selected within the initial polytope
    of the partition, if `qinit \in (r'\E \E', r'\E \A', r'\A \E')`.

    The initial continuous state is `x_init` if `qinit == r'\A \A'`,
    and is asserted to correspond to `d_init['loc']`.
    The initial discrete state is `d_init` if `qinit == r'\A \A'`,
    and is used to find the initial transition in `ctrl`.

    @param env_inputs:
        has length `T + 1`
    @param sys_dyn:
        system dynamics
    @param T:
        number of simulation steps
    @param qinit:
        quantification of initial conditions
    @param d_init:
        initial discrete state,
        given when `qinit == r'\A \A'`
    @param x_init: initial continuous state,
        given when `qinit == r'\A \A'`
    @param show_traj:
        plot trajectory
    """
    N = disc_dynamics.disc_params['N']
    # initialization:
    #     pick initial continuous state
    #     consistent with
    #     initial controller state (discrete)
    if qinit == r'\E \A' or qinit == r'\A \E':
        # pick an initial discrete
        # system state given the
        # initial discrete environment state
        (nd, out) = ctrl.reaction(
            'Sinit', env_inputs[0])
        init_edges = ctrl.edges(
            'Sinit',
            data=True)
        for u, v, edge_data in init_edges:
            assert u == 'Sinit', u
            if v == nd:
                break
        assert v == nd, (v, nd)
        s0_part = edge_data['loc']
        assert s0_part == out['loc'], (s0_part, out)
    elif qinit == r'\E \E':
        # pick an initial discrete state
        init_edges = ctrl.edges(
            'Sinit',
            data=True)
        u, v, edge_data = next(
            iter(init_edges))
        assert u == 'Sinit', u
        s0_part = edge_data['loc']
        nd = v
    elif qinit == r'\A \A':
        assert d_init is not None
        assert x_init is not None
        s0_part = _abs.find_controller.find_discrete_state(
            x_init, disc_dynamics.ppp)
        assert s0_part == d_init['loc'], (s0_part, d_init)
        # find the machine node with
        # `d_init` as discrete state
        init_edges = ctrl.edges(
            'Sinit',
            data=True)
        for u, v, edge_data in init_edges:
            assert u == 'Sinit', u
            if edge_data == d_init:
                break
        assert edge_data == d_init, (edge_data, d_init)
        nd = v
    # initialize continuous state for
    # the cases that the initial
    # continuous state is not given
    if qinit in (r'\E \E', r'\E \A', r'\A \E'):
        # find initial polytope
        init_poly = disc_dynamics.ppp.regions[
            s0_part].list_poly[0]
        # disc_dynamics.ppp[s0_part][0]
        #
        # pick an initial continuous state
        x_init = pick_point_in_polytope(init_poly)
        # assert equal
        s0_part_ = _abs.find_controller.find_discrete_state(
            x_init, disc_dynamics.ppp)
        assert s0_part == s0_part_, (s0_part, s0_part_)
    x = [x_init[0]]
    y = [x_init[1]]
    for i in range(T):
        (nd, out) = ctrl.reaction(
            nd, env_inputs[i + 1])
        x0 = np.array([x[i * N], y[i * N]])
        start = s0_part
        end = disc_dynamics.ppp2ts.index(out['loc'])
        u = _abs.find_controller.get_input(
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
        s0_part = _abs.find_controller.find_discrete_state(
            point, disc_dynamics.ppp)
        s0_loc = disc_dynamics.ppp2ts[s0_part]
        assert s0_loc == out['loc'], (s0_loc, out['loc'])
        print(f'outputs:\n    {out}\n')
    _assert_pyplot()
    if show_traj:
        _plt.plot(x, label='x')
        _plt.plot(y, '--', label='y')
        _plt.xlabel('time')
        _plt.legend()
        _plt.grid()
        _plt.show()


def pick_point_in_polytope(poly):
    """Return a point in polytope `poly`."""
    poly_vertices = pc.extreme(poly)
    n_extreme = poly_vertices.shape[0]
    x = sum(poly_vertices) / n_extreme
    return x
