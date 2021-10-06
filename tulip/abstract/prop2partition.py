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
Proposition preserving partition module.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import warnings
import copy

import numpy as np
from scipy import sparse as sp
import polytope as pc
from polytope.plot import plot_partition

from tulip import transys as trs

# inline imports:
#
# from tulip.graphics import newax


__all__ = [
    'prop2part',
    'part2convex',
    'pwa_partition',
    'add_grid',
    'PropPreservingPartition',
    'PPP',
    'ppp2ts']


_hl = 40 * '-'
logger = logging.getLogger(__name__)


def prop2part(state_space, cont_props_dict):
    """

    Takes a domain `state_space` and a `list` of
    propositions `cont_props`, and returns
    a proposition-preserving partition of
    the state-space.


    Relevant
    ========
    `PropPreservingPartition`,
    `polytope.Polytope`


    @param state_space: problem domain
    @type state_space: `polytope.Polytope`
    @param cont_props_dict: propositions
    @type cont_props_dict: `dict` of `polytope.Polytope`
    @return: state-space quotient partition induced
        by propositions
    @rtype: `PropPreservingPartition`
    """
    first_poly = [] #Initial Region's polytopes
    first_poly.append(state_space)
    regions = [pc.Region(first_poly)]
    for cur_prop in cont_props_dict:
        cur_prop_poly = cont_props_dict[cur_prop]
        num_reg = len(regions)
        prop_holds_reg = []
        for i in range(num_reg): #i region counter
            region_now = regions[i].copy()
            #loop for prop holds
            prop_holds_reg.append(0)
            prop_now = regions[i].props.copy()
            dummy = region_now.intersect(cur_prop_poly)
            # does cur_prop hold in dummy ?
            if pc.is_fulldim(dummy):
                dum_prop = prop_now.copy()
                dum_prop.add(cur_prop)
                # is dummy a Polytope ?
                if len(dummy) == 0:
                    regions[i] = pc.Region([dummy], dum_prop)
                else:
                    # dummy is a Region
                    dummy.props = dum_prop.copy()
                    regions[i] = dummy.copy()
                prop_holds_reg[-1] = 1
            else:
                #does not hold in the whole region
                # (-> no need for the 2nd loop)
                regions.append(region_now)
                continue
            #loop for prop does not hold
            regions.append(pc.Region([], props=prop_now) )
            dummy = region_now.diff(cur_prop_poly)
            if pc.is_fulldim(dummy):
                dum_prop = prop_now.copy()
                # is dummy a Polytope ?
                if len(dummy) == 0:
                    regions[-1] = pc.Region(
                        [pc.reduce(dummy)],
                        dum_prop)
                else:
                    # dummy is a Region
                    dummy.props = dum_prop.copy()
                    regions[-1] = dummy.copy()
            else:
                regions.pop()
        count = 0
        for hold_count in range(len(prop_holds_reg)):
            if prop_holds_reg[hold_count]==0:
                regions.pop(hold_count-count)
                count+=1
    mypartition = PropPreservingPartition(
        domain = copy.deepcopy(state_space),
        regions = regions,
        prop_regions = copy.deepcopy(cont_props_dict))
    mypartition.adj = pc.find_adjacent_regions(mypartition).copy()
    return mypartition


def part2convex(ppp):
    """Refine partition so that cells be convex.

    Takes a proposition-preserving partition, and
    generates another proposition-preserving partition,
    such that each cell in the new partition is
    a convex polytope.

    @type ppp: `PropPreservingPartition`
    @return: refinement into convex polytopes and
        map from new to old Regions
    @rtype: (`PropPreservingPartition`, list)
    """
    cvxpart = PropPreservingPartition(
        domain=copy.deepcopy(ppp.domain),
        prop_regions=copy.deepcopy(ppp.prop_regions))
    new2old = []
    for i in range(len(ppp.regions)):
        simplified_reg = pc.union(
            ppp.regions[i],
            ppp.regions[i],
            check_convex=True)
        for j in range(len(simplified_reg)):
            region_now = pc.Region(
                [simplified_reg[j]],
                ppp.regions[i].props)
            cvxpart.regions.append(region_now)
            new2old += [i]
    cvxpart.adj = pc.find_adjacent_regions(cvxpart).copy()
    return (cvxpart, new2old)


def pwa_partition(pwa_sys, ppp, abs_tol=1e-5):
    """This function takes:

    - a piecewise affine system `pwa_sys` and
    - a proposition-preserving partition `ppp`
      whose domain is a subset of the domain of `pwa_sys`

    and returns a *refined* proposition preserving partition
    where in each region a unique subsystem of `pwa_sys` is active.


    Reference
    =========
    Modified from Petter Nilsson's code
    implementing merge algorithm in:

    Nilsson et al.
    `Temporal Logic Control of Switched Affine Systems with an
    Application in Fuel Balancing`, ACC 2012.


    Relevant
    ========
    `discretize`


    @type pwa_sys: `hybrid.PwaSysDyn`
    @type ppp: `PropPreservingPartition`
    @return: new partition and associated maps:
        - new partition `new_ppp`
        - map of `new_ppp.regions` to `pwa_sys.list_subsys`
        - map of `new_ppp.regions` to `ppp.regions`
    @rtype: `(PropPreservingPartition, list, list)`
    """
    if pc.is_fulldim(ppp.domain.diff(pwa_sys.domain)):
        raise Exception(
            'piecewise-affine system '
            'is not defined everywhere in '
            'the state-space')
    # for each subsystem's domain, cut it into pieces
    # each piece is the intersection with
    # a unique Region in ppp.regions
    new_list = []
    subsys_list = []
    parents = []
    for i, subsys in enumerate(pwa_sys.list_subsys):
        for j, region in enumerate(ppp.regions):
            isect = region.intersect(subsys.domain)
            if pc.is_fulldim(isect):
                rc, xc = pc.cheby_ball(isect)
                if rc < abs_tol:
                    warnings.warn(
                        'One of the regions in '
                        'the refined PPP is '
                        'too small, this may cause '
                        'numerical problems')
                # not Region yet, but Polytope ?
                if len(isect) == 0:
                    isect = pc.Region([isect])
                # label with AP
                isect.props = region.props.copy()
                # store new Region
                new_list.append(isect)
                # keep track of original Region in ppp.regions
                parents.append(j)
                # index of subsystem active within isect
                subsys_list.append(i)
    # compute spatial adjacency matrix
    n = len(new_list)
    adj = sp.lil_matrix((n, n), dtype=np.int8)
    for i, ri in enumerate(new_list):
        pi = parents[i]
        for j, rj in enumerate(new_list[0:i]):
            pj = parents[j]
            if (ppp.adj[pi, pj] == 1) or (pi == pj):
                if pc.is_adjacent(ri, rj):
                    adj[i, j] = 1
                    adj[j, i] = 1
        adj[i, i] = 1
    new_ppp = PropPreservingPartition(
        domain = ppp.domain,
        regions = new_list,
        adj = adj,
        prop_regions = ppp.prop_regions)
    return (new_ppp, subsys_list, parents)


def add_grid(
        ppp,
        grid_size=None,
        num_grid_pnts=None,
        abs_tol=1e-10):
    """Refine proposition-preserving partition using grids.

    This function takes a proposition-preserving
    partition `ppp`, and the size of the grid, or
    the number of grids, and returns a refined
    proposition-preserving partition with grids.

    Input:

    - `ppp`: a `PropPreservingPartition` object
    - `grid_size`: the size of the grid,
      - type: `float`,
        or `list of `float`
    - `num_grid_pnts`: the number of grids
      for each dimension,
      - type: `int` or `list` of `int`

    Output:

      - A `PropPreservingPartition` object with grids

    Note: There could be numerical instabilities when
    the continuous propositions in `ppp` do not align well with
    the grid, resulting in very small regions.
    Performace significantly degrades without `glpk`.
    """
    if grid_size != None and num_grid_pnts != None:
        raise Exception(
            'Only one of '
            'the grid size or number of '
            'grid points parameters is '
            'allowed to be given.')
    if grid_size == None and num_grid_pnts == None:
        raise Exception(
            'At least one of the '
            'grid size or number of '
            'grid points parameters '
            'must be given.')
    dim = len(ppp.domain.A[0])
    domain_bb = ppp.domain.bounding_box
    size_list = list()
    if grid_size != None:
        if isinstance(grid_size, list):
            if len(grid_size) == dim:
                size_list = grid_size
            else:
                raise Exception(
                    '`grid_size` is not '
                    'given in a correct format.')
        elif isinstance(grid_size, float):
            for i in range(dim):
                size_list.append(grid_size)
        else:
            raise Exception(
                'grid_size is not given in '
                'a correct format.')
    else:
        if isinstance(num_grid_pnts, list):
            if len(num_grid_pnts) == dim:
                for i in range(dim):
                    if isinstance( num_grid_pnts[i], int):
                        grid_size=(
                                float(domain_bb[1][i]) -
                                float(domain_bb[0][i])
                            ) / num_grid_pnts[i]
                        size_list.append(grid_size)
                    else:
                        raise Exception(
                            '`num_grid_pnts` is not '
                            'given in a correct format.')
            else:
                raise Exception(
                    '`num_grid_pnts` is not given '
                    'in a correct format.')
        elif isinstance(num_grid_pnts, int):
            for i in range(dim):
                grid_size=(
                        float(domain_bb[1][i]) -
                        float(domain_bb[0][i])
                    ) / num_grid_pnts
                size_list.append(grid_size)
        else:
            raise Exception(
                '`num_grid_pnts` is not given '
                'in a correct format.')
    j = 0
    list_grid = dict()
    while j < dim:
        list_grid[j] = compute_interval(
            float(domain_bb[0][j]),
            float(domain_bb[1][j]),
            size_list[j],
            abs_tol)
        if j > 0:
            if j == 1:
                re_list = list_grid[j - 1]
                re_list = product_interval(
                    re_list, list_grid[j])
            else:
                re_list = product_interval(
                    re_list, list_grid[j])
        j += 1
    new_list = []
    parent = []
    for i in range(len(re_list)):
        temp_list = list()
        j = 0
        while j < dim * 2:
            temp_list.append(
                [re_list[i][j],
                re_list[i][j + 1]])
            j = j + 2
        for j in range(len(ppp.regions)):
            tmp = pc.box2poly(temp_list)
            isect = tmp.intersect(ppp.regions[j], abs_tol)
            # if pc.is_fulldim(isect):
            rc, xc = pc.cheby_ball(isect)
            if rc > abs_tol / 2:
                if rc < abs_tol:
                    print(
                        'Warning: '
                        'One of the regions in '
                        'the refined PPP is too small'
                        ', this may cause numerical problems')
                if len(isect) == 0:
                    isect = pc.Region([isect], [])
                isect.props = ppp.regions[j].props.copy()
                new_list.append(isect)
                parent.append(j)
    adj = sp.lil_matrix(
        (len(new_list), len(new_list)),
        dtype=np.int8)
    for i in range(len(new_list)):
        adj[i, i] = 1
        for j in range(i + 1, len(new_list)):
            if (ppp.adj[parent[i], parent[j]] == 1 or
                    parent[i] == parent[j]):
                if pc.is_adjacent(new_list[i], new_list[j]):
                    adj[i, j] = 1
                    adj[j, i] = 1
    return PropPreservingPartition(
        domain=ppp.domain,
        regions=new_list,
        adj=adj,
        prop_regions=ppp.prop_regions)


#### Helper functions ####


def compute_interval(
        low_domain, high_domain, size,
        abs_tol=1e-7):
    """Compute intervals for each dimension."""
    list_g = list()
    i = low_domain
    while True:
        if (i + size + abs_tol) >= high_domain:
            list_g.append([i, high_domain])
            break
        else:
            list_g.append([i, i + size])
        i = i + size
    return list_g


def product_interval(list1, list2):
    """Combine all intervals, for any two interval lists."""
    new_list = list()
    for m in range(len(list1)):
        for n in range(len(list2)):
            new_list.append(list1[m] + list2[n])
    return new_list


class PropPreservingPartition(pc.MetricPartition):
    """Partition class with following fields:

    Attributes:

    - `domain`: the domain we want to partition
      - type: `Polytope`

    - `regions`: `Region`s of
      proposition-preserving partition
      - type: `list` of `Region`

    - `adj`: sparse matrix showing which
      regions are adjacent.
      The order of `Region`s is the
      same as in the `list` `regions`.

      - type: `scipy` `lil` `sparse`

    - `prop_regions`: map from atomic
      proposition symbols to continuous subsets

      - type: `dict` of `Polytope` or `Region`


    Relevant
    ========
    `prop2part`
    """
    def __init__(
            self,
            domain=None,
            regions=[],
            adj=None,
            prop_regions=None,
            check=True):
        # super().__init__(adj)
        if prop_regions is None:
            self.prop_regions = None
        else:
            try:
                # don't call it
                # use try because it should work
                # vs hasattr, which would look like normal selection
                prop_regions.keys
            except:
                raise TypeError(
                    '`prop_regions` must be `dict`.'
                    f'Got instead: {type(prop_regions)}')
                raise TypeError(msg)
            self.prop_regions = copy.deepcopy(prop_regions)
        n = len(regions)
        if hasattr(adj, 'shape'):
            m, k = adj.shape
            if m != k:
                raise ValueError('adj must be square')
            if m != n:
                msg = "adj size doesn't agree with number of regions"
                raise ValueError(msg)
        self.regions = regions[:]
        if check:
            for region in regions:
                if not region <= domain:
                    raise ValueError(
                        f'Partition: Region:\n\n{region}\n'
                        'is not subset of given domain:\n\t'
                        f'{domain}')
            self.is_symbolic()
        self.domain = domain
        super(PropPreservingPartition, self).__init__(domain)
        self.adj = adj

    def reg2props(self, region_index):
        return self.regions[region_index].props.copy()

    # TODO: iterator over pairs
    # TODO: use `networkx` graph to store partition

    def is_symbolic(self):
        """Check that the set of preserved predicates
        are bijectively mapped to the symbols.

        Symbols = Atomic Propositions
        """
        if self.prop_regions is None:
            logging.warning(
                'No continuous propositions defined.')
            return
        for region in self.regions:
            if region.props <= set(self.prop_regions):
                continue
            raise ValueError(
                'Partitions: Region labeled with propositions:\n\t'
                f'{region.props}\n'
                'not all of which are in the '
                'continuous atomic propositions:\n\t'
                f'{set(self.prop_regions)}')

    def preserves_predicates(self):
        """Return `True` if each `Region` <= Predicates for the
        predicates in `prop_regions.values`,
        where `prop_regions` is a bijection to
        "continuous" propositions of the specification's alphabet.

        Note
        ====
        1. `prop_regions` in practice need not be injective.
            It doesnt hurt - though creates unnecessary redundancy.

        2. The specification alphabet is fixed an user-defined.
            It should be distinguished from the auxiliary alphabet
            generated automatically during abstraction,
            which defines another partition with
            its own bijection to TS.
        """
        all_props = set(self.prop_regions)
        for region in self.regions:
            # Propositions True in Region
            for prop in region.props:
                preimage = self.prop_regions[prop]
                if not region <= preimage:
                    return False
            # Propositions False in Region
            for prop in all_props.difference(region.props):
                preimage = self.prop_regions[prop]
                if region.intersect(preimage).volume > pc.polytope.ABS_TOL:
                    return False
        return True

    def __str__(self):
        """Get informal string representation."""
        s = (
            f'\n{_hl}\n'
            'Proposition Preserving Partition:\n' +
            _hl + 2 * '\n' +
            f'Domain: {self.domain}\n')
        for j, region in enumerate(self.regions):
            s += f'Region: {j}\n'
            if self.prop_regions is not None:
                s += '\t Propositions: '
                active_props = ' '.join(region.props)
                if active_props:
                    s += f'{active_props}\n'
                else:
                    s += '{}\n'
            s += str(region)
        if hasattr(self.adj, 'toarray'):
            s += 'Adjacency matrix:\n'
            s += f'{self.adj.toarray()}\n'
        return s

    def plot(
            self,
            trans=None,
            ppp2trans=None,
            only_adjacent=False,
            ax=None,
            plot_numbers=True,
            color_seed=None):
        """For details see `polytope.plot.plot_partition`."""
        return plot_partition(
            self, trans,
            ppp2trans, only_adjacent,
            ax, plot_numbers, color_seed)

    def plot_props(self, ax=None, text_color='yellow'):
        """Plot labeled regions of continuous propositions."""
        try:
            from tulip.graphics import newax
        except:
            logger.error('failed to import graphics')
            return
        if ax is None:
            ax, fig = newax()
        l, u = self.domain.bounding_box
        ax.set_xlim(l[0,0], u[0,0])
        ax.set_ylim(l[1,0], u[1,0])
        for prop, poly in self.prop_regions.items():
            isect_poly = poly.intersect(self.domain)
            isect_poly.plot(
                ax,
                color='none',
                hatch='/')
            isect_poly.text(
                prop, ax,
                color=text_color)
        return ax


class PPP(PropPreservingPartition):
    """Alias to `PropPreservingPartition`.

    See that for details.
    """
    def __init__(self, **args):
        PropPreservingPartition.__init__(self, **args)


def ppp2ts(part):
    """Derive transition system from proposition preserving partition.

    @param part: labeled polytopic partition from
        which to derive the transition system
    @type part: `PropPreservingPartition`

    @return: `(ts, state_map)`
        finite transition system labeled with propositions
        from the given partition, and map of
        polytope indices to transition system states.

    @rtype: `(transys.FTS, dict)`
    """
    # generate transition system and add transitions
    ofts = trs.FTS()
    adj = part.adj  # `sp.lil_matrix`
    n = adj.shape[0]
    ofts_states = range(n)
    ofts_states = trs.prepend_with(ofts_states, 's')
    ofts.states.add_from(ofts_states)
    ofts.transitions.add_adj(adj, ofts_states)
    # decorate TS with state labels
    atomic_propositions = set(part.prop_regions)
    ofts.atomic_propositions.add_from(atomic_propositions)
    for state, region in zip(ofts_states, part.regions):
        state_prop = region.props.copy()
        ofts.states.add(state, ap=state_prop)
    return (ofts, ofts_states)
