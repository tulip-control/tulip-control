# Copyright (c) 2011-2015 by California Institute of Technology
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
Algorithms related to discretization of continuous dynamics.

See Also
========
L{find_controller}
"""
from __future__ import absolute_import

import logging
logger = logging.getLogger(__name__)

import os
import warnings
import pprint
from copy import deepcopy
import multiprocessing as mp

import numpy as np
from scipy import sparse as sp
import polytope as pc

from polytope.plot import plot_partition, plot_transition_arrow
from tulip import transys as trs
from tulip.hybrid import LtiSysDyn, PwaSysDyn

from .prop2partition import (PropPreservingPartition,
                             pwa_partition, part2convex)
from .feasible import is_feasible, solve_feasible
from .plot import plot_ts_on_partition

# inline imports:
#
# inline: import matplotlib.pyplot as plt

debug = False

class AbstractSwitched(object):
    """Abstraction of SwitchedSysDyn, with mode-specific and common info.

    Attributes:

      - ppp: merged partition, if any
          Preserves both propositions and dynamics

      - ts: common TS, if any

      - ppp2ts: map from C{ppp.regions} to C{ts.states}

      - modes: dict of {mode: AbstractPwa}

      - ppp2modes: map from C{ppp.regions} to C{modes[mode].ppp.regions}
        of the form:

        {mode: list}

        where C{list} has same indices as C{ppp.regions} and
        elements in each C{list} are indices of regions in
        each C{modes[mode].ppp.regions}.

        type: dict

    Each partition corresponds to some mode.
    (for switched systems)

    In each mode a L{PwaSysDyn} is active.
    """
    def __init__(
        self, ppp=None, ts=None, ppp2ts=None,
        modes=None, ppp2modes=None
    ):
        if modes is None:
            modes = dict()

        self.ppp = ppp
        self.ts = ts
        self.ppp2ts = ppp2ts
        self.modes = modes
        self.ppp2modes = ppp2modes

    def __str__(self):
        s = 'Abstraction of switched system\n'
        s += str('common PPP:\n') + str(self.ppp)
        s += str('common ts:\n') + str(self.ts)

        for mode, ab in self.modes.iteritems():
            s += 'mode: ' + str(mode)
            s += ', with abstraction:\n' + str(ab)

        return s

    def ppp2pwa(self, mode, i):
        """Return original C{Region} containing C{Region} C{i} in C{mode}.

        @param mode: key of C{modes}

        @param i: Region index in common partition C{ppp.regions}.

        @return: tuple C{(j, region)} of:

                - index C{j} of C{Region} and
                - C{Region} object

            in C{modes[mode].ppp.regions}
        """
        region_idx = self.ppp2modes[mode][i]
        ab = self.modes[mode]
        return ab.ppp2pwa(region_idx)

    def ppp2sys(self, mode, i):
        """Return index of active PWA subsystem in C{mode},

        @param mode: key of C{modes}

        @param i: Region index in common partition C{ppp.regions}.

        @return: tuple C{(j, subsystem)} of:

                - index C{j} of PWA C{subsystem}
                - L{LtiSysDyn} object C{subsystem}
        """
        region_idx = self.ppp2modes[mode][i]
        ab = self.modes[mode]
        return ab.ppp2sys(region_idx)

    def plot(self, show_ts=False, only_adjacent=False):
        """Plot mode partitions and merged partition, if one exists.

        For details see L{AbstractPwa.plot}.
        """
        axs = []
        color_seed = 0

        # merged partition exists ?
        if self.ppp is not None:
            for mode in self.modes:
                env_mode, sys_mode = mode
                edge_label = {'env_actions':env_mode,
                              'sys_actions':sys_mode}

                ax = _plot_abstraction(
                    self, show_ts=False, only_adjacent=False,
                    color_seed=color_seed
                )
                plot_ts_on_partition(
                    self.ppp, self.ts, self.ppp2ts,
                    edge_label, only_adjacent, ax
                )
                axs += [ax]

        # plot mode partitions
        for mode, ab in self.modes.iteritems():
            ax = ab.plot(show_ts, only_adjacent, color_seed)
            ax.set_title('Abstraction for mode: ' + str(mode))
            axs += [ax]

        #if isinstance(self.ts, dict):
        #    for ts in self.ts:
        #        ax = ts.plot()
        #        axs += [ax]
        return axs

class AbstractPwa(object):
    """Discrete abstraction of PWA dynamics, with attributes:

      - ppp: Partition into Regions.
          Each Region corresponds to
          a discrete state of the abstraction

          type: L{PropPreservingPartition}

      - ts: Finite transition system abstracting the continuous system.
          Each state corresponds to a Region in C{ppp.regions}.
          It can be fed into discrete synthesis algorithms.

          type: L{FTS}

      - ppp2ts: bijection between C{ppp.regions} and C{ts.states}.
          Has common indices with C{ppp.regions}.
          Elements are states in C{ts.states}.
          (usually each state is a str)

          type: list of states

      - pwa: system dynamics

          type: L{PwaSysDyn}

      - pwa_ppp: partition preserving both:

            - propositions and
            - domains of PWA subsystems

          Used for non-conservative planning.
          If just L{LtiSysDyn}, then the only difference
          of C{pwa_ppp} from C{orig_ppp} is convexification.

          type: L{PropPreservingPartition}

      - orig_ppp: partition preserving only propositions
          i.e., agnostic of dynamics

          type: L{PropPreservingPartition}

      - disc_params: parameters used in discretization that
          should be passed to the controller refinement
          to ensure consistency

          type: dict

    If any of the above is not given,
    then it is initialized to None.

    Notes
    =====
      1. There could be some redundancy in ppp and ofts,
         in that they are both decorated with propositions.
         This might be useful to keep each of
         them as functional units on their own
         (possible to change later).

      2. The 'Pwa' in L{AbstractPwa} includes L{LtiSysDyn}
         as a special case.
    """
    def __init__(
        self, ppp=None, ts=None, ppp2ts=None,
        pwa=None, pwa_ppp=None, ppp2pwa=None, ppp2sys=None,
        orig_ppp=None, ppp2orig=None,
        disc_params=None
    ):
        if disc_params is None:
            disc_params = dict()

        self.ppp = ppp
        self.ts = ts
        self.ppp2ts = ppp2ts

        self.pwa = pwa
        self.pwa_ppp = pwa_ppp
        self._ppp2pwa = ppp2pwa
        self._ppp2sys = ppp2sys

        self.orig_ppp = orig_ppp
        self._ppp2orig = ppp2orig

        # original_regions -> pwa_ppp
        # ppp2orig -> ppp2pwa_ppp
        # ppp2pwa -> ppp2pwa_sys

        self.disc_params = disc_params

    def __str__(self):
        s = str(self.ppp)
        s += str(self.ts)

        s += 30 * '-' + '\n'

        s += 'Map PPP Regions ---> TS states:\n'
        s += self._ppp2other_str(self.ppp2ts) + '\n'

        s += 'Map PPP Regions ---> PWA PPP Regions:\n'
        s += self._ppp2other_str(self._ppp2pwa) + '\n'

        s += 'Map PPP Regions ---> PWA Subsystems:\n'
        s += self._ppp2other_str(self._ppp2sys) + '\n'

        s += 'Map PPP Regions ---> Original PPP Regions:\n'
        s += self._ppp2other_str(self._ppp2orig) + '\n'

        s += 'Discretization Options:\n\t'
        s += pprint.pformat(self.disc_params) +'\n'

        return s

    def ts2ppp(self, state):
        region_index = self.ppp2ts.index(state)
        region = self.ppp[region_index]
        return (region_index, region)

    def ppp2trans(self, region_index):
        """Return the transition set constraint and active subsystem,

        for non-conservative planning.
        """
        reg_idx, pwa_region = self.ppp2pwa(region_index)
        sys_idx, sys = self.ppp2sys(region_index)
        return pwa_region, sys

    def ppp2pwa(self, region_index):
        """Return dynamics and predicate-preserving region
        and its index for PWA subsystem active in given region.

        The returned region is the C{trans_set} used for
        non-conservative planning.

        @param region_index: index in C{ppp.regions}.

        @rtype: C{(i, pwa.pwa_ppp[i])}
        """
        j = self._ppp2pwa[region_index]
        pwa_region = self.pwa_ppp[j]
        return (j, pwa_region)

    def ppp2sys(self, region_index):
        """Return index and PWA subsystem active in indexed region.

        Semantics: j-th sub-system is active in i-th Region,
        where C{j = ppp2pwa[i]}

        @param region_index: index in C{ppp.regions}.

        @rtype: C{(i, pwa.list_subsys[i])}
        """
        # LtiSysDyn ?
        if self._ppp2sys is None:
            return (0, self.pwa)

        subsystem_idx = self._ppp2sys[region_index]
        subsystem = self.pwa.list_subsys[subsystem_idx]
        return (subsystem_idx, subsystem)

    def ppp2orig(self, region_index):
        """Return index and region of original partition.

        The original partition is w/o any dynamics,
        not even the PWA domains, only the polytopic predicates.

        @param region_index: index in C{ppp.regions}.

        @rtype: C{(i, orig_ppp.regions[i])}
        """
        j = self._ppp2orig[region_index]
        orig_region = self.orig_ppp[j]
        return (j, orig_region)

    def _ppp2other_str(self, ppp2other):
        if ppp2other is None:
            return ''

        s = ''
        for i, other in enumerate(ppp2other):
            s += '\t\t' + str(i) + ' -> ' + str(other) + '\n'
        return s

    def _debug_str_(self):
        s = str(self.ppp)
        s += str(self.ts)

        s += '(PWA + Prop)-Preserving Partition'
        s += str(self.pwa_ppp)

        s += 'Original Prop-Preserving Partition'
        s += str(self.orig_ppp)
        return s

    def plot(self, show_ts=False, only_adjacent=False,
             color_seed=None):
        """Plot partition and optionally feasible transitions.

        @param show_ts: plot feasible transitions on partition
        @type show_ts: bool

        @param only_adjacent: plot feasible transitions only
            between adjacent regions. This reduces clutter,
            but if horizon > 1 and not all horizon used,
            then some transitions could be hidden.
        @param only_adjacent: bool
        """
        ax = _plot_abstraction(self, show_ts, only_adjacent,
                               color_seed)
        return ax

    def verify_transitions(self):
        logger.info('verifying transitions...')

        for from_state, to_state in self.ts.transitions():
            i, from_region = self.ts2ppp(from_state)
            j, to_region = self.ts2ppp(to_state)

            trans_set, sys = self.ppp2trans(i)

            params = {'N', 'close_loop', 'use_all_horizon'}
            disc_params = {k:v for k,v in self.disc_params.iteritems()
                           if k in params}

            s0 = solve_feasible(from_region, to_region, sys,
                                trans_set=trans_set, **disc_params)

            msg = str(i) + ' ---> ' + str(j)

            if not from_region <= s0:
                logger.error('incorrect transition: ' + msg)

                isect = from_region.intersect(s0)
                ratio = isect.volume /from_region.volume
                logger.error('intersection volume: ' + str(ratio) + ' %')
            else:
                logger.info('correct transition: ' + msg)

def _plot_abstraction(ab, show_ts, only_adjacent, color_seed):
    if ab.ppp is None or ab.ts is None:
        warnings.warn('Either ppp or ts is None.')
        return

    if show_ts:
        ts = ab.ts
        ppp2ts = ab.ppp2ts
    else:
        ts = None
        ppp2ts = None

    ax = ab.ppp.plot(
        ts, ppp2ts, only_adjacent=only_adjacent,
        color_seed=color_seed
    )

    #ax = self.ts.plot()

    return ax

def discretize(
    part, ssys, N=10, min_cell_volume=0.1,
    closed_loop=True, conservative=False,
    max_num_poly=5, use_all_horizon=False,
    trans_length=1, remove_trans=False,
    abs_tol=1e-7,
    plotit=False, save_img=False, cont_props=None,
    plot_every=1
):
    """Refine the partition and establish transitions
    based on reachability analysis.

    Reference
    =========
    U{[NOTM12]
    <http://tulip-control.sourceforge.net/doc/bibliography.html#notm12>}

    See Also
    ========
    L{prop2partition.pwa_partition}, L{prop2partition.part2convex}

    @param part: L{PropPreservingPartition} object
    @param ssys: L{LtiSysDyn} or L{PwaSysDyn} object
    @param N: horizon length
    @param min_cell_volume: the minimum volume of cells in the resulting
        partition.
    @param closed_loop: boolean indicating whether the `closed loop`
        algorithm should be used. default True.
    @param conservative: if true, force sequence in reachability analysis
        to stay inside starting cell. If false, safety
        is ensured by keeping the sequence inside a convexified
        version of the original proposition preserving cell.
    @param max_num_poly: maximum number of polytopes in a region to use in
        reachability analysis.
    @param use_all_horizon: in closed loop algorithm: if we should look
        for reachability also in less than N steps.
    @param trans_length: the number of polytopes allowed to cross in a
        transition.  a value of 1 checks transitions
        only between neighbors, a value of 2 checks
        neighbors of neighbors and so on.
    @param remove_trans: if True, remove found transitions between
        non-neighbors.
    @param abs_tol: maximum volume for an "empty" polytope

    @param plotit: plot partitioning as it evolves
    @type plotit: boolean,
        default = False

    @param save_img: save snapshots of partitioning to PDF files,
        requires plotit=True
    @type save_img: boolean,
        default = False

    @param cont_props: continuous propositions to plot
    @type cont_props: list of C{Polytope}

    @rtype: L{AbstractPwa}
    """
    start_time = os.times()[0]

    orig_ppp = part
    min_cell_volume = (min_cell_volume /np.finfo(np.double).eps
        *np.finfo(np.double).eps)

    ispwa = isinstance(ssys, PwaSysDyn)
    islti = isinstance(ssys, LtiSysDyn)

    if ispwa:
        (part, ppp2pwa, part2orig) = pwa_partition(ssys, part)
    else:
        part2orig = range(len(part))

    # Save original polytopes, require them to be convex
    if conservative:
        orig_list = None
        orig = [0]
    else:
        (part, new2old) = part2convex(part) # convexify
        part2orig = [part2orig[i] for i in new2old]

        # map new regions to pwa subsystems
        if ispwa:
            ppp2pwa = [ppp2pwa[i] for i in new2old]

        remove_trans = False # already allowed in nonconservative
        orig_list = []
        for poly in part:
            if len(poly) == 0:
                orig_list.append(poly.copy())
            elif len(poly) == 1:
                orig_list.append(poly[0].copy())
            else:
                raise Exception("discretize: "
                    "problem in convexification")
        orig = range(len(orig_list))

    # Cheby radius of disturbance set
    # (defined within the loop for pwa systems)
    if islti:
        if len(ssys.E) > 0:
            rd = ssys.Wset.chebR
        else:
            rd = 0.

    # Initialize matrix for pairs to check
    IJ = part.adj.copy()
    IJ = IJ.todense()
    IJ = np.array(IJ)
    logger.debug("\n Starting IJ: \n" + str(IJ) )

    # next line omitted in discretize_overlap
    IJ = reachable_within(trans_length, IJ,
                          np.array(part.adj.todense()) )

    # Initialize output
    num_regions = len(part)
    transitions = np.zeros(
        [num_regions, num_regions],
        dtype = int
    )
    sol = deepcopy(part.regions)
    adj = part.adj.copy()
    adj = adj.todense()
    adj = np.array(adj)

    # next 2 lines omitted in discretize_overlap
    if ispwa:
        subsys_list = list(ppp2pwa)
    else:
        subsys_list = None
    ss = ssys

    # init graphics
    if plotit:
        try:
            import matplotlib.pyplot as plt

            plt.ion()
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.axis('scaled')
            ax2.axis('scaled')
            file_extension = 'pdf'
        except:
            logger.error('failed to import matplotlib')
            plt = None
    else:
    	plt = None

    iter_count = 0

    # List of how many "new" regions
    # have been created for each region
    # and a list of original number of neighbors
    #num_new_reg = np.zeros(len(orig_list))
    #num_orig_neigh = np.sum(adj, axis=1).flatten() - 1

    progress = list()

    # Do the abstraction
    while np.sum(IJ) > 0:
        ind = np.nonzero(IJ)
        # i,j swapped in discretize_overlap
        i = ind[1][0]
        j = ind[0][0]
        IJ[j, i] = 0
        si = sol[i]
        sj = sol[j]

        si_tmp = deepcopy(si)
        sj_tmp = deepcopy(sj)

        #num_new_reg[i] += 1
        #print(num_new_reg)

        if ispwa:
            ss = ssys.list_subsys[subsys_list[i]]
            if len(ss.E) > 0:
                rd, xd = pc.cheby_ball(ss.Wset)
            else:
                rd = 0.

        if conservative:
            # Don't use trans_set
            trans_set = None
        else:
            # Use original cell as trans_set
            trans_set = orig_list[orig[i]]

        S0 = solve_feasible(
            si, sj, ss, N, closed_loop,
            use_all_horizon, trans_set, max_num_poly
        )

        msg = '\n Working with partition cells: ' + str(i) + ', ' + str(j)
        logger.info(msg)

        msg = '\t' + str(i) +' (#polytopes = ' +str(len(si) ) +'), and:\n'
        msg += '\t' + str(j) +' (#polytopes = ' +str(len(sj) ) +')\n'

        if ispwa:
            msg += '\t with active subsystem: '
            msg += str(subsys_list[i]) + '\n'

        msg += '\t Computed reachable set S0 with volume: '
        msg += str(S0.volume) + '\n'

        logger.debug(msg)

        #logger.debug('si \cap s0')
        isect = si.intersect(S0)
        vol1 = isect.volume
        risect, xi = pc.cheby_ball(isect)

        #logger.debug('si \ s0')
        diff = si.diff(S0)
        vol2 = diff.volume
        rdiff, xd = pc.cheby_ball(diff)

        # if pc.is_fulldim(pc.Region([isect]).intersect(diff)):
        #     logging.getLogger('tulip.polytope').setLevel(logging.DEBUG)
        #     diff = pc.mldivide(si, S0, save=True)
        #
        #     ax = S0.plot()
        #     ax.axis([0.0, 1.0, 0.0, 2.0])
        #     ax.figure.savefig('./img/s0.pdf')
        #
        #     ax = si.plot()
        #     ax.axis([0.0, 1.0, 0.0, 2.0])
        #     ax.figure.savefig('./img/si.pdf')
        #
        #     ax = isect.plot()
        #     ax.axis([0.0, 1.0, 0.0, 2.0])
        #     ax.figure.savefig('./img/isect.pdf')
        #
        #     ax = diff.plot()
        #     ax.axis([0.0, 1.0, 0.0, 2.0])
        #     ax.figure.savefig('./img/diff.pdf')
        #
        #     ax = isect.intersect(diff).plot()
        #     ax.axis([0.0, 1.0, 0.0, 2.0])
        #     ax.figure.savefig('./img/diff_cap_isect.pdf')
        #
        #     logger.error('Intersection \cap Difference != \emptyset')
        #
        #     assert(False)

        if vol1 <= min_cell_volume:
            logger.warning('\t too small: si \cap Pre(sj), ' +
                           'so discard intersection')
        if vol1 <= min_cell_volume and isect:
            logger.warning('\t discarded non-empty intersection: ' +
                           'consider reducing min_cell_volume')
        if vol2 <= min_cell_volume:
            logger.warning('\t too small: si \ Pre(sj), so not reached it')

        # We don't want our partitions to be smaller than the disturbance set
        # Could be a problem since cheby radius is calculated for smallest
        # convex polytope, so if we have a region we might throw away a good
        # cell.
        if (vol1 > min_cell_volume) and (risect > rd) and \
           (vol2 > min_cell_volume) and (rdiff > rd):

            # Make sure new areas are Regions and add proposition lists
            if len(isect) == 0:
                isect = pc.Region([isect], si.props)
            else:
                isect.props = si.props.copy()

            if len(diff) == 0:
                diff = pc.Region([diff], si.props)
            else:
                diff.props = si.props.copy()

            # replace si by intersection (single state)
            isect_list = pc.separate(isect)
            sol[i] = isect_list[0]

            # cut difference into connected pieces
            difflist = pc.separate(diff)

            difflist += isect_list[1:]
            n_isect = len(isect_list) -1

            num_new = len(difflist)

            # add each piece, as a new state
            for region in difflist:
                sol.append(region)

                # keep track of PWA subsystems map to new states
                if ispwa:
                    subsys_list.append(subsys_list[i])
            n_cells = len(sol)
            new_idx = xrange(n_cells-1, n_cells-num_new-1, -1)

            """Update transition matrix"""
            transitions = np.pad(transitions, (0,num_new), 'constant')

            transitions[i, :] = np.zeros(n_cells)
            for r in new_idx:
                #transitions[:, r] = transitions[:, i]
                # All sets reachable from start are reachable from both part's
                # except possibly the new part
                transitions[i, r] = 0
                transitions[j, r] = 0

            # sol[j] is reachable from intersection of sol[i] and S0
            if i != j:
                transitions[j, i] = 1

                # sol[j] is reachable from each piece os S0 \cap sol[i]
                #for k in xrange(n_cells-n_isect-2, n_cells):
                #    transitions[j, k] = 1

            """Update adjacency matrix"""
            old_adj = np.nonzero(adj[i, :])[0]

            # reset new adjacencies
            adj[i, :] = np.zeros([n_cells -num_new])
            adj[:, i] = np.zeros([n_cells -num_new])
            adj[i, i] = 1

            adj = np.pad(adj, (0, num_new), 'constant')

            for r in new_idx:
                adj[i, r] = 1
                adj[r, i] = 1
                adj[r, r] = 1

                if not conservative:
                    orig = np.hstack([orig, orig[i]])

            # adjacencies between pieces of isect and diff
            for r in new_idx:
                for k in new_idx:
                    if r is k:
                        continue

                    if pc.is_adjacent(sol[r], sol[k]):
                        adj[r, k] = 1
                        adj[k, r] = 1

            msg = ''
            if logger.getEffectiveLevel() <= logging.DEBUG:
                msg += '\t\n Adding states ' + str(i) + ' and '
                for r in new_idx:
                    msg += str(r) + ' and '
                msg += '\n'
                logger.debug(msg)

            for k in np.setdiff1d(old_adj, [i,n_cells-1]):
                # Every "old" neighbor must be the neighbor
                # of at least one of the new
                if pc.is_adjacent(sol[i], sol[k]):
                    adj[i, k] = 1
                    adj[k, i] = 1
                elif remove_trans and (trans_length == 1):
                    # Actively remove transitions between non-neighbors
                    transitions[i, k] = 0
                    transitions[k, i] = 0

                for r in new_idx:
                    if pc.is_adjacent(sol[r], sol[k]):
                        adj[r, k] = 1
                        adj[k, r] = 1
                    elif remove_trans and (trans_length == 1):
                        # Actively remove transitions between non-neighbors
                        transitions[r, k] = 0
                        transitions[k, r] = 0

            """Update IJ matrix"""
            IJ = np.pad(IJ, (0,num_new), 'constant')
            adj_k = reachable_within(trans_length, adj, adj)
            sym_adj_change(IJ, adj_k, transitions, i)

            for r in new_idx:
                sym_adj_change(IJ, adj_k, transitions, r)

            if logger.getEffectiveLevel() <= logging.DEBUG:
                msg = '\n\n Updated adj: \n' + str(adj)
                msg += '\n\n Updated trans: \n' + str(transitions)
                msg += '\n\n Updated IJ: \n' + str(IJ)
                logger.debug(msg)

            logger.info('Divided region: ' + str(i) + '\n')
        elif vol2 < abs_tol:
            logger.info('Found: ' + str(i) + ' ---> ' + str(j) + '\n')
            transitions[j,i] = 1
        else:
            if logger.level <= logging.DEBUG:
                msg = '\t Unreachable: ' + str(i) + ' --X--> ' + str(j) + '\n'
                msg += '\t\t diff vol: ' + str(vol2) + '\n'
                msg += '\t\t intersect vol: ' + str(vol1) + '\n'
                logger.debug(msg)
            else:
                logger.info('\t unreachable\n')
            transitions[j,i] = 0

        # check to avoid overlapping Regions
        if debug:
            tmp_part = PropPreservingPartition(
                domain=part.domain,
                regions=sol, adj=sp.lil_matrix(adj),
                prop_regions=part.prop_regions
            )
            assert(tmp_part.is_partition() )

        n_cells = len(sol)
        progress_ratio = 1 - float(np.sum(IJ) ) /n_cells**2
        progress += [progress_ratio]

        msg = '\t total # polytopes: ' + str(n_cells) + '\n'
        msg += '\t progress ratio: ' + str(progress_ratio) + '\n'
        logger.info(msg)

        iter_count += 1

        # no plotting ?
        if not plotit:
            continue
        if plt is None or plot_partition is None:
            continue
        if iter_count % plot_every != 0:
            continue

        tmp_part = PropPreservingPartition(
            domain=part.domain,
            regions=sol, adj=sp.lil_matrix(adj),
            prop_regions=part.prop_regions
        )

        # plot pair under reachability check
        ax2.clear()
        si_tmp.plot(ax=ax2, color='green')
        sj_tmp.plot(ax2, color='red', hatch='o', alpha=0.5)
        plot_transition_arrow(si_tmp, sj_tmp, ax2)

        S0.plot(ax2, color='none', hatch='/', alpha=0.3)
        fig.canvas.draw()

        # plot partition
        ax1.clear()
        plot_partition(tmp_part, transitions.T, ax=ax1, color_seed=23)

        # plot dynamics
        ssys.plot(ax1, show_domain=False)

        # plot hatched continuous propositions
        part.plot_props(ax1)

        fig.canvas.draw()

        # scale view based on domain,
        # not only the current polytopes si, sj
        l,u = part.domain.bounding_box
        ax2.set_xlim(l[0,0], u[0,0])
        ax2.set_ylim(l[1,0], u[1,0])

        if save_img:
            fname = 'movie' +str(iter_count).zfill(3)
            fname += '.' + file_extension
            fig.savefig(fname, dpi=250)
        plt.pause(1)

    new_part = PropPreservingPartition(
        domain=part.domain,
        regions=sol, adj=sp.lil_matrix(adj),
        prop_regions=part.prop_regions
    )

    # check completeness of adjacency matrix
    if debug:
        tmp_part = deepcopy(new_part)
        tmp_part.compute_adj()

    # Generate transition system and add transitions
    ofts = trs.FTS()

    adj = sp.lil_matrix(transitions.T)
    n = adj.shape[0]
    ofts_states = range(n)

    ofts.states.add_from(ofts_states)

    ofts.transitions.add_adj(adj, ofts_states)

    # Decorate TS with state labels
    atomic_propositions = set(part.prop_regions)
    ofts.atomic_propositions.add_from(atomic_propositions)
    for state, region in zip(ofts_states, sol):
        state_prop = region.props.copy()
        ofts.states.add(state, ap=state_prop)

    param = {
        'N':N,
        'trans_length':trans_length,
        'closed_loop':closed_loop,
        'conservative':conservative,
        'use_all_horizon':use_all_horizon,
        'min_cell_volume':min_cell_volume,
        'max_num_poly':max_num_poly
    }

    ppp2orig = [part2orig[x] for x in orig]

    end_time = os.times()[0]
    msg = 'Total abstraction time: ' +\
          str(end_time - start_time) + '[sec]'
    print(msg)
    logger.info(msg)

    if save_img and plt is not None:
        fig, ax = plt.subplots(1, 1)
        plt.plot(progress)
        ax.set_xlabel('iteration')
        ax.set_ylabel('progress ratio')
        ax.figure.savefig('progress.pdf')

    return AbstractPwa(
        ppp=new_part,
        ts=ofts,
        ppp2ts=ofts_states,
        pwa=ssys,
        pwa_ppp=part,
        ppp2pwa=orig,
        ppp2sys=subsys_list,
        orig_ppp=orig_ppp,
        ppp2orig=ppp2orig,
        disc_params=param
    )

def reachable_within(trans_length, adj_k, adj):
    """Find cells reachable within trans_length hops.
    """
    if trans_length <= 1:
        return adj_k

    k = 1
    while k < trans_length:
        adj_k = np.dot(adj_k, adj)
        k += 1
    adj_k = (adj_k > 0).astype(int)

    return adj_k

def sym_adj_change(IJ, adj_k, transitions, i):
    horizontal = adj_k[i, :] -transitions[i, :] > 0
    vertical = adj_k[:, i] -transitions[:, i] > 0

    IJ[i, :] = horizontal.astype(int)
    IJ[:, i] = vertical.astype(int)

# DEFUNCT until further notice
def discretize_overlap(closed_loop=False, conservative=False):
    """default False.

    UNDER DEVELOPMENT; function signature may change without notice.
    Calling will result in NotImplementedError.
    """
    raise NotImplementedError
#
#         if rdiff < abs_tol:
#             logger.info("Transition found")
#             transitions[i,j] = 1
#
#         elif (vol1 > min_cell_volume) & (risect > rd) & \
#                 (num_new_reg[i] <= num_orig_neigh[i]+1):
#
#             # Make sure new cell is Region and add proposition lists
#             if len(isect) == 0:
#                 isect = pc.Region([isect], si.props)
#             else:
#                 isect.props = si.props.copy()
#
#             # Add new state
#             sol.append(isect)
#             size = len(sol)
#
#             # Add transitions
#             transitions = np.hstack([transitions, np.zeros([size - 1, 1],
#                                     dtype=int) ])
#             transitions = np.vstack([transitions, np.zeros([1, size],
#                                     dtype=int) ])
#
#             # All sets reachable from orig cell are reachable from both cells
#             transitions[size-1,:] = transitions[i,:]
#             transitions[size-1,j] = 1   # j is reachable from new cell
#
#             # Take care of adjacency
#             old_adj = np.nonzero(adj[i,:])[0]
#
#             adj = np.hstack([adj, np.zeros([size - 1, 1], dtype=int) ])
#             adj = np.vstack([adj, np.zeros([1, size], dtype=int) ])
#             adj[i,size-1] = 1
#             adj[size-1,i] = 1
#             adj[size-1,size-1] = 1
#
#             for k in np.setdiff1d(old_adj,[i,size-1]):
#                 if pc.is_adjacent(sol[size-1],sol[k],overlap=True):
#                     adj[size-1,k] = 1
#                     adj[k, size-1] = 1
#                 else:
#                     # Actively remove (valid) transitions between non-neighbors
#                     transitions[size-1,k] = 0
#                     transitions[k,size-1] = 0
#
#             # Assign original proposition cell to new state and update counts
#             if not conservative:
#                 orig = np.hstack([orig, orig[i]])
#             print(num_new_reg)
#             num_new_reg = np.hstack([num_new_reg, 0])
#             num_orig_neigh = np.hstack([num_orig_neigh, np.sum(adj[size-1,:])-1])
#
#             logger.info("\n Adding state " + str(size-1) + "\n")
#
#             # Just add adjacent cells for checking,
#             # unless transition already found
#             IJ = np.hstack([IJ, np.zeros([size - 1, 1], dtype=int) ])
#             IJ = np.vstack([IJ, np.zeros([1, size], dtype=int) ])
#             horiz2 = adj[size-1,:] - transitions[size-1,:] > 0
#             verti2 = adj[:,size-1] - transitions[:,size-1] > 0
#             IJ[size-1,:] = horiz2.astype(int)
#             IJ[:,size-1] = verti2.astype(int)
#         else:
#             logger.info("No transition found, intersect vol: " + str(vol1) )
#             transitions[i,j] = 0
#
#     new_part = PropPreservingPartition(
#                    domain=part.domain,
#                    regions=sol, adj=np.array([]),
#                    trans=transitions, prop_regions=part.prop_regions,
#                    original_regions=orig_list, orig=orig)
#     return new_part

def multiproc_discretize(q, mode, ppp, cont_dyn, disc_params):
    global logger
    logger = mp.log_to_stderr()

    name = mp.current_process().name
    print('Abstracting mode: ' + str(mode) + ', on: ' + str(name))

    absys = discretize(ppp, cont_dyn, **disc_params)

    q.put((mode, absys))
    print('Worker: ' + str(name) + 'finished.')

def multiproc_get_transitions(
    q, absys, mode, ssys, params
):
    global logger
    logger = mp.log_to_stderr()

    name = mp.current_process().name
    print('Merged transitions for mode: ' + str(mode) + ', on: ' + str(name))

    trans = get_transitions(absys, mode, ssys, **params)

    q.put((mode, trans))
    print('Worker: ' + str(name) + 'finished.')

def multiproc_discretize_switched(
    ppp, hybrid_sys, disc_params=None,
    plot=False, show_ts=False, only_adjacent=True
):
    """Parallel implementation of discretize_switched.

    Uses the multiprocessing package.
    """
    logger.info('parallel discretize_switched started')

    modes = hybrid_sys.modes
    mode_nums = hybrid_sys.disc_domain_size

    q = mp.Queue()

    mode_args = dict()
    for mode in modes:
        cont_dyn = hybrid_sys.dynamics[mode]
        mode_args[mode] = (q, mode, ppp, cont_dyn, disc_params[mode])

    jobs = [mp.Process(target=multiproc_discretize, args=args)
            for args in mode_args.itervalues()]
    for job in jobs:
        job.start()

    # flush before join:
    #   http://stackoverflow.com/questions/19071529/
    abstractions = dict()
    for job in jobs:
        mode, absys = q.get()
        abstractions[mode] = absys

    for job in jobs:
        job.join()

    # merge their domains
    (merged_abstr, ap_labeling) = merge_partitions(abstractions)
    n = len(merged_abstr.ppp)
    logger.info('Merged partition has: ' + str(n) + ', states')

    # find feasible transitions over merged partition
    for mode in modes:
        cont_dyn = hybrid_sys.dynamics[mode]
        params = disc_params[mode]

        mode_args[mode] = (q, merged_abstr, mode, cont_dyn, params)

    jobs = [mp.Process(target=multiproc_get_transitions, args=args)
            for args in mode_args.itervalues()]

    for job in jobs:
        job.start()

    trans = dict()
    for job in jobs:
        mode, t = q.get()
        trans[mode] = t

    # merge the abstractions, creating a common TS
    merge_abstractions(merged_abstr, trans,
                       abstractions, modes, mode_nums)

    if plot:
        plot_mode_partitions(merged_abstr, show_ts, only_adjacent)

    return merged_abstr

def discretize_switched(
    ppp, hybrid_sys, disc_params=None,
    plot=False, show_ts=False, only_adjacent=True
):
    """Abstract switched dynamics over given partition.

    @type ppp: L{PropPreservingPartition}

    @param hybrid_sys: dynamics of switching modes
    @type hybrid_sys: L{SwitchedSysDyn}

    @param disc_params: discretization parameters passed to L{discretize} for
		each mode. See L{discretize} for details.
    @type disc_params: dict (keyed by mode) of dicts.

    @param plot: save partition images
    @type plot: bool

    @param show_ts, only_adjacent: options for L{AbstractPwa.plot}.

    @return: abstracted dynamics,
        some attributes are dict keyed by mode
    @rtype: L{AbstractSwitched}
    """
    if disc_params is None:
        disc_params = {'N':1, 'trans_length':1}

    logger.info('discretizing hybrid system')

    modes = hybrid_sys.modes
    mode_nums = hybrid_sys.disc_domain_size

    # discretize each abstraction separately
    abstractions = dict()
    for mode in modes:
        logger.debug(30*'-'+'\n')
        logger.info('Abstracting mode: ' + str(mode))

        cont_dyn = hybrid_sys.dynamics[mode]

        absys = discretize(
            ppp, cont_dyn,
            **disc_params[mode]
        )
        logger.debug('Mode Abstraction:\n' + str(absys) +'\n')

        abstractions[mode] = absys

    # merge their domains
    (merged_abstr, ap_labeling) = merge_partitions(abstractions)
    n = len(merged_abstr.ppp)
    logger.info('Merged partition has: ' + str(n) + ', states')

    # find feasible transitions over merged partition
    trans = dict()
    for mode in modes:
        cont_dyn = hybrid_sys.dynamics[mode]

        params = disc_params[mode]

        trans[mode] = get_transitions(
            merged_abstr, mode, cont_dyn,
            N=params['N'], trans_length=params['trans_length']
        )

    # merge the abstractions, creating a common TS
    merge_abstractions(merged_abstr, trans,
                       abstractions, modes, mode_nums)

    if plot:
        plot_mode_partitions(merged_abstr, show_ts, only_adjacent)

    return merged_abstr

def plot_mode_partitions(swab, show_ts, only_adjacent):
    """Save each mode's partition and final merged partition.
    """
    axs = swab.plot(show_ts, only_adjacent)

    if not axs:
        logger.error('failed to plot the partitions.')
        return

    n = len(swab.modes)
    assert(len(axs) == 2*n)

    # annotate
    for ax in axs:
        plot_annot(ax)

    # save mode partitions
    for ax, mode in zip(axs[:n], swab.modes):
        fname = 'merged_' + str(mode) + '.pdf'
        ax.figure.savefig(fname)

    # save merged partition
    for ax, mode in zip(axs[n:], swab.modes):
        fname = 'part_' + str(mode) + '.pdf'
        ax.figure.savefig(fname)

def plot_annot(ax):
    fontsize = 5

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    ax.set_xlabel('$v_1$', fontsize=fontsize+6)
    ax.set_ylabel('$v_2$', fontsize=fontsize+6)

def merge_abstractions(merged_abstr, trans, abstr, modes, mode_nums):
    """Construct merged transitions.

    @type merged_abstr: L{AbstractSwitched}
    @type abstr: dict of L{AbstractPwa}
    """
    # TODO: check equality of atomic proposition sets
    aps = abstr[modes[0]].ts.atomic_propositions

    logger.info('APs: ' + str(aps))

    sys_ts = trs.FTS()

    # create stats
    n = len(merged_abstr.ppp)
    states = range(n)
    sys_ts.states.add_from(states)

    sys_ts.atomic_propositions.add_from(aps)

    # copy AP labels from regions to discrete states
    ppp2ts = states
    for (i, state) in enumerate(ppp2ts):
        props =  merged_abstr.ppp[i].props
        sys_ts.states[state]['ap'] = props

    # create mode actions
    sys_actions = [str(s) for e,s in modes]
    env_actions = [str(e) for e,s in modes]

    # no env actions ?
    if mode_nums[0] == 0:
        actions_per_mode = {
            (e,s):{'sys_actions':str(s)}
            for e,s in modes
        }
        sys_ts.sys_actions.add_from(sys_actions)
    elif mode_nums[1] == 0:
        # no sys actions
        actions_per_mode = {
            (e,s):{'env_actions':str(e)}
            for e,s in modes
        }
        sys_ts.env_actions.add_from(env_actions)
    else:
        actions_per_mode = {
            (e,s):{'env_actions':str(e), 'sys_actions':str(s)}
            for e,s in modes
        }
        sys_ts.env_actions.add_from([str(e) for e,s in modes])
        sys_ts.sys_actions.add_from([str(s) for e,s in modes])

    for mode in modes:
        env_sys_actions = actions_per_mode[mode]
        adj = trans[mode]

        sys_ts.transitions.add_adj(
            adj = adj,
            adj2states = states,
            **env_sys_actions
        )

    merged_abstr.ts = sys_ts
    merged_abstr.ppp2ts = ppp2ts

def get_transitions(
    abstract_sys, mode, ssys, N=10,
    closed_loop=True,
    trans_length=1
):
    """Find which transitions are feasible in given mode.

    Used for the candidate transitions of the merged partition.

    @rtype: scipy.sparse.lil_matrix
    """
    logger.info('checking which transitions remain feasible after merging')
    part = abstract_sys.ppp

    # Initialize matrix for pairs to check
    IJ = part.adj.copy()
    if trans_length > 1:
        k = 1
        while k < trans_length:
            IJ = np.dot(IJ, part.adj)
            k += 1
        IJ = (IJ > 0).astype(int)

    # Initialize output
    n = len(part)
    transitions = sp.lil_matrix((n, n), dtype=int)

    # Do the abstraction
    n_checked = 0
    n_found = 0
    while np.sum(IJ) > 0:
        n_checked += 1

        ind = np.nonzero(IJ)
        i = ind[1][0]
        j = ind[0][0]
        IJ[j,i] = 0

        logger.debug('checking transition: ' + str(i) + ' -> ' + str(j))

        si = part[i]
        sj = part[j]

        # Use original cell as trans_set
        trans_set = abstract_sys.ppp2pwa(mode, i)[1]
        active_subsystem = abstract_sys.ppp2sys(mode, i)[1]

        trans_feasible = is_feasible(
            si, sj, active_subsystem, N,
            closed_loop = closed_loop,
            trans_set = trans_set
        )

        if trans_feasible:
            transitions[i, j] = 1
            msg = '\t Feasible transition.'
            n_found += 1
        else:
            transitions[i, j] = 0
            msg = '\t Not feasible transition.'
        logger.debug(msg)
    logger.info('Checked: ' + str(n_checked))
    logger.info('Found: ' + str(n_found))
    logger.info('Survived merging: ' + str(float(n_found) / n_checked) + ' % ')

    return transitions

def multiproc_merge_partitions(abstractions):
    """LOGTIME in #processors parallel merging.

    Assuming sufficient number of processors.

    UNDER DEVELOPMENT; function signature may change without notice.
    Calling will result in NotImplementedError.
    """
    raise NotImplementedError

def merge_partitions(abstractions):
    """Merge multiple abstractions.

    @param abstractions: keyed by mode
    @type abstractions: dict of L{AbstractPwa}

    @return: (merged_abstraction, ap_labeling)
        where:
            - merged_abstraction: L{AbstractSwitched}
            - ap_labeling: dict
    """
    if len(abstractions) == 0:
        warnings.warn('Abstractions empty, nothing to merge.')
        return

    # consistency check
    for ab1 in abstractions.itervalues():
        for ab2 in abstractions.itervalues():
            p1 = ab1.ppp
            p2 = ab2.ppp

            if p1.prop_regions != p2.prop_regions:
                msg = 'merge: partitions have different sets '
                msg += 'of continuous propositions'
                raise Exception(msg)

            if not (p1.domain.A == p2.domain.A).all() or \
            not (p1.domain.b == p2.domain.b).all():
                raise Exception('merge: partitions have different domains')

            # check equality of original PPP partitions
            if ab1.orig_ppp == ab2.orig_ppp:
                logger.info('original partitions happen to be equal')

    init_mode = abstractions.keys()[0]
    all_modes = set(abstractions)
    remaining_modes = all_modes.difference(set([init_mode]))

    print('init mode: ' + str(init_mode))
    print('all modes: ' + str(all_modes))
    print('remaining modes: ' + str(remaining_modes))

    # initialize iteration data
    prev_modes = [init_mode]

   	# Create a list of merged-together regions
    ab0 = abstractions[init_mode]
    regions = list(ab0.ppp)
    parents = {init_mode:range(len(regions) )}
    ap_labeling = {i:reg.props for i,reg in enumerate(regions)}
    for cur_mode in remaining_modes:
        ab2 = abstractions[cur_mode]
        r = merge_partition_pair(
            regions, ab2, cur_mode, prev_modes,
            parents, ap_labeling
        )
        regions, parents, ap_labeling = r
        prev_modes += [cur_mode]
    new_list = regions

    # build adjacency based on spatial adjacencies of
    # component abstractions.
    # which justifies the assumed symmetry of part1.adj, part2.adj
	# Basically, if two regions are either 1) part of the same region in one of
	# the abstractions or 2) adjacent in one of the abstractions, then the two
	# regions are adjacent in the switched dynamics.
    n_reg = len(new_list)

    adj = np.zeros([n_reg, n_reg], dtype=int)
    for i, reg_i in enumerate(new_list):
        for j, reg_j in enumerate(new_list[0:i]):
            touching = False
            for mode in abstractions:
                pi = parents[mode][i]
                pj = parents[mode][j]

                part = abstractions[mode].ppp

                if (part.adj[pi, pj] == 1) or (pi == pj):
                    touching = True
                    break

            if not touching:
                continue

            if pc.is_adjacent(reg_i, reg_j):
                adj[i,j] = 1
                adj[j,i] = 1
        adj[i,i] = 1

    ppp = PropPreservingPartition(
        domain=ab0.ppp.domain,
        regions=new_list,
        prop_regions=ab0.ppp.prop_regions,
        adj=adj
    )

    abstraction = AbstractSwitched(
        ppp=ppp,
        modes=abstractions,
        ppp2modes=parents,
    )

    return (abstraction, ap_labeling)

def merge_partition_pair(
    old_regions, ab2,
    cur_mode, prev_modes,
    old_parents, old_ap_labeling
):
    """Merge an Abstraction with the current partition iterate.

    @param old_regions: A list of C{Region} that is from either:
        1. The ppp of the first (initial) L{AbstractPwa} to be merged.
        2. A list of already-merged regions
    @type old_regions: list of C{Region}

    @param ab2: Abstracted piecewise affine dynamics to be merged into the
    @type ab2: L{AbstractPwa}

    @param cur_mode: mode to be merged
    @type cur_mode: tuple

    @param prev_modes: list of modes that have already been merged together
    @type prev_modes: list of tuple

    @param old_parents: dict of modes that have already been merged to dict of
        indices of new regions to indices of regions
    @type old_parents: dict of modes to list of region indices in list
        C{old_regions} or dict of region indices to regions in original ppp for
        that mode

    @param old_ap_labeling: dict of states of already-merged modes to sets of
        propositions for each state
    @type old_ap_labeling: dict of tuples to sets

    @return: the following:
        - C{new_list}, list of new regions
        - C{parents}, same as input param C{old_parents}, except that it
          includes the mode that was just merged and for list of regions in
          return value C{new_list}
        - C{ap_labeling}, same as input param C{old_ap_labeling}, except that it
          includes the mode that was just merged.
    """
    logger.info('merging partitions')

    part2 = ab2.ppp

    modes = prev_modes + [cur_mode]

    new_list = []
    parents = {mode:dict() for mode in modes}
    ap_labeling = dict()

    for i in xrange(len(old_regions)):
        for j in xrange(len(part2)):
            isect = pc.intersect(old_regions[i],
                                 part2[j])
            rc, xc = pc.cheby_ball(isect)

            # no intersection ?
            if rc < 1e-5:
                continue
            logger.info('merging region: A' + str(i) +
                        ', with: B' + str(j))

            # if Polytope, make it Region
            if len(isect) == 0:
                isect = pc.Region([isect])

            # label the Region with propositions
            isect.props = old_regions[i].props.copy()

            new_list.append(isect)
            idx = new_list.index(isect)

            # keep track of parents
            for mode in prev_modes:
                parents[mode][idx] = old_parents[mode][i]
            parents[cur_mode][idx] = j

            # union of AP labels from parent states
            ap_label_1 = old_ap_labeling[i]
            ap_label_2 = ab2.ts.states[j]['ap']

            logger.debug('AP label 1: ' + str(ap_label_1))
            logger.debug('AP label 2: ' + str(ap_label_2))

            # original partitions may be different if pwa_partition used
            # but must originate from same initial partition,
            # i.e., have same continuous propositions, checked above
            #
            # so no two intersecting regions can have different AP labels,
            # checked here
            if ap_label_1 != ap_label_2:
                msg = 'Inconsistent AP labels between intersecting regions\n'
                msg += 'of partitions of switched system.'
                raise Exception(msg)

            ap_labeling[idx] = ap_label_1

    return new_list, parents, ap_labeling
