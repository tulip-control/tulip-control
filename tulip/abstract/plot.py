# Copyright (c) 2011 by California Institute of Technology
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

Functions:
    - plot_partition
    - plot_trajectory
"""
import logging
logger = logging.getLogger(__name__)
from warnings import warn

import numpy as np
import networkx as nx

from tulip.polytope import cheby_ball, bounding_box

try:
    import matplotlib
except Exception, e:
    logger.error(e)
    matplotlib = None

try:
    from tulip.polytope.polytope import _get_patch
except Exception, e:
    logger.error(e)
    matplotlib = None

try:
    from tulip.graphics import newax
except Exception, e:
    logger.error(e)
    matplotlib = None

def plot_partition(ppp, trans=None, plot_numbers=True,
                   ax=None, color_seed=None, nodelist=None,
                   show=False):
    """Plots 2D PropPreservingPartition using matplotlib

    @type ppp: PropPreservingPartition
    
    @param trans: Transition matrix. If used,
        then transitions in C{ppp}are shown with arrows.
    
    @param plot_numbers: If True,
        then annotate each Region center with its number.
    
    @param show: If True, then show the plot.
        Otherwise return axis object.
        Axis object is good for creating custom plots.
    
    @param ax: axes where to plot
    
    @param color_seed: seed for reproducible random coloring
    
    @param nodelist: order mapping ppp indices to trans states
    @type nodelist: list of trans states
    
    @param show: call mpl.pyplot.show before returning
    
    see also
    --------
    abstract.prop2partition.PropPreservingPartition, plot_trajectory
    """
    if matplotlib is None:
        warn('matplotlib not found')
        return
    
    if isinstance(trans, nx.MultiDiGraph):
        if nodelist is None:
            n = len(trans.states)
            nodelist = ['s' +str(x) for x in xrange(n)]
        trans = np.array(nx.to_numpy_matrix(trans,
                         nodelist=nodelist) )
    
    l,u = bounding_box(ppp.domain)
    arr_size = (u[0,0]-l[0,0])/50.0
    reg_list = ppp.list_region
    
    # new figure ?
    if ax is None:
        ax, fig = newax()
    
    ax.set_xlim(l[0,0],u[0,0])
    ax.set_ylim(l[1,0],u[1,0])
    
    # repeatable coloring ?
    if color_seed is not None:
        prng = np.random.RandomState(color_seed)
    else:
        prng = np.random.RandomState()
    
    # plot polytope patches
    for i in xrange(len(reg_list)):
        reg = reg_list[i]
        
        # select random color,
        # same color for all polytopes in each region
        col = prng.rand(3)
        
        # single polytope or region ?
        if len(reg) == 0:
            ax.add_patch(_get_patch(reg, col) )
        else:
            for poly2 in reg.list_poly:
                ax.add_patch(_get_patch(poly2, col) )
    
    # plot transition arrows between patches
    for (i, reg) in enumerate(reg_list):
        rc, xc = cheby_ball(reg)
        
        
        if plot_numbers:
            ax.text(xc[0], xc[1], str(i), color='red')
        
        if trans is None:
            continue
        
        for j in np.nonzero(trans[:,i] )[0]:
            reg1 = reg_list[j]
            
            rc, xc = cheby_ball(reg)
            rc1, xc1 = cheby_ball(reg1)
            
            if np.sum(np.abs(xc1-xc)) < 1e-7:
                continue
            
            x = xc[0]
            y = xc[1]
            dx = xc1[0] - xc[0]
            dy = xc1[1] - xc[1]
            arr = matplotlib.patches.Arrow(
                float(x), float(y), float(dx), float(dy),
                width=arr_size, color='black'
            )
            ax.add_patch(arr)
    
    if show:
        matplotlib.pyplot.show()
    
    return ax
    
def plot_trajectory(ppp, x0, u_seq, ssys,
                    ax=None, color_seed=None):
    """Plots a PropPreservingPartition and the trajectory generated by x0
    input sequence u_seq.

    @type ppp: PropPreservingPartition
    @param x0: initial state
    @param u_seq: matrix where each row contains an input
    @param ssys: system dynamics
    @param show: if True, then show plot.
        Otherwise, return axis object
    @param color_seed: see polt_partition
    
    see also
    --------
    plot_partition, plot
    """
    if ax is None:
        ax, fig = newax()
    
    plot_partition(plot_numbers=False, ax=ax, show=False)
    
    A = ssys.A
    B = ssys.B
    
    if ssys.K != None:
        K = ssys.K
    else:
        K = np.zeros(x0.shape)
    
    x = x0.flatten()
    x_arr = x0
    for i in range(u_seq.shape[0]):
        x = np.dot(A, x).flatten() +\
            np.dot(B, u_seq[i, :] ).flatten() +\
            K.flatten()
        x_arr = np.vstack([x_arr, x.flatten()])
    
    ax.plot(x_arr[:,0], x_arr[:,1], 'o-')
    
    return ax
