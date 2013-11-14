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
Functions for plotting Polytopes and Partitions.
The functions can be accessed by

> from tulip.polytope.plot import *

Functions:

    - get_patch
    - plot
    - plot_partition
    - plot_trajectory
"""
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from polytope import dimension, extreme, cheby_ball, bounding_box, is_fulldim

def get_patch(poly1, color="blue"):
    """Takes a Polytope and returns a Matplotlib Patch Polytope 
    that can be added to a plot
    
    Example::

    > # Plot Polytope objects poly1 and poly2 in the same plot
    > import matplotlib.pyplot as plt
    > fig = plt.figure()
    > ax = fig.add_subplot(111)
    > p1 = get_patch(poly1, color="blue")
    > p2 = get_patch(poly2, color="yellow")
    > ax.add_patch(p1)
    > ax.add_patch(p2)
    > ax.set_xlim(xl, xu) # Optional: set axis max/min
    > ax.set_ylim(yl, yu) 
    > plt.show()
    """
    V = extreme(poly1)
    rc,xc = cheby_ball(poly1)
    x = V[:,1] - xc[1]
    y = V[:,0] - xc[0]
    mult = np.sqrt(x**2 + y**2)
    x = x/mult
    angle = np.arccos(x)
    corr = np.ones(y.size) - 2*(y < 0)
    angle = angle*corr
    ind = np.argsort(angle) 

    patch = matplotlib.patches.Polygon(V[ind,:], True, color=color)
    return patch
    
def plot(poly1, show=True):
    """Plots 2D polytope or region using matplotlib.
    
    @type: poly1: Polytope or Region
    """
    if not is_fulldim(poly1):
        print("Cannot plot empty polytope")
        return
    
    if dimension(poly1) != 2:
        print("Cannot plot polytopes of dimension larger than 2")
        return
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    if len(poly1) == 0:
        poly = get_patch(poly1)
        ax.add_patch(poly)
    else:
        for poly2 in poly1.list_poly:
            poly = get_patch(poly2, color=np.random.rand(3))
            ax.add_patch(poly)
    
    l,u = bounding_box(poly1)
    ax.set_xlim(l[0,0],u[0,0])
    ax.set_ylim(l[1,0],u[1,0])
    if show:
        plt.show()

def plot_partition(ppp, trans=None, plot_numbers=True,
                   show=True, ax=None, color_seed=None):
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
    
    see also
    --------
    abstract.prop2part.PropPreservingPartition
    """
    l,u = bounding_box(ppp.domain)
    arr_size = (u[0,0]-l[0,0])/50.0
    reg_list = ppp.list_region
    
    # new figure ?
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.set_xlim(l[0,0],u[0,0])
    ax.set_ylim(l[1,0],u[1,0])
    
    # repeatable coloring ?
    if color_seed is not None:
        prng = np.random.RandomState(color_seed)
    
    # plot polytope patches
    for i in xrange(len(reg_list)):
        reg = reg_list[i]
        
        # select random color,
        # same color for all polytopes in each region
        col = prng.rand(3)
        
        # single polytope or region ?
        if len(reg) == 0:
            ax.add_patch(get_patch(reg, col) )
        else:
            for poly2 in reg.list_poly:
                ax.add_patch(get_patch(poly2, col) )
    
    # plot transition arrows between patches
    for i in xrange(len(reg_list) ):
        reg = reg_list[i]
        rc, xc = cheby_ball(reg)
        
        if trans is not None:
            for j in np.nonzero(trans[:,i] )[0]:
                reg1 = reg_list[j]
                rc1, xc1 = cheby_ball(reg1)
                
                if not np.sum(np.abs(xc1-xc)) < 1e-7:
                    x = xc[0]
                    y = xc[1]
                    dx = xc1[0] - xc[0]
                    dy = xc1[1] - xc[1]
                    arr = matplotlib.patches.Arrow(
                        float(x), float(y), float(dx), float(dy),
                        width=arr_size, color='black'
                    )
                    ax.add_patch(arr)
        if plot_numbers:
            ax.text(xc[0], xc[1], str(i),color='red')            
    
    if show:
        plt.show()
    else:
        return ax
    
def plot_trajectory(ppp, x0, u_seq, ssys, show):
    """Plots a PropPreservingPartition and the trajectory generated by x0
    input sequence u_seq.

    @type ppp: PropPreservingPartition
    @param x0: initial state
    @param u_seq: matrix where each row contains an input
    @param ssys: system dynamics
    @param show: if True, then show plot.
        Otherwise, return axis object
    """
    ax = plot_partition(plot_numbers=False, show=False)    
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
    
    if show:
        plt.show()
