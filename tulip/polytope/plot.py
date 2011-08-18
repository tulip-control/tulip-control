#
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

"""
Functions for plotting Polytopes and Partitions. The functions
can be accessed by

> from tulip.polytope.plot import *

Functions: 
    - get_patch
	- plot
	- plot_partition
	- plot_trajectory

Created by Petter Nilsson, 8/4/11
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from polytope import dimension, extreme, cheby_ball, bounding_box, is_fulldim

def get_patch(poly1, color="blue"):
    """Takes a Polytope and returns a Matplotlib Patch Polytope 
    that can be added to a plot
    
    Example:
    > # Plot Polytope objects poly1 and poly2 in the same plot
    > import matploytlib.pyplot as plt
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
    
def plot(poly1):    
    """Plots a 2D polytope or a region using matplotlib.
    
    Input:
    - `poly1`: Polytope or Region
    """
    if is_fulldim(poly1):
        
        if dimension(poly1) != 2:
            print "Can not plot polytopes of dimension larger than 2"
            return
        
        if len(poly1) == 0:
    
            poly = get_patch(poly1)
            l,u = bounding_box(poly1)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.add_patch(poly)        
        
            ax.set_xlim(l[0,0],u[0,0])
            ax.set_ylim(l[1,0],u[1,0])
            plt.show()
        
        else:
            l,u = bounding_box(poly1)
        
            fig = plt.figure()
            ax = fig.add_subplot(111)
                
            for poly2 in poly1.list_poly:
                poly = get_patch(poly2, color=np.random.rand(3))
                ax.add_patch(poly)
        
            ax.set_xlim(l[0,0],u[0,0])
            ax.set_ylim(l[1,0],u[1,0])
            plt.show()
    else:
        print "Cannot plot empty polytope"
        
def plot_partition(ppp, plot_transitions=False, plot_numbers=True, show=True):
    """Plots a 2D PropPreservingPartition object using matplotlib
    
    Input:
    - `ppp`: A PropPreservingPartition object
    - `plot_transitions`: If True, represent transitions in `ppp` with arrows.
                          Requires transitions to be stored in `ppp`.
    - `plot_numbers`: If True, plot the number of the Region in the center of 
                      each Region.
    - `show`: If True, show the plot. Otherwise return axis object. Axis object is good
              for creating custom plots.
    """

    l,u = bounding_box(ppp.domain)
    arr_size = (u[0,0]-l[0,0])/50.0
    reg_list = ppp.list_region
    trans = ppp.trans
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(l[0,0],u[0,0])
    ax.set_ylim(l[1,0],u[1,0])

    for i in range(len(reg_list)):
        reg = reg_list[i]        
        if len(reg) == 0:
            ax.add_patch(get_patch(reg, np.random.rand(3)))      
        else:
            col = np.random.rand(3)
            for poly2 in reg.list_poly:  
                ax.add_patch(get_patch(poly2, color=col))

    for i in range(len(reg_list)):
        reg = reg_list[i]
        rc, xc = cheby_ball(reg)
        if plot_transitions:
            for j in np.nonzero(trans[:,i])[0]:
                reg1 = reg_list[j]
                rc1,xc1 = cheby_ball(reg1)
                if not np.sum(np.abs(xc1-xc)) < 1e-7:
                    x = xc[0]
                    y = xc[1]
                    dx = xc1[0] - xc[0]
                    dy = xc1[1] - xc[1]
                    arr = matplotlib.patches.Arrow(float(x),float(y),float(dx),float(dy),width=arr_size,color='black')
                    ax.add_patch(arr)
        if plot_numbers:
            ax.text(xc[0], xc[1], str(i),color='red')            
    
    if show:
        plt.show()
    else:
        return ax
    
def plot_trajectory(ppp, x0, u_seq, ssys):
    """Plots a PropPreservingPartition and the trajectory generated by x0
    input sequence u_seq.
    
    Input:
    - `ppp`: a PropPreservingPartition object
    - `x0`: initial state
    - `u_seq`: matrix where each row contains an input
    - `ssys`: system dynamics
    - `show`: if True, show plot. if false, return axis object
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
        x = np.dot(A,x).flatten() + np.dot(B,u_seq[i,:]).flatten() + K.flatten()
        x_arr = np.vstack([x_arr, x.flatten()])
    
    ax.plot(x_arr[:,0], x_arr[:,1], 'o-')
    
    if show:
        plt.show()
        
    
