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
Functions for plotting Polytopes.
The functions can be accessed by

>>> from tulip.polytope.plot import *

Functions:

    - get_patch
    - plot
"""
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from polytope import dimension, extreme, cheby_ball, is_fulldim

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

def newax():
    """Instantiate new figure and axes.
    """
    #TODO mv to pyvectorized
    fig = plt.figure()
    ax = fig.add_subplot(111)
    return ax

def plot(poly1, ax=None, color=np.random.rand(3)):
    """Plots 2D polytope or region using matplotlib.
    
    @type: poly1: Polytope or Region
    
    see also
    --------
    plot_partition
    """
    #TODO optional arg for text label
    if not is_fulldim(poly1):
        print("Cannot plot empty polytope")
        return
    
    if dimension(poly1) != 2:
        print("Cannot plot polytopes of dimension larger than 2")
        return
    
    if ax is None:
        ax = newax()
    
    if len(poly1) == 0:
        poly = get_patch(poly1, color=color)
        ax.add_patch(poly)
    else:
        for poly2 in poly1.list_poly:
            poly = get_patch(poly2, color=color)
            ax.add_patch(poly)
    
    # affect view
    #l,u = bounding_box(poly1)
    
    #ax.set_xlim(l[0,0], u[0,0])
    #ax.set_ylim(l[1,0], u[1,0])
    
    return ax
