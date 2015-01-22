# Copyright (c) 2013-2014 by California Institute of Technology
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
Convenience functions for plotting

WARNING: The public functions dimension, newax, dom2vec, quiver will
eventually be removed.  Their use in new applications is discouraged.

They come from https://github.com/johnyf/pyvectorized
"""
from __future__ import division

import logging
logger = logging.getLogger(__name__)

from warnings import warn
from itertools import izip_longest

import numpy as np
try:
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import axes3d
except Exception as e:
    logger.error(e)

# from mayavi import mlab

def dimension(ndarray):
    """dimension of ndarray  (DEPRECATED)
    
    - ndim == 1:
        dimension = 1
    - ndim == 2:
        dimension = shape[0]
    """
    if ndarray.ndim < 2:
        return ndarray.ndim
    return ndarray.shape[0]

def newax(subplots=(1, 1), fig=None,
          mode='list', dim=2):
    """Create (possibly multiple) new axes handles.  (DEPRECATED)
    
    @param fig: attach axes to this figure
    @type fig: figure object,
        should be consistent with C{dim}
    
    @param subplots: number or layout of subplots
    @type subplots: int or
        2-tuple of subplot layout
    
    @param mode: return the axes shaped as a
        vector or as a matrix.
        This is a convenience for later iterations
        over the axes.
    @type mode: 'matrix' | ['list']
    
    @param dim: plot dimension:
        
            - if dim == 2, then use matplotlib
            - if dim == 3, then use mayavi
        
        So the figure type depends on dim.
    
    @return: C{(ax, fig)} where:
        - C{ax}: axes created
        - C{fig}: parent of ax
    @rtype: list or list of lists,
        depending on C{mode} above
    """
    # layout or number of axes ?
    try:
        subplot_layout = tuple(subplots)
    except:
        subplot_layout = (1, subplots)
    
    # reasonable layout ?
    if len(subplot_layout) != 2:
        raise Exception('newax:'
                        'subplot layout should be 2-tuple or int.')
    
    # which figure ?
    if fig is None:
        fig = plt.figure()
    
    # create subplot(s)
    (nv, nh) = subplot_layout
    n = np.prod(subplot_layout)
    
    try:
        dim = tuple(dim)
    except:
        # all same dim
        dim = [dim] * n
    
    # matplotlib (2D) or mayavi (3D) ?
    ax = []
    for (i, curdim) in enumerate(dim):
        if curdim == 2:
            curax = fig.add_subplot(nv, nh, i + 1)
            ax.append(curax)
        else:
            curax = fig.add_subplot(nv, nh, i + 1, projection='3d')
            ax.append(curax)
                      
        if curdim > 3:
            warn('ndim > 3, but plot limited to 3.')
    
    if mode is 'matrix':
        ax = list(_grouper(nh, ax))
    
    # single axes ?
    if subplot_layout == (1, 1):
        ax = ax[0]
    
    return (ax, fig)

def dom2vec(domain, resolution):
    """Matrix of column vectors for meshgrid points.  (DEPRECATED)
    
    Returns a matrix of column vectors for the meshgrid
    point coordinates over a parallelepiped domain
    with the given resolution.
    
    Example
    =======
    >>> domain = [0, 1, 0,2]
    >>> resolution = [4, 5]
    >>> q = domain2vec(domain, resolution)
    
    @param domain: extremal values of parallelepiped
    @type domain: [xmin, xmax, ymin, ymax, ...]
    
    @param resolution: # points /dimension
    @type resolution: [nx, ny, ...]
    
    @return: q = matrix of column vectors (meshgrid point coordinates)
    @rtype: [#dim x #points]
        
    See also vec2meshgrid, domain2meshgrid, meshgrid2vec.
    """
    domain = _grouper(2, domain)
    lambda_linspace = lambda (dom, res): np.linspace(dom[0], dom[1], res)
    axis_grids = map(lambda_linspace, zip(domain, resolution))
    pnt_coor = np.meshgrid(*axis_grids)
    q = np.vstack(map(np.ravel, pnt_coor))
    
    return q

def quiver(x, v, ax=None, **kwargs):
    """Multi-dimensional quiver.  (DEPRECATED)
    
    Plot v columns at points in columns of x
    in axes ax with plot formatting options in kwargs.
    
    >>> import numpy as np
    >>> import matplotlib as mpl
    >>> from pyvectorized import quiver, dom2vec
    >>> x = dom2vec([0, 10, 0, 11], [20, 20])
    >>> v = np.vstack(np.sin(x[1, :] ), np.cos(x[2, :] ) )
    >>> quiver(mpl.gca(), x, v)
    
    see also
        matplotlib.quiver, mayavi.quiver3
    
    @param x: points where vectors are based
        each column is a coordinate tuple
    @type x: 2d lil | numpy.ndarray
    
    @param v: vectors which to base at points x
    @type v: 2d lil | numpy.ndarray
    
    @param ax: axes handle, e.g., ax = gca())
    
    @param x: matrix of points where vectors are plotted
    @type x: [#dim x #points]
    
    @param v: matrix of column vectors to plot at points x
    @type v: [#dim x #points]
    
    @param kwargs: plot formatting
    
    @return: handle to plotted object(s)
    """
    # multiple axes ?
    try:
        fields = [quiver(x, v, i, **kwargs) for i in ax]
        return fields
    except:
        pass
    
    if not ax:
        ax = plt.gca()
    
    dim = dimension(x)
    
    if dim < 2:
        raise Exception('ndim < 2')
    elif dim < 3:
        h = ax.quiver(x[0, :], x[1, :],
                      v[0, :], v[1, :], **kwargs)
    else:
        raise NotImplementedError
        
        from mayavi.mlab import quiver3d
        
        if ax:
            print('axes arg ignored, mayavi used')
        
        h = quiver3d(x[0, :], x[1, :], x[2, :],
                     v[0, :], v[1, :], v[2, :], **kwargs)
    
    if dim > 3:
        warn('quiver:ndim #dimensions > 3,' +
             'plotting only 3D component.')
    
    return h

def _grouper(n, iterable, fillvalue=None):
    """grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx
    """
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)
