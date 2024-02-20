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
"""Convenience functions for plotting.

WARNING: The public functions:
    `dimension`, `newax`, `dom2vec`, `quiver`
will eventually be removed.
Their use in new applications is discouraged.
"""
import collections.abc as _abc
import itertools as _itr
import logging
import typing as _ty
import warnings as _warn

import numpy as np

try:
    import graphviz as _gv
except ImportError as error:
    _gv = None
    gv_error = error
try:
    import matplotlib as _mpl
    import matplotlib.pyplot as _plt
    import mpl_toolkits.mplot3d.axes3d
except ImportError as error:
    _mpl = None
    _plt = None
    mpl_error = error

import tulip._utils as _utl


__all__ = [
    'dimension',
    'newax',
    'dom2vec',
    'quiver']


if _ty.TYPE_CHECKING:
    _Digraph: _ty.TypeAlias = _gv.Digraph
    _Axes: _ty.TypeAlias = _mpl.axes.Axes
    _Figure: _ty.TypeAlias = _mpl.figure.Figure
else:
    _Digraph = _utl.get_type(_gv, 'Digraph')
    _Axes = _utl.get_type(_mpl, 'axes.Axes')
    _Figure = _utl.get_type(_mpl, 'figure.Figure')


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


PlotDimension = _ty.Literal[2, 3]


def newax(
        subplots:
            int |
            tuple[int, int]=(1, 1),
        fig=None,
        mode:
            _ty.Literal[
                'list', 'matrix']='list',
        dim:
            PlotDimension |
            _abc.Iterable[PlotDimension]
            =2
        ) -> tuple[
            _Axes,
            _Figure]:
    """Create (possibly multiple) new axes handles.  (DEPRECATED)

    @param fig:
        attach axes to this figure
    @type fig:
        figure object,
        should be consistent with `dim`
    @param subplots:
        number or
        layout of subplots
    @param mode:
        return the axes shaped as a
        vector or as a matrix.
        This is a convenience for later iterations
        over the axes.
    @param dim:
        plot dimension:
        - if `dim == 2`, then use `matplotlib`
        So the figure type depends on `dim`.

        If `dim` is an iterable,
        the each item specifies the dimension of
        the corresponding subplot.
    @return:
        `(ax, fig)` where:
        - `ax`: axes created
        - `fig`: parent of `ax`
        The returned value's type
        depends on `mode` above
    """
    _assert_pyplot()
    # layout or number of axes ?
    match subplots:
        case (_, _):
            subplot_layout = tuple(subplots)
        case int():
            subplot_layout = (1, subplots)
        case _:
            raise TypeError(
                'Expected 2-`tuple` or `int` '
                'as value for parameter '
                '`subplots`. Got instead: '
                f'{subplots = }')
    # which figure ?
    if fig is None:
        fig = _plt.figure()
    # create subplot(s)
    nv, nh = subplot_layout
    n = np.prod(subplot_layout)
    match dim:
        case _abc.Iterable():
            dim = tuple(dim)
        case _:
            # all subplots have the
            # same number of
            # spatial dimensions
            dim = n * [dim]
    dim_ok = all(
        x in (2, 3)
        for x in dim)
    if not dim_ok:
        raise ValueError(
            'Expected dimension 2 or 3, '
            f'but: {dim = }')
    # matplotlib (2D)
    ax = list()
    for i, curdim in enumerate(dim):
        if curdim == 2:
            curax = fig.add_subplot(nv, nh, i + 1)
            ax.append(curax)
        else:
            raise ValueError(
                '`ndim >= 3`, but plot limited to 2.')
    if mode == 'matrix':
        ax = list(_grouper(nh, ax))
    # single axes ?
    if subplot_layout == (1, 1):
        ax = ax[0]
    return (ax, fig)


def dom2vec(
        domain:
            list[float],
        resolution:
            list[int]
        ) -> np.ndarray:
    """Matrix of column vectors for meshgrid points.  (DEPRECATED)

    Returns a matrix of column vectors for the meshgrid
    point coordinates over a parallelepiped domain
    with the given resolution.

    Example
    =======

    ```python
    domain = [0, 1, 0,2]
    resolution = [4, 5]
    q = domain2vec(domain, resolution)
    ```

    @param domain:
        extremal values of parallelepiped,
        arranged as:
        `[xmin, xmax, ymin, ymax, ...]`
    @param resolution:
        # points / dimension,
        arranged as:
        `[nx, ny, ...]`
    @return:
        q = matrix of column vectors
        (meshgrid point coordinates),
        of shape:
        [#dim x #points]

    See also:
        `vec2meshgrid`,
        `domain2meshgrid`,
        `meshgrid2vec`
    """
    domain_grouped = _grouper(2, domain)
    def lambda_linspace(dom, res):
        return np.linspace(
            dom[0], dom[1], res)
    axis_grids = map(
        lambda_linspace,
        domain_grouped, resolution)
    pnt_coor = np.meshgrid(*axis_grids)
    q = np.vstack(list(map(
        np.ravel, pnt_coor)))
    return q


def quiver(
        x:
            np.ndarray,
        v:
            np.ndarray,
        ax:
            _Axes |
            None=None,
        **kwargs):
    """Multi-dimensional quiver.  (DEPRECATED)

    Plot v columns at points in columns of x
    in axes ax with plot formatting options in kwargs.

    ```python
    import numpy as np
    import matplotlib as mpl

    x = dom2vec([0, 10, 0, 11], [20, 20])
    v = np.vstack(np.sin(x[1, :] ), np.cos(x[2, :]))
    quiver(mpl.gca(), x, v)
    ```

    see also
        `matplotlib.quiver`

    @param x:
        points where vectors are based
        each column is a coordinate tuple
        [#dim x #points]
        (can be 2d lil)
    @param v:
        matrix of column vectors to
        plot at points x
        [#dim x #points]
        (can be 2d lil)
    @param ax:
        axes handle, e.g., `ax = gca()`
    @param kwargs:
        plot formatting
    @return:
        handle to plotted object(s)
    """
    _assert_pyplot()
    if not ax:
        ax = _plt.gca()
    dim = dimension(x)
    if dim == 2:
        h = ax.quiver(
            x[0, :], x[1, :],
            v[0, :], v[1, :],
            **kwargs)
    else:
        raise ValueError(
            'ndim #dimensions > 3,'
            'plotting only 3D component.')
    return h


def _grouper(
        n:
            int,
        iterable:
            _abc.Iterable,
        fillvalue:
            str |
            None=None
        ) -> _abc.Iterator:
    """Yield chunks of length `n`.

    Pad the last chunk using `fillvalue`,
    to be of length `n`.

    Example:

    ```python
    chunks = grouper(3, 'ABCDEFG', 'x')
    chunks_ = [
        ('A', 'B', 'C'),
        ('D', 'E', 'F'),
        ('G', 'x', 'x')]
    assert chunks == chunks_, (
        chunks, chunks_)
    ```
    """
    args = [iter(iterable)] * n
    return _itr.zip_longest(fillvalue=fillvalue, *args)


def networkx_to_graphviz(graph) -> _Digraph:
    """Convert `networkx` `graph` to `graphviz.Digraph`."""
    _assert_graphviz()
    if graph.is_directed():
        gv_graph = _gv.Digraph()
    else:
        gv_graph = _gv.Graph()
    for u, d in graph.nodes(data=True):
        gv_graph.node(
            str(u), **d)
    for u, v, d in graph.edges(data=True):
        gv_graph.edge(
            str(u), str(v), **d)
    return gv_graph


def _assert_graphviz():
    """Raise `ImportError` if `graphviz` missing."""
    if _gv is not None:
        return
    raise ImportError(
        'Could not import the Python package '
        '`graphviz`, which can be installed from '
        'the Python Package Index (PyPI) '
        'with `pip install graphviz`'
        ) from gv_error


def _assert_pyplot():
    """Raise `ImportError` if `matplotlib` missing."""
    if _plt is not None:
        return
    raise ImportError(
        'Could not import `matplotlib`. '
        '`matplotlib` can be installed from '
        'the Python Package Index (PyPI) '
        'with `pip install matplotlib`'
        ) from mpl_error
