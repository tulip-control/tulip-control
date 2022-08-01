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
"""Convert labeled graph to GraphViz DOT, with custom filtering."""
import collections.abc as _abc
import functools as _ft
import io
import logging
import re
import textwrap as _tw
import typing as _ty

import networkx as nx
import numpy as np
# inline:
#
# import webcolors

import tulip.graphics as _graphics


logger = logging.getLogger(__name__)


def _states2dot_str(
        graph,
        to_dot_graph,
        wrap=10,
        tikz=False,
        rankdir='TB'):
    """Copy nodes to given graph,

    with attributes for dot export.
    """
    # TODO generate LaTeX legend table for edge labels
    states = graph.states
    # get labeling def
    if hasattr(graph, '_state_label_def'):
        label_def = graph._state_label_def
    if hasattr(graph, '_state_dot_label_format'):
        label_format = graph._state_dot_label_format
    else:
        label_format = {
            'type?label': '',
            'separator': r'\\n'}
    for u, d in graph.nodes(data=True):
        # initial state ?
        is_initial = u in states.initial
        is_accepting = _is_accepting(graph, u)
        # state annotation
        node_dot_label = _form_node_label(
            u, d, label_def,
            label_format, wrap, tikz=tikz)

        # node_dot_label = fill(
        #     str(state),
        #     width=wrap)
        rim_color = d.get('color', 'black')
        if tikz:
            _state2tikz(
                graph, to_dot_graph, u,
                is_initial, is_accepting, rankdir,
                rim_color, d, node_dot_label)
        else:
            _state2dot(
                graph, to_dot_graph, u,
                is_initial, is_accepting,
                rim_color, d, node_dot_label)


def _state2dot(
        graph, to_dot_graph, state,
        is_initial, is_accepting,
        rim_color, d, node_dot_label):
    if is_initial:
        _add_incoming_edge(to_dot_graph, state)
    normal_shape = graph.dot_node_shape['normal']
    accept_shape = graph.dot_node_shape.get('accepting', '')
    shape = accept_shape if is_accepting else normal_shape
    corners = 'rounded' if shape == 'rectangle' else ''
    rim_color = '"' + _format_color(rim_color, 'dot') + '"'
    fc = d.get('fillcolor', 'none')
    filled = '' if fc == 'none' else 'filled'
    if fc == 'gradient':
        # top/bottom colors
        # not supported for dot
        lc = d.get(
            'left_color',
            d['top_color'])
        rc = d.get(
            'right_color',
            d['bottom_color'])
        if isinstance(lc, str):
            fillcolor = lc
        elif isinstance(lc, dict):
            fillcolor = list(lc.keys())[0]
        else:
            raise TypeError(
                'left_color must be str or dict.')
        if isinstance(rc, str):
            fillcolor += ':' + rc
        elif isinstance(rc, dict):
            fillcolor += ':' + list(rc.keys())[0]
        else:
            raise TypeError(
                'right_color must be str or dict.')
    else:
        fillcolor = _format_color(fc, 'dot')
    if corners and filled:
        node_style = f'"{corners}, {filled}"'
    elif corners:
        node_style = f'"{corners}"'
    else:
        node_style = f'"{filled}"'
    to_dot_graph.add_node(
        state,
        label=node_dot_label,
        shape=shape,
        style=node_style,
        color=rim_color,
        fillcolor=f'"{fillcolor}"')


def _state2tikz(
        graph, to_pydot_graph, state,
        is_initial, is_accepting, rankdir,
        rim_color, d, node_dot_label):
    style = 'state'
    if rankdir == 'LR':
        init_dir = 'initial left'
    elif rankdir == 'RL':
        init_dir = 'initial right'
    elif rankdir == 'TB':
        init_dir = 'initial above'
    elif rankdir == 'BT':
        init_dir = 'initial below'
    else:
        raise ValueError('Unknown rankdir')
    if is_initial:
        style += f', initial by arrow, {init_dir}, initial text='
    if is_accepting:
        style += ', accepting'
    if graph.dot_node_shape['normal'] == 'rectangle':
        style += ', shape = rectangle, rounded corners'
    # darken the rim
    if 'black' in rim_color:
        c = _format_color(rim_color, 'tikz')
    else:
        c = _format_color(rim_color, 'tikz') + '!black!30'
    style += ', draw = ' + c
    fill = d.get('fillcolor')
    if fill == 'gradient':
        s = {'top_color', 'bottom_color',
             'left_color', 'right_color'}
        for x in s:
            if x in d:
                style += f', {x} = ' + _format_color(
                    d[x], 'tikz')
    elif fill is not None:
        # not gradient
        style += ', fill = ' + _format_color(fill, 'tikz')
    else:
        logger.debug('fillcolor is None')
    to_dot_graph.add_node(
        state,
        texlbl=node_dot_label,
        style=style)


def _format_color(
        color:
            str |
            dict,
        prog:
            _ty.Literal[
                'tikz',
                'dot']
            ='tikz'):
    """Encode color in syntax for given program.

    @param color:
      - `str` for single color or
      - `dict` for weighted color mix
    """
    if isinstance(color, str):
        return color
    if not isinstance(color, dict):
        raise Exception(
            'color must be str or dict')
    if prog == 'tikz':
        s = '!'.join(
            f'{k}!{v}'
            for k, v in
                color.items())
    elif prog == 'dot':
        t = sum(color.values())
        try:
            import webcolors
            # mix them
            result = np.array(
                (0.0, 0.0, 0.0))
            for c, w in color.items():
                result += w / t * np.array(
                    webcolors.name_to_rgb(c))
            s = webcolors.rgb_to_hex(result)
        except:
            logger.warning(
                'failed to import webcolors')
            s = ':'.join(
                f'{k};{v / t}'
                for k, v in
                    color.items())
    else:
        raise ValueError(
            f'Unknown program: {prog}. '
            'Available options are: '
            "'dot' or 'tikz'.")
    return s


def _place_initial_states(trs_graph, pd_graph, tikz):
    empty = nx.DiGraph()
    init_subg = _graphics.networkx_to_graphviz(empty)
    init_subg.graph_attr['rank'] = 'source'
    for node in trs_graph.states.initial:
        init_subg.node(str(node))
        phantom_node = f'phantominit{node}'
        init_subg.node(str(phantom_node))
    pd_graph.subgraph(init_subg)


def _add_incoming_edge(
        g,
        state):
    phantom_node = f'phantominit{state}'
    g.add_node(
        phantom_node,
        label='""',
        shape='none',
        width='0')
    g.add_edge(phantom_node, state)


def _form_node_label(state, state_data, label_def,
                     label_format, width=10, tikz=False):
    # node itself
    state_str = str(state)
    state_str = state_str.replace("'", "")
    # rm parentheses to reduce size of
    # states in figure
    if tikz:
        state_str = state_str.replace('(', '')
        state_str = state_str.replace(')', '')
    # make indices subscripts
    if tikz:
        pattern = r' ([a-zA-Z] \d+) '
        make_subscript = lambda x: x.group(0)[0] + '_' + x.group(0)[1:]
        state_str = re.sub(
            pattern,
            make_subscript,
            state_str,
            flags=re.VERBOSE)
    # SVG requires breaking the
    # math environment into
    # one math env per line.
    # Just make 1st line math env
    # if latex:
    #    state_str = f'${state_str}$'
    #    state_str = _tw.fill(state_str, width=width)
    node_dot_label = state_str
    # newline between state name and
    # label, only if state is labeled
    if len(state_data) != 0:
        node_dot_label += r'\\n'
    # add node annotations from
    # action, AP sets etc
    # other key,values in state attr_dict ignored
    pieces = list()
    for label_type, label_value in state_data.items():
        if label_type not in label_def:
            continue
        # label formatting
        type_name = label_format[label_type]
        sep_type_value = label_format['type?label']
        # avoid turning strings to lists,
        # or non-iterables to lists
        if isinstance(label_value, str):
            label_str = _tw.fill(
                label_value,
                width=width)
        elif isinstance(label_value, _abc.Iterable):
                # and not str
            s = ', '.join([str(x) for x in label_value])
            label_str = r'\\{' + _tw.fill(
                s,
                width=width) + r'\\}'
        else:
            label_str = _tw.fill(
                str(label_value),
                width=width)
        pieces.append(type_name + sep_type_value + label_str)
    sep_label_sets = label_format['separator']
    node_dot_label += sep_label_sets.join(pieces)
    if tikz:
        # replace LF by latex newline
        node_dot_label = node_dot_label.replace(
            r'\\n', r'\\\\ ')
        # dot2tex math mode does not
        # handle newlines properly
        node_dot_label = (
            rf'$\\begin{{matrix}} {node_dot_label}'
            r'\\end{matrix}$')
    return node_dot_label


def _is_accepting(graph, state):
    """accepting state ?"""
    # no accepting states defined ?
    if not hasattr(graph.states, 'accepting'):
        return False
    return state in graph.states.accepting


def _transitions2dot_str(
        trans,
        to_dot_graph,
        tikz=False
        ) -> str:
    """Convert transitions to dot str."""
    graph = trans.graph
    graph_hasattr = _ft.partial(
        hasattr, graph)
    has_attr = all(map(graph_hasattr, [
        '_transition_label_def',
        '_transition_dot_label_format',
        '_transition_dot_mask']))
    if not has_attr:
        return
    # get labeling def
    label_def = graph._transition_label_def
    label_format = graph._transition_dot_label_format
    label_mask = graph._transition_dot_mask
    edges = graph.edges(
        data=True,
        keys=True)
    for u, v, key, edge_data in edges:
        edge_dot_label = _form_edge_label(
            edge_data, label_def,
            label_format, label_mask, tikz)
        edge_color = edge_data.get('color', 'black')
        to_dot_graph.add_edge(
            u, v,
            key=key,
            label=edge_dot_label,
            color=edge_color)


def _form_edge_label(edge_data, label_def,
                     label_format, label_mask, tikz):
    label = ''
        # dot label for edge
    sep_label_sets = label_format['separator']
    for label_type, label_value in edge_data.items():
        if label_type not in label_def:
            continue
        # masking defined ?
        # custom filter hiding based on value
        if label_type in label_mask:
            # not show ?
            if not label_mask[label_type](label_value):
                continue
        # label formatting
        if label_type in label_format:
            type_name = label_format[label_type]
            sep_type_value = label_format['type?label']
        else:
            type_name = ':'
            sep_type_value = r',\\n'
        # format iterable containers using
        # mathematical set notation: {...}
        if isinstance(label_value, str):
            # str is Iterable:
            # avoid turning it to list
            label_str = label_value
        elif isinstance(label_value, _abc.Iterable):
            s = ', '.join([str(x) for x in label_value])
            label_str = r'\\{' + _tw.fill(s) + r'\\}'
        else:
            label_str = str(label_value)
        if tikz:
            type_name = r'\mathrm' + '{' + type_name + '}'
        label += (type_name + sep_type_value +
                  label_str + sep_label_sets)
    if tikz:
        label = r'\\begin{matrix}' + label + r'\\end{matrix}'
    label = f'"{label}"'
    return label


def _graph2dot(
        graph:
            'LabeledDiGraph',
        wrap=10,
        tikz=False,
        rankdir='TB'
        ) -> str:
    """Convert (possibly labeled) state graph to dot str."""
    dummy_nx_graph = nx.MultiDiGraph()
    _states2dot_str(
        graph, dummy_nx_graph,
        wrap=wrap,
        tikz=tikz,
        rankdir=rankdir)
    _transitions2dot_str(
        graph.transitions,
        dummy_nx_graph,
        tikz=tikz)
    dot_graph = _graphics.networkx_to_graphviz(
        dummy_nx_graph)
    _place_initial_states(
        graph, dot_graph, tikz)
    dot_graph.graph_attr['overlap'] = 'false'
    # dot_graph.graph_attr['size'] = '"0.25,1"'
    # dot_graph.graph_attr['ratio'] = '"compress"'
    dot_graph.graph_attr['nodesep'] = '0.5'
    dot_graph.graph_attr['ranksep'] = '0.1'
    return dot_graph


def graph2dot_str(
        graph:
            'LabeledDiGraph',
        wrap=10,
        tikz=False
        ) -> str:
    """Convert graph to DOT string.

    @param wrap:
        textwrap width
    """
    dot_graph = _graph2dot(
        graph,
        wrap=wrap,
        tikz=tikz)
    return dot_graph.source


def save_dot(
        graph:
            'LabeledDiGraph',
        path,
        fileformat,
        rankdir,
        prog,
        wrap,
        tikz=False
        ) -> bool:
    """Save state graph to dot file.

    @return:
        `True` upon success
    """
    dot_graph = _graph2dot(
        graph,
        wrap=wrap,
        tikz=tikz,
        rankdir=rankdir)
    if dot_graph is None:
        # graph2dot must have
        # printed warning already
        return False
    dot_graph.graph_attr['rankdir'] = rankdir
    dot_graph.graph_attr['splines'] = 'true'
    dot_graph.render(
        filename=path,
        format=fileformat,
        engine=prog,
        quiet=tikz)
            # tikz labels can raise
            # graphviz warnings
    return True


def plot_dot(
        graph:
            nx.Graph,
        prog:
            _ty.Literal[
                'dot',
                'neato',
                'circo',
                'twopi',
                'fdp',
                'sfdp']
            ='dot',
        rankdir:
            _ty.Literal[
                'LR',
                'TB']
            ='LR',
        wrap=10,
        ax=None):
    """Plot a networkx graph using dot.

    No files written or deleted from the disk.

    Note that all networkx graph classes are inherited
    from `networkx.Graph`.


    See Also
    ========
    `dot` documentation

    @param graph:
        to plot
    @param prog:
        GraphViz program to use
    @param rankdir:
        direction to layout nodes
    @param ax:
        axes
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError(
            'graph not networkx class.'
            f'Got instead: {type(graph)}')
    dot_graph = _graphics.networkx_to_graphviz(graph)
    dot_graph.graph_attr['rankdir'] = rankdir
    dot_graph.graph_attr['splines'] = 'true'
    dot_graph.graph_attr['bgcolor'] = 'gray'
    # layout
    png_str = dot_graph.pipe(
        format='png',
        engine=prog,
        encoding='utf-8')
    # installed ?
    try:
        from IPython.display import display, Image
        logger.debug('IPython installed.')
        # called by IPython ?
        try:
            cfg = get_ipython().config
            logger.debug(
                'Script called by IPython.')
            # Caution!!! : not ordinary dict,
            # but IPython.config.loader.Config
            #
            # qtconsole ?
            if cfg['IPKernelApp']:
                logger.debug(
                    'Within IPython QtConsole.')
                display(Image(data=png_str))
                return True
        except:
            print(
                'IPython installed, '
                'but not called from it.')
    except ImportError:
        logger.warning(
            'IPython not found.\n'
            'So loaded dot images not inline.')
    # not called from
    # IPython QtConsole,
    # try Matplotlib...
    # installed ?
    try:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
    except:
        logger.debug(
            'Matplotlib not installed.')
        logger.warning(
            'Neither IPython QtConsole nor '
            'Matplotlib available.')
        return None
    logger.debug(
        'Matplotlib installed.')
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    sio = io.BytesIO()
    sio.write(png_str)
    sio.seek(0)
    img = mpimg.imread(sio)
    ax.imshow(
        img,
        aspect='equal')
    plt.show(block=False)
    return ax
