# Copyright (c) 2013 by California Institute of Technology
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
Convert labeled graph to dot using
pydot and custom filtering
"""
import logging
logger = logging.getLogger(__name__)

import re
from collections import Iterable
from textwrap import fill
from cStringIO import StringIO

import networkx as nx
from networkx.utils import make_str

try:
    import pydot
except ImportError:
    logger.error('pydot package not found.\nHence dot export not unavailable.')
    pydot = None

try:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    matplotlib = True
except ImportError:
    logger.error('matplotlib package not found.\nSo no loading of dot plots.')
    matplotlib = None

try:
    from IPython.display import display, Image
    IPython = True
except ImportError:
    logger.error('IPython not found.\nSo loaded dot images not inline.')
    IPython = None

def _states2dot_str(graph, to_pydot_graph, wrap=10,
                    latex=False, tikz=False):
    """Copy nodes to given Pydot graph, with attributes for dot export.
    """
    # TODO option to replace with int to reduce size,
    # TODO generate separate LaTeX legend table (PNG option ?)
    
    states = graph.states
    
    # get labeling def
    if hasattr(graph, '_state_label_def'):
        label_def = graph._state_label_def
    if hasattr(graph, '_state_dot_label_format'):
        label_format = graph._state_dot_label_format
    else:
        label_format = {'type?label':'', 'separator':'\n'}
    
    for (state, state_data) in graph.nodes_iter(data=True):
        # initial state ?
        is_initial = state in states.initial
        is_accepting = _is_accepting(graph, state)
        
        # state annotation
        node_dot_label = _form_node_label(
            state, state_data, label_def,
            label_format, wrap, latex, tikz=tikz
        )
    
        #node_dot_label = fill(str(state), width=wrap)
    
        # state boundary color
        rim_color = 'black'
        if state_data.has_key('color'):
            rim_color = state_data['color']
        
        # state interior color
        fill_color = None
        if state_data.has_key('fillcolor'):
            fill_color = state_data['fillcolor']
        
        if tikz:
            _state2tikz(graph, to_pydot_graph, state,
                        is_initial, is_accepting,
                        rim_color, fill_color, node_dot_label)
        else:
            _state2dot(graph, to_pydot_graph, state,
                       is_initial, is_accepting,
                       rim_color, fill_color, node_dot_label)

def _state2dot(graph, to_pydot_graph, state,
               is_initial, is_accepting,
               rim_color, fill_color, node_dot_label):
    if is_initial:
        _add_incoming_edge(to_pydot_graph, state)
    
    shape = graph.dot_node_shape['normal']
    if is_accepting:
        shape = graph.dot_node_shape['accepting']
    
    corners = ''
    if shape is 'rectangle':
        corners = 'rounded'
    
    rim_color = '"' + rim_color + '"'
    
    filled = ''
    if fill_color is not None:
        filled = ',filled'
    
    if fill_color is None:
        fill_color = 'none'
    
    node_style = '"' + corners + filled + '"'
    
    to_pydot_graph.add_node(
        state,
        label=node_dot_label,
        shape=shape,
        style=node_style,
        color=rim_color,
        fillcolor=fill_color
    )

def _state2tikz(graph, to_pydot_graph, state,
                is_initial, is_accepting,
                rim_color, fill_color, node_dot_label):
    style = 'state'
    
    if is_initial:
        style += ', initial by arrow, initial right, initial text='
    if is_accepting:
        style += ', accepting'
    
    if graph.dot_node_shape['normal'] is 'rectangle':
        style += ', shape = rectangle, rounded corners'
    
    if 'black' not in rim_color:
        tmp_rim_color = rim_color + '!black!30'
    else:
        tmp_rim_color = rim_color
    style += ', draw = ' + tmp_rim_color
    
    if fill_color is not None:
        s = {'top color', 'bottom color', 'left color', 'right color'}
        if any(x in fill_color for x in s):
            style += ', ' + fill_color
        else:
            style += ', fill = ' + fill_color
    
    to_pydot_graph.add_node(
        state,
        texlbl=node_dot_label,
        style=style
    )

def _place_initial_states(trs_graph, pd_graph):
    init_subg = pydot.Subgraph('initial')
    init_subg.set_rank('source')
    
    for node in trs_graph.states.initial:
        pd_node = pydot.Node(make_str(node) )
        init_subg.add_node(pd_node)
    
    pd_graph.add_subgraph(init_subg)

def _add_incoming_edge(g, state):
    phantom_node = 'phantominit' +str(state)
    
    g.add_node(phantom_node, label='""', shape='none', width='0')
    g.add_edge(phantom_node, state)

def _form_node_label(state, state_data, label_def,
                     label_format, width=10, latex=False, tikz=False):
    # node itself
    state_str = str(state)
    state_str = state_str.replace("'", "")
    
    # rm parentheses to reduce size of states in fig
    if tikz:
        state_str = state_str.replace('(', '')
        state_str = state_str.replace(')', '')
    
    # make indices subscripts
    if latex or tikz:
        pattern = '([a-zA-Z]\d+)'
        make_subscript = lambda x: x.group(0)[0] + '_' + x.group(0)[1:]
        state_str = re.sub(pattern, make_subscript, state_str)
    
    # SVG requires breaking the math environment into
    # one math env per line. Just make 1st line math env
    if latex:
        state_str = '$' + state_str + '$'
        state_str = fill(state_str, width=width)
    node_dot_label = state_str
    
    # newline between state name and label, only if state is labeled
    if len(state_data) != 0:
        node_dot_label += '\n'
    
    # add node annotations from action, AP sets etc
    # other key,values in state attr_dict ignored
    pieces = list()
    for (label_type, label_value) in state_data.iteritems():
        if label_type not in label_def:
            continue
        
        # label formatting
        type_name = label_format[label_type]
        sep_type_value = label_format['type?label']
        
        # avoid turning strings to lists,
        # or non-iterables to lists
        if isinstance(label_value, str):
            label_str = fill(label_value, width=width)
        elif isinstance(label_value, Iterable): # and not str
            s = ', '.join([str(x) for x in label_value])
            label_str = '{' + fill(s, width=width) + '}'
        else:
            label_str = fill(str(label_value), width=width)
        
        pieces.append(type_name + sep_type_value + label_str)
    
    sep_label_sets = label_format['separator']
    node_dot_label += sep_label_sets.join(pieces)
    
    if latex:
        node_dot_label = node_dot_label.replace(r'{', r'\\{')
        node_dot_label = node_dot_label.replace(r'}', r'\\}')
    
    if tikz:
        node_dot_label = node_dot_label.replace(r'{', r'\{')
        node_dot_label = node_dot_label.replace(r'}', r'\}')
        
        # replace LF by latex newline
        node_dot_label = node_dot_label.replace('\n', '\\\\\\\\ ')
        
        # dot2tex math mode doesn't handle newlines properly
        node_dot_label = '$\\\\begin{matrix} ' + node_dot_label + '\\end{matrix}$'
    
    return node_dot_label

def _is_accepting(graph, state):
    """accepting state ?
    """
    # check if accepting states defined
    if not hasattr(graph.states, 'accepting'):
        return False
    
    # check state is accepting
    if state in graph.states.accepting:
        return True
    return False

def _transitions2dot_str(trans, to_pydot_graph, latex, tikz=False):
    """Convert transitions to dot str.
    
    @rtype: str
    """
    if not hasattr(trans.graph, '_transition_label_def'):
        return
    if not hasattr(trans.graph, '_transition_dot_label_format'):
        return
    if not hasattr(trans.graph, '_transition_dot_mask'):
        return
    
    # get labeling def
    label_def = trans.graph._transition_label_def
    label_format = trans.graph._transition_dot_label_format
    label_mask = trans.graph._transition_dot_mask
    
    for (u, v, key, edge_data) in \
    trans.graph.edges_iter(data=True, keys=True):
        edge_dot_label = _form_edge_label(
            edge_data, label_def,
            label_format, label_mask, latex, tikz
        )
        to_pydot_graph.add_edge(u, v, key=key,
                                label=edge_dot_label)

def _form_edge_label(edge_data, label_def,
                     label_format, label_mask, latex, tikz):
    edge_dot_label = '"'
    sep_label_sets = label_format['separator']
    
    for (label_type, label_value) in edge_data.iteritems():
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
            sep_type_value = ','
        
        if isinstance(label_value, str):
            # str is Iterable: avoid turning it to list
            label_str = label_value
        elif isinstance(label_value, Iterable):
            s = ', '.join([str(x) for x in label_value])
            label_str = '{' + fill(s) + '}'
        else:
            label_str = str(label_value)
        
        edge_dot_label += type_name +sep_type_value
        edge_dot_label += label_str +sep_label_sets
    edge_dot_label += '"'
    
    if latex:
        edge_dot_label = edge_dot_label.replace(r'{', r'\\{')
        edge_dot_label = edge_dot_label.replace(r'}', r'\\}')
    
    if tikz:
        edge_dot_label = edge_dot_label.replace(r'{', r'\{')
        edge_dot_label = edge_dot_label.replace(r'}', r'\}')
    
    return edge_dot_label

def _pydot_missing():
    if pydot is None:
        msg = 'Attempted calling _to_pydot.\n'
        msg += 'Unavailable due to pydot not installed.\n'
        logger.warn(msg)
        return True
    
    return False
    
def _graph2pydot(graph, wrap=10, latex=False, tikz=False):
    """Convert (possibly labeled) state graph to dot str.
    
    @type graph: L{LabeledDiGraph}
    
    @rtype: str
    """
    if _pydot_missing():
        return None
    
    dummy_nx_graph = nx.MultiDiGraph()
    
    _states2dot_str(graph, dummy_nx_graph, wrap=wrap, latex=latex, tikz=tikz)
    _transitions2dot_str(graph.transitions, dummy_nx_graph, latex, tikz=tikz)
    
    pydot_graph = nx.to_pydot(dummy_nx_graph)
    _place_initial_states(graph, pydot_graph)
    
    pydot_graph.set_overlap('false')
    #pydot_graph.set_size('"0.25,1"')
    #pydot_graph.set_ratio('"compress"')
    #pydot_graph.set_nodesep(0.1)
    #pydot_graph.set_ranksep(0.1)
    
    return pydot_graph

def graph2dot_str(graph, wrap=10, latex=False, tikz=False):
    """Convert graph to dot string.
    
    Requires pydot.
    
    @type graph: L{LabeledDiGraph}
    
    @param wrap: textwrap width
    
    @rtype: str
    """
    pydot_graph = _graph2pydot(graph, wrap=wrap,
                               latex=latex, tikz=tikz)
    
    return pydot_graph.to_string()

def save_dot(graph, path, fileformat, rankdir, prog, wrap, latex, tikz=False):
    """Save state graph to dot file.
    
    @type graph: L{LabeledDiGraph}
    
    @return: True upon success
    @rtype: bool
    """
    pydot_graph = _graph2pydot(graph, wrap=wrap,
                               latex=latex, tikz=tikz)
    if pydot_graph is None:
        # graph2dot must have printed warning already
        return False
    pydot_graph.set_rankdir(rankdir)
    pydot_graph.set_splines('true')
    pydot_graph.write(path, format=fileformat, prog=prog)
    return True

def plot_pydot(graph, prog='dot', rankdir='LR', wrap=10, ax=None):
    """Plot a networkx or pydot graph using dot.
    
    No files written or deleted from the disk.
    
    Note that all networkx graph classes are inherited
    from networkx.Graph
    
    See Also
    ========
    dot & pydot documentation
    
    @param graph: to plot
    @type graph: networkx.Graph | pydot.Graph
    
    @param prog: GraphViz programto use
    @type prog: 'dot' | 'neato' | 'circo' | 'twopi'
        | 'fdp' | 'sfdp' | etc
    
    @param rankdir: direction to layout nodes
    @type rankdir: 'LR' | 'TB'
    
    @param ax: axes
    """
    if pydot is None:
        msg = 'Using plot_pydot requires that pydot be installed.'
        logger.warn(msg)
        return
    
    try:
        pydot_graph = _graph2pydot(graph, wrap=wrap)
    except:
        if isinstance(graph, nx.Graph):
            pydot_graph = nx.to_pydot(graph)
        else:
            raise TypeError('graph not networkx or pydot class.' +
                'Got instead: ' +str(type(graph) ) )
    pydot_graph.set_rankdir(rankdir)
    pydot_graph.set_splines('true')
    pydot_graph.set_bgcolor('gray')
    
    png_str = pydot_graph.create_png(prog=prog)
    
    # installed ?
    if IPython:
        logger.debug('IPython installed.')
        
        # called by IPython ?
        try:
            cfg = get_ipython().config
            logger.debug('Script called by IPython.')
            
            # Caution!!! : not ordinary dict,
            # but IPython.config.loader.Config
            
            # qtconsole ?
            if cfg['IPKernelApp']:
                logger.debug('Within IPython QtConsole.')
                display(Image(data=png_str) )
                return True
        except:
            print('IPython installed, but not called from it.')
    else:
        logger.debug('IPython not installed.')
    
    # not called from IPython QtConsole, try Matplotlib...
    
    # installed ?
    if matplotlib:
        logger.debug('Matplotlib installed.')
        
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        
        sio = StringIO()
        sio.write(png_str)
        sio.seek(0)
        img = mpimg.imread(sio)
        ax.imshow(img, aspect='equal')
        plt.show(block=False)
        
        return ax
    else:
        logger.debug('Matplotlib not installed.')
    
    logger.warn('Neither IPython QtConsole nor Matplotlib available.')
    return None
