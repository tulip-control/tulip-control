# Copyright (c) 2013-2015 by California Institute of Technology
# and 2014 The Regents of the University of Michigan
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
# 3. Neither the name of the copyright holder(s) nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
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
"""Base classes for labeled directed graphs"""
from __future__ import absolute_import
import logging
import os
import copy
from pprint import pformat
from collections import Iterable
import warnings
import networkx as nx
from tulip.transys.mathset import SubSet, TypedDict
# inline imports:
#
# from tulip.transys.export import graph2dot
# from tulip.transys.export import save_d3
# from tulip.transys.export import graph2dot


logger = logging.getLogger(__name__)


def label_is_desired(attr_dict, desired_dict):
    """Return True if all labels match.

    Supports symbolic evaluation, if label type is callable.
    """
    if not isinstance(attr_dict, TypedDict):
        raise Exception('attr_dict must be TypedDict' +
                        ', instead: ' + str(type(attr_dict)))
    if attr_dict == desired_dict:
        return True
    # different keys ?
    mismatched_keys = set(attr_dict).symmetric_difference(desired_dict)
    if mismatched_keys:
        return False
    # any labels have symbolic semantics ?
    label_def = attr_dict.allowed_values
    for type_name, value in attr_dict.iteritems():
        logger.debug('Checking label type:\n\t' + str(type_name))
        type_def = label_def[type_name]
        desired_value = desired_dict[type_name]
        if hasattr(type_def, '__call__'):
            logger.debug('Found label semantics:\n\t' + str(type_def))
            # value = guard
            if not type_def(value, desired_value):
                return False
            else:
                continue
        # no guard semantics given,
        # then by convention: guard is singleton {cur_val},
        if not value == desired_value:
            test_common_bug(value, desired_value)
            return False
    return True


def test_common_bug(value, desired_value):
    logger.debug('Label value:\n\t' + str(value))
    logger.debug('Desired value:\n\t' + str(desired_value))
    if (
        isinstance(value, (set, list)) and
        isinstance(desired_value, (set, list)) and
        value.__class__ != desired_value.__class__
    ):
        msg = (
            'Set SubLabel:\n\t' + str(value) +
            'compared to list SubLabel:\n\t' + str(desired_value) +
            'Did you mix sets & lists when setting AP labels ?')
        raise Exception(msg)


class States(object):
    """Methods to manage states and initial states."""

    def __init__(self, graph):
        """Initialize C{States}.

        @type graph: L{LabeledDiGraph}
        """
        self.graph = graph
        self.initial = []

    def __getitem__(self, state):
        return self.graph.node[state]

    def __call__(self, *args, **kwargs):
        """Return list of states.

        For more details see L{LabeledDiGraph.nodes}
        """
        return self.graph.nodes(*args, **kwargs)

    def __str__(self):
        return 'States:\n' + pformat(self(data=False))

    def __len__(self):
        """Total number of states."""
        return self.graph.number_of_nodes()

    def __iter__(self):
        return iter(self())

    def __ior__(self, new_states):
        # TODO carefully test this
        self.add_from(new_states)
        return self

    def __contains__(self, state):
        """Return True if state in states."""
        return state in self.graph

    @property
    def initial(self):
        """ Return L{SubSet} of initial states."""
        return self._initial

    @initial.setter
    def initial(self, states):
        s = SubSet(self)
        s |= states
        self._initial = s

    def _warn_if_state_exists(self, state):
        if state in self:
            logger.debug('State already exists.')

    def _single_state2singleton(self, state):
        """Convert to a singleton list, if argument is a single state.

        Otherwise return given argument.
        """
        if state in self:
            states = [state]
        else:
            states = state
        return states

    def add(self, new_state, attr_dict=None, check=True, **attr):
        """Wraps L{LabeledDiGraph.add_node},

        which wraps C{networkx.MultiDiGraph.add_node}.
        """
        self._warn_if_state_exists(new_state)
        logger.debug('Adding new id: ' + str(new_state))
        self.graph.add_node(new_state, attr_dict, check, **attr)

    def add_from(self, new_states, check=True, **attr):
        """Wraps L{LabeledDiGraph.add_nodes_from},

        which wraps C{networkx.MultiDiGraph.add_nodes_from}.
        """
        self.graph.add_nodes_from(new_states, check, **attr)

    def remove(self, state):
        """Remove C{state} from states (including initial).

        Wraps C{networkx.MultiDiGraph.remove_node}.
        """
        if state in self.initial:
            self.initial.remove(state)
        self.graph.remove_node(state)

    def remove_from(self, states):
        """Remove a list of states.

        Iterates C{States.remove} to imitate
        C{networkx.MultiDiGraph.remove_nodes_from},
        handling also initial states.
        """
        for state in states:
            self.remove(state)

    def post(self, states=None):
        """Direct successor set (1-hop) for given states.

        Edge labels are ignored.

        If multiple states provided,
        then union Post(s) for s in states provided.

        See Also
        ========
          - L{pre}
          - Def. 2.3, p.23 U{[BK08]
            <http://tulip-control.sourceforge.net/doc/bibliography.html#bk08>}

        @param states:
          - None, so initial states returned
          - single state or
          - set of states or

        @rtype: set
        """
        if states is None:
            return set(self.initial)
        states = self._single_state2singleton(states)
        successors = set()
        for state in states:
            successors |= set(self.graph.successors(state))
        return successors

    def pre(self, states):
        """Return direct predecessors (1-hop) of given state.

        See Also
        ========
          - L{post}
          - Def. 2.3, p.23 U{[BK08]
            <http://tulip-control.sourceforge.net/doc/bibliography.html#bk08>}

        @rtype: set
        """
        states = self._single_state2singleton(states)
        predecessors = set()
        for state in states:
            predecessors |= set(self.graph.predecessors(state))
        return predecessors

    def forward_reachable(self, state):
        """Return states reachable from given state.

        Iterated post(), a wrapper of networkx.descendants.
        """
        descendants = nx.descendants(self, state)
        return descendants

    def backward_reachable(self, state):
        """Return states from which the given state can be reached.

        A wrapper of networkx.ancestors.
        """
        ancestors = nx.ancestors(self, state)
        return ancestors

    def paint(self, state, color):
        """Color the given state.

        The state is filled with given color,
        rendered with dot when plotting and saving.

        @param state: valid system state

        @param color: with which to paint C{state}
        @type color: str of valid dot color
        """
        self.graph.node[state]['style'] = 'filled'
        self.graph.node[state]['fillcolor'] = color

    def find(self, states=None, with_attr_dict=None, **with_attr):
        """Filter by desired states and by desired state labels.

        Examples
        ========
        Assume that the system is:

        >>> import transys as trs
        >>> ts = trs.FTS()
        >>> ts.atomic_propositions.add('p')
        >>> ts.states.add('s0', ap={'p'})

          - To find the label of a single state C{'s0'}:

              >>> a = ts.states.find(['s0'] )
              >>> (s0_, label) = a[0]
              >>> print(label)
              {'ap': set(['p'])}

          - To find all states with a specific label C{{'p'}}:

              >>> ts.states.add('s1', ap={'p'})
              >>> b = ts.states.find(with_attr_dict={'ap':{'p'} } )
              >>> states = [state for (state, label_) in b]
              >>> print(set(states) )
              {'s0', 's1'}

          - To find all states in subset C{M} labeled with C{{'p'}}:

              >>> ts.states.add('s2', ap={'p'})
              >>> M = {'s0', 's2'}
              >>> b = ts.states.find(M, {'ap': {'p'} } )
              >>> states = [state for (state, label_) in b]
              >>> print(set(states) )
              {'s0', 's2'}

        @param states: subset of states over which to search
        @type states: 'any' (default)
            | iterable of valid states
            | single valid state

        @param with_attr_dict: label with which to filter the states
        @type with_attr_dict: {sublabel_type : desired_sublabel_value, ...}
            | leave empty, to allow any state label (default)

        @param with_attr: label key-value pairs which take
            precedence over C{with_attr_dict}.

        @rtype: list of labeled states
        @return: [(C{state}, C{label}),...]
            where:
                - C{state} \\in C{states}
                - C{label}: dict
        """
        if with_attr_dict is None:
            with_attr_dict = with_attr
        else:
            try:
                with_attr_dict.update(with_attr)
            except AttributeError:
                raise Exception('with_attr_dict must be a dict')
        if states is not None:
            # singleton check
            if states in self:
                state = states
                msg = (
                    'LabeledStates.find got single state: ' +
                    str(state) + '\n'
                    'instead of Iterable of states.\n')
                states = [state]
                msg += 'Replaced given states = ' + str(state)
                msg += ' with states = ' + str(states)
                logger.debug(msg)
        found_state_label_pairs = []
        for state, attr_dict in self.graph.nodes_iter(data=True):
            logger.debug('Checking state_id = ' + str(state) +
                         ', with attr_dict = ' + str(attr_dict))
            if states is not None:
                if state not in states:
                    logger.debug('state_id = ' + str(state) + ', not desired.')
                    continue
            msg = (
                'Checking state label:\n\t attr_dict = ' +
                str(attr_dict) +
                '\n vs:\n\t desired_label = ' + str(with_attr_dict))
            logger.debug(msg)
            if not with_attr_dict:
                logger.debug('Any label acceptable.')
                ok = True
            else:
                ok = label_is_desired(attr_dict, with_attr_dict)
            if ok:
                logger.debug('Label Matched:\n\t' + str(attr_dict) +
                             ' == ' + str(with_attr_dict))
                state_label_pair = (state, dict(attr_dict))
                found_state_label_pairs.append(state_label_pair)
            else:
                logger.debug('No match for label---> state discarded.')
        return found_state_label_pairs

    def is_terminal(self, state):
        """Return True if state has no outgoing transitions.

        See Also
        ========
        Def. 2.4, p.23 U{[BK08]
        <http://tulip-control.sourceforge.net/doc/bibliography.html#bk08>}
        """
        return not bool(self.graph.successors(state))


class Transitions(object):
    """Methods for handling labeled transitions.

    Note that a directed edge is an ordered set of nodes.
    Unlike an edge, a transition is a labeled edge.
    """

    def __init__(self, graph, deterministic=False):
        """Initialize C{Transitions}.

        @type graph: L{LabeledDiGraph}
        """
        self.graph = graph
        self._deterministic = deterministic

    def __call__(self, **kwargs):
        """Return list of transitions.

        Wraps L{LabeledDiGraph.edges}.
        """
        return self.graph.edges(**kwargs)

    def __str__(self):
        return 'Transitions:\n' + pformat(self())

    def __len__(self):
        """Count transitions."""
        return self.graph.number_of_edges()

    def _breaks_determinism(self, from_state, sublabels):
        """Return True if adding transition conserves determinism."""
        if not self._deterministic:
            return

        if from_state not in self.graph.states:
            raise Exception('from_state \notin graph')

        same_labeled = self.find([from_state], with_attr_dict=sublabels)

        if same_labeled:
            msg = (
                'Candidate transition violates determinism.\n'
                'Existing transitions with same label:\n' +
                str(same_labeled)
            )
            raise Exception(msg)

    def add(self, from_state, to_state, attr_dict=None, check=True, **attr):
        """Wrapper of L{LabeledDiGraph.add_edge},

        which wraps C{networkx.MultiDiGraph.add_edge}.
        """
        # self._breaks_determinism(from_state, labels)
        self.graph.add_edge(from_state, to_state,
                            attr_dict=attr_dict, check=check, **attr)

    def add_from(self, transitions, attr_dict=None, check=True, **attr):
        """Wrapper of L{LabeledDiGraph.add_edges_from},

        which wraps C{networkx.MultiDiGraph.add_edges_from}.
        """
        self.graph.add_edges_from(transitions, attr_dict=attr_dict,
                                  check=check, **attr)

    def add_comb(self, from_states, to_states, attr_dict=None,
                 check=True, **attr):
        """Add an edge for each combination C{(u, v)},

        for C{u} in C{from_states} for C{v} in C{to_states}.
        """
        for u in from_states:
            for v in to_states:
                self.graph.add_edge(u, v, attr_dict=attr_dict,
                                    check=check, **attr)

    def remove(self, from_state, to_state, attr_dict=None, **attr):
        """Remove single transition.

        If only the states are passed,
        then all transitions between them are removed.

        If C{attr_dict}, C{attr} are also passed,
        then only transitions annotated with those labels are removed.

        Wraps L{LabeledDiGraph.remove_labeled_edge}.
        """
        self.graph.remove_labeled_edge(from_state, to_state, attr_dict, **attr)

    def remove_from(self, transitions):
        """Remove list of transitions.

        Each transition is either a:

          - 2-tuple: (u, v), or a
          - 3-tuple: (u, v, data)
        """
        self.graph.remove_labeled_edges_from(transitions)

    def add_adj(
            self, adj, adj2states, attr_dict=None,
            check=True, **attr
    ):
        """Add multiple labeled transitions from adjacency matrix.

        The label can be empty.
        For more details see L{add}.

        @param adj: new transitions represented by adjacency matrix.
        @type adj: scipy.sparse.lil (list of lists)

        @param adj2states: map from adjacency matrix indices to states.
            If value not a state, raise Exception.
            Use L{States.add}, L{States.add_from} to add states first.

            For example the 1st state in adj2states corresponds to
            the first node in C{adj}.

            States must have been added using:

               - sys.states.add, or
               - sys.states.add_from

            If C{adj2states} includes a state not in sys.states,
            no transition is added and an exception raised.
        @type adj2states: either of:
            - C{dict} from adjacency matrix indices to
              existing, or
            - C{list} of existing states
        """
        # square ?
        if adj.shape[0] != adj.shape[1]:
            raise Exception('Adjacency matrix must be square.')
        # check states exist, before adding any transitions
        for state in adj2states:
            if state not in self.graph:
                raise Exception(
                    'State: ' + str(state) + ' not found.'
                    ' Consider adding it with sys.states.add')
        # convert to format friendly for edge iteration
        nx_adj = nx.from_scipy_sparse_matrix(
            adj, create_using=nx.DiGraph())
        # add each edge using existing checks
        for i, j in nx_adj.edges_iter():
            si = adj2states[i]
            sj = adj2states[j]
            self.add(si, sj, attr_dict, check, **attr)

    def find(self, from_states=None, to_states=None,
             with_attr_dict=None, typed_only=False, **with_attr):
        """Find all edges between given states with given labels.

        Instead of having two separate methods to:

          - find all labels of edges between given states (s1, s2)

          - find all transitions (s1, s2, L) with given label L,
                possibly from some given state s1,
                i.e., the edges leading to the successor states
                Post(s1, a) = Post(s1) restricted by action a

        this method provides both functionalities.

        Preimage under edge labeling function L of given label,
        intersected with given subset of edges::
            L^{-1}(desired_label) \\cap (from_states x to_states)

        See Also
        ========
        L{add}, L{add_adj}

        @param from_states: edges must start from this subset of states
        @type from_states:
            - iterable of existing states, or
            - None (no constraint, default)

        @param to_states: edges must end in this subset of states
        @type to_states:
            - iterable of existing states, or
            - None (no constraint, default)

        @param with_attr_dict: edges must be annotated with these labels
        @type with_attr_dict:
            - {label_type : desired_label_value, ...}, or
            - None (no constraint, default)

        @param with_attr: label type-value pairs,
            take precedence over C{desired_label}.

        @return: set of transitions = labeled edges::
                (from_state, to_state, label)
        such that::
                (from_state, to_state )
                in from_states x to_states

        @rtype: list of transitions::
                = list of labeled edges
                = [(from_state, to_state, label),...]
        where:
          - C{from_state} in C{from_states}
          - C{to_state} in C{to_states}
          - C{label}: dict
        """
        if with_attr_dict is None:
            with_attr_dict = with_attr
        try:
            with_attr_dict.update(with_attr)
        except:
            raise TypeError('with_attr_dict must be a dict')
        found_transitions = []
        u_v_edges = self.graph.edges_iter(nbunch=from_states, data=True)
        if to_states is not None:
            u_v_edges = [(u, v, d)
                         for u, v, d in u_v_edges
                         if v in to_states]
        for u, v, attr_dict in u_v_edges:
            ok = True
            if not with_attr_dict:
                logger.debug('Any label is allowed.')
            elif not attr_dict:
                logger.debug('No labels defined.')
            else:
                logger.debug('Checking guard.')
                ok = label_is_desired(attr_dict, with_attr_dict)
            if ok:
                logger.debug('Transition label matched desired label.')
                transition = (u, v, dict(attr_dict))
                found_transitions.append(transition)
        return found_transitions


class LabeledDiGraph(nx.MultiDiGraph):
    """Directed multi-graph with constrained labeling.

    Provides facilities to define labeling functions on
    vertices and edges, with given co-domains,
    so that labels are type-checked, not arbitrary.

    Each state (or edge) is annotated with labels.
    Before removing a value from a label type,
    first make sure no state (or edge) is labeled with it.

    Multiple edges with the same C{attr_dict} are not possible.
    So the difference from C{networkx.MultiDiGraph} is that
    the C{dict} of edges between u,v is a bijection.

    Between two nodes either:

      - a single unlabeled edge exists (no labeling constraints), or
      - labeled edges exist

    but mixing labeled with unlabeled edges for the same
    edge is not allowed, to simplify and avoid confusion.


    Label types by example
    ======================

    Use a C{dict} for each label type you want to define,
    like this:

      >>> types = [
              {'name': 'drink',
               'values': {'tea', 'coffee'},
               'setter': True,
               'default': 'tea'}]

    This will create a label type named C{'drink'} that can
    take the values C{'tea'} and C{'coffee'}.

    Assuming this label type applies to nodes,
    you can now label a new node as:

      >>> g = LabeledDiGraph(types)
      >>> g.add_node(1, drink='coffee')

    If you omit the label when adding a new node,
    it gets the default value:

      >>> g.add_node(2)
      >>> g.node[2]
      {'drink': 'tea'}

    The main difference with vanilla C{networkx} is
    that the dict above includes type checking:

      >>> type(g.node[2])
      tulip.transys.mathset.TypedDict

    The C{'setter'} key with value C{True}
    creates also a field C{g.drink}.
    Be careful to avoid name conflicts with existing
    networkx C{MultiDiGraph} attributes.

    This allows us to add more values after creating
    the graph:

      >>> g.drink
      {'coffee', 'tea'}
      >>> g.drink.add('water')
      {'coffee', 'tea', 'water'}

    Finally, the graph will prevent us from
    accidentally using an untyped label name,
    by raising an C{AttributeError}:

      >>> g.add_node(3, day='Jan')
      AttributeError: ...

    To add untyped labels, do so explicitly:

      >>> g.add_node(3, day='Jan', check=False)
      >>> g.node[3]
      {'day': 'Jan', 'drink': 'tea'}


    Details on label types
    ======================

    Each label type is defined by a C{dict} that
    must have the keys C{'name'} and C{'values'}:

      - C{'name'}: with C{str} value

      - C{'values' : B} implements C{__contains__}
        used to check label validity.

        If you want the codomain C{B} to be
        extensible even after initialization,
        it must implement method C{add}.

    and optionally the keys:

      - C{'setter': C} with 3 possibilities:

        - if absent,
          then no C{setter} attribute is created

        - otherwise an attribute C{self.A}
          is created, pointing at:

            - the given co-domain C{B}
              if C{C is True}

            - C{C}, otherwise.

      - C{'default': d} is a value in C{B}
        to be returned for node and edge labels
        not yet explicitly specified by the user.

    @param node_label_types: applies to nodes, as described above.
    @type node_label_types: C{list} of C{dict}

    @param edge_label_types: applies to edges, as described above.
    @type node_label_types: C{list} of C{dict}

    @param deterministic: if True, then edge-label-deterministic


    Deprecated dot export
    =====================

    BEWARE: the dot interface will be separated from
    the class itself. Some basic style definitions as
    below may remain, but masking selected labels and
    other features will be accessible only via functions.

    For dot export subclasses must define:

        - _state_label_def
        - _state_dot_label_format

        - _transition_label_def
        - _transition_dot_label_format
        - _transition_dot_mask

    Note: this interface will be improved in the future.


    Credits
    =======

    Some code in overridden methods of C{networkx.MultiDiGraph}
    is adapted from C{networkx}, which is distributed under a BSD license.
    """

    def __init__(
            self,
            node_label_types=None,
            edge_label_types=None,
            deterministic=False):
        node_labeling, node_defaults = self._init_labeling(node_label_types)
        edge_labeling, edge_defaults = self._init_labeling(edge_label_types)

        self._state_label_def = node_labeling
        self._node_label_defaults = node_defaults

        self._transition_label_def = edge_labeling
        self._edge_label_defaults = edge_defaults

        # temporary hack until rename
        self._node_label_types = self._state_label_def
        self._edge_label_types = self._transition_label_def

        nx.MultiDiGraph.__init__(self)

        self.states = States(self)

        # todo: handle accepting states separately
        self.transitions = Transitions(self, deterministic)

        # export properties
        self.dot_node_shape = {'normal': 'circle'}
        self.default_layout = 'dot'

    def _init_labeling(self, label_types):
        """Initialize labeling.

        Note
        ====
        'state' will be renamed to 'node' in the future
        'transition' will be renamed to 'edge' in the future

        @param label_types: see L{__init__}.
        """
        labeling = dict()
        defaults = dict()
        if label_types is None:
            logger.debug('no label types passed')
            return labeling, defaults
        if not label_types:
            logger.warn('empty label types: %s' % str(label_types))
        # define the labeling
        labeling = {d['name']: d['values'] for d in label_types}
        defaults = {d['name']: d.get('default') for d in label_types
                    if 'default' in d}
        setters = {d['name']: d.get('setter') for d in label_types
                   if 'setter' in d}
        for name, setter in setters.iteritems():
            # point to given values ?
            if setter is True:
                setter = labeling[name]
            setattr(self, name, setter)
        return labeling, defaults

    def _check_for_untyped_keys(self, typed_attr, type_defs, check):
        untyped_keys = set(typed_attr).difference(type_defs)
        msg = (
            'checking for untyped keys...\n' +
            'attribute dict: ' + str(typed_attr) + '\n' +
            'type definitions: ' + str(type_defs) + '\n' +
            'untyped_keys: ' + str(untyped_keys))
        logger.debug(msg)
        if untyped_keys:
            msg = (
                'The following edge attributes:\n' +
                str({k: typed_attr[k] for k in untyped_keys}) + '\n' +
                'are not allowed.\n' +
                'Currently the allowed attributes are:' +
                ', '.join([str(x) for x in type_defs]))
            if check:
                msg += ('\nTo set attributes not included ' +
                        'in the existing types, pass: check = False')
                raise AttributeError(msg)
            else:
                msg += '\nAllowed because you passed: check = True'
                logger.warning(msg)
        else:
            logger.debug('no untyped keys.')

    def is_consistent(self):
        """Check if labels are consistent with their type definitions.

        Use case: removing values from a label type
        can invalidate existing labels that use them.

        @rtype: bool
        """
        for node, attr_dict in self.nodes_iter(data=True):
            if not attr_dict.is_consistent():
                return False
        for node_i, node_j, attr_dict in self.edges_iter(data=True):
            if not attr_dict.is_consistent():
                return False
        return True

    def _update_attr_dict_with_attr(self, attr_dict, attr):
        if attr_dict is None:
            attr_dict = attr
        else:
            try:
                attr_dict.update(attr)
            except AttributeError:
                msg = 'The attr_dict argument must be a dictionary.'
                raise nx.NetworkXError(msg)
        return attr_dict

    def add_node(self, n, attr_dict=None, check=True, **attr):
        """Use a L{TypedDict} as attribute dict.

        Overrides C{networkx.MultiDiGraph.add_node},
        see that for details.

        Log warning if node already exists.
        All other functionality remains the same.

        @param check: if True and untyped keys are passed,
            then raise C{AttributeError}.
        """
        # avoid multiple additions
        if n in self:
            logger.debug('Graph already has node: ' + str(n))
        attr_dict = self._update_attr_dict_with_attr(attr_dict, attr)
        # define typed dict
        typed_attr = TypedDict()
        typed_attr.set_types(self._node_label_types)
        typed_attr.update(copy.deepcopy(self._node_label_defaults))
        # type checking happens here
        typed_attr.update(attr_dict)
        logger.debug('node typed_attr: ' + str(typed_attr))
        self._check_for_untyped_keys(typed_attr,
                                     self._node_label_types,
                                     check)
        nx.MultiDiGraph.add_node(self, n, attr_dict=typed_attr)

    def add_nodes_from(self, nodes, check=True, **attr):
        """Create or label multiple nodes.

        Overrides C{networkx.MultiDiGraph.add_nodes_from},
        for details see that and L{LabeledDiGraph.add_node}.
        """
        for n in nodes:
            try:
                n not in self.succ
                node = n
                attr_dict = attr
            except TypeError:
                node, ndict = n
                attr_dict = attr.copy()
                attr_dict.update(ndict)
            self.add_node(node, attr_dict=attr_dict, check=check)

    def add_edge(self, u, v, key=None, attr_dict=None, check=True, **attr):
        """Use a L{TypedDict} as attribute dict.

        Overrides C{networkx.MultiDiGraph.add_edge},
        see that for details.

          - Raise ValueError if C{u} or C{v} are not already nodes.
          - Raise Exception if edge (u, v, {}) exists.
          - Log warning if edge (u, v, attr_dict) exists.
          - Raise ValueError if C{attr_dict} contains
            typed key with invalid value.
          - Raise AttributeError if C{attr_dict} contains untyped keys,
            unless C{check=False}.

        Each label defines a different labeled edge.
        So to "change" the label, either:

            - remove the edge with this label, then add a new one, or
            - find the edge key, then use subscript notation:

                C{G[i][j][key]['attr_name'] = attr_value}

        Notes
        =====
        1. Argument C{key} has been removed compared to
           C{networkx.MultiDiGraph.add_edge}, because edges are defined
           by their labeling, i.e., multiple edges with same labeling
           are not allowed.

        @param check: raise C{AttributeError} if C{attr_dict}
            has untyped attribute keys, otherwise warn
        """
        # legacy
        if 'check_states' in attr:
            msg = 'saw keyword argument: check_states ' +\
                  'which is no longer available, ' +\
                  'firstly add the new nodes.'
            logger.warning(msg)
        # check nodes exist
        if u not in self.succ:
            raise ValueError('Graph does not have node u: ' + str(u))
        if v not in self.succ:
            raise ValueError('Graph does not have node v: ' + str(v))
        attr_dict = self._update_attr_dict_with_attr(attr_dict, attr)
        # define typed dict
        typed_attr = TypedDict()
        typed_attr.set_types(self._edge_label_types)
        typed_attr.update(copy.deepcopy(self._edge_label_defaults))
        # type checking happens here
        typed_attr.update(attr_dict)
        logger.debug('Given: attr_dict = ' + str(attr_dict))
        logger.debug('Stored in: typed_attr = ' + str(typed_attr))
        # may be possible to speedup using .succ
        existing_u_v = self.get_edge_data(u, v, default={})
        if dict() in existing_u_v.values():
            msg = (
                'Unlabeled transition: '
                'from_state-> to_state already exists,\n'
                'where:\t from_state = ' + str(u) + '\n'
                'and:\t to_state = ' + str(v) + '\n')
            raise Exception(msg)
        # check if same labeled transition exists
        if attr_dict in existing_u_v.values():
            msg = (
                'Same labeled transition:\n'
                'from_state---[label]---> to_state\n'
                'already exists, where:\n'
                '\t from_state = ' + str(u) + '\n'
                '\t to_state = ' + str(v) + '\n'
                '\t label = ' + str(typed_attr) + '\n')
            warnings.warn(msg)
            logger.warning(msg)
            return
        # self._breaks_determinism(from_state, labels)
        self._check_for_untyped_keys(typed_attr,
                                     self._edge_label_types,
                                     check)
        # the only change from nx in this clause is using TypedDict
        logger.debug('adding edge: ' + str(u) + ' ---> ' + str(v))
        if v in self.succ[u]:
            msg = 'there already exist directed edges with ' +\
                  'same end-points'
            logger.debug(msg)
            keydict = self.adj[u][v]
            # find a unique integer key
            if key is None:
                key = len(keydict)
                while key in keydict:
                    key -= 1
            datadict = keydict.get(key, typed_attr)
            datadict.update(typed_attr)
            keydict[key] = datadict
        else:
            logger.debug('first directed edge between these nodes')
            # selfloops work this way without special treatment
            key = 0
            keydict = {key: typed_attr}
            self.succ[u][v] = keydict
            self.pred[v][u] = keydict

    def add_edges_from(self, labeled_ebunch, attr_dict=None,
                       check=True, **attr):
        """Add multiple labeled edges.

        Overrides C{networkx.MultiDiGraph.add_edges_from},
        see that for details.

        Only difference is that only 2 and 3-tuple edges allowed.
        Keys cannot be specified, because a bijection is maintained.

        @param labeled_ebunch: iterable container of:

            - 2-tuples: (u, v), or
            - 3-tuples: (u, v, label)

          See also L{remove_labeled_edges_from}.
        """
        attr_dict = self._update_attr_dict_with_attr(attr_dict, attr)
        # process ebunch
        for e in labeled_ebunch:
            datadict = dict(attr_dict)
            # distinguish by number of elements given
            ne = len(e)
            if ne == 4:
                u, v, key, dd = e
            elif ne == 3:
                u, v, dd = e
                key = None
            elif ne == 2:
                u, v = e
                dd = {}
                key = None
            else:
                raise ValueError(
                    'Edge tuple %s must be a 2-, 3-, or 4-tuple .' % (e,))
            datadict.update(dd)
            self.add_edge(u, v, key=key, attr_dict=datadict, check=check)

    def remove_labeled_edge(self, u, v, attr_dict=None, **attr):
        """Remove single labeled edge.

        @param attr_dict: attributes with which to identify the edge.
        @type attr_dict: dict

        @param attr: keyword arguments with which to update C{attr_dict}.
        """
        if u not in self:
            return
        if v not in self[u]:
            return
        attr_dict = self._update_attr_dict_with_attr(attr_dict, attr)
        rm_keys = {key for key, data in self[u][v].iteritems()
                   if data == attr_dict}
        for key in rm_keys:
            self.remove_edge(u, v, key=key)

    def remove_labeled_edges_from(self, labeled_ebunch,
                                  attr_dict=None, **attr):
        """Remove labeled edges.

        Example
        =======
        >>> g = LabeledDiGraph()
        >>> g.add_edge(1, 2, day='Mon')
        >>> g.add_edge(1, 2, day='Tue')
        >>> edges = [(1, 2, {'day':'Mon'}),
                     (1, 2, {'day':'Tue'})]
        >>> g.remove_edges_from(edges)

        @param labeled_ebunch: iterable container of edge tuples
            Each edge tuple can be:

              - 2-tuple: (u, v) All edges between u and v are removed.
              - 3-tuple: (u, v, attr_dict) all edges between u and v
                  annotated with that C{attr_dict} are removed.
        """
        attr_dict = self._update_attr_dict_with_attr(attr_dict, attr)
        for e in labeled_ebunch:
            datadict = dict(attr_dict)
            ne = len(e)
            if ne == 3:
                u, v, dd = e
                datadict.update(dd)
            elif ne == 2:
                u, v = e
            else:
                raise ValueError(
                    'Edge tuple %s must be a 2- or 3-tuple .' % (e,))
            self.remove_labeled_edge(u, v, attr_dict=datadict)

    def has_deadends(self):
        """Return False if all nodes have outgoing edges.

        Edge labels are not taken into account.
        """
        for n in self:
            if not self.succ[n]:
                return True
        return False

    def remove_deadends(self):
        """Recursively delete nodes with no outgoing transitions."""
        n = len(self)
        s = {1}
        while s:
            s = {n for n in self if not self.succ[n]}
            self.states.remove_from(s)
        m = len(self)
        assert n == 0 or m > 0, 'removed all {n} nodes!'.format(n=n)
        assert n >= 0, 'added {n} nodes'.format(n=n)
        print('removed {r} nodes from '
              '{n} total'.format(r=n - m, n=n))

    def dot_str(self, wrap=10, **kwargs):
        """Return dot string.

        Requires pydot.
        """
        from tulip.transys.export import graph2dot
        return graph2dot.graph2dot_str(self, wrap, **kwargs)

    def save(self, filename=None, fileformat=None,
             rankdir='LR', prog=None,
             wrap=10, tikz=False):
        """Save image to file.

        Recommended file formats:

            - tikz (via dot2tex)
            - pdf
            - svg
            - dot
            - png

        Any other format supported by C{pydot.write} is available.

        Experimental:

            - html (uses d3.js)
            - 'scxml'

        Requires
        ========
          - graphviz dot: http://www.graphviz.org/
          - pydot: https://pypi.python.org/pypi/pydot

        and for tikz:

          - dot2tex: https://pypi.python.org/pypi/dot2tex
          - dot2texi: http://www.ctan.org/pkg/dot2texi
            (to automate inclusion)

        See Also
        ========
        L{plot}, C{pydot.Dot.write}

        @param filename: file path to save image to
            Default is C{self.name}, unless C{name} is empty,
            then use 'out.pdf'.

            If extension is missing '.pdf' is used.
        @type filename: str

        @param fileformat: replace the extension of C{filename}
            with this. For example::

                filename = 'fig.pdf'
                fileformat = 'svg'

            result in saving 'fig.svg'

        @param rankdir: direction for dot layout
        @type rankdir: str = 'TB' | 'LR'
            (i.e., Top->Bottom | Left->Right)

        @param prog: executable to call
        @type prog: dot | circo | ... see pydot.Dot.write

        @param wrap: max width of node strings
        @type wrap: int

        @param tikz: use tikz automata library in dot

        @rtype: bool
        @return: True if saving completed successfully, False otherwise.
        """
        if filename is None:
            if not self.name:
                filename = 'out'
            else:
                filename = self.name
        fname, fextension = os.path.splitext(filename)
        # default extension
        if not fextension or fextension is '.':
            fextension = '.pdf'
        if fileformat:
            fextension = '.' + fileformat
        filename = fname + fextension
        # drop '.'
        fileformat = fextension[1:]
        # check for html
        if fileformat is 'html':
            from tulip.transys.export import save_d3
            return save_d3.labeled_digraph2d3(self, filename)
        # subclass has extra export formats ?
        if hasattr(self, '_save'):
            if self._save(filename, fileformat):
                return True
        if prog is None:
            prog = self.default_layout
        from tulip.transys.export import graph2dot
        graph2dot.save_dot(self, filename, fileformat, rankdir,
                           prog, wrap, tikz=tikz)
        return True

    def plot(self, rankdir='LR', prog=None, wrap=10, ax=None):
        """Plot image using dot.

        No file I/O involved.
        Requires GraphViz dot and either Matplotlib or IPython.

        NetworkX does not yet support plotting multiple edges between 2 nodes.
        This method fixes that issue, so users don't need to look at files
        in a separate viewer during development.

        See Also
        ========
        L{save}

        Depends
        =======
        dot and either of IPython or Matplotlib
        """
        # anything to plot ?
        if not self.states:
            print(
                60 * '!' +
                "\nThe system doesn't have any states to plot.\n" +
                60 * '!')
            return
        if prog is None:
            prog = self.default_layout
        from tulip.transys.export import graph2dot
        return graph2dot.plot_pydot(self, prog, rankdir, wrap, ax=ax)


def str2singleton(ap_label):
    """If string, convert to set(string).

    Convention: singleton str {'*'}
    can be passed as str '*' instead.
    """
    if isinstance(ap_label, str):
        logger.debug('Saw str state label:\n\t' + str(ap_label))
        ap_label = {ap_label}
        logger.debug('Replaced with singleton:\n\t' + str(ap_label) + '\n')
    return ap_label


def prepend_with(states, prepend_str):
    """Prepend items with given string.

    Example
    =======
    >>> states = [0, 1]
    >>> prepend_str = 's'
    >>> states = prepend_with(states, prepend_str)
    >>> assert(states == ['s0', 's1'] )

    See Also
    ========
    L{tuple2ba}, L{tuple2fts}

    @param states: items prepended with string C{prepend_str}
    @type states: iterable

    @param prepend_str: text prepended to C{states}.  If None, then
        C{states} is returned without modification
    @type prepend_str: str or None
    """
    if not isinstance(states, Iterable):
        raise TypeError('states must be Iterable. Got:\n\t' +
                        str(states) + '\ninstead.')
    if not isinstance(prepend_str, str) and prepend_str is not None:
        raise TypeError('prepend_str must be of type str. Got:\n\t' +
                        str(prepend_str) + '\ninstead.')
    if prepend_str is None:
        return states
    return [prepend_str + str(s) for s in states]
