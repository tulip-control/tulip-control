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
import collections.abc as _abc
import copy
import logging
import os
import pprint as _pp
import typing as _ty

import networkx as nx

import tulip.transys.export.graph2dot as graph2dot
import tulip.transys.export.save_d3 as save_d3
import tulip.transys.mathset as _mset


__all__ = [
    'LabeledDiGraph',
    'prepend_with']


logger = logging.getLogger(__name__)


def label_is_desired(
        attr_dict:
            dict,
        desired_dict:
            dict
        ) -> bool:
    """Return True if all labels match.

    Supports symbolic evaluation, if label type is callable.
    """
    if not isinstance(attr_dict, _mset.TypedDict):
        raise TypeError(
            'attr_dict must be TypedDict'
            f', instead: {type(attr_dict)}')
    if attr_dict == desired_dict:
        return True
    # different keys ?
    mismatched_keys = set(attr_dict).symmetric_difference(
        desired_dict)
    if mismatched_keys:
        return False
    # any labels have symbolic semantics ?
    label_def = attr_dict.allowed_values
    for type_name, value in attr_dict.items():
        logger.debug(
            'Checking label type:\n'
            f'\t{type_name}')
        type_def = label_def[type_name]
        desired_value = desired_dict[type_name]
        if callable(type_def):
            logger.debug(
                'Found label semantics:\n'
                f'\t{type_def}')
            # value = guard
            if not type_def(value, desired_value):
                return False
            else:
                continue
        # no guard semantics given,
        # then by convention:
        # guard is singleton {cur_val},
        if value != desired_value:
            test_common_bug(value, desired_value)
            return False
    return True


def test_common_bug(
        value,
        desired_value
        ) -> None:
    logger.debug(
        f'Label value:\n\t{value}')
    logger.debug(
        'Desired value:\n'
        f'\t{desired_value}')
    if (
        isinstance(value, (set, list)) and
        isinstance(desired_value, (set, list)) and
        value.__class__ != desired_value.__class__
    ):
        msg = (
            'Set SubLabel:\n'
            f'\t{value}'
            'compared to list SubLabel:\n'
            f'\t{desired_value}'
            'Did you mix sets & lists when setting AP labels ?')
        raise Exception(msg)


class States:
    """Methods to manage states and initial states."""

    def __init__(
            self,
            graph:
                'LabeledDiGraph'):
        """Initialize `States`."""
        self.graph = graph
        self.initial = list()

    def __getitem__(self, state):
        return self.graph.nodes[state]

    def __call__(self, *args, **kwargs):
        """Return list of states.

        For more details see `LabeledDiGraph.nodes`
        """
        return self.graph.nodes(*args, **kwargs)

    def __str__(self):
        states = _pp.pformat(self(data=False))
        return f'States:\n{states}'

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
    def initial(self) -> _mset.SubSet:
        """ Return `SubSet` of initial states."""
        return self._initial

    @initial.setter
    def initial(self, states):
        s = _mset.SubSet(self)
        s.update(states)
        self._initial = s

    def _single_state2singleton(
            self,
            state
            ) -> list:
        """Convert to a singleton list, if argument is a single state.

        Otherwise return given argument.
        """
        if state in self:
            states = [state]
        else:
            states = state
        return states

    def add(
            self,
            new_state,
            attr_dict:
                dict |
                None=None,
            check:
                bool=True,
            **attr):
        """Wraps `LabeledDiGraph.add_node`,

        which wraps `networkx.MultiDiGraph.add_node`.
        """
        self.graph.add_node(
            new_state, attr_dict, check,
            **attr)

    def add_from(
            self,
            new_states:
                _abc.Iterable,
            check:
                bool=True,
            **attr):
        """Wraps `LabeledDiGraph.add_nodes_from`,

        which wraps `networkx.MultiDiGraph.add_nodes_from`.
        """
        self.graph.add_nodes_from(
            new_states, check,
            **attr)

    def remove(self, state):
        """Remove `state` from states (including initial).

        Wraps `networkx.MultiDiGraph.remove_node`.
        """
        if state in self.initial:
            self.initial.remove(state)
        self.graph.remove_node(state)

    def remove_from(
            self,
            states:
                _abc.Iterable):
        """Remove multiple states.

        Iterates `States.remove` to imitate
        `networkx.MultiDiGraph.remove_nodes_from`,
        handling also initial states.
        """
        any(map(self.remove, states))

    def post(
            self,
            states:
                _abc.Iterable |
                None=None
            ) -> set:
        """Direct successor set (1-hop) for given states.

        Edge labels are ignored.

        Union of Post(s) for s in given states.

        See Also
        ========
        - `pre`
        - Def. 2.3, p.23 [BK08](
            https://tulip-control.sourceforge.io/doc/bibliography.html#bk08)

        @param states:
          - `None`, so initial states returned
          - iterable of states
        """
        if states is None:
            return set(self.initial)
        return set().union(*map(
            self.graph.successors,
            states))

    def pre(
            self,
            states:
                _abc.Iterable
            ) -> set:
        """Return direct predecessors (1-hop) of given state.

        See Also
        ========
        - `post`
        - Def. 2.3, p.23 [BK08](
            https://tulip-control.sourceforge.io/doc/bibliography.html#bk08)
        """
        return set().union(*map(
            self.graph.predecessors,
            states))

    def forward_reachable(
            self,
            state
            ) -> set:
        """Return states reachable from `state`.

        Wrapper of `networkx.descendants()`.
        """
        return nx.descendants(self, state)

    def backward_reachable(
            self,
            state
            ) -> set:
        """Return states from which `state` can be reached.

        Wrapper of `networkx.ancestors()`.
        """
        return nx.ancestors(self, state)

    def paint(
            self,
            state,
            color:
                str):
        """Color the given state.

        The state is filled with given color,
        rendered with dot when plotting and saving.

        @param state:
            valid system state
        @param color:
            DOT color with which to paint `state`
        """
        self.graph.nodes[state]['style'] = 'filled'
        self.graph.nodes[state]['fillcolor'] = color

    def find(
            self,
            states=None,
            with_attr_dict:
                dict |
                None=None,
            **with_attr
            ) -> list[tuple]:
        """Filter by desired states and by desired state labels.

        Examples
        ========
        Assume that the system is:

        ```python
        import transys as trs

        ts = trs.FTS()
        ts.atomic_propositions.add('p')
        ts.states.add('s0', ap={'p'})
        ```

        - To find the label of a single state `'s0'`:

          ```python
          >>> a = ts.states.find(['s0'])
          >>> s0_, label = a[0]
          >>> print(label)
          {'ap': set(['p'])}
          ```

        - To find all states with a specific label `{'p'}`:

          ```python
          >>> ts.states.add('s1', ap={'p'})
          >>> b = ts.states.find(with_attr_dict={'ap':{'p'}})
          >>> states = [state for state, label_ in b]
          >>> print(set(states))
          {'s0', 's1'}
          ```

        - To find all states in subset `M` labeled with `{'p'}`:

          ```python
          >>> ts.states.add('s2', ap={'p'})
          >>> M = {'s0', 's2'}
          >>> b = ts.states.find(M, {'ap': {'p'}})
          >>> states = [state for state, label_ in b]
          >>> print(set(states))
          {'s0', 's2'}
          ```

        @param states:
            subset of states over which to search,
            or a specific state
        @param with_attr_dict:
            label with which to filter the states,
            of the form:
            `{sublabel_type : desired_sublabel_value, ...}`
            (if `None`, then any state label is allowed)
        @param with_attr:
            label key-value pairs which take
            precedence over `with_attr_dict`.
        @return:
            `[(state, label),...]`
            where:
            - `state` \\in `states`
            - `label`: dict
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
                    '`LabeledStates.find` got '
                    f'single state: {state},\n'
                    'instead of `Iterable` of states.\n')
                states = [state]
                msg += f'Replaced given states = {state}'
                msg += f' with states = {states}'
                logger.debug(msg)
        found_state_label_pairs = list()
        for state, attr_dict in self.graph.nodes(data=True):
            logger.debug(
                f'Checking state_id = {state}'
                f', with attr_dict = {attr_dict}')
            if states is not None:
                if state not in states:
                    logger.debug(
                        f'state_id = {state}, not desired.')
                    continue
            logger.debug(
                'Checking state label:\n'
                f'\t attr_dict = {attr_dict}\n'
                ' vs:\n'
                f'\t desired_label = {with_attr_dict}')
            if not with_attr_dict:
                logger.debug('Any label acceptable.')
                ok = True
            else:
                typed_attr = _mset.TypedDict()
                typed_attr.set_types(self.graph._node_label_types)
                typed_attr.update(attr_dict)
                ok = label_is_desired(typed_attr, with_attr_dict)
            if ok:
                logger.debug(
                    f'Label Matched:\n\t{attr_dict}'
                    f' == {with_attr_dict}')
                state_label_pair = (state, dict(attr_dict))
                found_state_label_pairs.append(state_label_pair)
            else:
                logger.debug('No match for label---> state discarded.')
        return found_state_label_pairs

    def is_terminal(
            self,
            state
            ) -> bool:
        """Return `True` if `state` has no outgoing transitions.

        See Also
        ========
        Def. 2.4, p.23 [BK08](
            https://tulip-control.sourceforge.io/doc/bibliography.html#bk08)
        """
        return self.graph.out_degree(state) == 0


class Transitions:
    """Methods for handling labeled transitions.

    Note that a directed edge is an ordered set of nodes.
    Unlike an edge, a transition is a labeled edge.
    """

    def __init__(
            self,
            graph:
                'LabeledDiGraph',
            deterministic:
                bool=False):
        """Initialize `Transitions`."""
        self.graph = graph
        self._deterministic = deterministic

    def __call__(self, **kwargs):
        """Return list of transitions.

        Wraps `LabeledDiGraph.edges()`.
        """
        return self.graph.edges(**kwargs)

    def __str__(self):
        transitions = _pp.pformat(self())
        return f'Transitions:\n{transitions}'

    def __len__(self):
        """Count transitions."""
        return self.graph.number_of_edges()

    def _breaks_determinism(
            self,
            from_state,
            sublabels:
                dict
            ) -> bool:
        """Return `True` if adding transition conserves determinism."""
        if not self._deterministic:
            return False
        if from_state not in self.graph.states:
            raise ValueError(
                r'from_state \notin graph')
        same_labeled = self.find(
            [from_state],
            with_attr_dict=sublabels)
        if same_labeled:
            raise ValueError(
                'Candidate transition violates determinism.\n'
                'Existing transitions with same label:\n'
                f'{same_labeled}')
        return True

    def add(
            self,
            from_state,
            to_state,
            attr_dict:
                dict |
                None=None,
            check:
                bool=True,
            **attr):
        """Wrapper of `LabeledDiGraph.add_edge`,

        which wraps `networkx.MultiDiGraph.add_edge`.
        """
        # self._breaks_determinism(from_state, labels)
        self.graph.add_edge(
            from_state, to_state,
            attr_dict=attr_dict,
            check=check,
            **attr)

    def add_from(
            self,
            transitions:
                _abc.Iterable,
            attr_dict:
                dict |
                None=None,
            check:
                bool=True,
            **attr):
        """Wrapper of `LabeledDiGraph.add_edges_from`,

        which wraps `networkx.MultiDiGraph.add_edges_from`.
        """
        self.graph.add_edges_from(
            transitions,
            attr_dict=attr_dict,
            check=check,
            **attr)

    def add_comb(
            self,
            from_states:
                _abc.Iterable,
            to_states:
                _abc.Iterable,
            attr_dict:
                dict |
                None=None,
            check:
                bool=True,
            **attr):
        """Add an edge for each combination `(u, v)`,

        for `u` in `from_states` for `v` in `to_states`.
        """
        for u in from_states:
            for v in to_states:
                self.graph.add_edge(
                    u, v,
                    attr_dict=attr_dict,
                    check=check,
                    **attr)

    def remove(
            self,
            from_state,
            to_state,
            attr_dict:
                dict |
                None=None,
            **attr):
        """Remove single transition.

        If only the states are passed,
        then all transitions between them are removed.

        If `attr_dict`, `attr` are also passed,
        then only transitions annotated with
        those labels are removed.

        Wraps `LabeledDiGraph.remove_labeled_edge()`.
        """
        self.graph.remove_labeled_edge(
            from_state, to_state, attr_dict,
            **attr)

    def remove_from(
            self,
            transitions:
                _abc.Iterable[tuple]):
        """Remove multiple transitions.

        Each transition is either a:

        - 2-tuple: `(u, v)`, or
        - 3-tuple: `(u, v, data)`
        """
        self.graph.remove_labeled_edges_from(
            transitions)

    def add_adj(
            self,
            adj:
                'scipy.sparse.lil_array',
            adj2states:
                dict |
                list,
            attr_dict:
                dict |
                None=None,
            check:
                bool=True,
            **attr):
        """Add multiple labeled transitions from adjacency matrix.

        The label can be empty.
        For more details see `add`.

        @param adj:
            new transitions represented by adjacency matrix.
        @param adj2states:
            map from adjacency matrix indices to states.
            If value not a state, raise Exception.
            Use `States.add`, `States.add_from` to add states first.

            For example the 1st state in adj2states corresponds to
            the first node in `adj`.

            States must have been added using:

            - `sys.states.add`, or
            - `sys.states.add_from`

            If `adj2states` includes a state not in sys.states,
            no transition is added and an exception raised.

            Either:
            - `dict` that maps adjacency matrix indices
              to existing states, or
            - `list` of existing states
        """
        # square ?
        if adj.shape[0] != adj.shape[1]:
            raise ValueError(
                'Adjacency matrix must be square.')
        # check states exist,
        # before adding any transitions
        for state in adj2states:
            if state in self.graph:
                continue
            raise ValueError(
                f'State: {state} not found.'
                ' Consider adding it with sys.states.add')
        # convert to format friendly for
        # edge iteration
        nx_adj = nx.from_scipy_sparse_array(
            adj, create_using=nx.DiGraph())
        # add each edge using existing checks
        for i, j in nx_adj.edges():
            si = adj2states[i]
            sj = adj2states[j]
            self.add(
                si, sj, attr_dict, check,
                **attr)

    def find(
            self,
            from_states:
                _abc.Iterable |
                None=None,
            to_states:
                _abc.Iterable |
                None=None,
            with_attr_dict:
                dict |
                None=None,
            typed_only:
                bool=False,
            **with_attr
            ) -> list:
        r"""Find all edges between given states with given labels.

        Instead of having two separate methods to:

        - find all labels of edges between given states (s1, s2)

        - find all transitions (s1, s2, L) with given label L,
          possibly from some given state s1,
          i.e., the edges leading to the successor states
          Post(s1, a) = Post(s1) restricted by action a

        this method provides both functionalities.

        Preimage under edge labeling function L of given label,
        intersected with given subset of edges:

        ```
        L^{-1}(desired_label) \\cap (from_states x to_states)
        ```

        See Also
        ========
        `add`, `add_adj`

        @param from_states:
            edges must start from this subset of states
            (existing states)
        @param to_states:
            edges must end in this subset of states
            (existing states)
        @param with_attr_dict:
            edges must be annotated with these labels,
            of the form:
            `{label_type : desired_label_value, ...}`
        @param with_attr:
            label type-value pairs,
            take precedence over `desired_label`.
        @return:
            transitions = labeled edges:
                `(from_state, to_state, label)`
            such that:
                (from_state, to_state)
                in from_states \X to_states
            where:
              - `from_state \in from_states`
              - `to_state \in to_states`
              - `label`: dict
        """
        if with_attr_dict is None:
            with_attr_dict = with_attr
        try:
            with_attr_dict.update(with_attr)
        except:
            raise TypeError(
                '`with_attr_dict` must be a `dict`')
        found_transitions = list()
        u_v_edges = self.graph.edges(
            nbunch=from_states,
            data=True)
        if to_states is not None:
            u_v_edges = [
                (u, v, d)
                for u, v, d in u_v_edges
                if v in to_states]
        for u, v, attr_dict in u_v_edges:
            ok = True
            if not with_attr_dict:
                logger.debug(
                    'Any label is allowed.')
            elif not attr_dict:
                logger.debug(
                    'No labels defined.')
            else:
                logger.debug(
                    'Checking guard.')
                typed_attr = _mset.TypedDict()
                typed_attr.set_types(
                    self.graph._edge_label_types)
                typed_attr.update(attr_dict)
                ok = label_is_desired(
                    typed_attr, with_attr_dict)
            if ok:
                logger.debug(
                    'Transition label matched desired label.')
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

    Multiple edges with the same `attr_dict` are not possible.
    So the difference from `networkx.MultiDiGraph` is that
    the `dict` of edges between u,v is a bijection.

    Between two nodes either:

      - a single unlabeled edge exists (no labeling constraints), or
      - labeled edges exist

    but mixing labeled with unlabeled edges for the same
    edge is not allowed, to simplify and avoid confusion.


    Label types by example
    ======================

    Use a `dict` for each label type you want to define,
    like this:

    ```python
    types = [
        {'name': 'drink',
         'values': {'tea', 'coffee'},
         'setter': True,
         'default': 'tea'}]
    ```

    This will create a label type named `'drink'` that can
    take the values `'tea'` and `'coffee'`.

    Assuming this label type applies to nodes,
    you can now label a new node as:

    ```python
    g = LabeledDiGraph(types)
    g.add_node(1, drink='coffee')
    ```

    If you omit the label when adding a new node,
    it gets the default value:

    ```python
    >>> g.add_node(2)
    >>> g.nodes[2]
    {'drink': 'tea'}
    ```

    The main difference with vanilla `networkx` is
    that the dict above includes type checking:

    ```python
    >>> type(g.nodes[2])
    tulip.transys.mathset.TypedDict
    ```

    The `'setter'` key with value `True`
    creates also a field `g.drink`.
    Be careful to avoid name conflicts with existing
    networkx `MultiDiGraph` attributes.

    This allows us to add more values after creating
    the graph:

    ```python
    >>> g.drink
    {'coffee', 'tea'}
    >>> g.drink.add('water')
    {'coffee', 'tea', 'water'}
    ```

    Finally, the graph will prevent us from
    accidentally using an untyped label name,
    by raising an `AttributeError`:

    ```python
    >>> g.add_node(3, day='Jan')
    AttributeError: ...
    ```

    To add untyped labels, do so explicitly:

    ```python
    >>> g.add_node(3, day='Jan', check=False)
    >>> g.nodes[3]
    {'day': 'Jan', 'drink': 'tea'}
    ```


    Details on label types
    ======================

    Each label type is defined by a `dict` that
    must have the keys `'name'` and `'values'`:

      - `'name'`: with `str` value

      - `'values' : B` implements `__contains__`
        used to check label validity.

        If you want the codomain `B` to be
        extensible even after initialization,
        it must implement method `add`.

    and optionally the keys:

      - `'setter': C` with 3 possibilities:

        - if absent,
          then no `setter` attribute is created

        - otherwise an attribute `self.A`
          is created, pointing at:

            - the given co-domain `B`
              if `C is True`

            - `C`, otherwise.

      - `'default': d` is a value in `B`
        to be returned for node and edge labels
        not yet explicitly specified by the user.

    @param node_label_types:
        applies to nodes, as described above.
    @param edge_label_types:
        applies to edges, as described above.
    @param deterministic:
        if True, then edge-label-deterministic


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

    Some code in overridden methods of `networkx.MultiDiGraph`
    is adapted from `networkx`, which is distributed under a BSD license.
    """

    LabelTypes = list[dict]

    def __init__(
            self,
            node_label_types:
                LabelTypes |
                None=None,
            edge_label_types:
                LabelTypes |
                None=None,
            deterministic:
                bool=False):
        node_labeling, node_defaults = self._init_labeling(
            node_label_types)
        edge_labeling, edge_defaults = self._init_labeling(
            edge_label_types)
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
        self.transitions = Transitions(
            self, deterministic)
        # export properties
        self.dot_node_shape = dict(
            normal='circle')
        self.default_layout = 'dot'

    def add_label_types(
            self,
            label_types:
                LabelTypes,
            is_edge:
                bool):
        """Add label_types to node or edge depending on is_edge param

        @param label_types:
            see `__init__`.
        @param is_edge:
            whether to add label_types to node (False) or edge (True)
        """
        labeling, defaults = self._init_labeling(label_types)
        if is_edge:
            self._edge_label_types.update(labeling)
            self._edge_label_defaults.update(defaults)
        else:
            self._node_label_types.update(labeling)
            self._node_label_defaults.update(defaults)

    def _init_labeling(
            self,
            label_types:
                list[dict] |
                None
            ) -> tuple[
                dict[str, ...],
                dict[str, ...]]:
        """Initialize labeling.

        Note
        ====
        'state' will be renamed to 'node' in the future
        'transition' will be renamed to 'edge' in the future

        @param label_types:
            see `__init__`.
        """
        labeling = dict()
        defaults = dict()
        if label_types is None:
            logger.debug(
                'no label types passed')
            return labeling, defaults
        if not label_types:
            logger.warning(
                f'empty label types: {label_types}')
        # define the labeling
        labeling = {
            d['name']:
                d['values']
            for d in label_types}
        defaults = {
            d['name']:
                d.get('default')
            for d in label_types
            if 'default' in d}
        setters = {
            d['name']:
                d.get('setter')
            for d in label_types
            if 'setter' in d}
        for name, setter in setters.items():
            # point to given values ?
            if setter is True:
                setter = labeling[name]
            setattr(self, name, setter)
        return labeling, defaults

    def _check_for_untyped_keys(
            self,
            typed_attr:
                dict,
            type_defs:
                dict[str, ...],
            check:
                bool):
        untyped_keys = set(typed_attr).difference(type_defs)
        logger.debug(
            'checking for untyped keys...\n'
            f'attribute dict: {typed_attr}\n'
            f'type definitions: {type_defs}\n'
            f'untyped_keys: {untyped_keys}')
        if untyped_keys:
            edge_attrs = {
                k: typed_attr[k]
                for k in untyped_keys}
            allowed_attrs = ', '.join(map(
                str, type_defs))
            msg = (
                'The following edge attributes:\n'
                f'{edge_attrs}\n'
                'are not allowed.\n'
                'Currently the allowed attributes are:'
                f'{allowed_attrs}')
            if check:
                msg += ('\nTo set attributes not included '
                        'in the existing types, pass: check = False')
                raise AttributeError(msg)
            else:
                msg += '\nAllowed because you passed: check = False'
                logger.warning(msg)
        else:
            logger.debug('no untyped keys.')

    def is_consistent(self) -> bool:
        """Check if labels are consistent with their type definitions.

        Use case: removing values from a label type
        can invalidate existing labels that use them.
        """
        for node, attr_dict in self.nodes(data=True):
            if not attr_dict.is_consistent():
                return False
        edges = self.edges(data=True)
        for node_i, node_j, attr_dict in edges:
            if not attr_dict.is_consistent():
                return False
        return True

    def _update_attr_dict_with_attr(
            self,
            attr_dict:
                dict |
                None,
            attr:
                dict):
        if attr_dict is None:
            attr_dict = attr
        else:
            try:
                attr_dict.update(attr)
            except AttributeError:
                msg = 'The `attr_dict` argument must be a dictionary.'
                raise nx.NetworkXError(msg)
        return attr_dict

    def add_node(
            self,
            n,
            attr_dict:
                dict |
                None=None,
            check:
                bool=True,
            **attr):
        """Use a `TypedDict` as attribute dict.

        Overrides `networkx.MultiDiGraph.add_node`,
        see that for details.

        Log warning if node already exists.
        All other functionality remains the same.

        @param check:
            if True and untyped keys are passed,
            then raise `AttributeError`.
        """
        attr_dict = self._update_attr_dict_with_attr(
            attr_dict, attr)
        # define typed dict
        typed_attr = _mset.TypedDict()
        typed_attr.set_types(self._node_label_types)
        typed_attr.update(
            copy.deepcopy(
                self._node_label_defaults))
        # type checking happens here
        typed_attr.update(attr_dict)
        self._check_for_untyped_keys(
            typed_attr,
            self._node_label_types,
            check)
        nx.MultiDiGraph.add_node(
            self, n,
            **typed_attr)

    def add_nodes_from(
            self,
            nodes:
                _abc.Iterable,
            check:
                bool=True,
            **attr):
        """Create or label multiple nodes.

        Overrides `networkx.MultiDiGraph.add_nodes_from`,
        for details see that and `LabeledDiGraph.add_node`.
        """
        for n in nodes:
            try:
                n not in self._succ
                node = n
                attr_dict = attr
            except TypeError:
                node, ndict = n
                attr_dict = attr.copy()
                attr_dict.update(ndict)
            self.add_node(
                node,
                attr_dict=attr_dict,
                check=check)

    def add_edge(
            self,
            u,
            v,
            key=None,
            attr_dict:
                dict |
                None=None,
            check:
                bool=True,
            **attr):
        """Use a `TypedDict` as attribute dict.

        Overrides `networkx.MultiDiGraph.add_edge`,
        see that for details.

          - Raise ValueError if `u` or `v` are not already nodes.
          - Raise Exception if edge (u, v, {}) exists.
          - Log warning if edge (u, v, attr_dict) exists.
          - Raise ValueError if `attr_dict` contains
            typed key with invalid value.
          - Raise AttributeError if `attr_dict` contains untyped keys,
            unless `check=False`.

        Each label defines a different labeled edge.
        So to "change" the label, either:

            - remove the edge with this label, then add a new one, or
            - find the edge key, then use subscript notation:

                `G[i][j][key]['attr_name'] = attr_value`

        Notes
        =====

        @param check:
            raise `AttributeError` if `attr_dict`
            has untyped attribute keys, otherwise warn
        """
        # legacy
        if 'check_states' in attr:
            logger.warning(
                'saw keyword argument: check_states '
                'which is no longer available, '
                'firstly add the new nodes.')
        # check nodes exist
        if u not in self._succ:
            raise ValueError(
                f'Graph does not have node u: {u}')
        if v not in self._succ:
            raise ValueError(
                f'Graph does not have node v: {v}')
        attr_dict = self._update_attr_dict_with_attr(
            attr_dict, attr)
        # define typed dict
        typed_attr = _mset.TypedDict()
        typed_attr.set_types(self._edge_label_types)
        typed_attr.update(
            copy.deepcopy(
                self._edge_label_defaults))
        # type checking happens here
        typed_attr.update(attr_dict)
        existing_u_v = self.get_edge_data(
            u, v,
            default=dict())
        if dict() in existing_u_v.values():
            raise Exception(
                'Unlabeled transition: '
                'from_state-> to_state already exists,\n'
                f'where:\t from_state = {u}\n'
                f'and:\t to_state = {v}\n')
        # check if same labeled transition exists
        if attr_dict in existing_u_v.values():
            logger.warning(
                'Same labeled transition:\n'
                'from_state---[label]---> to_state\n'
                'already exists, where:\n'
                f'\t from_state = {u}\n'
                f'\t to_state = {v}\n'
                f'\t label = {typed_attr}\n')
            return
        # self._breaks_determinism(from_state, labels)
        self._check_for_untyped_keys(
            typed_attr,
            self._edge_label_types,
            check)
        # the only change from nx in
        # this clause is using TypedDict
        logger.debug(f'adding edge: {u} ---> {v}')
        if key is None:
            key = self.new_edge_key(u, v)
        if v in self._succ[u]:
            keydict = self._adj[u][v]
            datadict = keydict.get(key, typed_attr)
            datadict.update(typed_attr)
            nx.MultiDiGraph.add_edge(
                self, u, v, key,
                **datadict)
        else:
            # selfloops work this way
            # without special treatment
            nx.MultiDiGraph.add_edge(
                self, u, v,
                **typed_attr)

    def add_edges_from(
            self,
            labeled_ebunch:
                _abc.Iterable,
            attr_dict:
                dict |
                None=None,
            check:
                bool=True,
            **attr):
        """Add multiple labeled edges.

        Overrides `networkx.MultiDiGraph.add_edges_from`,
        see that for details.

        Only difference is that only 2 and 3-tuple edges allowed.
        Keys cannot be specified, because a bijection is maintained.

        @param labeled_ebunch:
            iterable container of:

            - 2-tuples: (u, v), or
            - 3-tuples: (u, v, label)

          See also `remove_labeled_edges_from`.
        """
        attr_dict = self._update_attr_dict_with_attr(
            attr_dict, attr)
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
                dd = dict()
                key = None
            else:
                raise ValueError(
                    f'Edge tuple {e} must '
                    'be a 2-, 3-, or 4-tuple .')
            datadict.update(dd)
            self.add_edge(
                u, v,
                key=key,
                attr_dict=datadict,
                check=check)

    def remove_labeled_edge(
            self,
            u,
            v,
            attr_dict:
                dict |
                None=None,
            **attr):
        """Remove single labeled edge.

        @param attr_dict:
            attributes with which to identify the edge.
        @param attr:
            keyword arguments with which to update `attr_dict`.
        """
        if u not in self:
            return
        if v not in self[u]:
            return
        attr_dict = self._update_attr_dict_with_attr(
            attr_dict, attr)
        rm_keys = {
            key
            for key, data in
                self[u][v].items()
            if data == attr_dict}
        for key in rm_keys:
            self.remove_edge(
                u, v,
                key=key)

    def remove_labeled_edges_from(
            self,
            labeled_ebunch:
                _abc.Iterable,
            attr_dict:
                dict |
                None=None,
            **attr):
        """Remove labeled edges.

        Example
        =======

        ```python
        g = LabeledDiGraph()
        g.add_edge(1, 2, day='Mon')
        g.add_edge(1, 2, day='Tue')
        edges = [
            (1, 2, {'day':'Mon'}),
            (1, 2, {'day':'Tue'})]
        g.remove_edges_from(edges)
        ```

        @param labeled_ebunch:
            iterable container of edge `tuple`s
            Each edge `tuple` can be:

            - 2-`tuple`: `(u, v)` All edges between
              `u` and `v` are removed.
            - 3-`tuple`: `(u, v, attr_dict)` all edges
              between `u` and `v` annotated with
              that `attr_dict` are removed.
        """
        attr_dict = self._update_attr_dict_with_attr(
            attr_dict, attr)
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
                            f'Edge tuple {e} must '
                            'be a 2- or 3-tuple .')
            self.remove_labeled_edge(
                u, v,
                attr_dict=datadict)

    def has_deadends(self) -> bool:
        """Return `False` if all nodes have outgoing edges.

        Edge labels are not taken into account.
        """
        for _, d in self.out_degree():
            if d == 0:
                return True
        return False

    def remove_deadends(self):
        """Recursively delete nodes with no outgoing transitions."""
        n = len(self)
        s = {1}
        while s:
            s = {u for u, d in self.out_degree()
                 if d == 0}
            self.states.remove_from(s)
        m = len(self)
        assert n == 0 or m > 0, (
            f'removed all {n} nodes!\n'
            ' Please check env_init and env_safety to avoid trivial'
            ' realizability. Alternatively, you can set "rm_deadends = 0"'
            ' in the options for "synthesize" to get the trivial strategy.')
        assert n >= 0, f'added {n} nodes'
        print(f'removed {n - m} nodes from '
              f'{n} total')

    def dot_str(
            self,
            wrap:
                int=10,
            **kwargs
            ) -> str:
        """Return dot string."""
        return graph2dot.graph2dot_str(
            self, wrap,
            **kwargs)

    def save(
            self,
            filename:
                str |
                None=None,
            fileformat:
                _ty.Literal[
                    'pdf',
                    'svg',
                    'png',
                    'dot'] |
                None=None,
            rankdir:
                _ty.Literal[
                    'LR',
                    'TB']
                ='LR',
            prog=None,
            wrap:
                int=10,
            tikz:
                bool=False
            ) -> bool:
        """Save image to file.

        Recommended file formats:

        - tikz (via dot2tex)
        - pdf
        - svg
        - dot
        - png

        Experimental:

        - html (uses d3.js)
        - 'scxml'

        Requires
        ========
          - graphviz `dot`: <http://www.graphviz.org>
          - Python package `graphviz`:
            <https://pypi.org/project/graphviz>

        and for tikz:

        - `dot2tex`: <https://pypi.python.org/pypi/dot2tex>
        - `dot2texi`: <http://www.ctan.org/pkg/dot2texi>
          (to automate inclusion)

        See Also
        ========
        `plot`

        @param filename:
            file path to save image to
            Default is `self.name`, unless `name` is empty,
            then use 'out.pdf'.

            If extension is missing '.pdf' is used.
        @param fileformat:
            replace the extension of `filename`
            with this. For example:

            ```python
            filename = 'fig.pdf'
            fileformat = 'svg'
            ```

            result in saving `'fig.svg'`.
        @param rankdir:
            direction for dot layout
        @param prog:
            executable to call
        @type prog:
            dot | circo | ... read GraphViz
            documentation
        @param wrap:
            max width of node strings
        @param tikz:
            use tikz automata library in dot
        @return:
            `True` if saving completed successfully,
            `False` otherwise.
        """
        if filename is None:
            if not self.name:
                filename = 'out'
            else:
                filename = self.name
        fname, fextension = os.path.splitext(filename)
        # default extension
        if not fextension or fextension == '.':
            fextension = '.pdf'
        if fileformat:
            fextension = f'.{fileformat}'
        filename = fname + fextension
        # drop '.'
        fileformat = fextension[1:]
        # check for html
        if fileformat == 'html':
            return save_d3.labeled_digraph2d3(self, filename)
        # subclass has extra export formats ?
        if hasattr(self, '_save'):
            if self._save(filename, fileformat):
                return True
        if prog is None:
            prog = self.default_layout
        graph2dot.save_dot(
            self, filename,
            fileformat, rankdir,
            prog, wrap,
            tikz=tikz)
        return True

    def plot(
            self,
            rankdir:
                _ty.Literal[
                    'LR',
                    'TB']
                ='LR',
            prog:
                _ty.Literal[
                    'dot',
                    'neato',
                    'twopi',
                    'circo',
                    'sfdp'] |
                None=None,
            wrap:
                int=10,
            ax=None):
        """Plot image using dot.

        No file I/O involved.
        Requires GraphViz dot and either Matplotlib or IPython.

        NetworkX does not yet support plotting multiple edges between 2 nodes.
        This method fixes that issue, so users don't need to look at files
        in a separate viewer during development.

        See Also
        ========
        `save`

        Depends
        =======
        dot and either of IPython or Matplotlib
        """
        # anything to plot ?
        if not self.states:
            hline = 60 * '!'
            print(
                f"{hline}\n"
                "The system does not have "
                f"any states to plot.\n{hline}")
            return
        if prog is None:
            prog = self.default_layout
        return graph2dot.plot_dot(
            self, prog, rankdir, wrap, ax=ax)


def str2singleton(ap_label) -> set:
    """If string, convert to set(string).

    Convention: singleton str {'*'}
    can be passed as str '*' instead.
    """
    if isinstance(ap_label, str):
        logger.debug(
            'Saw str state label:\n'
            f'\t{ap_label}')
        ap_label = {ap_label}
        logger.debug(
            'Replaced with singleton:\n'
            f'\t{ap_label}\n')
    return ap_label


def prepend_with(
        states:
            _abc.Iterable,
        prepend_str:
            str |
            None
        ) -> list[str]:
    """Prepend items with given string.

    Example
    =======

    ```python
    states = [0, 1]
    prepend_str = 's'
    states = prepend_with(states, prepend_str)
    assert states == ['s0', 's1']
    ```

    See Also
    ========
    `tuple2ba`, `tuple2fts`

    @param states:
        items prepended with string `prepend_str`
    @param prepend_str:
        text prepended to `states`.  If None, then
        `states` is returned without modification
    """
    if not isinstance(states, _abc.Iterable):
        raise TypeError(
            'states must be Iterable. '
            f'Got:\n\t{states}\ninstead.')
    if not isinstance(prepend_str, str) and prepend_str is not None:
        raise TypeError(
            '`prepend_str` must be of type `str`. '
            f'Got:\n\t{prepend_str}\ninstead.')
    if prepend_str is None:
        return states
    return [f'{prepend_str}{s}' for s in states]
