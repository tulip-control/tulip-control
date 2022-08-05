# Copyright (c) 2013-2015 by California Institute of Technology
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
"""Mathematical Sets and Power Sets."""
import collections.abc as _abc
import itertools as _itr
import logging
import operator as _op
import pprint as _pp
import random
import typing as _ty
import warnings


__all__ = [
    'MathSet',
    'SubSet',
    'PowerSet',
    'TypedDict']


_logger = logging.getLogger(__name__)


def compare_lists(
        list1:
            list,
        list2:
            list
        ) -> bool:
    """Compare list contents, ignoring ordering.

    Hashability of elements not assumed, incurring `O(N**2)`

    Relevant
    ========
    `MathSet`

    @return:
        `True` if a bijection exists between the lists.
        Note that this takes into account multiplicity of elements.
    """
    if not isinstance(list1, list):
        raise TypeError(
            f'Not a list, instead list1:\n\t{list1}')
    if not isinstance(list2, list):
        raise TypeError(
            f'Not a list, instead list2:\n\t{list2}')
    dummy_list = list(list1)
    same_lists = True
    for item in list2:
        try:
            dummy_list.remove(item)
        except:
            # unique element not found to be rm'd
            same_lists = False
            break
    # anything remaining ?
    same_lists = same_lists and not bool(dummy_list)
    return same_lists


class MathSet:
    """Mathematical set, allows unhashable elements.

    Examples
    ========

    ```python
    s = MathSet(['a', 1, [1,2], {'a', 'b'} ] )
    ```

    Then print(s) shows how the elements were separately stored
    in a set and list, to optimize contains operations:

    ```python
    >>> print(s)
    MathSet(['a', 1, [1, 2], set(['a', 'b'])])
    ```

    Set operations similar to the builtin type are supported:

    ```python
    p = MathSet()
    p.add(1)
    p |= [1, 2]
    p |= {3, 4}
    p |= [[1, 2], '5', {'a': 1}]
    p.add_from([5, 6, '7', {8, 9}])
    p.remove(1)
    print(p)
    ```

    The output is:

    ```python
    MathSet([2, 3, 4, 5, 6, '5', '7', [1, 2], {'a': 1}, set([8, 9])])
    ```

    Relevant
    ========
    `SubSet`, `PowerSet`, set
    """

    def __init__(
            self,
            iterable:
                _abc.Iterable=None):
        """Initialize by adding elements from iterable.

        Example
        =======

        ```python
        s = MathSet([1, 2, 'a', {3, 4}])
        ```

        @param iterable:
            iterable from which to initialize the set S
            which underlies the PowerSet 2^S
        """
        if iterable is None:
            iterable = list()
        self._delete_all()
        self.add_from(iterable)

    def __repr__(self):
        return (
            f'MathSet({_pp.pformat(list(self._set))}'
            f'{self._list})')

    def _debug_repr(self) -> str:
        set_str = ', '.join(map(
            repr, self._set))
        return f'MathSet({{{set_str}}} +{self._list})'

    def __or__(
            self,
            other:
                _abc.Iterable
            ) -> 'MathSet':
        """Union with another mathematical set.

        See Also
        ========
        `__ior__`

        @param other:
            any other mathematical set.
        @return:
            self | iterable
        """
        s = MathSet(self)
        s.add_from(other)
        return s

    def __mul__(
            self,
            other:
                'MathSet'
            ) -> 'MathSet':
        """Return the Cartesian product with another `MathSet`.

        Example
        =======

        ```python
        a = MathSet([1, 2])
        b = MathSet([3, 4])
        c = a * b
        print(type(c))
        print(c)
        ```

        If we prefer a CartesianProduct returned instead:

        ```
        c = a.cartesian(b)
        ```

        See Also
        ========
        `cartesian`

        @param other:
            set with which to take Cartesian product
        @return:
            Cartesian product of `self` with `other`.
            (explicit construction)
        """
        cartesian = [
            (x, y)
            for x in self
            for y in other]
        return MathSet(cartesian)

    def update(
            self,
            iterable:
                _abc.Iterable
            ) -> None:
        self.add_from(iterable)

    def __ior__(
            self,
            iterable:
                _abc.Iterable
            ) -> 'MathSet':
        """Union with of MathSet with iterable.

        Example
        =======

        ```python
        >>> s = MathSet([1, 2])
        >>> s |= [3, 4]  # much cleaner and familiar
        >>> print(s)
        set([1, 2, 3, 4]) U []
        ```

        See Also
        ========
        `__or__`

        @param iterable:
            any mathematical set.
        @return:
            self | iterable
        """
        self.add_from(iterable)
        return self

    def __sub__(
            self,
            rm_items:
                _abc.Iterable
            ) -> 'MathSet':
        s = MathSet(self)
        print(f'{s =}')
        print(rm_items)
        for item in rm_items:
            if item in s:
                print(f'Removing...: {(item)}')
                s.remove(item)
        return s

    def __isub__(
            self,
            rm_items:
                _abc.Iterable
            ) -> 'MathSet':
        """Delete multiple elements."""
        for item in rm_items:
            if item in self:
                self.remove(item)
        return self

    def __eq__(
            self,
            other:
                'MathSet'
            ) -> bool:
        if not isinstance(other, MathSet):
            raise TypeError(
                'For now comparison only to another MathSet.\n'
                f'Got:\n\t{other}\n of type: '
                f'{type(other)}, instead.')
        same_lists = compare_lists(
            self._list, other._list)
        return (
            self._set == other._set and
            same_lists)

    def __contains__(
            self,
            item
            ) -> bool:
        if isinstance(item, _abc.Hashable):
            try:
                return item in self._set
            except TypeError:
                _logger.error(
                    'UnHashable items within Hashable.')
        return item in self._list

    def __iter__(self):
        return iter(self._list + list(self._set))

    def __len__(self):
        """Number of elements in set."""
        return len(self._set) + len(self._list)

    def _filter_hashables(
            self,
            iterable:
                _abc.Iterable
            ) -> _abc.Iterable:
        return filter(
            lambda x:
                isinstance(x, _abc.Hashable),
            iterable)

    def _filter_unhashables(
            self,
            iterable:
                _abc.Iterable
            ) -> _abc.Iterable:
        return list(filter(
            lambda x:
                not isinstance(x, _abc.Hashable),
            iterable))

    def _delete_all(self) -> None:
        self._set = set()
        self._list = list()

    def add(
            self,
            item
            ) -> None:
        """Add element to mathematical set.

        Example
        =======

        ```python
        >>> s = MathSet()
        >>> s.add(1)
        set([1]) U []
        ```

        See Also
        ========
        `add_from`, `__ior__`, `remove`

        @param item:
            the new set element
            (if hashable it is stored in a Python set,
            otherwise stored in a list)
        """
        if isinstance(item, _abc.Hashable):
            try:
                self._set.add(item)
                return
            except TypeError:
                _logger.error(
                    'UnHashable items within Hashable.')
        if item not in self._list:
            self._list.append(item)
        else:
            _logger.warning(
                'item already in MathSet.')

    def add_from(
            self,
            iterable:
                _abc.Iterable
            ) -> None:
        """Add multiple elements to mathematical set.

        Equivalent to |=

        Example
        =======

        ```python
        s = MathSet()
        s.add_from([1, 2, {3}])
        ```

        is equivalent to:

        ```python
        s = MathSet()
        s |= [1, 2, {3}]
        ```

        See Also
        ========
        `add`, `__ior__`, `remove`

        @param iterable:
            new MathSet elements
        """
        if not isinstance(iterable, _abc.Iterable):
            raise TypeError(
                'Can only add elements to MathSet from Iterable.\n'
                f'Got:\n\t{iterable}\n instead.')
        if isinstance(iterable, MathSet):
            self._set |= set(iterable._set)
            self._list = list(unique(self._list + iterable._list))
            return
        # speed up
        if isinstance(iterable, set):
            self._set |= iterable
            return
        # filter to optimize storage
        try:
            self._set |= set(self._filter_hashables(iterable))
            self._list = list(unique(
                self._list + self._filter_unhashables(iterable)))
            return
        except:
            # ...if contents of elements in iterable are mutable
            self._list = list(unique(self._list + list(iterable)))

    def remove(
            self,
            item
            ) -> None:
        """Remove existing element from mathematical set.

        Example
        =======

        ```python
        >>> p = MathSet([1, 2] )
        >>> p.remove(1)
        >>> p
        set([2]) U []
        ```

        See Also
        ========
        `add`, `add_from`, `__or__`

        @param item:
            An item already in the set.
            For adding items, see add.
        """
        if item not in self:
            warnings.warn(
                'Set element not in set S.\n'
                'Maybe you targeted another element for removal ?')
        if isinstance(item, _abc.Hashable):
            try:
                self._set.remove(item)
                return
            except:
                _logger.debug(
                    f'item: {item}, contains unhashables.')
        self._list.remove(item)

    def pop(self) -> object:
        """Remove and return random MathSet element.

        Raises KeyError if MathSet is empty.
        """
        if not self:
            raise KeyError(
                'Nothing to pop: `MathSet` is empty.')
        if self._set and self._list:
            if random.randint(0, 1):
                return self._set.pop()
            else:
                return self._list.pop()
        elif self._set and not self._list:
            return self._set.pop()
        elif self._list and not self._set:
            return self._list.pop()
        else:
            raise Exception(
                'Bug in empty `MathSet`: not `self` above '
                'should not reaching this point.')

    def intersection(
            self,
            iterable:
                _abc.Iterable
            ) -> 'MathSet':
        """Return intersection with iterable.

        @param iterable:
            find common elements with `self`
        @return:
            intersection of `self` with `iterable`
        """
        s = MathSet()
        for item in iterable:
            if item in self:
                s.add(item)
        return s

    def intersects(
            self,
            iterable:
                _abc.Iterable
            ) -> bool:
        """Check intersection with iterable.

        Checks the existence of common elements with iterable.

        ```python
        s = MathSet()
        s.add(1)
        r = [1, 2]
        assert s.intersects(r)
        ```

        @param iterable:
            with which to check intersection
        @return:
            `True` if `self` has common element with `iterable`.
            Otherwise `False`.
        """
        return any(map(
            self.__contains__, iterable))


class SubSet(MathSet):
    r"""Subset of selected MathSet, or other Iterable.

    Prior to adding new elements,
    it checks that they are in its superset.

    Example
    =======

    ```python
    >>> superset = [1, 2]
    >>> s = SubSet(superset)
    >>> s |= [1, 2]
    >>> print(s)
    SubSet([[1, 2]])
    >>> s.add(3)
    # raises exception because 3 \\notin [1,2]
    ```

    See Also
    ========
    `MathSet`, `PowerSet`
    """

    def __init__(
            self,
            superset:
                _abc.Container,
            iterable:
                _abc.Iterable |
                None=None):
        """Define the superset of this set.

        @param superset:
            This SubSet checked vs `superset`
        @param iterable:
            elements to add to subset
        """
        self._superset = superset
        super().__init__([])
        if not isinstance(superset, _abc.Container):
            raise TypeError(
                'superset must be Iterable,\n'
                f'Got instead:\n\t{superset}')

    def __repr__(self):
        return (
            f'SubSet({_pp.pformat(list(self._set))}'
            f'{self._list})')

    def _debug_repr(self) -> str:
        set_str = ', '.join(map(
            repr, self._set))
        return f'SubSet({{{set_str}}} +{self._list})'

    @property
    def superset(self) -> _abc.Iterable:
        return self._superset

    def add(
            self,
            new_element
            ) -> None:
        """Add state to subset.

        Extends MathSet.add with subset relation checking.

        Example
        =======
        `new_initial_state` should already be a state.
        First use states.add to include it in set of states,
        then states.add_initial.

        See Also
        ========
        `MathSet.add`
        """
        if new_element not in self._superset:
            raise ValueError(
                'New element state \\notin superset.\n'
                'Add it first to states using e.g. sys.states.add()\n'
                'The new element is:\n'
                f'\t{new_element}\n'
                f'and superset:\n\t{self._superset}')
        super().add(new_element)

    def add_from(
            self,
            new_elements:
                _abc.Iterable
            ) -> None:
        """Add multiple new elements to subset.

        Extends MathSet.add_from with subset relation checking.

        Note
        ====
        It would be sufficient to extend only `.add` provided
        `MathSet.add_from` called `.add` iteratively.
        However `MathSet.add_from` filters states, which is
        arguably more efficient. So both `.add` and `.add_from`
        need to be extended here.

        See Also
        ========
        `add`, `__ior__`
        """
        if is_subset(new_elements, self._superset):
            super().add_from(new_elements)
            return
        raise ValueError(
            f'All new_elements:\n\t{new_elements}'
            '\nshould already be \\in '
            f'self.superset = {self._superset}')


class CartesianProduct:
    """List of MathSets, with Cartesian semantics."""

    def __init__(self):
        self.mathsets = list()

    def __contains__(
            self,
            element:
                _abc.Iterable
            ) -> bool:
        # TODO check ordered
        if not isinstance(element, _abc.Iterable):
            raise TypeError(
                'Argument element must be `Iterable`, otherwise cannot '
                'recover which item in it belongs to which set in the '
                'Cartesian product.')
        pairs = zip(self.mathsets, elements)
        return all(map(
            _op.contains, pairs))

    def add(
            self,
            mathset
            ) -> None:
        self.mathsets.append(mathset)

    def add_from(
            self,
            mathsets:
                _abc.Iterable
            ) -> None:
        self.mathsets.extend(mathsets)

    def remove(
            self,
            mathset
            ) -> None:
        self.mathsets.remove(mathset)

    def remove_from(
            self,
            mathsets:
                _abc.Iterable
            ) -> None:
        any(map(self.remove, mathsets))


def unique(
        iterable:
            _abc.Iterable
        ) -> (
            set |
            list):
    """Return unique elements.

    Note
    ====
    Always returning a list for consistency was tempting,
    however this defeats the purpose of creating this function
    to achieve brevity elsewhere in the code.

    @return:
        iterable with duplicates removed, as `set` if possible.
        - If all items in `iterable` are hashable,
            then returns `set`.
        - If iterable contains unhashable item,
            then returns `list` of unique elements.
    """
    # hashable items ?
    try:
        return set(iterable)
    except TypeError:
        pass
    unique_items = list()
    for item in iterable:
        if item not in unique_items:
            unique_items.append(item)
    return unique_items


def contains_multiple(
        iterable:
            _abc.Collection
        ) -> bool:
    """Does iterable contain any item multiple times ?"""
    return len(iterable) != len(unique(iterable))


def is_subset(
        small_iterable:
            _abc.Iterable,
        big_iterable:
            _abc.Iterable
        ) -> bool:
    """Comparison for handling list <= set, and lists with unhashable items."""
    # asserts removed when compiling with optimization on...
    # it would have been elegant to use instead:
    #   assert(isinstance(big_iterable, Iterable))
    # since the error msg is succintly stated by the assert itself
    if not isinstance(big_iterable, _abc.Iterable):
        raise TypeError(
            'big_iterable must be an `Iterable`, '
            'otherwise subset relation undefined.\n'
            f'Got:\n\t{big_iterable}\ninstead.')
    if not isinstance(small_iterable, _abc.Iterable):
        raise TypeError(
            'small_iterable must be Iterable, '
            'otherwise subset relation undefined.\n'
            f'Got:\n\t{small_iterable}\ninstead.')
    # nxor
    if isinstance(small_iterable, str) != isinstance(big_iterable, str):
        raise TypeError(
            'Either both or none of `small_iterable`, '
            '`big_iterable` should be strings.\n'
            'Otherwise subset relation between string '
            'and non-string may introduce bugs.\nGot:\n\t'
            f'{small_iterable},\t{big_iterable}'
            '\ninstead.')
    if not isinstance(big_iterable, _abc.Container):
        big_iterable = list(big_iterable)
    if isinstance(big_iterable, list):
        try:
            big_iterable = set(big_iterable)
        except TypeError:
            pass
    return all(map(
        big_iterable.__contains__,
        small_iterable))


def powerset(
        iterable:
            _abc.Iterable
        ) -> _abc.Iterable:
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)

    From <http://docs.python.org/2/library/itertools.html>,
    also in <https://pypi.python.org/pypi/more-itertools>
    """
    s = list(iterable)
    return _itr.chain.from_iterable(
        _itr.combinations(s, r)
        for r in range(len(s) + 1))


class PowerSet:
    """Efficiently store power set of a mathematical set.

    Set here isn't necessarily a Python set,
    i.e., it may comprise of unhashable elements.

    Example
    =======
    Specify the mathematical set S underlying the PowerSet.

    ```python
    S = [[1, 2], '3', {'a': 1}, 1]
    p = PowerSet(S)

    q = PowerSet()
    q.math_set = S
    ```

    Add new element to underlying set `S`.

    ```python
    p.math_set.add({3: 'a'})
    ```

    Add multiple new elements to underlying set `S`.

    ```python
    p.math_set.add_from({3, 'a'})
    p.math_set |= [1, 2]
    ```

    Remove existing element from set `S`.

    ```python
    p.remove(1)
    ```

    See Also
    ========
    `MathSet`, `SubSet`, `is_subset`

    @param iterable:
        mathematical set `S` of elements,
        on which this `2^S` defined.
    """

    def __init__(
            self,
            iterable:
                _abc.Iterable |
                None=None):
        """Create new `PowerSet` over elements contained in `iterable`.

        This powerset is `2^iterable`.

        @param iterable:
            contains elements of set `iterable`
            underlying the `PowerSet`.
        """
        if iterable is None:
            iterable = list()
        self.math_set = MathSet(iterable)

    def __get__(self, instance, value):
        return self()

    def __repr__(self):
        return f'PowerSet({self.math_set} )'

    def __contains__(self, item):
        r"""Is item \\in 2^iterable = this powerset(iterable)."""
        if not isinstance(item, _abc.Iterable):
            raise Exception(
                f'Not iterable:\n\t{item},\n'
                'this is a powerset, so it contains (math) sets.')
        return is_subset(item, self.math_set)

    def __iter__(self):
        return powerset(self.math_set)

    def __len__(self):
        return 2 ** len(self.math_set)

    def __add__(self, other):
        if not isinstance(other, PowerSet):
            raise TypeError(
                'Addition defined only between PowerSets.\n'
                f'Got instead:\n\t other = {other}')
        list1 = self.math_set
        list2 = other.math_set
        union = list1 | list2
        return PowerSet(union)

    def __eq__(self, other):
        if not isinstance(other, PowerSet):
            raise TypeError(
                'Can only compare to another PowerSet.')
        return other.math_set == self.math_set

    def __setattr__(self, name, value):
        expected_type = (
            name != 'math_set' or
            isinstance(value, MathSet))
        if expected_type:
            object.__setattr__(self, name, value)
            return
        raise TypeError(
            'PowerSet.math_set must be of class MathSet.\n'
            f'Got instead:\n\t{value}'
            f'\nof class:\nt\t{type(value)}')


class TypedDict(dict):
    """dict subclass where values can be constrained by key.

    For each key, a domain can optionally be defined,
    which restricts the admissible values that can be
    paired with that key.

    Example
    =======

    ```python
    d = TypedDict()
    allowed_values = {
        'name': {'Maria', 'John'},
        'age': range(122)}
    default_values = {
        'name': 'Maria',
        'age': 30}
    d.set_types(allowed_types)
    d.update(default_values)
    ```
    """
    # credits for debugging this go here:
    #   <http://stackoverflow.com/questions/2060972/>

    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)
        self.allowed_values = dict()

    def __setitem__(
            self,
            i,
            y
            ) -> None:
        """Raise ValueError if value y not allowed for key i."""
        valid_y = True
        if hasattr(self, 'allowed_values') and i in self.allowed_values:
            valid_y = False
            if self.allowed_values[i] is None:
                valid_y = True
            else:
                try:
                    if y in self.allowed_values[i]:
                        valid_y = True
                except:
                    valid_y = False
        if not valid_y:
            raise ValueError(
                f'key: {i}, cannot be'
                f' assigned value: {y}\n'
                'Admissible values are:\n\t'
                f'{self.allowed_values[i]}')
        super().__setitem__(i, y)

    def __str__(self):
        return f'TypedDict({dict.__str__(self)})'

    def update(
            self,
            *args,
            **kwargs
            ) -> None:
        if args:
            if len(args) > 1:
                raise TypeError(
                    'update expected at most 1 arguments, '
                    f'got {len(args)}')
            other = dict(args[0])
            for key in other:
                self[key] = other[key]
        for key in kwargs:
            self[key] = kwargs[key]

    def setdefault(
            self,
            key,
            value:
                _ty.Optional=None):
        if key not in self:
            self[key] = value
        return self[key]

    def set_types(
            self,
            allowed_values
            ) -> None:
        """Restrict values the key can be paired with.

        @param allowed_values: dict of the form:

            ```python
            {key: values}
            ```

            `values` must implement `__contains__`
            to enable checking validity of values.

            If `values` is `None`,
            then any value is allowed.
        """
        self.allowed_values = allowed_values

    def is_consistent(self) -> bool:
        """Check if typed keys have consistent values.

        Use case: changing the object that allowed_values
        points to can invalidate the assigned values.
        """
        for k, v in self:
            if k in self.allowed_values:
                if v not in self.allowed_values[k]:
                    return False
        return True
