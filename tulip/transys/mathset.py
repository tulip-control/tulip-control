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
"""Mathematical Sets and Power Sets"""
from __future__ import print_function

import logging
import warnings
from itertools import chain, combinations
from collections import Iterable, Hashable, Container
from pprint import pformat
from random import randint


logger = logging.getLogger(__name__)


def compare_lists(list1, list2):
    """Compare list contents, ignoring ordering.

    Hashability of elements not assumed, incurring O(N**2)

    See Also
    ========
    L{MathSet}

    @type list1: list
    @type list2: list

    @return: True if a bijection exists between the lists.
        Note that this takes into account multiplicity of elements.
    @rtype: bool
    """
    if not isinstance(list1, list):
        raise TypeError('Not a list, instead list1:\n\t' + str(list1))

    if not isinstance(list2, list):
        raise TypeError('Not a list, instead list2:\n\t' + str(list2))
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


class MathSet(object):
    """Mathematical set, allows unhashable elements.

    Examples
    ========
    >>> s = MathSet(['a', 1, [1,2], {'a', 'b'} ] )

    Then print(s) shows how the elements were separately stored
    in a set and list, to optimize contains operations:

    >>> print(s)
    MathSet(['a', 1, [1, 2], set(['a', 'b'])])

    Set operations similar to the builtin type are supported:

    >>> p = MathSet()
    >>> p.add(1)
    >>> p |= [1, 2]
    >>> p |= {3, 4}
    >>> p |= [[1, 2], '5', {'a':1} ]
    >>> p.add_from([5, 6, '7', {8, 9} ] )
    >>> p.remove(1)
    >>> p
    MathSet([2, 3, 4, 5, 6, '5', '7', [1, 2], {'a': 1}, set([8, 9])])

    See Also
    ========
    L{SubSet}, L{PowerSet}, set
    """

    def __init__(self, iterable=[]):
        """Initialize by adding elements from iterable.

        Example
        =======
        >>> s = MathSet([1, 2, 'a', {3, 4} ] )

        @param iterable: iterable from which to initialize the set S
            which underlies the PowerSet 2^S
        @type iterable: iterable, any element types allowed
        """
        self._delete_all()
        self.add_from(iterable)

    def __repr__(self):
        return 'MathSet(' + pformat(list(self._set) + self._list) + ')'

    def _debug_repr(self):
        set_str = ', '.join([repr(i) for i in self._set])
        return 'MathSet({' + set_str + '} +' + str(self._list) + ')'

    def __or__(self, other):
        """Union with another mathematical set.

        See Also
        ========
        L{__ior__}

        @param other: any other mathematical set.
        @type other: iterable, elements not restricted to hashable

        @return: self | iterable
        @rtype: MathSet
        """
        s = MathSet(self)
        s.add_from(other)
        return s

    def __mul__(self, other):
        """Return the Cartesian product with another C{MathSet}.

        Example
        =======
        >>> a = MathSet([1, 2] )
        >>> b = MathSet([3, 4] )
        >>> c = a *b
        >>> print(type(c) )

        >>> print(c)


        If we prefer a CartesianProduct returned instead:
        >>> c = a.cartesian(b)

        See Also
        ========
        L{cartesian}

        @param other: set with which to take Cartesian product
        @type other: MathSet

        @return: Cartesian product of C{self} with C{other}.
        @rtype: C{MathSet} (explicit construction)
        """
        cartesian = [(x, y) for x in self for y in other]
        return MathSet(cartesian)

    def __ior__(self, iterable):
        """Union with of MathSet with iterable.

        Example
        =======
        >>> s = MathSet([1, 2] )
        >>> s |= [3, 4] # much cleaner & familiar
        >>> print(s)
        set([1, 2, 3, 4]) U []

        See Also
        ========
        L{__or__}

        @param iterable: any mathematical set.
        @type iterable: iterable, elements not restricted to hashable

        @return: self | iterable
        @rtype: MathSet
        """
        self.add_from(iterable)
        return self

    def __sub__(self, rm_items):
        s = MathSet(self)
        print('s = ' + str(s))
        print(rm_items)
        for item in rm_items:
            if item in s:
                print('Removing...: ' + str(item))
                s.remove(item)
        return s

    def __isub__(self, rm_items):
        """Delete multiple elements."""
        for item in rm_items:
            if item in self:
                self.remove(item)
        return self

    def __eq__(self, other):
        if not isinstance(other, MathSet):
            raise TypeError(
                'For now comparison only to another MathSet.\n'
                'Got:\n\t' + str(other) + '\n of type: ' +
                str(type(other)) + ', instead.')
        same_lists = compare_lists(self._list, other._list)
        return (self._set == other._set) and same_lists

    def __contains__(self, item):
        if isinstance(item, Hashable):
            try:
                return item in self._set
            except:
                logger.error('UnHashable items within Hashable.')
        return item in self._list

    def __iter__(self):
        return iter(self._list + list(self._set))

    def __len__(self):
        """Number of elements in set."""
        return len(self._set) + len(self._list)

    def _filter_hashables(self, iterable):
        return filter(lambda x: isinstance(x, Hashable), iterable)

    def _filter_unhashables(self, iterable):
        return list(filter(lambda x: not isinstance(x, Hashable), iterable))

    def _delete_all(self):
        self._set = set()
        self._list = list()

    def add(self, item):
        """Add element to mathematical set.

        Example
        =======
        >>> s = MathSet()
        >>> s.add(1)
        set([1]) U []

        See Also
        ========
        L{add_from}, L{__ior__}, L{remove}

        @param item: the new set element
        @type item: anything, if hashable it is stored in a Python set,
            otherwise stored in a list.
        """
        if isinstance(item, Hashable):
            try:
                self._set.add(item)
                return
            except TypeError:
                logger.error('UnHashable items within Hashable.')
        if item not in self._list:
            self._list.append(item)
        else:
            logger.warning('item already in MathSet.')

    def add_from(self, iterable):
        """Add multiple elements to mathematical set.

        Equivalent to |=

        Example
        =======
        >>> s = MathSet()
        >>> s.add_from([1, 2, {3} ] )

        is equivalent to:

        >>> s = MathSet()
        >>> s |= [1, 2, {3} ]

        See Also
        ========
        L{add}, L{__ior__}, L{remove}

        @param iterable: new MathSet elements
        @type iterable: iterable containing (possibly not hashable) elements
        """
        if not isinstance(iterable, Iterable):
            raise TypeError(
                'Can only add elements to MathSet from Iterable.\n'
                'Got:\n\t' + str(iterable) + '\n instead.')
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

    def remove(self, item):
        """Remove existing element from mathematical set.

        Example
        =======
        >>> p = MathSet([1, 2] )
        >>> p.remove(1)
        >>> p
        set([2]) U []

        See Also
        ========
        L{add}, L{add_from}, L{__or__}

        @param item: An item already in the set.
            For adding items, see add.
        """
        if item not in self:
            warnings.warn(
                'Set element not in set S.\n'
                'Maybe you targeted another element for removal ?')
        if isinstance(item, Hashable):
            try:
                self._set.remove(item)
                return
            except:
                logger.debug('item: ' + str(item) + ', contains unhashables.')
        self._list.remove(item)

    def pop(self):
        """Remove and return random MathSet element.

        Raises KeyError if MathSet is empty.
        """
        if not self:
            raise KeyError('Nothing to pop: MathSet is empty.')
        if self._set and self._list:
            if randint(0, 1):
                return self._set.pop()
            else:
                return self._list.pop()
        elif self._set and not self._list:
            return self._set.pop()
        elif self._list and not self._set:
            return self._list.pop()
        else:
            raise Exception('Bug in empty MathSet: not self above' +
                            'should not reaching this point.')

    def intersection(self, iterable):
        """Return intersection with iterable.

        @param iterable: find common elements with C{self}
        @type iterable: C{Iterable}

        @return: intersection of C{self} with C{iterable}
        @rtype: C{MathSet}
        """
        s = MathSet()
        for item in iterable:
            print(item)
            if item in self:
                print('Adding...\n')
                s.add(item)
        return s

    def intersects(self, iterable):
        """Check intersection with iterable.

        Checks the existence of common elements with iterable.

        >>> s = MathSet()
        >>> s.add(1)
        >>> r = [1,2]
        >>> s.intersects(r)
        True

        @param iterable: with which to check intersection
        @type iterable: C{Iterable}

        @return: C{True} if C{self} has common element with C{iterable}.
            Otherwise C{False}.
        @rtype: C{bool}
        """
        for item in iterable:
            if item in self:
                return True
        return False


class SubSet(MathSet):
    """Subset of selected MathSet, or other Iterable.

    Prior to adding new elements,
    it checks that they are in its superset.

    Example
    =======
    >>> superset = [1, 2]
    >>> s = SubSet(superset)
    >>> s |= [1, 2]
    >>> print(s)
    SubSet([[1, 2]])
    >>> s.add(3)
    raises exception because 3 \\notin [1,2]

    See Also
    ========
    L{MathSet}, L{PowerSet}
    """

    def __init__(self, superset, iterable=None):
        """Define the superset of this set.

        @param superset: This SubSet checked vs C{superset}
        @type superset: Iterable

        @param iterable: elements to add to subset
        @type iterable: Iterable
        """
        self._superset = superset
        super(SubSet, self).__init__([])
        if not isinstance(superset, Container):
            raise TypeError('superset must be Iterable,\n'
                            'Got instead:\n\t' + str(superset))

    def __repr__(self):
        return 'SubSet(' + pformat(list(self._set) + self._list) + ')'

    def _debug_repr(self):
        set_str = ', '.join([repr(i) for i in self._set])
        return 'SubSet({' + set_str + '} +' + str(self._list) + ')'

    @property
    def superset(self):
        return self._superset

    def add(self, new_element):
        """Add state to subset.

        Extends MathSet.add with subset relation checking.

        Example
        =======
        C{new_initial_state} should already be a state.
        First use states.add to include it in set of states,
        then states.add_initial.

        See Also
        ========
        L{MathSet.add}
        """
        if new_element not in self._superset:
            raise Exception(
                'New element state \\notin superset.\n'
                'Add it first to states using e.g. sys.states.add()\n'
                'FYI: new element:\n\t' + str(new_element) + '\n'
                'and superset:\n\t' + str(self._superset))
        super(SubSet, self).add(new_element)

    def add_from(self, new_elements):
        """Add multiple new elements to subset.

        Extends MathSet.add_from with subset relation checking.

        Note
        ====
        It would be sufficient to extend only .add provided
        MathSet.add_from called .add iteratively.
        However MathSet.add_from filters states, which is
        arguably more efficient. So both .add and .add_from
        need to be extended here.

        See Also
        ========
        L{add}, L{__ior__}
        """
        if not is_subset(new_elements, self._superset):
            raise Exception('All new_elements:\n\t' + str(new_elements) +
                            '\nshould already be \\in ' +
                            'self.superset = ' + str(self._superset))
        super(SubSet, self).add_from(new_elements)


class CartesianProduct(object):
    """List of MathSets, with Cartesian semantics."""

    def __init__(self):
        self.mathsets = []

    def __contains__(self, element):
        # TODO check ordered
        if not isinstance(element, Iterable):
            raise TypeError(
                'Argument element must be Iterable, otherwise cannot '
                'recover which item in it belongs to which set in the '
                'Cartesian product.')
        for idx, item in enumerate(element):
            if item not in self.mathsets[idx]:
                return False
        return True

    def __mul__(self, mathsets):
        """Multiply Cartesian products."""
        if not isinstance(mathsets, list):
            raise TypeError('mathsets given must be a list of MathSet.')

    def add(self, mathset):
        self.mathsets += [mathset]

    def add_from(self, mathsets):
        self.mathsets += mathsets

    def remove(self, mathset):
        self.mathsets.remove(mathset)

    def remove_from(self, mathsets):
        for mathset in mathsets:
            self.remove(mathset)


def unique(iterable):
    """Return unique elements.

    Note
    ====
    Always returning a list for consistency was tempting,
    however this defeats the purpose of creating this function
    to achieve brevity elsewhere in the code.

    @return: iterable with duplicates removed, as C{set} if possible.
    @rtype:
        - If all items in C{iterable} are hashable,
            then returns C{set}.
        - If iterable contains unhashable item,
            then returns C{list} of unique elements.
    """
    # hashable items ?
    try:
        unique_items = set(iterable)
    except:
        unique_items = []
        for item in iterable:
            if item not in unique_items:
                unique_items.append(item)
    return unique_items


def contains_multiple(iterable):
    """Does iterable contain any item multiple times ?"""
    return len(iterable) != len(unique(iterable))


def is_subset(small_iterable, big_iterable):
    """Comparison for handling list <= set, and lists with unhashable items.
    """
    # asserts removed when compiling with optimization on...
    # it would have been elegant to use instead:
    #   assert(isinstance(big_iterable, Iterable))
    # since the error msg is succintly stated by the assert itself
    if not isinstance(big_iterable, (Iterable, Container)):
        raise TypeError('big_iterable must be either Iterable or Container, '
                        'otherwise subset relation undefined.\n'
                        'Got:\n\t' + str(big_iterable) + '\ninstead.')
    if not isinstance(small_iterable, Iterable):
        raise TypeError('small_iterable must be Iterable, '
                        'otherwise subset relation undefined.\n'
                        'Got:\n\t' + str(small_iterable) + '\ninstead.')
    # nxor
    if isinstance(small_iterable, str) != isinstance(big_iterable, str):
        raise TypeError('Either both or none of small_iterable, '
                        'big_iterable should be strings.\n'
                        'Otherwise subset relation between string '
                        'and non-string may introduce bugs.\nGot:\n\t' +
                        str(small_iterable) + ',\t' + str(big_iterable) +
                        '\ninstead.')
    try:
        # first, avoid object duplication
        if not isinstance(small_iterable, set):
            small_iterable = set(small_iterable)
        if not isinstance(big_iterable, set):
            big_iterable = set(big_iterable)
        return small_iterable <= big_iterable
    except TypeError:
        # not all items hashable...
        try:
            # list to avoid: unhashable \in set ? => error
            if not isinstance(big_iterable, list):
                # avoid object duplication
                big_iterable = list(big_iterable)
        except:
            logger.error('Could not convert big_iterable to list.')

        for item in small_iterable:
            if item not in big_iterable:
                return False
        return True
    except:
        raise Exception('Failed to compare iterables.')


def powerset(iterable):
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)

    From http://docs.python.org/2/library/itertools.html,
    also in https://pypi.python.org/pypi/more-itertools
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r)
                               for r in range(len(s) + 1))


class PowerSet(object):
    """Efficiently store power set of a mathematical set.

    Set here isn't necessarily a Python set,
    i.e., it may comprise of unhashable elements.

    Example
    =======
    Specify the mathematical set S underlying the PowerSet.

    >>> S = [[1, 2], '3', {'a':1}, 1]
    >>> p = PowerSet(S)

    >>> q = PowerSet()
    >>> q.math_set = S

    Add new element to underlying set S.

    >>> p.math_set.add({3: 'a'} )

    Add multiple new elements to underlying set S.

    >>> p.math_set.add_from({3, 'a'} )
    >>> p.math_set |= [1,2]

    Remove existing element from set S.

    >>> p.remove(1)

    See Also
    ========
    L{MathSet}, L{SubSet}, L{is_subset}

    @param iterable: mathematical set S of elements, on which this 2^S defined.
    @type iterable: iterable container
    """

    def __init__(self, iterable=None):
        """Create new PowerSet over elements contained in S = C{iterable}.

        This powerset is 2^S.

        @param iterable: contains elements of set S underlying the PowerSet.
        @type iterable: iterable of elements which can be hashable or not.
        """
        if iterable is None:
            iterable = []
        self.math_set = MathSet(iterable)

    def __get__(self, instance, value):
        return self()

    def __repr__(self):
        return 'PowerSet(' + str(self.math_set) + ' )'

    def __contains__(self, item):
        """Is item \\in 2^iterable = this powerset(iterable)."""
        if not isinstance(item, Iterable):
            raise Exception('Not iterable:\n\t' + str(item) + ',\n'
                            'this is a powerset, so it contains (math) sets.')

        return is_subset(item, self.math_set)

    def __iter__(self):
        return powerset(self.math_set)

    def __len__(self):
        return 2 ** len(self.math_set)

    def __add__(self, other):
        if not isinstance(other, PowerSet):
            raise TypeError('Addition defined only between PowerSets.\n'
                            'Got instead:\n\t other = ' + str(other))
        list1 = self.math_set
        list2 = other.math_set
        union = list1 | list2
        return PowerSet(union)

    def __eq__(self, other):
        if not isinstance(other, PowerSet):
            raise TypeError('Can only compare to another PowerSet.')

        return other.math_set == self.math_set

    def __setattr__(self, name, value):
        if name is 'math_set' and not isinstance(value, MathSet):
            msg = (
                'PowerSet.math_set must be of class MathSet.\n'
                'Got instead:\n\t' + str(value) +
                '\nof class:\nt\t' + str(type(value)))
            raise Exception(msg)
        object.__setattr__(self, name, value)


class TypedDict(dict):
    """dict subclass where values can be constrained by key.

    For each key, a domain can optionally be defined,
    which restricts the admissible values that can be
    paired with that key.

    Example
    =======

    >>> d = TypedDict()
    >>> allowed_values = {'name': {'Maria', 'John'},
                          'age': range(122)}
    >>> default_values = {'name': 'Maria',
                          'age': 30}
    >>> d.set_types(allowed_types)
    >>> d.update(default_values)
    """
    # credits for debugging this go here:
    #   http://stackoverflow.com/questions/2060972/

    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)
        self.allowed_values = dict()

    def __setitem__(self, i, y):
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
            msg = (
                'key: ' + str(i) + ', cannot be'
                ' assigned value: ' + str(y) + '\n'
                'Admissible values are:\n\t'
                + str(self.allowed_values[i]))
            raise ValueError(msg)
        super(TypedDict, self).__setitem__(i, y)

    def __str__(self):
        return 'TypedDict(' + dict.__str__(self) + ')'

    def update(self, *args, **kwargs):
        if args:
            if len(args) > 1:
                raise TypeError("update expected at most 1 arguments, "
                                "got %d" % len(args))
            other = dict(args[0])
            for key in other:
                self[key] = other[key]
        for key in kwargs:
            self[key] = kwargs[key]

    def setdefault(self, key, value=None):
        if key not in self:
            self[key] = value
        return self[key]

    def set_types(self, allowed_values):
        """Restrict values the key can be paired with.

        @param allowed_values: dict of the form::

                {key : values}

            C{values} must implement C{__contains__}
            to enable checking validity of values.

            If C{values} is C{None},
            then any value is allowed.
        """
        self.allowed_values = allowed_values

    def is_consistent(self):
        """Check if typed keys have consistent values.

        Use case: changing the object that allowed_values
        points to can invalidate the assigned values.

        @rtype: bool
        """
        for k, v in self:
            if k in self.allowed_values:
                if v in self.allowed_values[k]:
                    return False
        return True
