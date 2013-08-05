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
Mathematical Sets and Power Sets
"""
from itertools import chain, combinations
from collections import Iterable, Hashable
import warnings
#from scipy.sparse import lil_matrix # is this really needed ?

hl = 60 *'-'
debug = False

def compare_lists(list1, list2):
    """Compare list contents, ignoring ordering.
    
    Hashability of elements not assumed, incurring O(N**2)
    
    see also
    --------
    MathSet
    
    @type list1: list
    @type list2: list
    
    @return: True if a bijection exists between the lists.
        Note that this takes into account multiplicity of elements.
    @rtype: bool
    """
    if not isinstance(list1, list):
        raise TypeError('Not a list, instead list1:\n\t' +str(list1) )
    
    if not isinstance(list2, list):
        raise TypeError('Not a list, instead list2:\n\t' +str(list2) )
    
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
    same_lists = same_lists and len(dummy_list) == 0
    
    return same_lists

class MathSet(object):
    """Mathematical set, allows unhashable elements.
    
    examples
    --------
    >>> s = MathSet(['a', 1, [1,2], {'a', 'b'} ] )
    
    Then print(s) shows how the elements were separately stored
    in a set and list, to optimize contains operations:
    
    >>> print(s)
    set(['a', 1]) U [[1, 2], set(['a', 'b'])]
    
    Set operations similar to the builtin type are supported:
    >>> p = MathSet()
    >>> p.add(1)
    >>> p |= [1, 2]
    >>> p |= {3, 4}
    >>> p |= [[1, 2], '5', {'a':1} ]
    >>> p.add_from([5, 6, '7', {8, 9} ] )
    >>> p.remove(1)
    >>> p
    set([2, 3, 4, 5, 6, '5', '7']) U [[1, 2], {'a': 1}, set([8, 9])]
    
    see also
    --------
    PowerSet
    """    
    def __init__(self, iterable=[]):
        """Initialize by adding elements from iterable.
        
        example
        -------
        >>> s = MathSet([1, 2, 'a', {3, 4} ] )
        
        @param iterable: iterable from which to initialize the set S
            which underlies the PowerSet 2^S
        @type iterable: iterable, any element types allowed
        """
        self._delete_all()
        self.add_from(iterable)
    
    def __repr__(self):
        return str(self._set) +' U ' +str(self._list)
    
    def __str__(self):
        return self.__repr__()
    
    def __call__(self):
        return list(self._set) +self._list
    
    #def __get__(self, instance, iterable):
    #    return self()
    
    #def __set__(self, instance, iterable):
    #    self._delete_all()
    #    self.add_from(iterable)
    
    def __or__(self, other):
        """Union with another mathematical set.
        
        see also
        --------
        __ior__
        
        @param other: any other mathematical set.
        @type other: iterable, elements not restricted to hashable
        
        @return: self | iterable
        @rtype: MathSet
        """
        s = MathSet(self)
        s.add_from(other)
        return s
    
    def __ior__(self, iterable):
        """Union with of MathSet with iterable.
        
        example
        -------
        
        
        see also
        --------
        __or__
        
        @param iterable: any mathematical set.
        @type iterable: iterable, elements not restricted to hashable
        
        @return: self | iterable
        @rtype: MathSet
        """
        self.add_from(iterable)
        return self
    
    def __eq__(self, other):
        if not isinstance(other, MathSet):
            raise TypeError('For now comparison only to another MathSet.\n' +
                            'Got:\n\t' +str(other) +'\n of type: ' +
                            str(type(other) ) +', instead.')
        
        same_lists = compare_lists(self._list, other._list)
        
        return (self._set == other._set) and same_lists
    
    def __contains__(self, item):
        if isinstance(item, Hashable):
            return item in self._set
        else:
            return item in self._list
    
    def __iter__(self):
        return iter(self() )
    
    def __len__(self):
        """Number of elements in set."""
        return len(self._set) +len(self._list)
    
    def _filter_hashables(self, iterable):
        return filter(lambda x: isinstance(x, Hashable), iterable)
    
    def _filter_unhashables(self, iterable):
        return filter(lambda x: not isinstance(x, Hashable), iterable)
    
    def _delete_all(self):
        self._set = set()
        self._list = list()
    
    def add(self, item):
        """Add element to mathematical set.
        
        example
        -------
        >>> s = MathSet()
        >>> s.add(1)
        set([1]) U []
        
        see also
        --------
        add_from, __ior__, remove
        
        @param item: the new set element
        @type item: anything, if hashable it is stored in a Python set,
            otherwise stored in a list.
        """
        if isinstance(item, Hashable):
            self._set.add(item)
        else:
            if item not in self._list:
                self._list.append(item)
            else:
                warnings.warn('item already in MathSet.')
    
    def add_from(self, iterable):
        """Add multiple elements to mathematical set.
        
        Equivalent to |=
        
        example
        -------
        >>> s = MathSet()
        >>> s.add_from([1, 2, {3} ] )
        
        is equivalent to:
        
        >>> s = MathSet()
        >>> s |= [1, 2, {3} ]
        
        see also
        --------
        add, __ior__, remove
        
        @param iterable: new MathSet elements
        @type iterable: iterable containing (possibly not hashable) elements
        """
        if not isinstance(iterable, Iterable):
            raise TypeError('Can only add elements to MathSet from Iterable.\n' +
                            'Got:\n\t' +str(iterable) +'\n instead.')
        
        if isinstance(iterable, MathSet):
            self._set |= set(iterable._set)
            self._list = list(unique(self._list +
                          self._filter_unhashables(iterable) ) )
            return
        
        # speed up
        if isinstance(iterable, set):
            self._set |= iterable
            return
        
        # filter to optimize storage
        self._set |= set(self._filter_hashables(iterable) )
        self._list = list(unique(self._list +
                          self._filter_unhashables(iterable) ) )
    
    def remove(self, item):
        """Remove existing element from mathematical set.
        
        example
        -------
        >>> p = MathSet([1, 2] )
        >>> p.remove(1)
        >>> p
        set([2]) U []
        
        see also
        --------
        add, add_from, __or__
        
        @param item: An item already in the set.
            For adding items, see add.
        """
        if item not in self:
            warnings.warn('Set element not in set S.\n'+
                          'Maybe you targeted another element for removal ?')
        
        if isinstance(item, Hashable):
            self._set.remove(item)
        else:
            self._list.remove(item)    

def unique(iterable):
    """Return unique elements.
    
    note
    ----
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
    return len(iterable) != len(unique(iterable) )

def is_subset(small_iterable, big_iterable):
    """Comparison for handling list <= set, and lists with unhashable items.
    """
    # asserts removed when compiling with optimization on...
    # it would have been elegant to use instead:
    #   assert(isinstance(big_iterable, Iterable))
    # since the error msg is succintly stated by the assert itself
    if not isinstance(big_iterable, Iterable):
        raise TypeError('big_iterable must be Iterable, '
                        'otherwise subset relation undefined.\n'
                        'Got:\n\t' +str(big_iterable) +'\ninstead.')
        
    if not isinstance(small_iterable, Iterable):
        raise TypeError('small_iterable must be Iterable, '
                        'otherwise subset relation undefined.\n'
                        'Got:\n\t' +str(big_iterable) +'\ninstead.')
    
    # nxor
    if isinstance(small_iterable, str) != isinstance(big_iterable, str):
        raise TypeError('Either both or none of small_iterable, '
                        'big_iterable should be strings.\n'
                        'Otherwise subset relation between string '
                        'and non-string may introduce bugs.\nGot:\n\t' +
                        str(big_iterable) +',\t' +str(small_iterable) +
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
    
        # list to avoid: unhashable \in set ? => error
        if not isinstance(big_iterable, list):
            # avoid object duplication
            big_iterable = list(big_iterable)
        
        for item in small_iterable:
            if item not in big_iterable:
                return False
        return True
    except:
        raise Exception('Failed to compare iterables.')

def powerset(iterable):
        """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
        
        From:
            http://docs.python.org/2/library/itertools.html,
        also in:
            https://pypi.python.org/pypi/more-itertools
        """
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1) )

class PowerSet(object):
    """Efficiently store power set of a mathematical set.
    
    Set here isn't necessarily a Python set,
    i.e., it may comprise of unhashable elements.
      
    example
    -------
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
    
    see also
    --------
    is_subset
    
    @param iterable: mathematical set S of elements, on which this 2^S defined.
    @type iterable: iterable container
    """    
    def __init__(self, iterable=[]):
        """Create new PowerSet over elements contained in S = C{iterable}.
        
        This powerset is 2^S.
        
        @param iterable: contains elements of set S underlying the PowerSet.
        @type iterable: iterable of elements which can be hashable or not.
        """
        self.math_set = MathSet(iterable)
    
    def __get__(self, instance, value):
        return self()
    
    def __str__(self):
        return 'PowerSet(' +str(self.math_set) +' )'
    
    def __contains__(self, item):
        """Is item \\in 2^iterable = this powerset(iterable)."""
        if not isinstance(item, Iterable):
            raise Exception('Not iterable:\n\t' +str(item) +',\n'
                            'this is a powerset, so it contains (math) sets.')
        
        return is_subset(item, self.math_set)
    
    def __call__(self):
        """Return the powerset as list of subsets, each subset as tuple."""
        return list(powerset(self.math_set) )
    
    def __iter__(self):
        return iter(self() )
    
    def __len__(self):
        return 2**len(self.math_set)
    
    def __add__(self, other):
        if not isinstance(other, PowerSet):
            raise TypeError('Addition defined only between PowerSets.\n' +
                            'Got instead:\n\t other = ' +str(other) )
        
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
            raise Exception('math_set attribute of PowerSet must be of ' +
                          'class MathSet. Given:\n\t' +str(value) )
        
        object.__setattr__(self, name, value)
