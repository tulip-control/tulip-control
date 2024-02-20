# Copyright (c) 2020 by California Institute of Technology
# and University of Texas at Austin
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
"""Cost Module for state or transition cost/weight"""
import collections.abc as _abc
import functools as _ft
import itertools as _itr
import operator as _op
import typing as _ty


class ValidTransitionCost:
    """A class for defining valid transition cost."""

    def __contains__(self, other):
        try:
            other + 0
            other >= 0
        except TypeError:
            return False
        return (
            other + 0 >= 0 and
            other + 0 == other)


@_ft.total_ordering
class VectorCost:
    """Cost with addition and comparison operations."""

    def __init__(
            self,
            value:
                int |
                float |
                _abc.Iterable):
        if isinstance(value, int | float):
            value = [value]
        if not isinstance(value, _abc.Iterable):
            raise TypeError(
                'Expected iterable, '
                f'got instead: {value}, '
                f'of type {type(value)}')
        self._value = list(value)

    def __str__(self):
        return str(self._value)

    def __getitem__(self, key):
        return self._value[key]

    def __iter__(self):
        return iter(self._value)

    def __len__(self):
        return len(self._value)

    def _convert(
            self,
            other:
                _ty.Union[
                    int,
                    float,
                    'VectorCost']
            ) -> 'VectorCost':
        if isinstance(other, int | float):
            repeated = _itr.repeat(
                other, len(self))
            other = VectorCost(repeated)
        self._assert_equal_len(other)
        return other

    def __add__(
            self,
            other):
        other = self._convert(other)
        self._assert_equal_len(other)
        pairs = zip(self, other)
        return VectorCost(
            _itr.starmap(_op.add, pairs))

    def _assert_equal_len(
            self,
            other):
        """Raise `ValueError` if lengths differ."""
        if len(self) == len(other):
            return
        raise ValueError(
            'Mismatch of lengths: '
            f'{len(self) = } and '
            f'{len(other) = }')

    def __radd__(
            self,
            other):
        return self.__add__(other)

    def __eq__(self, other):
        other = self._convert(other)
        pairs = zip(self, other)
        return all(_itr.starmap(
            _op.eq, pairs))

    def __gt__(self, other):
        other = self._convert(other)
        for a, b in zip(self, other):
            if a == b:
                continue
            return a > b
        return True
