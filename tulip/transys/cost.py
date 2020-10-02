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

from collections import Iterable


class ValidTransitionCost(object):
    """A class for defining valid transition cost."""

    def __contains__(self, obj):
        try:
            if obj + 0 >= 0 and obj + 0 == obj:
                return True
            return False
        except Exception:
            return False


class VectorCost(object):
    """A class for defining a vector cost, with addition and comparision operations"""

    def __init__(self, value):
        if isinstance(value, int) or isinstance(value, float):
            value = [value]

        assert isinstance(value, Iterable)
        self._value = value

    def __str__(self):
        return str(self._value)

    def __getitem__(self, key):
        return self._value[key]

    def __iter__(self):
        return iter(self._value)

    def __len__(self):
        return len(self._value)

    def _convert(self, obj):
        if isinstance(obj, int) or isinstance(obj, float):
            obj = VectorCost([obj] * len(self))
        assert len(self) == len(obj)
        return obj

    def __add__(self, obj):
        obj = self._convert(obj)
        return VectorCost([self[i] + obj[i] for i in range(len(self))])

    def __radd__(self, obj):
        return self.__add__(obj)

    def __eq__(self, obj):
        obj = self._convert(obj)
        for i in range(len(self)):
            if not (self[i] == obj[i]):
                return False
        return True

    def __gt__(self, obj):
        obj = self._convert(obj)
        for i in range(len(self)):
            if self[i] > obj[i]:
                return True
            elif obj[i] > self[i]:
                return False
        return False

    def __ne__(self, obj):
        return not (self.__eq__(obj))

    def __ge__(self, obj):
        return self.__gt__(obj) or self.__eq__(obj)

    def __lt__(self, obj):
        return not (self.__ge__(obj))

    def __le__(self, obj):
        return self.__lt__(obj) or self.__eq__(obj)
