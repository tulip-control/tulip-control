#!/usr/bin/env python
from __future__ import print_function
from tulip.spec.form import LTL
from tulip.interfaces import lily

f = LTL('([]<>(!a)) -> ([](a -> <> b) && [](a -> !X b))',
        input_variables={'a': 'boolean'},
        output_variables={'b': 'boolean'})
# Try appending
#
#     && [](!b -> !X b)
#
# to the above LTL formula. The result will not be realizable.

M = lily.synthesize(f)
if M is not None:
    print(M)
else:
    print('Not realizable.')
