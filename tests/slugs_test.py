#!/usr/bin/env python
"""Tests for the interface with slugs."""

import logging
logger = logging.getLogger(__name__)

import nose.tools as nt
from tulip.interfaces import slugs

import jtlvint_test


def bitfield_to_int_test():
    var = 'a'
    dom = (0, 30)

    # test int values
    bools = {'a@0.0.30': 0, 'a@1': 1, 'a@2': 1,
             'a@3': 0, 'a@4': 1, 'a@5': 0}
    n = slugs._bitfield_to_int(var, dom, bools)
    logger.debug(n)
    assert(n == 22)

    # test str values
    bools = {'a@0.0.30': '0', 'a@1': '1', 'a@2': '1',
             'a@3': '0', 'a@4': '1', 'a@5': '0'}
    n = slugs._bitfield_to_int(var, dom, bools)
    logger.debug(n)
    assert(n == 22)

    # range
    for n in xrange(30):
        bits = list(bin(n))[2:]
        bits.reverse()  # little-endian
        d = {'a@{i}'.format(i=i): v for i, v in enumerate(bits)}
        d['a@0.0.30'] = d.pop('a@0')
        m = slugs._bitfield_to_int(var, dom, d)
        logger.debug((n, m))
        assert(n == m)


def replace_bitfield_with_int_test():
    vrs = {'c': 'boolean', 'd': (0, 5)}
    line = 'State 0 with rank 0 -> <c:1, d@0.0.5:0, d@1:1, d@2:1>'
    r = slugs._replace_bitfield_with_int(line, vrs)
    logger.debug(r)
    assert(r == 'State 0 with rank 0 -> <c:1, d:6, >')

    vrs = {'c': 'boolean', 'd': (0, 5)}
    line = 'State 4 with rank 0 -> <d@0.0.5:0, d@1:1, d@2:1, c:1>'
    r = slugs._replace_bitfield_with_int(line, vrs)
    logger.debug(r)
    assert(r == 'State 4 with rank 0 -> <d:6, c:1>')

    vrs = {'ca': 'boolean', 'df': (0, 5)}
    line = 'State 4 with rank 0 -> <ca:0, df@0.0.5:1, df@1:0, df@2:0>'
    r = slugs._replace_bitfield_with_int(line, vrs)
    logger.debug(r)
    assert(r == 'State 4 with rank 0 -> <ca:0, df:1, >')

    vrs = {'ca': 'boolean', 'df': (2, 5)}
    line = 'State 4 with rank 0 -> <ca:0, df@0.2.5:1, df@1:1, df@2:0>'
    r = slugs._replace_bitfield_with_int(line, vrs)
    logger.debug(r)
    assert(r == 'State 4 with rank 0 -> <ca:0, df:3, >')

    vrs = {'c': 'boolean', 'd': (0, 5)}
    line = 'State 4 with rank 0 -> <d@0.0.10:0, d@1:1, d@2:1, c:1>'
    with nt.assert_raises(ValueError):
        slugs._replace_bitfield_with_int(line, vrs)


class basic_test(jtlvint_test.basic_test):
    def setUp(self):
        super(basic_test, self).setUp()
        self.check_realizable = lambda x: slugs.synthesize(x) is not None
        self.synthesize = slugs.synthesize
