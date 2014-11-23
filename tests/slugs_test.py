#!/usr/bin/env python
"""Tests for the interface with slugs."""
import logging
logger = logging.getLogger(__name__)
from tulip.interfaces import slugs
import jtlvint_test


def bitfields_to_ints_test():
    t = {'a': (0, 30)}

    # test int values
    bits = {'a@0.0.30': 0, 'a@1': 1, 'a@2': 1,
            'a@3': 0, 'a@4': 1, 'a@5': 0}
    n = slugs._bitfields_to_ints(bits, t)
    logger.debug(n)
    assert n == {'a': 22}

    # test str values
    bits = {'a@0.0.30': '0', 'a@1': '1', 'a@2': '1',
            'a@3': '0', 'a@4': '1', 'a@5': '0'}
    n = slugs._bitfields_to_ints(bits, t)
    logger.debug(n)
    assert n == {'a': 22}

    # range
    for n in xrange(30):
        bits = list(bin(n).lstrip('0b').zfill(6))
        bits.reverse()  # little-endian
        d = {'a@{i}'.format(i=i): v for i, v in enumerate(bits)}
        d['a@0.0.30'] = d.pop('a@0')
        t = {'a': (0, 30)}
        print(d)
        m = slugs._bitfields_to_ints(d, t)
        logger.debug((n, m))
        assert m == {'a': n}


class basic_test(jtlvint_test.basic_test):
    def setUp(self):
        super(basic_test, self).setUp()
        self.check_realizable = lambda x: slugs.synthesize(x) is not None
        self.synthesize = slugs.synthesize
