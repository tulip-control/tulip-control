#!/usr/bin/env python
"""
Tests for the interface with Lily.
"""
from nose.tools import raises
from tulip.spec.form import LTL, GRSpec
from tulip.interfaces import lily


def check_realizable(f, expected):
    M = lily.synthesize(f)
    if expected:
        assert M is not None
    else:
        assert M is None

def test_realizable():
    for f, expected in [(LTL('([]<>a) -> ([]<>(a && b))',
                             input_variables={'a': 'boolean'},
                             output_variables={'b': 'boolean'}),
                         True),
                        (LTL('[]<>(a && b)',
                             input_variables={'a': 'boolean'},
                             output_variables={'b': 'boolean'}),
                         False)]:
        yield check_realizable, f, expected

def test_GRSpec():
    f = GRSpec(env_vars={'a'}, sys_vars={'b'},
               env_init=['!a'], sys_init=['!b'],
               env_safety=['a -> X !a', '!a -> X a'],
               sys_prog=['a && b'])
    M = lily.synthesize(f)
    assert M is not None

@raises(TypeError)
def test_nonbool():
    f = LTL('[](a -> <>b)',
            input_variables={'a': (0,1)},
            output_variables={'b': 'boolean'})
    M = lily.synthesize(f)
