#!/usr/bin/env python
"""Tests for gr1 fragment untilities."""
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('tulip.ltl_parser_log').setLevel(logging.WARNING)
logging.getLogger('tulip.spec.form').setLevel(logging.WARNING)
logging.getLogger('omega').setLevel(logging.WARNING)
from nose.tools import assert_raises
from tulip import spec, synth
from tulip.spec import parser, transformation
from tulip.spec import gr1_fragment as gr1


def test_split_gr1():
    # init
    f = '(x > 0) & (y + 1 < 2)'
    d = gr1.split_gr1(f)
    assert d['init'] == ['( x > 0 ) & ( ( y + 1 ) < 2 )'], d
    assert d['G'] == [''], d
    assert d['GF'] == [], d
    # safety
    f = '[]((x > 0) & (z = 3 + y))'
    d = gr1.split_gr1(f)
    assert d['init'] == [''], d
    assert d['G'] == ['( ( x > 0 ) & ( z = ( 3 + y ) ) )'], d
    assert d['GF'] == [], d
    # recurrence
    f = '[]<>(x > 0)'
    d = gr1.split_gr1(f)
    assert d['init'] == [''], d
    assert d['G'] == [''], d
    assert d['GF'] == ['( x > 0 )'], d
    # all together
    f = (
        '(x > 0) & (y + 1 < 2) & '
        '[]( (X y) > 0) & '
        '[]<>((z - x <= 0) | (p -> q))')
    d = gr1.split_gr1(f)
    assert d['init'] == ['( x > 0 ) & ( ( y + 1 ) < 2 )'], d
    assert d['G'] == ['( ( X y ) > 0 )'], d
    assert d['GF'] == ['( ( ( z - x ) <= 0 ) | ( p -> q ) )'], d
    # not in fragment
    with assert_raises(AssertionError):
        gr1.split_gr1('[]( [] p )')
    with assert_raises(AssertionError):
        gr1.split_gr1('<>( [] p )')
    with assert_raises(AssertionError):
        gr1.split_gr1('(X p ) & ( [] p )')
    with assert_raises(AssertionError):
        gr1.split_gr1('[]<> ( x & (X y) )')


def test_has_operator():
    t = parser.parse(' [](x & y) ')
    g = transformation.Tree.from_recursive_ast(t)
    assert gr1.has_operator(t, g, {'&'}) == '&'
    assert gr1.has_operator(t, g, {'G'}) == 'G'


def test_name_conflict():
    assert_raises(ValueError, gr1.stability_to_gr1, 'a', aux='a')


def test_stability():
    s = gr1.stability_to_gr1('p', aux='a')

    assert isinstance(s, spec.GRSpec)
    assert 'aux' not in s.sys_vars
    assert 'a' in s.sys_vars
    assert 'p' in s.sys_vars

    s.moore = False
    s.plus_one = False
    s.qinit = '\A \E'

    # p && X[]!p
    s0 = spec.GRSpec(
        sys_vars={'p'}, sys_init={'p'},
        sys_safety={'p -> X !p',
                    '!p -> X !p'},
        moore=False,
        plus_one=False,
        qinit='\A \E'
    )
    assert not synth.is_realizable('omega', s | s0)

    # !p && X[]p
    s1 = spec.GRSpec(
        sys_vars={'p'}, sys_init={'!p'},
        sys_safety={'!p -> X p',
                    'p -> X p'},
        moore=False,
        plus_one=False,
        qinit='\A \E'
    )
    assert synth.is_realizable('omega', s | s1)

    # []<>p && []<>!p
    s2 = spec.GRSpec(
        sys_vars={'p'},
        sys_prog={'p', '!p'},
        moore=False,
        plus_one=False,
        qinit='\A \E'
    )
    assert not synth.is_realizable('omega', s | s2)

    # env b can prevent !p, but has tp <> become !b,
    # releasing sys to set p
    #
    # env: b && []<>!b
    # sys: !p && []((b && !p) -> X!p)
    s3 = spec.GRSpec(
        env_vars={'b'}, env_init={'b'},
        env_prog={'!b'},
        sys_vars={'p'}, sys_init={'!p'},
        sys_safety={'(b && !p) -> X !p'},
        moore=False,
        plus_one=False,
        qinit='\A \E')

    assert synth.is_realizable('omega', s | s3)

    s3.env_prog = []
    assert not synth.is_realizable('omega', s | s3)

    # s4 = s | s3
    # print(s4.pretty() )
    # mealy = synth.synthesize('omega', s4)
    # mealy.save()


def test_response():
    s = gr1.response_to_gr1('p', 'q')

    assert isinstance(s, spec.GRSpec)
    assert 'p' in s.sys_vars
    assert 'q' in s.sys_vars

    s.moore = False
    s.plus_one = False
    s.qinit = '\A \E'

    # p && []!q
    s0 = spec.GRSpec(
        sys_vars={'p', 'q'},
        sys_init={'p'},
        sys_safety={'!q'},
        moore=False,
        plus_one=False,
        qinit='\A \E'
    )
    assert not synth.is_realizable('omega', s | s0)

    # []!p && []!q
    s1 = spec.GRSpec(
        sys_vars={'p', 'q'},
        sys_safety={'!p && !q'},
        moore=False,
        plus_one=False,
        qinit='\A \E'
    )
    assert synth.is_realizable('omega', s | s1)

    # p && q
    s2 = spec.GRSpec(
        sys_vars={'p', 'q'},
        sys_init={'p && q'},
        moore=False,
        plus_one=False,
        qinit='\A \E'
    )
    assert synth.is_realizable('omega', s | s2)

    # alternating p, alternating q
    s3 = spec.GRSpec(
        sys_vars={'p', 'q'},
        sys_safety={
            'p -> X !p',
            '!p -> X p',
            'p -> X q',
            'q -> X ! q'},
        moore=False,
        plus_one=False,
        qinit='\A \E'
    )
    assert synth.is_realizable('omega', s | s3)
    # print((s | s2).pretty() )


def test_eventually():
    s = gr1.eventually_to_gr1('p', aux='c')

    assert isinstance(s, spec.GRSpec)
    assert 'aux' not in str(s)
    assert 'c' in s.sys_vars
    assert 'p' in s.sys_vars

    s.moore = False
    s.plus_one = False
    s.qinit = '\A \E'

    # []!p
    s0 = spec.GRSpec(
        sys_vars={'p'},
        sys_safety={'!p'},
        moore=False,
        plus_one=False,
        qinit='\A \E'
    )
    assert not synth.is_realizable('omega', s | s0)

    # !p && []<>p && []<>!p
    s1 = spec.GRSpec(
        sys_vars={'p'},
        sys_init={'!p'},
        sys_prog={'!p', 'p'},
        moore=False,
        plus_one=False,
        qinit='\A \E'
    )
    assert synth.is_realizable('omega', s | s1)

    # s2 = s | s1
    # print(s4.pretty() )
    # mealy = synth.synthesize('omega', s4)
    # mealy.save()


def test_until():
    s = gr1.until_to_gr1('p', 'q', aux='c')

    assert isinstance(s, spec.GRSpec)
    assert 'aux' not in str(s)
    assert 'c' in s.sys_vars
    assert 'p' in s.sys_vars
    assert 'q' in s.sys_vars

    s.moore = False
    s.plus_one = False
    s.qinit = '\A \E'

    # []!q
    s0 = spec.GRSpec(
        sys_vars={'q'},
        sys_safety={'!q'},
        moore=False,
        plus_one=False,
        qinit='\A \E'
    )
    assert not synth.is_realizable('omega', s | s0)

    # !q && <>q
    s1 = spec.GRSpec(
        sys_vars={'q'},
        sys_init={'!q'},
        sys_prog={'q'},
        moore=False,
        plus_one=False,
        qinit='\A \E'
    )
    assert synth.is_realizable('omega', s | s1)

    # !q && []!p && <>q
    s1 = spec.GRSpec(
        sys_vars={'q'},
        sys_init={'!q'},
        sys_safety={'!p'},
        sys_prog={'q'},
        moore=False,
        plus_one=False,
        qinit='\A \E'
    )
    assert not synth.is_realizable('omega', s | s1)


if __name__ == '__main__':
    test_response()
