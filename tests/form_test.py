#!/usr/bin/env python
"""
Tests for logic formulas module.
"""
import logging
logging.basicConfig(level=logging.ERROR)

from nose.tools import assert_raises
from tulip import spec, synth

def test_name_conflict():
    assert_raises(ValueError, spec.form.stability_to_gr1, 'a', aux='a')

def test_stability():
    s = spec.form.stability_to_gr1('p', aux='a')
        
    assert(isinstance(s, spec.GRSpec) )
    assert('aux' not in s.sys_vars)
    assert('a' in s.sys_vars)
    assert('p' in s.sys_vars)
    
    # p && X[]!p
    s0 = spec.GRSpec(
        sys_vars={'p'}, sys_init={'p'},
        sys_safety={'p -> X !p',
                    '!p -> X !p'}
    )
    assert(not synth.is_realizable('gr1c', s | s0) )
    
    # !p && X[]p
    s1 = spec.GRSpec(
        sys_vars={'p'}, sys_init={'!p'},
        sys_safety={'!p -> X p',
                    'p -> X p'}
    )
    assert(synth.is_realizable('gr1c', s | s1) )
    
    # []<>p && []<>!p
    s2 = spec.GRSpec(
        sys_vars={'p'},
        sys_prog={'p', '!p'}
    )
    assert(not synth.is_realizable('gr1c', s | s2) )
    
    # env b can prevent !p, but has tp <> become !b,
    # releasing sys to set p
    #
    # env: b && []<>!b
    # sys: !p && []((b && !p) -> X!p)
    s3 = spec.GRSpec(
        env_vars={'b'}, env_init={'b'},
        env_prog={'!b'},
        sys_vars={'p'}, sys_init={'!p'},
        sys_safety={'(b && !p) -> X !p'}
    )
    
    assert(synth.is_realizable('gr1c', s | s3) )
    
    s3.env_prog = []
    assert(not synth.is_realizable('gr1c', s | s3) )
    
    #s4 = s | s3
    #print(s4.pretty() )
    #mealy = synth.synthesize('gr1c', s4)
    #mealy.save()

def test_response():
    s = spec.form.response_to_gr1('p', 'q')
    
    assert(isinstance(s, spec.GRSpec) )
    assert('p' in s.sys_vars)
    assert('q' in s.sys_vars)
    
    # p && []!q
    s0 = spec.GRSpec(
        sys_vars={'p', 'q'}, sys_init={'p'},
        sys_safety={'!q'}
    )
    assert(not synth.is_realizable('gr1c', s | s0) )
    
    # []!p && []!q
    s1 = spec.GRSpec(
        sys_vars={'p', 'q'}, sys_safety={'!p && !q'}
    )
    assert(synth.is_realizable('gr1c', s | s1) )
    
    # p && q
    s2 = spec.GRSpec(
        sys_vars={'p', 'q'}, sys_init={'p && q'},
    )
    assert(synth.is_realizable('gr1c', s | s2) )
    
    # alternating p, alternating q
    s3 = spec.GRSpec(
        sys_vars={'p', 'q'}, sys_safety={
            'p -> X !p',
            '!p -> X p',
            'p -> X q',
            'q -> X ! q'
        }
    )
    assert(synth.is_realizable('gr1c', s | s3) )
    
    #print((s | s2).pretty() )

def test_eventually():
    s = spec.form.eventually_to_gr1('p', aux='c')
    
    assert(isinstance(s, spec.GRSpec) )
    assert('aux' not in str(s) )
    assert('c' in s.sys_vars)
    assert('p' in s.sys_vars)
    
    # []!p
    s0 = spec.GRSpec(
        sys_vars={'p'}, sys_safety={'!p'}
    )
    assert(not synth.is_realizable('gr1c', s | s0) )
    
    # !p && []<>p && []<>!p
    s1 = spec.GRSpec(
        sys_vars={'p'}, sys_init={'!p'},
        sys_prog={'!p', 'p'}
    )
    assert(synth.is_realizable('gr1c', s | s1) )
    
    #s2 = s | s1
    #print(s4.pretty() )
    #mealy = synth.synthesize('gr1c', s4)
    #mealy.save()

if __name__ == '__main__':
    test_response()
