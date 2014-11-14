#!/usr/bin/env python
"""Tests for the tulip.spec subpackage."""
import logging
#logging.basicConfig(level=logging.DEBUG)
import copy
import nose.tools as nt
from tulip.spec.parser import parse


def parse_parse_check(formula, expected_length):
    # If expected_length is None, then the formula is malformed, and
    # thus we expect parsing to fail.
    if expected_length is not None:
        assert len(parse(formula)) == expected_length
    else:
        nt.assert_raises(Exception, parse, formula)


def parse_parse_test():
    for (formula, expected_len) in [("G p", 2),
                                    ("p G", None)]:
        yield parse_parse_check, formula, expected_len


def full_name_operators_test():
    formulas = {
        'always eventually p':'( G ( F p ) )',
        'ALwaYs EvenTUAlly(p)':'( G ( F p ) )',
        '(p and q) UNtIl (q or ((p -> w) and not (z implies b))) and always next g':
        '( ( p & q ) U ( ( q | ( ( p -> w ) & ( ! ( z -> b ) ) ) ) & ( G ( X g ) ) ) )'
    }
    
    for f, correct in formulas.iteritems():
        ast = parse(f, full_operators=True)
        #ast.write('hehe.png')
        assert(str(ast) == correct)

def test_to_labeled_graph():
    f = ('( ( p & q ) U ( ( q | ( ( p -> w ) & ( ! ( z -> b ) ) ) ) & '
         '( G ( X g ) ) ) )')
    tree = parse(f)
    assert(len(tree) == 18)
    nodes = {'p', 'q', 'w', 'z', 'b', 'g', 'G', 'U', 'X', '&', '|', '!', '->'}
    
    g = ast.ast_to_labeled_graph(tree, detailed=False)
    labels = {d['label'] for u, d in g.nodes_iter(data=True)}
        
    print(labels)
    assert(labels == nodes)

