#!/usr/bin/env python
"""Tests for the tulip.spec subpackage."""
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('ltl_parser_log').setLevel(logging.ERROR)
import nose.tools as nt
from tulip.spec import ast, plyparser
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
        'always eventually p': '( G ( F p ) )',
        'ALwaYs EvenTUAlly(p)': '( G ( F p ) )',
        ('(p and q) UNtIl (q or ((p -> w) and '
         'not (z implies b))) and always next g'):
        ('( ( p & q ) U ( ( q | ( ( p -> w ) & '
         '( ! ( z -> b ) ) ) ) & ( G ( X g ) ) ) )')}

    for f, correct in formulas.iteritems():
        tree = parse(f, full_operators=True)
        print(tree)
        # g.write('hehe.png')
        assert tree.flatten() == correct


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
def test_ast_nodes():
    nodes = ast.make_nodes()
    # test Terminal
    t = nodes.Terminal('a')
    assert t.value == 'a'
    assert repr(t) == "Terminal('a')"
    assert str(t) == 'a'
    assert len(t) == 1
    t1 = nodes.Terminal('a')
    assert t == t1
    t2 = nodes.Terminal('b')
    assert t != t2
    assert t.flatten() == 'a'
    # values and operators must be strings
    nt.assert_raises(TypeError, nodes.Terminal, 2)
    nt.assert_raises(TypeError, nodes.Unary, 2, 'v')
    nt.assert_raises(TypeError, nodes.Binary, 2, 'v')
    # test Unary
    u = nodes.Unary('!', t)
    assert u.operator == '!'
    assert u.operand is t
    assert repr(u) == "Unary('!', Terminal('a'))"
    assert str(u) == '!'
    assert len(u) == 2
    assert u.flatten() == '( ! a )'
    # test Binary
    v = nodes.Binary('+', t, t2)
    assert v.operator == '+'
    assert v.left is t
    assert v.right is t2
    assert repr(v) == "Binary('+', Terminal('a'), Terminal('b'))"
    assert len(v) == 3
    assert v.flatten() == "( a + b )"
    # different operator map
    opmap = {'!': '!', '+': '+'}
    nodes = ast.make_nodes(opmap)
    assert nodes.Node.opmap is opmap
    assert nodes.Terminal.opmap is opmap
    assert nodes.Unary.opmap is opmap
    assert nodes.Binary.opmap is opmap


def test_fol_nodes():
    nodes = ast.make_fol_nodes()
    # test Var
    v = nodes.Var('a')
    assert v.value == 'a'
    nt.assert_raises(TypeError, nodes.Var, 2)
    # test Bool
    b = nodes.Bool('TRue')
    print(b.value)
    assert b.value == 'True'
    assert b.flatten() == 'True'
    nt.assert_raises(TypeError, nodes.Bool, 2)
    nt.assert_raises(TypeError, nodes.Bool, 'bee')


def test_lex():
    # catch token precedence errors
    # for example if "EVENTUALLY" is defined before "FALSE",
    # then "False" is parsed as "EVENTUALLY", "NAME", where
    # "NAME" is equal to "alse".
    s = 'False'
    lexer = plyparser.Lexer()
    lexer.lexer.input(s)
    r = list(lexer.lexer)
    assert len(r) == 1
    (tok, ) = r
    assert tok.value == 'False'
    # another case, which needs that "NAME" be defined
    # before "NEXT"
    s = 'X0reach'
    lexer.lexer.input(s)
    r = list(lexer.lexer)
    assert len(r) == 1
    (tok, ) = r
    assert tok.value == 'X0reach'


def lexer_token_precedence_test():
    s = 'False'
    r = parse(s)
    assert isinstance(r, ast.nodes.Bool)
    assert r.value == 'False'
    s = 'a'
    r = parse(s)
    assert isinstance(r, ast.nodes.Var)
    assert r.value == 'a'
    s = '"a"'
    r = parse(s)
    assert isinstance(r, ast.nodes.Str)
    assert r.value == 'a'
    s = '1'
    r = parse(s)
    assert isinstance(r, ast.nodes.Num)
    assert r.value == '1'
    s = '[] a'
    r = parse(s)
    assert isinstance(r, ast.nodes.Unary)
    assert r.operator == 'G'
    assert isinstance(r.operand, ast.nodes.Var)
    assert r.operand.value == 'a'
    s = 'a U b'
    r = parse(s)
    assert isinstance(r, ast.nodes.Binary)
    assert r.operator == 'U'
    assert isinstance(r.left, ast.nodes.Var)
    assert r.left.value == 'a'
    assert isinstance(r.right, ast.nodes.Var)
    assert r.right.value == 'b'
    s = '( a )'
    r = parse(s)
    assert isinstance(r, ast.nodes.Var)
    s = "(a ' = 1)"
    r = parse(s)
    assert isinstance(r, ast.nodes.Comparator)
    assert r.operator == '='
    x = r.left
    assert isinstance(x, ast.nodes.Unary)
    assert x.operator == 'X'
    assert isinstance(x.operand, ast.nodes.Var)
    assert x.operand.value == 'a'
    y = r.right
    assert isinstance(y, ast.nodes.Num)
    assert y.value == '1'
