import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('tulip.ltl_parser_log').setLevel(logging.ERROR)
# import nose.tools as nt
from tulip.spec.ast import nodes
from tulip.spec.parser import parse
from tulip.spec import transformation as tx


def test_to_labeled_graph():
    f = ('( ( p & q ) U ( ( q | ( ( p -> w ) & ( ! ( z -> b ) ) ) ) & '
         '( G ( X g ) ) ) )')
    tree = parse(f)
    assert len(tree) == 18
    nodes = {'p', 'q', 'w', 'z', 'b', 'g', 'G',
             'U', 'X', '&', '|', '!', '->'}
    g = tx.Tree.from_recursive_ast(tree)
    h = tx.ast_to_labeled_graph(g, detailed=False)
    labels = {d['label'] for u, d in h.nodes_iter(data=True)}
    assert labels == nodes


def tree_from_recursive_ast_test():
    # (1 - x) + y
    x = nodes.Var('x')
    one = nodes.Num('1')
    minus = nodes.Operator('-', one, x)
    y = nodes.Var('y')
    plus = nodes.Operator('+', minus, y)
    # convert to Tree
    g = tx.Tree.from_recursive_ast(plus)
    assert len(g) == 5
    assert set(g.nodes()) == set([x, one, minus, y, plus])
    assert g.has_edge(plus, y)
    assert g.has_edge(plus, minus)
    assert g.has_edge(minus, one)
    assert g.has_edge(minus, x)


def tree_to_recursive_ast_test():
    g = tx.Tree()
    x = nodes.Var('x')
    one = nodes.Num('1')
    minus = nodes.Operator('-', one, x)
    y = nodes.Var('y')
    # (1 - x) + y
    plus = nodes.Operator('+', minus, y)
    g.add_nodes_from([plus, x, y, one, minus])
    # g is different than forula:
    # x - (z + 1)
    z = nodes.Var('z')
    g.root = minus
    g.add_edge(minus, x, key=0)
    g.add_edge(minus, plus, key=1)
    g.add_edge(plus, z, key=0)
    g.add_edge(plus, one, key=1)
    t = g.to_recursive_ast()
    # -
    assert isinstance(t, nodes.Operator)
    assert t.operator == '-'
    # must be a new object
    assert t is not minus
    assert len(t.operands) == 2
    # x
    u = t.operands[0]
    assert isinstance(u, nodes.Var)
    assert u.value == 'x'
    assert u is not x
    # +
    u = t.operands[1]
    assert isinstance(u, nodes.Operator)
    assert u.operator == '+'
    assert u is not plus
    assert len(u.operands) == 2
    # z
    v = u.operands[0]
    assert isinstance(v, nodes.Var)
    assert v.value == 'z'
    assert v is not z
    # 1
    u = u.operands[1]
    assert isinstance(u, nodes.Num)
    assert u.value == '1'
    assert u is not one


def test_str_to_int():
    r = parse('a = "hehe"')
    g = tx.Tree.from_recursive_ast(r)
    var_str2int = {'a': ['hehe', 'haha']}
    tx.sub_constants(g, var_str2int)
    f = g.to_recursive_ast()
    s = f.flatten()
    assert s == '( a = 0 )'

    x = '(loc = "s2") -> X((((env_alice = "left") && (env_bob = "bright"))))'
    var_str2int = {
        'loc': ['s0', 's2'],
        'env_alice': ['left', 'right'],
        'env_bob': ['bleft', 'bright']}
    r = parse(x)
    print(repr(r))
    g = tx.Tree.from_recursive_ast(r)
    print(g)
    tx.sub_constants(g, var_str2int)
    print(str(g))
    f = g.to_recursive_ast()
    print(repr(f))
    s = f.flatten()
    print(s)
    assert s == ('( ( loc = 1 ) -> '
                 '( X ( ( env_alice = 0 ) & ( env_bob = 1 ) ) ) )')
