import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('tulip.ltl_parser_log').setLevel(logging.ERROR)
#import nose.tools as nt
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


def Tree_test():
    pass


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
