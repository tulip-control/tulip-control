import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('tulip.ltl_parser_log').setLevel(logging.ERROR)
from nose.tools import raises
#from tulip.spec.parser import parse
from tulip import spec
from tulip.spec import translation as ts
from tulip.spec import form


def test_translate_ast_to_gr1c():
    x = '(loc = "s2") -> X((((env_alice = "left") && (env_bob = "bright"))))'
    s = spec.GRSpec(sys_vars={'loc': ['s0', 's2'],
                              'env_alice': ['left', 'right'],
                              'env_bob': ['bleft', 'bright']},
                    sys_safety=[x])
    s.str_to_int()
    sint = s._bool_int[x]
    print(repr(sint))
    rint = s.ast(sint)
    print(repr(rint))
    r = ts.translate_ast(rint, 'gr1c')
    print(repr(r))
    print(r.flatten())
    assert r.flatten() == ("( ( loc = 1 ) -> "
                           "( ( env_alice' = 0 ) & ( env_bob' = 1 ) ) )")


@raises(TypeError)
def check_translate_unrecognized_types(spc):
    ts.translate(spc, 'gr1c')

def test_translate_unrecognized_types():
    for spc in [form.LTL(), 'a -> b']:
        yield check_translate_unrecognized_types, spc
