# Copyright (c) 2011-2015 by California Institute of Technology
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the California Institute of Technology nor
#    the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CALTECH
# OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
#
"""AST subclasses to translate to each syntax of:

- gr1c:
  <https://tulip-control.github.io/gr1c/md_spc_format.html>
- JTLV:
  <http://jtlv.ysaar.net/>
- SMV:
  <https://web.archive.org/web/20201231070353/
   https://nusmv.fbk.eu/NuSMV/userman/v21/nusmv_3.html>
- SPIN:
  <http://spinroot.com/spin/Man/ltl.html>
  <http://spinroot.com/spin/Man/operators.html>
- python (Boolean formulas only)
- WRING:
  <https://web.archive.org/web/20160321085248/
   http://vlsi.colorado.edu/~rbloem/wring.html>
  (see top of file: LTL.pm)
"""
import collections.abc as _abc
import logging
# import pprint
import re
import typing as _ty

import tulip.spec.ast as _ast
import tulip.spec.form as _form


__all__ = [
    'translate']


_logger = logging.getLogger(__name__)
Node = _ast.NodeSpec
Nodes = _ast.NodesSpec


def make_jtlv_nodes() -> Nodes:
    opmap = {
        'False':
            'FALSE',
        'True':
            'TRUE',
        '!':
            '!',
        '|':
            '|',
        '&':
            '&',
        '->':
            '->',
        '<->':
            '<->',
        'G':
            '[]',
        'F':
            '<>',
        'X':
            'next',
        'U':
            'U',
        '<':
            '<',
        '<=':
            '<=',
        '=':
            '=',
        '>=':
            '>=',
        '>':
            '>',
        '!=':
            '!='}
    nodes = _ast.make_fol_nodes(opmap)
    class Str(
            nodes.Str):
        def flatten(self, **kw):
            return f'({self})'
    class Var(
            nodes.Var):
        def flatten(self, env_vars=None, sys_vars=None, **kw):
            v = self.value
            if v in env_vars:
                player = 'e'
            elif v in sys_vars:
                player = 's'
            else:
                raise ValueError(
                    f'{v} neither env nor sys var')
            return f'({player}.{v})'
    nodes.Str = Str
    nodes.Var = Var
    return nodes


def make_gr1c_nodes(
        opmap:
            _ast.OpMap |
            None=None
        ) -> Nodes:
    if opmap is None:
        opmap = {
            'False':
                'False',
            'True':
                'True',
            '!':
                '!',
            '|':
                '|',
            '&':
                '&',
            '->':
                '->',
            '<->':
                '<->',
            'G':
                '[]',
            'F':
                '<>',
            'X':
                '',
            '<':
                '<',
            '<=':
                '<=',
            '=':
                '=',
            '>=':
                '>=',
            '>':
                '>',
            '!=':
                '!='}
    nodes = _ast.make_fol_nodes(opmap)
    class Var(
            nodes.Var):
        def flatten(self, prime=None, **kw):
            prm = "'" if prime else ''
            return f'{self.value}{prm}'
    class Unary(
            nodes.Unary):
        def flatten(self, *arg, **kw):
            if self.operator == 'X':
                kw.update(prime=True)
                return self.operands[0].flatten(*arg, **kw)
            return super().flatten(*arg, **kw)
    nodes.Var = Var
    nodes.Unary = Unary
    return nodes


def make_slugs_nodes() -> Nodes:
    """Simple translation, unisigned arithmetic only.

    For signed arithmetic use Promela instead.
    """
    opmap = {
        'False':
            'FALSE',
        'True':
            'TRUE',
        '!':
            '!',
        '|':
            '|',
        '&':
            '&',
        '->':
            '->',
        '<->':
            '<->',
        'G':
            '[]',
        'F':
            '<>',
        'X':
            '',
        '<':
            '<',
        '<=':
            '<=',
        '=':
            '=',
        '>=':
            '>=',
        '>':
            '>',
        '!=':
            '!=',
        '+':
            '+',
        '-':
            '-'}
    return make_gr1c_nodes(opmap)


def make_promela_nodes() -> Nodes:
    opmap = dict(_ast.OPMAP)
    opmap.update({
        'True':
            'true',
        'False':
            'false',
        'G':
            '[]',
        'F':
            '<>',
        'R':
            'V',
        '=':
            '=='})
    return _ast.make_fol_nodes(opmap)


def make_smv_nodes() -> Nodes:
    opmap = {
        'X':
            'X',
        'G':
            'G',
        'F':
            'F',
        'U':
            'U',
        'R':
            'V'}
    return _ast.make_fol_nodes(opmap)


def make_wring_nodes() -> Nodes:
    opmap = {
        'False':
            '0',
        'True':
            '1',
        '!':
            '!',
        '|':
            '+',
        '&':
            '*',
        '->':
            '->',
        '<->':
            '<->',
        'xor':
            '^',
        'G':
            'G',
        'F':
            'F',
        'X':
            'X',
        'U':
            'U',
        'R':
            'R',
        'V':
            'V'}
    nodes = _ast.make_fol_nodes(opmap)
    class Var(
            nodes.Var):
        def flatten(self, *arg, **kw):
            if ('env_vars' in kw) or ('sys_vars' in kw):
                env_vars = kw['env_vars']
                sys_vars = kw['sys_vars']
                if self.value in env_vars:
                    this_type = env_vars[self.value]
                elif self.value in sys_vars:
                    this_type = sys_vars[self.value]
                else:
                    raise TypeError(
                        f'"{self.value}" is not defined '
                        f'as a variable in {env_vars} nor {sys_vars}')
                if this_type != 'boolean':
                    raise TypeError(
                        f'"{self.val}" is not Boolean, but {this_type}')
            return f'({self.value}=1)'
    nodes.Var = Var
    return nodes


def make_python_nodes() -> Nodes:
    opmap = {
        'True':
            'True',
        'False':
            'False',
        '!':
            'not',
        '&':
            'and',
        '|':
            'or',
        '^':
            '^',
        '=':
            '==',
        '!=':
            '!=',
        '<':
            '<',
        '<':
            '<',
        '>=':
            '>=',
        '<=':
            '<=',
        '>':
            '>',
        '+':
            '+',
        '-':
            '-'}
    nodes = _ast.make_fol_nodes(opmap)
    class Imp(
            nodes.Binary):
        def flatten(self, *arg, **kw):
            l = self.operands[0].flatten()
            r = self.operands[1].flatten()
            return f'((not ({l})) or {r})'
    class BiImp(
            nodes.Binary):
        def flatten(self, *arg, **kw):
            l = self.operands[0].flatten()
            r = self.operands[1].flatten()
            return f'({l} == {r})'
    nodes.Imp = Imp
    nodes.BiImp = BiImp
    return nodes


lang2nodes = {
    'jtlv':
        make_jtlv_nodes(),
    'gr1c':
        make_gr1c_nodes(),
    'slugs':
        make_slugs_nodes(),
    'promela':
        make_promela_nodes(),
    'smv':
        make_smv_nodes(),
    'python':
        make_python_nodes(),
    'wring':
        make_wring_nodes()}


def _to_jtlv(
        d:
            dict[str, ...]
        ) -> tuple[
            str,
            str]:
    """Return specification as list of two strings [assumption, guarantee].

    Format is that of JTLV.
    """
    _logger.info('translate to jtlv...')
    f = _jtlv_str
    parts = [f(d['env_init'], 'valid initial env states', ''),
             f(d['env_safety'], 'safety assumption on environment', '[]'),
             f(d['env_prog'], 'justice assumption on environment', '[]<>')]
    assumption = ' & \n'.join(filter(None, parts))
    parts = [f(d['sys_init'], 'valid initial system states', ''),
             f(d['sys_safety'], 'safety requirement on system', '[]'),
             f(d['sys_prog'], 'progress requirement on system', '[]<>')]
    guarantee = ' & \n'.join(filter(None, parts))
    return (assumption, guarantee)


def _jtlv_str(
        m:
            _abc.Iterable[str],
        comment:
            str,
        prefix:
            str='[]<>'
        ) -> str:
    # no clauses ?
    if not m:
        return ''
    w = list()
    for x in m:
        _logger.debug(f'translate clause: {x}')
        if not x:
            continue
        # collapse any whitespace between any
        # "next" operator that precedes parenthesis
        if prefix == '[]':
            c = re.sub(r'next\s*\(', 'next(', x)
        else:
            c = x
        w.append(f'\t{prefix}({c})')
    formula = ' & \n'.join(w)
    return f'-- {comment}\n{formula}'


def _to_gr1c(
        d:
            dict[str, ...]
        ) -> str:
    """Dump to gr1c specification string.

    Cf. `interfaces.gr1c`.
    """
    def _to_gr1c_print_vars(vardict):
        output = ''
        for var, dom in vardict.items():
            if dom == 'boolean':
                output += ' ' + var
            elif isinstance(dom, tuple) and len(dom) == 2:
                output += ' %s [%d, %d]' % (var, dom[0], dom[1])
            elif isinstance(dom, list) and len(dom) > 0:
                int_dom = convert_domain(dom)
                output += ' %s [%d, %d]' % (var, int_dom[0], int_dom[1])
            else:
                raise ValueError(
                    f'Domain "{dom}" not supported by gr1c.')
        return output
    _logger.info('translate to gr1c...')
    output = (
        'ENV:' + _to_gr1c_print_vars(d['env_vars']) + ';\n' +
        'SYS:' + _to_gr1c_print_vars(d['sys_vars']) + ';\n' +
        # env
        _gr1c_str(d['env_init'], 'ENVINIT', '') +
        _gr1c_str(d['env_safety'], 'ENVTRANS', '[]') +
        _gr1c_str(d['env_prog'], 'ENVGOAL', '[]<>') + '\n' +
        # sys
        _gr1c_str(d['sys_init'], 'SYSINIT', '') +
        _gr1c_str(d['sys_safety'], 'SYSTRANS', '[]') +
        _gr1c_str(d['sys_prog'], 'SYSGOAL', '[]<>'))
    return output


def _to_wring(
        d:
            dict[str, ...]
        ) -> str:
    """Dump to LTL formula in Wring syntax

    Assume that d is a dictionary describing a GR(1) formula in the
    manner of tulip.spec.form.GRSpec; e.g., it should have a key named
    'env_init'. Compare with _to_gr1c().
    """
    assumption = ''
    if d['env_init']:
        assumption += ' * '.join([f'({f})' for f in d['env_init']])
    if d['env_safety']:
        if len(assumption) > 0:
            assumption += ' * '
        assumption += ' * '.join([f'G({f})' for f in d['env_safety']])
    if d['env_prog']:
        if len(assumption) > 0:
            assumption += ' * '
        assumption += ' * '.join([f'G(F({f}))' for f in d['env_prog']])
    guarantee = ''
    if d['sys_init']:
        guarantee += ' * '.join([f'({f})' for f in d['sys_init']])
    if d['sys_safety']:
        if len(guarantee) > 0:
            guarantee += ' * '
        guarantee += ' * '.join([f'G({f})' for f in d['sys_safety']])
    if d['sys_prog']:
        if len(guarantee) > 0:
            guarantee += ' * '
        guarantee += ' * '.join([f'G(F({f}))' for f in d['sys_prog']])
    # Put the parts together, simplifying in special cases
    if guarantee:
        if assumption:
            return f'(({assumption}) -> ({guarantee}))'
        else:
            return f'({guarantee})'
    else:
        return 'G(1)'


def convert_domain(
        dom:
            list[str]
        ) -> (
            bool |
            tuple[int, int]):
    """Return equivalent integer domain if `dom` contais strings."""
    # not a string variable ?
    if not isinstance(dom, list):
        return dom
    return (0, len(dom) - 1)


def _gr1c_str(
        s:
            _abc.Iterable[str],
        name:
            str='SYSGOAL',
        prefix:
            str='[]<>'
        ) -> str:
    if not s:
        return f'{name}:;\n'
    f = '\n& '.join(f'{prefix}({x})' for x in s)
    return f'{name}: {f};\n'


def _to_slugs(
        d:
            dict[str, ...]
        ) -> str:
    """Return structured slugs spec."""
    f = _slugs_str
    return (
        _format_slugs_vars(d['env_vars'], 'INPUT') +
        _format_slugs_vars(d['sys_vars'], 'OUTPUT') +
        # env
        f(d['env_safety'], 'ENV_TRANS') +
        f(d['env_prog'], 'ENV_LIVENESS') +
        f(d['env_init'], 'ENV_INIT', sep='&') +
        # sys
        f(d['sys_safety'], 'SYS_TRANS') +
        f(d['sys_prog'], 'SYS_LIVENESS') +
        f(d['sys_init'], 'SYS_INIT', sep='&'))


def _slugs_str(
        r:
            _abc.Iterable[str],
        name:
            str,
        sep:
            str='\n'
        ) -> str:
    if not r:
        return f'[{name}]\n'
    sep = f' {sep} '
    f = sep.join(x for x in r if x)
    return f'[{name}]\n{f}\n\n'


def _format_slugs_vars(
        vardict:
            dict[str, ...],
        name:
            str
        ) -> str:
    a = list()
    for var, dom in vardict.items():
        if dom == 'boolean':
            a.append(var)
        elif isinstance(dom, tuple) and len(dom) == 2:
            a.append(f'{var}: {dom[0]}...{dom[1]}')
        else:
            raise ValueError(
                f'unknown domain type: {dom}')
    vars = '\n'.join(a)
    return f'[{name}]\n{vars}\n\n'


to_lang = {
    'jtlv':
        _to_jtlv,
    'gr1c':
        _to_gr1c,
    'slugs':
        _to_slugs,
    'wring':
        _to_wring}


def translate(
        spec:
            '_form.GRSpec',
        lang:
            _ty.Literal[
                'gr1c',
                'slugs',
                'jtlv',
                'wring']
        ) -> str:
    """Return str or tuple in tool format.

    Consult the respective documentation in `tulip.interfaces`
    concerning formats and links to further reading.

    @return:
        spec formatted for input to tool; the type of the return
        value depends on the tool:
        - `str` if gr1c or slugs
        - (assumption, guarantee), where each element of the tuple is `str`
    """
    if not isinstance(spec, _form.GRSpec):
        raise TypeError(
            'translate requires first argument (spec) to be of type GRSpec')
    spec.check_syntax()
    spec.str_to_int()
    # pprint.pprint(spec._bool_int)
    d = {p: [translate_ast(spec.ast(spec._bool_int[x]), lang).flatten(
             env_vars=spec.env_vars, sys_vars=spec.sys_vars)
         for x in getattr(spec, p)] for p in spec._parts}
    # pprint.pprint(d)
    d['env_vars'] = spec.env_vars
    d['sys_vars'] = spec.sys_vars
    return to_lang[lang](d)


def translate_ast(
        tree:
            Node,
        lang:
            _ty.Literal[
                'gr1c',
                'slugs',
                'jtlv',
                'promela',
                'smv',
                'python',
                'wring']
        ) -> Node:
    """Return AST of formula `tree`.

    @return:
        tree using AST nodes of `lang`
    """
    if lang == 'python':
        return _ast_to_python(tree, lang2nodes[lang])
    return _ast_to_lang(tree, lang2nodes[lang])


def _ast_to_lang(
        u:
            Node,
        nodes:
            Nodes
        ) -> Node:
    cls = getattr(nodes, type(u).__name__)
    if hasattr(u, 'value'):
        return cls(u.value)
    elif hasattr(u, 'operator'):
        xyz = [_ast_to_lang(x, nodes) for x in u.operands]
        return cls(u.operator, *xyz)
    raise TypeError(
        f'Unknown node type "{type(u).__name__}"')


def _ast_to_python(
        u:
            Node,
        nodes:
            Nodes
        ) -> Node:
    cls = getattr(nodes, type(u).__name__)
    if hasattr(u, 'value'):
        return cls(u.value)
    elif not hasattr(u, 'operands'):
        raise TypeError(
            f'AST node: {type(u).__name__}'
            ', is neither terminal nor operator.')
    elif len(u.operands) == 1:
        assert u.operator == '!'
        return cls(u.operator, _ast_to_python(u.operands[0], nodes))
    elif len(u.operands) == 2:
        assert u.operator in {
            '&', '|', '^', '->', '<->',
            '>', '>=', '=', '!=', '<=', '<',
            '+', '-', '*', '/'}
        if u.operator == '->':
            cls = nodes.Imp
        elif u.operator == '<->':
            cls = nodes.BiImp
        return cls(u.operator,
                   _ast_to_python(u.operands[0], nodes),
                   _ast_to_python(u.operands[1], nodes))
    raise ValueError(
        f'Operator: {u}, is neither unary nor binary.')
