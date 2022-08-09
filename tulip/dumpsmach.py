# Copyright (c) 2014-2015 by California Institute of Technology
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
"""Code generation and exporting of controllers from TuLiP

Routines in this module are cross-cutting in the sense that they
concern multiple aspects of solutions created by TuLiP and accordingly
should not be placed under a specific subpackage, like tulip.transys.
"""
import itertools as _itr
import time

import tulip.transys as _trs


__all__ = [
    'write_python_case',
    'python_case']


def write_python_case(
        filename:
            str,
        *args,
        **kwargs):
    """Convenience wrapper for writing output of python_case to file.

    @param filename:
        Name of file in which to place the code generated
        by `python_case`.
    """
    with open(filename, 'w') as f:
        f.write(python_case(*args, **kwargs))


def python_case(
        M:
            _trs.MealyMachine,
        classname:
            str="TulipStrategy",
        start='Sinit'
        ) -> str:
    """Export MealyMachine as Python class based on flat if-else block.

    Usage documentation for the generated code is included in the output.
    Consult the docstrings of the class and move() method.

    @param start:
        initial node in `M`
    @return:
        The returned string is valid Python code and can, for
        example, be:
          - saved directly into a ".*.py" file, or
          - passed to "exec".
    """
    tab = 4 * ' '
    node_to_int = dict([(s, i) for i, s in enumerate(M)])
    input_vars = [input_var for input_var in M.inputs] if M.inputs else []
    input_args = ', '.join(input_vars)
    input_args_str = "'"+"', '".join(input_vars)+"'"
    code = (
        'class {classname}:\n'
        '{t}"""Mealy transducer.\n'
        '\n'
        '{t}Internal states are integers, the current state\n'
        '{t}is stored in the attribute "state".\n'
        '{t}To take a transition, call method "move".\n'
        '\n'
        '{t}The names of input variables are stored in the\n'
        '{t}attribute "input_vars".\n'
        '\n'
        '{t}Automatically generated by tulip.dumpsmach on {date}\n'
        '{t}To learn more about TuLiP, visit http://tulip-control.org\n'
        '{t}"""\n'
        '{t}def __init__(self):\n'
        '{t2}self.state = {sinit}\n'
        '{t2}self.input_vars = [{input_args_str}]\n'
        '\n'
        '{t}def move(self, {input_args}) -> dict:\n'
        '{t2}"""Given inputs, take move and return outputs.\n'
        '\n'
        '{t2}@return:\n'
        '{t2}    dictionary with keys of the output variable names:\n'
        '{t2}    {outputs}\n'
        '{t2}"""\n'
        '{t2}output = dict()\n'
        ).format(
            classname=classname,
            t=tab,
            t2=2*tab,
            date=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            sinit=node_to_int[start],
            input_args_str=input_args_str,
            input_args=input_args,
            outputs=list(map(str, M.outputs)))
    # cached generator
    ifs = lambda: _itr.chain(['if'], _itr.repeat('elif'))
    proj = lambda d, keys: ((k, d[k]) for k in d if k in keys)
    proj = lambda d, keys: ((k, "'"+d[k]+"'" if isinstance(d[k], str) else d[k])
                            for k in d if k in keys)
    # generate selection statements
    c = list()
    for u, ifu in zip(M, ifs()):
        edges = list()
        for (_, w, d), ifw in zip(M.edges(u, data=True), ifs()):
            if M.inputs:
                guard = ' and '.join(
                    f'({k} == {v})'
                    for k, v in proj(d, M.inputs))
            else:
                guard = 'True'
            outputs = ''.join(
                f'{4 * tab}output["{k}"] = {v}\n'
                for k, v in proj(d, M.outputs))
            target_id = node_to_int[w]
            edges.append(
                f'{3 * tab}{ifw} {guard}:\n'
                f'{4 * tab}self.state = {target_id}\n'
                '\n'
                f'{outputs}')
        # handle invalid inputs or dead-end
        if edges and M.inputs:
            edges.append(
                f'{3 * tab}else:\n'
                f'{4 * tab}self._error({input_args})\n')
        elif not edges:
            edges.append(
                f'{3 * tab}raise RuntimeError('
                '"Reached dead-end state !")\n')
        # each state
        c.append((
            '{t2}{ifu} self.state == {node_id}:\n'
            '{edges}').format(
                t2=2*tab,
                ifu=ifu,
                node_id=node_to_int[u],
                edges=''.join(edges)))
    code += ''.join(c) + (
            '{t2}else:\n'
            '{t3}raise AssertionError("Unrecognized internal state: " + '
            'str(self.state))\n'
            '{t2}return output\n'
            '\n'
            '{t}def _error(self, {input_args}):\n'
            '{t2}raise ValueError("Unrecognized input: " + ('
            '{inputs}).format({args}))\n').format(
                t=tab,
                t2=2*tab,
                t3=3*tab,
                input_args=input_args,
                inputs=''.join(
                    '\n{t}"{v} = {{{v}}}; "'.format(v=v, t=3*tab)
                    for v in M.inputs),
                args=','.join('\n{t}{v}={v}'.format(v=v, t=4*tab)
                              for v in M.inputs))
    return code
