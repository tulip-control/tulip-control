# Copyright (c) 2013 by California Institute of Technology
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
"""Convert Finite State Machines to State Chart XML (SCXML)."""
import tulip.transys as _trs


def mealy2scxml(
        mealy:
            '_trs.MealyMachine'
        ) -> str:
    """Convert Mealy machine to SCXML.

    Using `examples/transys/machine_examples`:

    ```python
    from machine_examples import garage_counter
    from tulip.transys.export import machine2scxml


    m = garage_counter()
    s = machine2scxml.mealy2scxml(m)
    f = open('mealy.scxml', 'w')
    f.write(s)
    f.close()
    ```

    See Also
    ========
    `transys.machines.mealy`

    @param mealy:
        machine to export as SCXML
    @return:
        SCXML string
    """
    def indent(
            n:
                int
            ) -> str:
        return '\n' + n * '\t'
    def transitions_str(
            from_state,
            mealy:
                '_trs.MealyMachine'
            ) -> str:
        s = ''
        trans = mealy.transitions.find([from_state])
        n = 2
        for from_state_, to_state, sublabel_dict in trans:
            s += f'{indent(n)}<transition '
            n = n + 1
            s += (
                f'{indent(n)}event="input_present" '
                f'{indent(n)}cond="{sublabel_dict}" '
                f'{indent(n)}target="{to_state}"/>')
            n = n - 1
            s += f'{indent(n)}</transition>'
        return s
    s = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<scxml xmlns="http://www.w3.org/2005/07/scxml" '
        ' version="1.0" ')
    if len(mealy.states.initial) != 1:
        raise Exception(
            'Must have exactly 1 initial state.\n'
            f'Got instead:\n\t{mealy.states.initial()}')
    initial_state = mealy.states.initial()[0]
    s += f'initial="{initial_state}">\n'
    for state in mealy.states():
        s += (
            f'\t<state id="{state}">'
            f'{transitions_str(state, mealy)}\n'
            '\t</state>\n')
    s += '</scxml>'
    return s
