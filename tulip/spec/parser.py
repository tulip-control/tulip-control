# Copyright (c) 2011-2014 by California Institute of Technology
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
"""LTL parser.

Recognizes formulas in the syntax of:
- JTLV
- SPIN
- SMV
- gr1c
"""
import functools as _ft
import re

import tulip.spec.ast as _ast
import tulip.spec.lexyacc as lexyacc


__all__ = [
    'parse']


def parse(
        formula:
            str,
        full_operators:
            bool=False
        ) -> _ast.NodeSpec:
    """Return syntax tree for `formula`.

    The returned tree is "abstract",
    in that blankspace of `formula`
    cannot be reproduced from that `tree`.

    @param full_operators:
        replace full names of operators
        with their symbols (case-insensitive,
        each operator must be a separate word).
    """
    if full_operators:
        formula = _replace_full_name_operators(formula)
    parser = _make_parser()
    spec = parser.parse(formula)
    if spec is not None:
        return spec
    raise Exception(
        'Parsing formula:\n'
        f'{formula}\n'
        'failed')


@_ft.cache
def _make_parser() -> lexyacc.Parser:
    """Return parser.

    Memoizes the parser.
    """
    return lexyacc.Parser()


def _replace_full_name_operators(
        formula:
            str
        ) -> str:
    """Replace full operator names with symbols.

    Replaces full names with symbols,
    for temporal and Boolean operators.

    Each operator must be a word
    (as defined by `\b` in regexp).
    Substitution is case-insensitive.
    """
    subs = _ast.FULL_OPERATOR_NAMES.items()
    for name, symbol in subs:
        formula = re.sub(
            rf'(?i)\b{name}\b',
            symbol,
            formula)
    return formula
