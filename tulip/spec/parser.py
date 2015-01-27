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
"""LTL parser supporting JTLV, SPIN, SMV, and gr1c syntax"""
from __future__ import absolute_import
import re
from tulip.spec import ast, lexyacc


# cache
parsers = dict()


def parse(formula, full_operators=False):
    """Parse formula string and create abstract syntax tree (AST).

    @param full_operators: replace full names of operators
        with their symbols (case insensitive,
        each operator must be a separate word).
    @type full_operators: C{bool}
    """
    if full_operators:
        formula = _replace_full_name_operators(formula)
    if parsers.get('ply') is None:
        parsers['ply'] = lexyacc.Parser()
    spec = parsers['ply'].parse(formula)
    # did ply fail merely printing warnings ?
    if spec is None:
        raise Exception('Parsing formula:\n{f}\nfailed'.format(f=formula))
    return spec


def _replace_full_name_operators(formula):
    """Replace full names with symbols for temporal and Boolean operators.

    Each operator must be a word (as defined by \b in regexp).
    Substitution is case insensitive.
    """
    for name, symbol in ast.FULL_OPERATOR_NAMES.iteritems():
        formula = re.sub(r'\b(?i)' + name + r'\b', symbol, formula)
    return formula
