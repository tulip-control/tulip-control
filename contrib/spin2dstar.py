#!/usr/bin/env python
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
"""
Convert from most of SPIN syntax to LTL2DSTAR syntax.

Example usage:

  $ ./spin2dstar.py '([]<>a) -> ([]<>b)'


SCL; 3 Sep 2013
"""
import sys

import pyparsing as prs


SPIN_SYN = {"&": "&&",
            "|": "||",
            "!": "!",
            "->": "->",
            "<->": "<->",
            "[]": "[]",
            "<>": "<>",
            "X": "X",
            "U": "U"}

DSTAR_SYN = {"&": "&",
             "|": "|",
             "!": "!",
             "->": "i",
             "<->": "e",
             "[]": "G",
             "<>": "F",
             "X": "X",
             "U": "U"}


class AST:
    def __init__(self, sym="", left="", right=""):
        self.sym = sym
        self.left = left
        self.right = right

    def __str__(self):
        return f"({self.left}{self.sym}{self.right})"

    def infix_to_tree(self, toks):
        print(toks)

    def to_dstar(self, pretty=False):
        try:
            output = DSTAR_SYN[self.sym]
        except KeyError:
            output = self.sym
        if self.left != "":
            if pretty:
                output += " "
            output += self.left.to_dstar(pretty=pretty)
        if self.right != "":
            if pretty:
                output += " "
            output += self.right.to_dstar(pretty=pretty)
        return output


def reverse_lookup(d, value):
    """Return first key found with given value.

    Raise ValueError exception if no matches found.
    """
    for k, v in d.items():
        if v == value:
            return k

expr_stack = list()
def push_op(toks):
    canonized_tok = reverse_lookup(SPIN_SYN, toks[0])
    if canonized_tok in ("!", "[]", "<>", "X"):  # Unary
        if len(expr_stack) < 1:
            raise ValueError("Insufficient number of operands in expression stack")
        expr_stack.append(AST(sym=canonized_tok, right=expr_stack.pop()))
    else:  # Binary
        if len(expr_stack) < 2:
            raise ValueError("Insufficient number of operands in expression stack")
        expr_stack.append(AST(sym=canonized_tok, right=expr_stack.pop(), left=expr_stack.pop()))

def push_ident(toks):
    expr_stack.append(AST(toks[0]))


def spin_to_AST(inform):
    spin_G = prs.Literal(SPIN_SYN["[]"])
    spin_F = prs.Literal(SPIN_SYN["<>"])
    spin_X = prs.Literal(SPIN_SYN["X"])
    spin_U = prs.Literal(SPIN_SYN["U"])
    spin_negate = prs.Literal(SPIN_SYN["!"])
    spin_and = prs.Literal(SPIN_SYN["&"])
    spin_or = prs.Literal(SPIN_SYN["|"])
    spin_implies = prs.Literal(SPIN_SYN["->"])
    spin_equiv = prs.Literal(SPIN_SYN["<->"])
    identifier = prs.Word(
        prs.alphas + "_",
        prs.alphanums + "_")
    form = prs.Forward()
    unary_or_less = prs.Forward()
    unary_or_less << ((spin_negate + unary_or_less).setParseAction(push_op)
                      | (spin_X + unary_or_less).setParseAction(push_op)
                      | (spin_G + unary_or_less).setParseAction(push_op)
                      | (spin_F + unary_or_less).setParseAction(push_op)
                      | identifier.setParseAction(push_ident)
                      | (prs.Suppress("(") + form + prs.Suppress(")")))
    form << ((unary_or_less + prs.ZeroOrMore(
        ((spin_U
         | spin_and
         | spin_or
         | spin_implies
         | spin_equiv) + form).setParseAction(push_op)))
         | unary_or_less)
    form.parseString(inform, parseAll=True)
    assert len(expr_stack) == 1
    return expr_stack.pop()

def spin_to_dstar(inform):
    expr = spin_to_AST(inform)
    return expr.to_dstar(pretty=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} FORMULA")
        exit(1)
    print(spin_to_dstar(sys.argv[1]) )
