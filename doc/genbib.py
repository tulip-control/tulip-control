#!/usr/bin/env python
"""Generate bibliography.rst

The input file is of the following form. Blank lines are ignored. New
lines are collapsed to a single space. Besides newlines, text for each
entry is copied without filtering, and thus reST syntax can be
used. Note that the key line must begin with '['.

[M55]
G. H. Mealy. `A Method for Synthesizing Sequential Circuits
<http://dx.doi.org/10.1002/j.1538-7305.1955.tb03788.x>`_. *Bell System
Technical Journal (BSTJ)*, Vol.34, No.5, pp. 1045 -- 1079, September,
1955.
"""
from __future__ import print_function
import sys
import io


def print_entry(out_f, bkey, entry_text):
    nl = '\n'
    idt = 4*' '
    if bkey is not None:
        bkey_canon = bkey.lower()
        out_f.write(':raw-html:`<a href="#'+bkey_canon+'" id="'+bkey_canon+
                    '">['+bkey+']</a>`'+nl+idt+'\\'+entry_text+2*nl)

if len(sys.argv) < 2:
    print('Usage: genbib.py FILE')
    sys.exit(1)

out_f = io.open('bibliography.rst', 'w')

with io.open(sys.argv[1], 'r') as f:
    bkey = None
    entry_text = None
    out_f.write(u'''Bibliography
============

.. role:: raw-html(raw)
   :format: html

''')
    for line in f:
        line = line.strip()
        if len(line) == 0:
            continue

        if line[0] == '[' and line:
            print_entry(out_f, bkey, entry_text)
            closing_sym = line.index(']')
            bkey = line[1:closing_sym]
            entry_text = u''
        elif bkey is None:
            ValueError('Entry text found without preceding key.')
        else:
            entry_text += line+' '

print_entry(out_f, bkey, entry_text)
