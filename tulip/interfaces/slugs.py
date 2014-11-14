# Copyright (c) 2014 by California Institute of Technology
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
Interface to the slugs implementation of GR(1) synthesis.

Relevant links:
  - U{slugs<https://github.com/LTLMoP/slugs>}
"""

from __future__ import absolute_import
import logging
logger = logging.getLogger(__name__)
import os
import re
import subprocess
import tempfile
import slugs
from tulip.interfaces import jtlv


def synthesize(spec):
    """Return strategy satisfying the specification C{spec}.

    @type spec: L{GRSpec}
    @return: If realizable return synthesized strategy, otherwise C{None}.
    @rtype: C{networkx.DiGraph}
    """
    struct = spec.to_slugs()
    s = slugs.convert_to_slugsin(struct, True)

    with tempfile.NamedTemporaryFile(delete=False) as fin:
        fin.write(s)

    logger.info('\n\n structured slugs:\n\n {struct}'.format(struct=struct) +
                '\n\n slugs in:\n\n {s}\n'.format(s=s))
    if not realizable:
        return None

    os.unlink(fin.name)
    # collect int vars
    vrs = dict(spec.sys_vars)
    vrs.update(spec.env_vars)
    vrs = {k: dom for k, dom in vrs.iteritems()
           if isinstance(dom, tuple) and len(dom) == 2}

    lines = [_replace_bitfield_with_int(line, vrs)
             for line in out.split('\n')]
    g = jtlv.jtlv_output_to_networkx(lines, spec)
    logger.debug(
        ('loaded strategy with vertices:\n  {v}\n'
         'and edges:\n {e}\n').format(
            v='\n  '.join(str(x) for x in g.nodes(data=True)),
            e=g.edges()
        )
    )
    return g


def _call_slugs(options):
    c = ['slugs'] + options
    logger.debug('Calling: ' + ' '.join(c))
    try:
        p = subprocess.Popen(
            c,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
    except OSError as e:
        if e.errno == os.errno.ENOENT:
            raise Exception('slugs not found in path.')
        else:
            raise

    out, err = p.communicate()
    msg = (
        '\n slugs return code: {c}\n\n'.format(c=p.returncode) +
        '\n slugs stdout:\n\n {out}\n\n'.format(out=out)
    )
    logger.debug(msg)

    # error ?
    if p.returncode != 0:
        raise Exception(msg)

    realizable = 'Specification is realizable' in out
    # check sanity
    if not realizable:
        assert('Specification is unrealizable' in out)
    return realizable, out


def _bitfield_to_int(var, dom, bools):
    """Return integer value of bitfield.

    @type var: str

    @type dom: 2-tuple of int

    @type bools: list of tuples
    """
    bits = dict(bools)

    # rename LSB
    lsb = '{var}@0.{min}.{max}'.format(var=var, min=dom[0], max=dom[1])
    name = '{var}@0'.format(var=var)
    if lsb not in bits:
        raise ValueError('"{lsb}" expected in {bits}'.format(
                         lsb=lsb, bits=bits))
    bits[name] = bits.pop(lsb)

    # note: little-endian
    s = ''.join(str(bits['{var}@{i}'.format(var=var, i=i)])
                for i in xrange(len(bits)-1, -1, -1))
    return int(s, 2)


def _replace_bitfield_with_int(line, vrs):
    """Convert bitfield representation to integers.

    @type line: str

    @type vrs: dict
    """
    for var, dom in vrs.iteritems():
        p = r'({var}@\w+|{var}@\w+\.{min}\.{max}):(\w+)'.format(
            var=var, min=dom[0], max=dom[1]
        )
        # [(varname, bool), ...]
        bools = re.findall(p, line)
        if not bools:
            continue

        i = _bitfield_to_int(var, dom, bools)

        # replace LSB with integer variable and its value
        k, v = bools[0]
        p = r'{key}\w*:{val}[,\s*]*'.format(key=re.escape(k), val=v)
        r = '{var}:{intval}, '.format(var=var, intval=i)
        line = re.sub(p, r, line)

        # erase other bits
        for key, val in bools[1:]:
            p = r'({key}\w*:{val}[,\s*]*)'.format(key=re.escape(key), val=val)
            line = re.sub(p, '', line)
    return line
