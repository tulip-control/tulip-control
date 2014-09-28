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

import math
import copy
import os
import re
import subprocess
import tempfile

import slugs
from . import jtlv


def synthesize(spec, only_realizability=False, options=None):
    """Return strategy satisfying the specification.
    
    @type spec: L{GRSpec}
    
    @type only_realizability: bool
    
    @param options: list of options for C{slugs},
        passed to C{subprocess.call}.
    
    @return: Return synthesized strategy if realizable,
        otherwise C{None}.
    @rtype: C{networkx.DiGraph}
    """
    if options is None:
        options = []
    
    struct = spec.to_slugs()
    s = slugs.convert_to_slugsin(struct, True)
    
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(s)
    
    logger.info('\n\n structured slugs:\n\n {struct}'.format(struct=struct) +
                '\n\n slugs in:\n\n {s}\n'.format(s=s))
    
    realizable, out = _call_slugs(f.name, options + ['--onlyRealizability'])
    
    if only_realizability:
        os.unlink(f.name)
        return realizable
    
    if realizable:
        __, out = _call_slugs(f.name, options)
        os.unlink(f.name)
        
        lines = [_bitwise_to_int_domain(line, spec)
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
    else:
        return None


def _call_slugs(f, options):
    c = ['slugs'] + options + ['{file}'.format(file=f)]
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

    
def _bool_to_int_val(var, dom, boolValDict):
    for boolVar, boolVal in boolValDict.iteritems():
        m = re.search(
            r'(' + var + r'@\w+\.' +
            str(dom[0]) + r'\.' + str(dom[1]) + r')',
            boolVar
        )
        if not m:
            continue
        
        min_int = dom[0]
        max_int = dom[1]
        boolValDict[boolVar.split('.')[0]] = boolValDict.pop(boolVar)
    
    assert(min_int >= 0)
    assert(max_int >= 0)
    
    # note: little-endian
    s = ''.join(boolValDict['{var}@{i}'.format(var=var, i=i)]
                for i in xrange(int(math.ceil(math.log(max_int))), -1, -1))
    return int(s, 2)


def _bitwise_to_int_domain(line, spec):
    """Convert bitwise representation to integer domain defined in spec."""
    allVars = dict(spec.sys_vars)
    allVars.update(spec.env_vars)
    
    for var, dom in allVars.iteritems():
        if not isinstance(dom, tuple) or len(dom) != 2:
            continue
            
        boolValDict = dict(re.findall(
            r'(' + var + r'@\w+|' + var + r'@\w+\.' +
            str(dom[0]) + r'\.' + str(dom[1]) + r'):(\w+)', line
        ))
        
        if not boolValDict:
            continue

        intVal = _bool_to_int_val(var, dom, copy.deepcopy(boolValDict))
        
        first = True
        for key, val in boolValDict.iteritems():
            if first:
                line = re.sub(
                    r'(' + re.escape(key) + r'\w*:' + str(val) + r')',
                    var + ':' + str(intVal), line
                )
                first=False
            else:
                line = re.sub(
                    r'(' + re.escape(key) + r'\w*:' + str(val) + r'[,]*)',
                    '', line
                )

    return line
