# Copyright (c) 2014, 2015 by California Institute of Technology
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
"""Interface to the slugs implementation of GR(1) synthesis.

Relevant links:
- [slugs](https://github.com/LTLMoP/slugs)
"""
import errno
import json
import logging
import os
import subprocess
import tempfile
import typing as _ty

import networkx as nx
import tulip.spec as _spec


# If this path begins with '/',
# then it is considered to be absolute.
# Otherwise, it is relative to
# the path of the `slugs` executable.
SLUGS_COMPILER_PATH = (
    '../tools/StructuredSlugsParser/'
    'compiler.py')
BDD_FILE = 'strategy_bdd.txt'
_logger = logging.getLogger(__name__)


def check_realizable(
        spec:
            _spec.GRSpec
        ) -> bool:
    """Decide realizability of specification.

    Consult the documentation of `synthesize` about parameters.

    @return:
        `True` if realizable,
        `False` if not, or an error occurs.
    """
    if isinstance(spec, _spec.GRSpec):
        assert not spec.moore
        assert not spec.plus_one
        struct = _spec.translate(spec, 'slugs')
    else:
        struct = spec
    with tempfile.NamedTemporaryFile(delete=False) as fin:
        fin.write(bytes(struct, 'utf-8'))
    realizable, out = _call_slugs(fin.name, synth=False)
    return realizable


def synthesize(
        spec:
            _spec.GRSpec,
        symbolic:
            bool=False
        ) -> (
            nx.DiGraph |
            None):
    """Return strategy satisfying the specification `spec`.

    @return:
        If realizable return synthesized strategy,
        otherwise `None`.
    """
    assert not spec.moore
    assert not spec.plus_one
    struct = _spec.translate(spec, 'slugs')
    with tempfile.NamedTemporaryFile(delete=False) as fin:
        fin.write(bytes(struct, 'utf-8'))
    realizable, out = _call_slugs(
        fin.name,
        synth=True,
        symbolic=symbolic)
    if not realizable:
        return None
    os.unlink(fin.name)
    # collect int vars
    vrs = dict(spec.sys_vars)
    vrs.update(spec.env_vars)
    dout = json.loads(out)
    g = nx.DiGraph()
    dvars = dout['variables']
    for stru, d in dout['nodes'].items():
        u = int(stru)
        state = dict(zip(dvars, d['state']))
        g.add_node(u, state=state)
        for v in d['trans']:
            g.add_edge(u, v)
    h = nx.DiGraph()
    for u, d in g.nodes(data=True):
        bit_state = d['state']
        int_state = _bitfields_to_ints(bit_state, vrs)
        h.add_node(u, state=int_state)
    for u, v in g.edges():
        h.add_edge(u, v)
    nodes = h.nodes(data=True)
    nodes = '\n  '.join(map(str, nodes))
    _logger.debug(
        f'loaded strategy with vertices:\n  {nodes}\n'
        f'and edges:\n {h.edges()}\n')
    return h


def _bitfields_to_ints(
        bit_state:
            dict[
                str,
                _ty.Literal[
                    0,
                    1]],
        vrs:
            dict
        ) -> dict[
            str,
            int]:
    """Convert bitfield representation to integers."""
    int_state = dict()
    for var, dom in vrs.items():
        if dom == 'boolean':
            int_state[var] = bit_state[var]
            continue
        bitnames = [
            f'{var}@{i}'
            for i in range(dom[1].bit_length())]
        bitnames[0] = f'{var}@0.{dom[0]}.{dom[1]}'
        bitvalues = [bit_state[b] for b in bitnames]
        # little-endian
        bitvalues = map(str, reversed(bitvalues))
        val = ''.join(bitvalues)
        int_state[var] = int(val, 2)
    return int_state


def _call_slugs(
        filename:
            str,
        synth:
            bool=True,
        symbolic:
            bool=True,
        slugs_compiler_path:
            str |
            None=None
        ) -> tuple[
            bool,
            str]:
    """Call `slugs` and return results.

    `slugs_compiler_path` is the path to
    the SlugsIn converter format.
    If `None` (default), then use the path
    as in the module-level identifier
    `SLUGS_COMPILER_PATH`.

    If this path begins with `/`,
    then it is considered to be absolute.
    Otherwise, it is relative to the path
    of the `slugs` executable.
    """
    if slugs_compiler_path is None:
        slugs_compiler_path = SLUGS_COMPILER_PATH
    slugs_path = None
    for exe_path in os.environ['PATH'].split(':'):
        if os.path.exists(os.path.join(exe_path, 'slugs')):
            slugs_path = os.path.join(exe_path, 'slugs')
            break
    if slugs_path is None:
        raise Exception('`slugs` not found in path.')
    if not os.path.isabs(slugs_compiler_path):
        slugs_compiler_path = os.path.abspath(
            os.path.join(
                os.path.dirname(slugs_path),
                slugs_compiler_path))
    if not os.path.exists(slugs_compiler_path):
        raise Exception('`slugs/compiler.py` not found.')
    with tempfile.NamedTemporaryFile(delete=False) as slugs_infile:
        subprocess.check_call(
            ['python', slugs_compiler_path, filename],
            stdout=slugs_infile,
            stderr=subprocess.STDOUT,
            universal_newlines=True)
    options = [slugs_path, slugs_infile.name]
    if synth:
        if symbolic:
            options.extend(['--symbolicStrategy', BDD_FILE])
        else:
            options.append('--explicitStrategy')
            options.append('--jsonOutput')
    else:
        # As of commit ad0cf12c14131fc6a20fe29edfe04d9aefd7c6d4
        # (tip of master branch of
        # <https://github.com/LTLMoP/slugs.git>
        # at the time of writing),
        # Slugs seems to default to checking realizability.
        # Trying to
        # provide `--onlyRealizability` leads to
        # the following error message from `slugs`:
        # > Error: Parameter '--onlyRealizability' is unknown.
        pass
    _logger.debug('Calling: ' + ' '.join(options))
    try:
        p = subprocess.Popen(
            options,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True)
    except OSError as e:
        if e.errno == errno.ENOENT:
            raise Exception('slugs not found in path.')
        else:
            raise
    out, err = p.communicate()
    msg = (
        f'\n slugs return code: {p.returncode}\n\n'
        f'\n slugs stderr: {err}\n\n'
        f'\n slugs stdout:\n\n {out}\n\n')
    _logger.debug(msg)
    # error ?
    if p.returncode != 0:
        raise Exception(msg)
    realizable = 'Specification is realizable' in err
    # check sanity
    if not realizable:
        assert 'Specification is unrealizable' in err
    return realizable, out
