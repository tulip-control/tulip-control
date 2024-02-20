# Copyright (c) 2015 by California Institute of Technology
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
"""Interface to `gr1py`.

<https://pypi.org/project/gr1py>
<https://github.com/slivingston/gr1py>
"""
import logging

import networkx as _nx

import tulip.interfaces.gr1c as _gr1c
import tulip.spec as _spec
try:
    import gr1py
    import gr1py.cli
except ImportError:
    gr1py = None


_logger = logging.getLogger(__name__)
_hl = 60 * '-'


def check_realizable(
        spec:
            _spec.GRSpec
        ) -> bool:
    """Decide realizability of specification.

    Consult the documentation of `synthesize` about parameters.

    @return:
        `True` if realizable,
        `False` if not, or
        an error occurs.
    """
    init_option = _gr1c.select_options(spec)
    tsys, exprtab = _spec_to_gr1py(spec)
    return gr1py.solve.check_realizable(
        tsys, exprtab,
        init_flags=init_option)


def synthesize(
        spec:
            _spec.GRSpec
        ) -> _nx.DiGraph:
    """Synthesize strategy realizing the given specification.

    cf. `tulip.interfaces.gr1c.synthesize`
    """
    init_option = _gr1c.select_options(spec)
    tsys, exprtab = _spec_to_gr1py(spec)
    strategy = gr1py.solve.synthesize(
        tsys, exprtab,
        init_flags=init_option)
    if strategy is None:
        return None
    s = gr1py.output.dumps_json(tsys.symtab, strategy)
    return _gr1c.load_aut_json(s)


def _spec_to_gr1py(
        spec:
            _spec.GRSpec
        ) -> tuple:
    if gr1py is None:
        raise ValueError(
            'Import of gr1py interface failed.\n'
            'Please verify installation of `gr1py`.')
    s = _spec.translate(spec, 'gr1c')
    _logger.info(
        f'\n{_hl}\n gr1py input:\n {s}\n{_hl}')
    tsys, exprtab = gr1py.cli.loads(s)
    return tsys, exprtab
