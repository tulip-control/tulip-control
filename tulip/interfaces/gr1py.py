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
"""Interface to C{gr1py}.

U{https://pypi.python.org/pypi/gr1py}
U{https://github.com/slivingston/gr1py}
"""
from __future__ import absolute_import
import logging
from tulip.spec import translate
from tulip.interfaces.gr1c import load_aut_json
from tulip.interfaces.gr1c import select_options
try:
    import gr1py
    import gr1py.cli
except ImportError:
    gr1py = None


logger = logging.getLogger(__name__)
_hl = 60 * '-'


def check_realizable(spec):
    """Decide realizability of specification.

    Consult the documentation of L{synthesize} about parameters.

    @return: True if realizable, False if not, or an error occurs.
    """
    init_option = select_options(spec)
    tsys, exprtab = _spec_to_gr1py(spec)
    return gr1py.solve.check_realizable(
        tsys, exprtab, init_flags=init_option)

def synthesize(spec):
    """Synthesize strategy realizing the given specification.

    cf. L{tulip.interfaces.gr1c.synthesize}
    """
    init_option = select_options(spec)
    tsys, exprtab = _spec_to_gr1py(spec)
    strategy = gr1py.solve.synthesize(
        tsys, exprtab, init_flags=init_option)
    if strategy is None:
        return None
    s = gr1py.output.dump_json(tsys.symtab, strategy)
    return load_aut_json(s)


def _spec_to_gr1py(spec):
    if gr1py is None:
        raise ValueError('Import of gr1py interface failed.\n'
                         'Please verify installation of "gr1py".')
    s = translate(spec, 'gr1c')
    logger.info('\n{hl}\n gr1py input:\n {s}\n{hl}'.format(s=s, hl=_hl))
    tsys, exprtab = gr1py.cli.loads(s)
    return tsys, exprtab
