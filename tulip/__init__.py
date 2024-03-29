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
TuLiP toolbox

The Temporal Logic Planning (TuLiP) Toolbox provides functions
for verifying and constructing control protocols.

Notes
=====
Citations are used throughout the documentation.  References
corresponding to these citations are defined in doc/bibliography.rst
of the TuLiP source distribution.  E.g., [BK08] used in various
docstrings is listed in doc/bibliography.rst as the book "Principles
of Model Checking" by Baier and Katoen (2008).
"""

try:
    import tulip._version as _version
    __version__ = _version.version
except ImportError:
    __version__ = None

import tulip.abstract
from tulip.abstract import *
import tulip.dumpsmach
from tulip.dumpsmach import *
import tulip.graphics
from tulip.graphics import *
import tulip.gridworld
from tulip.gridworld import *
import tulip.hybrid
from tulip.hybrid import *
import tulip.spec
from tulip.spec import *
import tulip.synth
from tulip.synth import *
import tulip.transys
from tulip.transys import *
