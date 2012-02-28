#!/usr/bin/env python
#
# Copyright (c) 2012 by California Institute of Technology
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
# $Id$
"""
Interface to gr1c

In general, functions defined here will raise CalledProcessError (from
the subprocess module) or OSError if an exception occurs while
interacting with the gr1c executable.

Most functions have a "verbose" argument.  0 means silent (the default
setting), positive means provide some status updates.
"""

import subprocess
import tempfile

from spec import GRSpec
from errorprint import printWarning, printError

GR1C_BIN_PREFIX=""


def check_syntax(spec_str, verbose=0):
    """Check whether given string has correct gr1c specification syntax.

    Return True if syntax check passed, False on error.
    """
    f = tempfile.TemporaryFile()
    f.write(spec_str)
    f.seek(0)
    p = subprocess.Popen([GR1C_BIN_PREFIX+"gr1c", "-s"],
                         stdin=f,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p.wait()
    if p.returncode == 0:
        return True
    else:
        if verbose > 0:
            print p.stdout.read()
        return False


def check_realizable(spec, verbose=0):
    """Decide realizability of specification defined by given GRSpec object.

    Return True if realizable, False if not, or an error occurs.
    """
    f = tempfile.TemporaryFile()
    f.write(spec.dumpgr1c())
    f.seek(0)
    p = subprocess.Popen([GR1C_BIN_PREFIX+"gr1c", "-r"],
                         stdin=f,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p.wait()
    if p.returncode == 0:
        return True
    else:
        if verbose > 0:
            print p.stdout.read()
        return False
