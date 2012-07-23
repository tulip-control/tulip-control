#!/usr/bin/env python
#
# Copyright (c) 2011, 2012 by California Institute of Technology
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
# $Id:$
# SPIN interface
from nusmvint import generateSolverInput
import os
from subprocess import Popen, PIPE, call
from contextlib import contextmanager
from errorprint import printWarning, printError

class SPINError(Exception):
    pass

@contextmanager
def cd(path):
    old = os.getcwd()
    if path:
        os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)

prv = lambda x: os.path.join("..", x)
class SPINInstance:
    SPIN_PATH="/home/nick/SURF/Spin/Src6.2.2/spin"
    def __init__(self, model="tmp.pml", out="tmp.aut",
                    path=SPIN_PATH, verbose=0):
        (self.model, self.out, self.path) = (model, out, path)
        try:
            os.mkdir("pan")
        except OSError:
            # PAN directory already exists
            pass
        with cd("pan"):
            call([self.path, "-a", prv(self.model)], stdout=PIPE)
            call("gcc -o pan pan.c", shell=True)
    def generateTrace(self, verbose=0):
        realizable = False
        with open(self.out, 'w') as f:
            oldcwd = os.getcwd()
            (model_dn, model_fn) = os.path.split(self.model)
            with cd(model_dn):
                pan = Popen(oldcwd + "/pan/pan -a", shell=True, stdout=PIPE)
                (out, err) = pan.communicate()
                if verbose > 0:
                    print out
                if err:
                    raise SPINError(err)
                if "acceptance cycle" in out:
                    realizable = True
                spin = Popen([self.path, "-t", "-p", "-g", "-w", model_fn],
                            stdout=f, stderr=PIPE)
                (out, err) = spin.communicate()
                if err:
                    raise SPINError(err)
                return realizable
                    
def computeStrategy(pml_file, aut_file, verbose=0):
    spin = SPINInstance(pml_file, aut_file, verbose=verbose)
    try:
        return spin.generateTrace(verbose)
    except SPINError as e:
        printError("SPIN error: " + e.message)
    
def generateSPINInput(*args, **kwargs):
    generateSolverInput(*args, solver='SPIN', **kwargs)
