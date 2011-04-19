#!/usr/bin/env python
#
# Copyright (c) 2011 by California Institute of Technology
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

""" print.py --- Print an error and warning message

Nok Wongpiromsarn (nok@cds.caltech.edu)
August 3, 2010
"""
import inspect

class bcolors:
    OKGREEN = '\033[1;92m'
    WARNING = '\033[1;95m'
    FAIL = '\033[1;91m'
    INFO = '\033[1;94m'
    ENDC = '\033[0m'

def printWarning(text, obj=None):
    tmp = "WARNING"
    modulename = str(inspect.getmodulename(inspect.getouterframes(inspect.currentframe())[1][1]))
    if (modulename != 'None'):
        tmp += " " + modulename + '.'
    if (obj is not None):
        tmp += obj.__class__.__name__ + '.'
    if (modulename != 'None'):
        tmp += str(inspect.getouterframes(inspect.currentframe())[1][3])
    tmp += ": " + text 
    print bcolors.WARNING + tmp + bcolors.ENDC

def printError(text, obj=None):
    tmp = "ERROR"
    modulename = str(inspect.getmodulename(inspect.getouterframes(inspect.currentframe())[1][1]))
    if (modulename != 'None'):
        tmp += " " + modulename + '.'
    if (obj is not None):
        tmp += obj.__class__.__name__ + '.'
    if (modulename != 'None'):
        tmp += str(inspect.getouterframes(inspect.currentframe())[1][3])
    tmp += ": " + text 
    print bcolors.FAIL + tmp + bcolors.ENDC

def printInfo(text):
    print bcolors.INFO + text + bcolors.ENDC
