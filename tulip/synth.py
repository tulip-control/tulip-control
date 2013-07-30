# Copyright (c) 2012, 2013 by California Institute of Technology
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
---------------------------------------------------------------------
Interface to library of synthesis tools, e.g. JTLV, gr1c
---------------------------------------------------------------------

TODO:
    allow transitions systems as input in addition to GRSpec objects
    
    Desired capabilities and relevant issues (from Scott):
        - Automatic selection of output type of transition system based on given arguments.
        - Inference of system variables and corresponding domains from a given (incomplete)
        - LTL formula and labeled transition system.
        - Available as tulip.synthesize or from tulip import synthesize.
    
"""


import os
from tulip import jtlvint
#from tulip import gr1cint


def synthesize(option, specs, sys=None):
    """
        Function to call the appropriate synthesis tool on the spec.
        Takes an option string and a GRSpec object
    """
    #here we need somehting like:
    #spec.import_PropPreservingPartition(sys)
        
    if option == 'gr1c':
        ctrl = gr1cint.synthesize(specs)
    if option == 'jtlv':
        ctrl = jtlvint.synthesize(specs)
    else:
        raise Exception('Undefined synthesis option. '+\
                        'Current options are \"jtlv\" and \"gr1c\"')
  


