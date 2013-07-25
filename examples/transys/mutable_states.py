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
Created on Wed Jul 24 23:25:28 2013

@author: ifilippi
"""

from tulip import transys as trs

def test_mutable_fts():
    ts = trs.FTS(mutable=True)
    
    s0 = [1, 3]
    s1 = ['f', 1, dict() ]
    
    ts.states.add(s0)
    ts.states.add(s1)
    
    ts.states.add_initial(s0)
    
    ts.transitions.add(s0, s1)
    print ts
    ts.plot()
    
    ts.states.remove(s0)
    ts.plot()
    
    return ts

def test_mutable_ba():
    ba = trs.BA(mutable=True)
    
    s0 = [1, 5]
    s1 = [{}, (1, 2), 'f', ['d', {} ] ]
    
    ba.states.add_from([s0, s1] )
    ba.states.add_initial(s0)
    ba.add_final_state(s1)
    
    ba.alphabet.add({'p', '!p'} )
    ba.transitions.add(s0, s1)
    ba.transitions.remove(s0, s1)
    
    ba.transitions.add_labeled(s0, s1, {'p', '!p'} )
    ba.plot()
    
    ba.transitions.remove_labeled(s0, s1, {'p', '!p'} )
    ba.plot()

if __name__ == '__main__':
    ts = test_mutable_fts()
    ba = test_mutable_ba()