# !/usr/bin/env python
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
----------------
Parallel Simulation Module
----------------
"""

import sys
import time, random
import threading

from tulip import *
from tulip.automaton import AutomatonState
from errorprint import printError


class Strategy(threading.Thread):
    """
    Simulate an single automaton as a strategy
    
    Several Strategy-threads can be executed in parallel to simulate
    an asynchronous composition of automata. Shared variables are used to
    communicate.

    The output is only to stdout as formatted prints.
    
    Arguments to __init__ constructor:

    - `D` -- an Automaton object generated from the jtlvint.synthesize or
        jtlv.computeStrategy function.
    - `V` -- a dictionary in which the keys correspond to variable names and the
        values are the corresponding valuations. All variables that are either
        written or read by the automaton associated with this Strategy must be
        included in this dictionary. Note that if this Strategy is to be executed
        in Parallel with other Strategies, then they must all be assigned the
        same dictionary, as this is how shared variables are implemented.
    - `X` -- a list of variable names. The environment variables of the
        automaton. All variable names in `X` must be keys in `V`.
    - `Y` -- a list of variable names. The system variables of the
        automaton. All variable names in `Y` must be keys in `V`.
    - `name` -- name to identify the Strategy in the screen output.
    - `Tmin` -- minimum time between transitions (in ms)
    - `Tmax` -- maximum time between transitions (in ms)
    - `runtime` -- time in seconds to run the thread;
                   if 0 (default), then run indefinitely.
        
    Note: The automaton sleeps a certain amount of time between transitions.
           This time is a value chosen uniformly between `Tmin` and `Tmax`.
    """

    def __init__(self, D, V, X, Y, name, Tmin=1000, Tmax=1000, runtime=0):
        threading.Thread.__init__(self)
        self.D = D # automaton
        self.V = V # dictionary of state variables and their current values
        self.X = X # list of names for environment variables
        self.Y = Y # list of names for system variables

        self.name = name # name to identify the automaton's ID

        self.Tmin = Tmin # minimum time between steps (in ms)
        self.Tmax = Tmax # maximum time between steps (in ms)

        self.runtime = runtime  # in seconds


    # Inherit run function from Thread class
    def run(self):
        # INITIALITY
        # get environment variables
        s0 = dict(filter(lambda (k,v): k in (self.X + self.Y), self.V.iteritems()))

        # apply inputs
        aut_state = self.D.findNextAutState(current_aut_state=None, env_state=s0)

        # check if initial conditions satisfy assumptions
        if (aut_state==-1):
            printError(self.name + ": No transition found.", obj=self)

        # CONSECUTION
        start_time = time.time()
        while True:
            if self.runtime > 0 and time.time() - start_time > self.runtime:
                break
            print '%s %04i\t: %s\n' % (self.name, aut_state.id, str(self.V)),

            #sleep for a given time (to emulate different processor frequencies)
            time.sleep(random.uniform(self.Tmin/1000, self.Tmax/1000)) 

            # get environment variables
            inputs = dict(filter(lambda (k,v): k in self.X, self.V.iteritems()))

            # apply inputs
            aut_state = self.D.findNextAutState(current_aut_state=aut_state, env_state=inputs)
            if (aut_state==-1):
                printError(self.name + ": No transition found.", obj=self)

            # check environment assumptions
            if(not isinstance(aut_state, AutomatonState)):
                printError(self.name + ": The environment violated its assumptions.", obj=self)

            # write system variables
            map(lambda (k,v): self.V.update({k:v}), dict(filter(lambda (k,v): k in self.Y, aut_state.state.iteritems())).iteritems())
