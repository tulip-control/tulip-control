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

import copy
import subprocess
import tempfile

from conxml import loadXML
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


def synthesize(spec, verbose=0):
    """Synthesize strategy.

    Return strategy as instance of Automaton class, or None if
    unrealizable or error occurs.
    """
    p = subprocess.Popen([GR1C_BIN_PREFIX+"gr1c", "-t", "tulip"],
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (stdoutdata, stderrdata) = p.communicate(spec.dumpgr1c())
    if p.returncode == 0:
        (prob, sys_dyn, aut) = loadXML(stdoutdata)
        return aut
    else:
        if verbose > 0:
            print stdoutdata
        return None


class GR1CSession:
    """Manage interactive session with gr1c.

    Given lists of environment and system variable names determine the
    order of values in state vectors for communication with the gr1c
    process.  Eventually there may be code to infer this directly from
    the spec file.

    Unless otherwise indicated, command methods return True on
    success, False if error.
    """
    def __init__(self, spec_filename, sys_vars, env_vars=[]):
        self.spec_filename = spec_filename
        self.sys_vars = sys_vars
        self.env_vars = env_vars
        if self.spec_filename is not None:
            self.p = subprocess.Popen([GR1C_BIN_PREFIX+"gr1c",
                                       "-i", self.spec_filename],
                                      stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT)
        else:
            self.p = None


    def iswinning(self, state):
        """Return True if given state is in winning set, False otherwise.

        state should be a dictionary with keys of variable names
        (strings) and values of the value taken by that variable in
        this state, e.g., as in nodes of the Automaton class.
        """
        state_vector = range(len(state))
        for ind in range(len(self.env_vars)):
            state_vector[ind] = state[self.env_vars[ind]]
        for ind in range(len(self.sys_vars)):
            state_vector[ind+len(self.env_vars)] = state[self.sys_vars[ind]]
        self.p.stdin.write("winning "+" ".join([str(i) for i in state_vector])+"\n")
        self.p.stdout.readline()
        if self.p.stdout.readline() == "True\n":
            return True
        else:
            return False


    def getindex(self, state, goal_mode):
        if goal_mode < 0 or goal_mode > self.numgoals()-1:
            raise ValueError("Invalid goal mode requested: "+str(goal_mode))
        state_vector = range(len(state))
        for ind in range(len(self.env_vars)):
            state_vector[ind] = state[self.env_vars[ind]]
        for ind in range(len(self.sys_vars)):
            state_vector[ind+len(self.env_vars)] = state[self.sys_vars[ind]]
        self.p.stdin.write("getindex "+" ".join([str(i) for i in state_vector])+" "+str(goal_mode)+"\n")
        self.p.stdout.readline()
        return int(self.p.stdout.readline()[:-1])

    def env_next(self, state):
        """Return list of possible next environment moves, given current state.

        Format of given state is same as for iswinning method.
        """
        state_vector = range(len(state))
        for ind in range(len(self.env_vars)):
            state_vector[ind] = state[self.env_vars[ind]]
        for ind in range(len(self.sys_vars)):
            state_vector[ind+len(self.env_vars)] = state[self.sys_vars[ind]]
        self.p.stdin.write("envnext "+" ".join([str(i) for i in state_vector])+"\n")
        self.p.stdout.readline()
        env_moves = []
        line = self.p.stdout.readline()
        while line != "---\n":
            env_moves.append(dict([(k, int(s)) for (k,s) in zip(self.env_vars, line.split())]))
            line = self.p.stdout.readline()
        return env_moves


    def sys_nextfeas(self, state, env_move, goal_mode):
        """Return list of next system moves consistent with some strategy.

        Format of given state and env_move is same as for iswinning
        method.
        """
        if goal_mode < 0 or goal_mode > self.numgoals()-1:
            raise ValueError("Invalid goal mode requested: "+str(goal_mode))
        state_vector = range(len(state))
        for ind in range(len(self.env_vars)):
            state_vector[ind] = state[self.env_vars[ind]]
        for ind in range(len(self.sys_vars)):
            state_vector[ind+len(self.env_vars)] = state[self.sys_vars[ind]]
        emove_vector = range(len(env_move))
        for ind in range(len(self.env_vars)):
            emove_vector[ind] = env_move[self.env_vars[ind]]
        self.p.stdin.write("sysnext "+" ".join([str(i) for i in state_vector])+" "+" ".join([str(i) for i in emove_vector])+" "+str(goal_mode)+"\n")
        self.p.stdout.readline()
        sys_moves = []
        line = self.p.stdout.readline()
        while line != "---\n":
            sys_moves.append(dict([(k, int(s)) for (k,s) in zip(self.sys_vars, line.split())]))
            line = self.p.stdout.readline()
        return sys_moves


    def sys_nexta(self, state, env_move):
        """Return list of possible next system moves, whether or not winning.

        Format of given state and env_move is same as for iswinning
        method.
        """
        state_vector = range(len(state))
        for ind in range(len(self.env_vars)):
            state_vector[ind] = state[self.env_vars[ind]]
        for ind in range(len(self.sys_vars)):
            state_vector[ind+len(self.env_vars)] = state[self.sys_vars[ind]]
        emove_vector = range(len(env_move))
        for ind in range(len(self.env_vars)):
            emove_vector[ind] = env_move[self.env_vars[ind]]
        self.p.stdin.write("sysnexta "+" ".join([str(i) for i in state_vector])+" "+" ".join([str(i) for i in emove_vector])+"\n")
        self.p.stdout.readline()
        sys_moves = []
        line = self.p.stdout.readline()
        while line != "---\n":
            sys_moves.append(dict([(k, int(s)) for (k,s) in zip(self.sys_vars, line.split())]))
            line = self.p.stdout.readline()
        return sys_moves


    def getvars(self):
        """Return string of environment and system variable names in order.

        Indices are indicated in parens.
        """
        self.p.stdin.write("var\n")
        self.p.stdout.readline()
        return self.p.stdout.readline()[:-1]

    def numgoals(self):
        self.p.stdin.write("numgoals\n")
        self.p.stdout.readline()
        return int(self.p.stdout.readline()[:-1])

    def reset(self, spec_filename=None):
        """Quit and start anew, reading spec from file with given name.

        If no filename given, then use previous one.
        """
        if self.p is not None:
            self.p.stdin.write("quit\n")
            returncode = self.p.wait()
            self.p = None
            if returncode != 0:
                self.spec_filename = None
                return False
        if spec_filename is not None:
            self.spec_filename = spec_filename
        if self.spec_filename is not None:
            self.p = subprocess.Popen([GR1C_BIN_PREFIX+"gr1c",
                                       "-i", self.spec_filename],
                                      stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT)
        else:
            self.p = None
        return True

    def close(self):
        """End session, and kill gr1c child process."""
        self.p.stdin.write("quit\n")
        returncode = self.p.wait()
        self.p = None
        if returncode != 0:
            return False
        else:
            return True
