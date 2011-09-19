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
""" 
------------------------------------
Jtlvint Module --- Interface to JTLV
------------------------------------

About JTLV, see http://jtlv.ysaar.net/
"""

import re, os, subprocess, sys
from prop2part import PropPreservingPartition
from polytope import Region
import rhtlp
import grgameint


def generateJTLVInput(env_vars={}, sys_disc_vars={}, spec=[], disc_props={}, \
                          disc_dynamics=PropPreservingPartition(), \
                          smv_file='tmp.smv', spc_file='tmp.spc', \
                          file_exist_option='a', verbose=0):
    """Generate JTLV input files: smv_file and spc_file.

    Input:

    - `env_vars`: a dictionary {str : str} or {str : list} whose keys are the names 
      of environment variables and whose values are their possible values, e.g., 
      boolean or {0, 2, ..., 5} or [0, 2, 3, 4, 5].
    - `sys_disc_vars`: a dictionary {str : str} or {str : list} whose keys are the 
      names of discrete system variables and whose values are their possible values.
    - `spec`: a list of two strings that represents system specification of the form
      assumption -> guarantee; the first string is the assumption and the second 
      string is the guarantee.
    - `disc_props`: a dictionary {str : str} whose keys are the symbols for 
      propositions on discrete variables and whose values are the actual propositions
      on discrete variables.
    - `disc_dynamics`: a PropPreservingPartition object that represents the 
      transition system obtained from the discretization procedure.
    - `smv_file`: a string that specifies the name of the resulting smv file.
    - `spc_file`: a string that specifies the name of the resulting spc file.
    - `file_exist_option`: a string that indicate what to do when the specified smv_file 
      or spc_file exists. Possible values are: 'a' (ask whether to replace or
      create a new file), 'r' (replace the existing file), 'n' (create a new file).
    - `verbose`: an integer that specifies the level of verbosity. If verbose is set to 0,
      this function will not print anything on the screen.
    """
    prob = rhtlp.SynthesisProb(env_vars={}, sys_disc_vars={}, disc_props={}, \
                       sys_cont_vars=[], cont_state_space=None, \
                       cont_props={}, sys_dyn=None, spec=['',''], verbose=verbose)
    prob.createProbFromDiscDynamics(env_vars=env_vars, sys_disc_vars=sys_disc_vars, \
                                         disc_props=disc_props, \
                                         disc_dynamics=disc_dynamics, spec=spec, \
                                         verbose=verbose)
    prob.toJTLVInput(smv_file=smv_file, spc_file=spc_file, \
                          file_exist_option=file_exist_option, verbose=verbose)
    return prob


###################################################################

def checkRealizability(smv_file='', spc_file='', aut_file='', heap_size='-Xmx128m', \
                           pick_sys_init=True, file_exist_option='a', verbose=0):
    """Determine whether the spec in smv_file and spc_file is realizable without 
    extracting an automaton.

    Input:

    - `smv_file`: a string that specifies the name of the smv file.
    - `spc_file`: a string that specifies the name of the spc file.
    - `aut_file`: a string that specifies the name of the file containing the output of JTLV
      (e.g. an initial state starting from which the spec is cannot be satisfied).
    - `heap_size`: a string that specifies java heap size. 
    - `pick_sys_init` is a boolean indicating whether the system can pick 
      its initial state (in response to the initial environment state).
    - `file_exist_option`: a string that indicate what to do when the specified aut_file 
      exists. Possible values are: 'a' (ask whether to replace or create a new file), 
      'r' (replace the existing file), 'n' (create a new file).
    - `verbose`: an integer that specifies the level of verbosity.
    """

    init_option = 1
    if (not pick_sys_init):
        init_option = 0
    realizable = computeStrategy(smv_file=smv_file, spc_file=spc_file, \
                                     aut_file=aut_file, heap_size=heap_size, \
                                     priority_kind=-1, init_option=init_option, \
                                     file_exist_option=file_exist_option, verbose=verbose)
    return realizable



###################################################################

def synthesize(env_vars={}, sys_disc_vars={}, spec='', disc_props={}, \
                   disc_dynamics=PropPreservingPartition(), \
                   smv_file='tmp.smv', spc_file='tmp.spc', \
                   aut_file='', heap_size='-Xmx128m', priority_kind=3, init_option=1, \
                   file_exist_option='a', verbose=0):
    """Compute an automaton satisfying `spec`. Return the realizability of `spec`.
    If `spec` is realizable, the resulting automaton will be stored in the
    `aut_file`. Otherwise, the counter examples will be stored.
    This function essentially combines ``generateJTLVInput`` and ``computeStrategy``

    Input:

    - `env_vars`: a dictionary {str : str} or {str : list} whose keys are the names 
      of environment variables and whose values are their possible values, e.g., 
      boolean or {0, 2, ..., 5} or [0, 2, 3, 4, 5].
    - `sys_disc_vars`: a dictionary {str : str} or {str : list} whose keys are the 
      names of discrete system variables and whose values are their possible values.
    - `spec`: a list of two strings that represents system specification of the form
      assumption -> guarantee; the first string is the assumption and the second 
      string is the guarantee.
    - `disc_props`: a dictionary {str : str} whose keys are the symbols for 
      propositions on discrete variables and whose values are the actual propositions
      on discrete variables.
    - `disc_dynamics`: a PropPreservingPartition object that represents the 
      transition system obtained from the discretization procedure.
    - `smv_file`: a string that specifies the name of the resulting smv file.
    - `spc_file`: a string that specifies the name of the resulting spc file.
    - `aut_file`: a string that specifies the name of the file containing the resulting 
      automaton.
    - `heap_size`: a string that specifies java heap size. 
    - `priority_kind`: a string of length 3 or an integer that specifies the type of 
      priority used in extracting the automaton. See the documentation of the 
      ``computeStrategy`` function for the possible values of `priority_kind`.
    - `init_option`: an integer in that specifies how to handle the initial state of 
      the system. See the documentation of the ``computeStrategy`` function for the 
      possible values of `init_option`.
    - `file_exist_option`: a string that indicate what to do when the specified smv_file 
      or spc_file exists. Possible values are: 'a' (ask whether to replace or
      create a new file), 'r' (replace the existing file), 'n' (create a new file).
    - `verbose`: an integer that specifies the level of verbosity. If verbose is set to 0,
      this function will not print anything on the screen.
    """
    generateJTLVInput(env_vars=env_vars, sys_disc_vars=sys_disc_vars, spec=spec, \
                          disc_props=disc_props, disc_dynamics=disc_dynamics, \
                          smv_file=smv_file, spc_file=spc_file, \
                          file_exist_option=file_exist_option, verbose=verbose)
    realizable = computeStrategy(smv_file=smv_file, spc_file=spc_file, \
                                        aut_file=aut_file, heap_size=heap_size, \
                                        priority_kind=priority_kind, \
                                        init_option=init_option, \
                                        file_exist_option=file_exist_option, \
                                        verbose=verbose)
    return realizable


###################################################################

def computeStrategy(smv_file, spc_file, aut_file='', heap_size='-Xmx128m', \
                        priority_kind=3, init_option=1, file_exist_option='a', verbose=0):
    """Compute an automaton satisfying the spec in smv_file and spc_file and store in 
    aut_file. Return the realizability of the spec.

    Input:

    - `smv_file`: a string that specifies the name of the smv file.
    - `spc_file`: a string that specifies the name of the spc file.
    - `aut_file`: a string that specifies the name of the file containing the resulting 
      automaton.
    - `heap_size`: a string that specifies java heap size. 
    - `priority_kind`: a string of length 3 or an integer that specifies the type of 
      priority used in extracting the automaton. Possible values of `priority_kind` are: 

        * 3 - 'ZYX'
        * 7 - 'ZXY'
        * 11 - 'YZX'
        * 15 - 'YXZ'
        * 19 - 'XZY'
        * 23 - 'XYZ'

      Here X means that the controller tries to disqualify one of the environment 
      assumptions, 
      Y means that the controller tries to advance with a finite path to somewhere, and
      Z means that the controller tries to satisfy one of his guarantees.
    - `init_option`: an integer in that specifies how to handle the initial state of 
      the system. Possible values of `init_option` are

        * 0 - The system has to be able to handle all the possible initial system
          states specified on the guarantee side of the specification.
        * 1 (default) - The system can choose its initial state, in response to the initial
          environment state. For each initial environment state, the resulting
          automaton contains exactly one initial system state, starting from which
          the system can satisfy the specification.
        * 2 - The system can choose its initial state, in response to the initial
          environment state. For each initial environment state, the resulting
          automaton contain all the possible initial system states, starting from which
          the system can satisfy the specification.
    - `file_exist_option`: a string that indicate what to do when the specified aut_file 
      exists. Possible values are: 'a' (ask whether to replace or create a new file), 
      'r' (replace the existing file), 'n' (create a new file).
    - `verbose`: an integer that specifies the level of verbosity.
    """
    realizable = grgameint.solveGame(smv_file=smv_file, \
                                         spc_file=spc_file, \
                                         aut_file=aut_file, \
                                         heap_size=heap_size, \
                                         priority_kind=priority_kind, \
                                         init_option=init_option, \
                                         file_exist_option=file_exist_option, \
                                         verbose=verbose)

    return realizable


###################################################################

def getCounterExamples(aut_file, verbose=0):
    """Return a list of dictionary, each representing a counter example.

    Input:

    - `aut_file`: a string containing the name of the file containing the
      counter examples generated by JTLV.
    """

    counter_examples = grgameint.getCounterExamples(aut_file=aut_file, verbose=verbose)
    return counter_examples


###################################################################

# Test case
#  * Default: Use init_option=1 with dynamics
#  * 1: Use init_option=2 with dynamics
#  * 2: Use init_option=0 with dynamics. This makes the spec unrealizable.
#  * 3: Use init_option=1 with no dynamics
if __name__ == "__main__":
    testfile = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'tmpspec', \
                                'testjtlvint')
    smvfile = testfile + '.smv'
    spcfile = testfile + '.spc'
    autfile = testfile + '.aut'
    print('Testing generateJTLVInput')
#    env_vars = {'park' : 'boolean', 'cellID' : '{0,...,3,4,5}'}
    env_vars = {'park' : 'boolean', 'cellID' : [0,1,2,3,4,5]}
    sys_disc_vars = {'gear' : '{-1...1}'}
    cont_props = ['X0', 'X1', 'X2', 'X3', 'X4']
    disc_dynamics = PropPreservingPartition()
    region0 = Region('p0', [1, 0, 0, 0, 0])
    region1 = Region('p1', [0, 1, 0, 0, 0])
    region2 = Region('p2', [1, 0, 1, 0, 0])
    region3 = Region('p3', [0, 0, 0, 1, 0])
    region4 = Region('p4', [0, 0, 0, 0, 1])
    region5 = Region('p5', [1, 0, 0, 1, 1])
    disc_dynamics.list_region = [region0, region1, region2, region3, region4, region5]
    disc_dynamics.num_regions = len(disc_dynamics.list_region)
    disc_dynamics.trans = [[1, 1, 0, 1, 0, 0], \
                         [1, 1, 1, 0, 1, 0], \
                         [0, 1, 1, 0, 0, 1], \
                         [1, 0, 0, 1, 1, 0], \
                         [0, 1, 0, 1, 1, 1], \
                         [0, 0, 1, 0, 1, 1]]
    disc_dynamics.list_prop_symbol = cont_props
    disc_dynamics.num_prop = len(disc_dynamics.list_prop_symbol)
    disc_props = {'Park' : 'park', \
                      'X0d' : 'cellID=0', \
                      'X1d' : 'cellID=1', \
                      'X2d' : 'cellID=2', \
                      'X3d' : 'cellID=3', \
                      'X4d' : 'cellID=4', \
                      'X5d' : 'gear = 1'}

    assumption = '[]<>(!park) & []<>(!X4d)'
    guarantee = '[]<>(X4d -> X4) & []<>X1 & [](Park -> X0)'
    spec = [assumption, guarantee]

    if ('3' in sys.argv): # For spec with no dynamics
        disc_dynamics=PropPreservingPartition()
        spec[1] = '[]<>(X0d -> X5d)'  
        newvarname = generateJTLVInput(env_vars=env_vars, sys_disc_vars=sys_disc_vars, \
                                           spec=spec, disc_props=disc_props, \
                                           disc_dynamics=disc_dynamics, \
                                           smv_file=smvfile, spc_file=spcfile, verbose=2)
    else:
        newvarname = generateJTLVInput(env_vars=env_vars, sys_disc_vars=sys_disc_vars, \
                                           spec=spec, disc_props=disc_props, \
                                           disc_dynamics=disc_dynamics, \
                                           smv_file=smvfile, spc_file=spcfile, verbose=2)
    print('DONE')
    print('================================\n')

    ####################################

    print('Testing checkRealizability')
    pick_sys_init = (not ('2' in sys.argv))
    realizability = checkRealizability(smv_file=smvfile, spc_file=spcfile, \
                                           aut_file='', heap_size='-Xmx128m', \
                                           pick_sys_init=pick_sys_init, \
                                           file_exist_option='a', verbose=3)
    print realizability
    print('DONE')
    print('================================\n')

    ####################################

    print('Testing computeStrategy')
    init_option = 1
    if ('1' in sys.argv):
        init_option = 2
    elif ('2' in sys.argv):
        init_option = 0
    realizability = computeStrategy(smv_file=smvfile, spc_file=spcfile, aut_file='', \
                                        heap_size='-Xmx128m', priority_kind='ZYX', \
                                        init_option=init_option, file_exist_option='a', \
                                        verbose=3)
    print realizability
    print('DONE')
    print('================================\n')

    ####################################

    print('Testing synthesize')

    realizability = synthesize(env_vars=env_vars, sys_disc_vars=sys_disc_vars, \
                                   spec=spec, disc_props=disc_props, \
                                   disc_dynamics=disc_dynamics, \
                                   smv_file=smvfile, spc_file=spcfile, \
                                   aut_file='', heap_size='-Xmx128m', priority_kind=3, \
                                   init_option=init_option, \
                                   file_exist_option='a', verbose=3)

    ####################################

    if ('2' in sys.argv):
        print('Testing getCounterExamples')
        counter_examples = getCounterExamples(aut_file=autfile, verbose=1)
