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
-----------------------------------------------
Receding horizon Temporal Logic Planning Module
-----------------------------------------------

Nok Wongpiromsarn (nok@cds.caltech.edu)

:Date: August 25, 2010
:Version: 0.1.0

minor refactoring by SCL <slivingston@caltech.edu>
3 May 2011.
"""

import sys, os, re, subprocess, copy

from errorprint import printWarning, printError, printInfo
from parsespec import parseSpec
from polytope_computations import Polytope, Region
from discretizeM import CtsSysDyn, discretizeM
from prop2part import PropPreservingPartition, prop2part2
import automaton
from spec import GRSpec
import rhtlputil
import grgameint


class SynthesisProb:
    """SynthesisProb class for specifying a planner synthesis problem.
    
    A SynthesisProb object contains the following fields:

    - `env_vars`: a dictionary {str : str} whose keys are the names 
      of environment variables and whose values are their possible values, e.g., 
      boolean or {0, 2, 3, 4, 5}.
    - `sys_vars`: a dictionary {str : str} whose keys are the 
      names of system variables and whose values are their possible values.
    - `spec`: a GRSpec object that specifies the specification of this synthesis problem.
    - `disc_cont_var`: the name of the continuous variable after the discretization.
    - `disc_dynamics`: a list of Region objects corresponding to the partition of the
      continuous state space.

    **Constructor**:

    **SynthesisProb** ([ `file` = ''[, `verbose` = 0]]): 
    construct this SynthesisProb object from file

    - `file`: the name of the rhtlp file to be parsed. If `file` is given,
      the rest of the inputs to this function will be ignored.

    **SynthesisProb** ([ `env_vars` = {}[, `sys_disc_vars` = {}[, `disc_props` = {}[, 
    `disc_dynamics` = None[, `spec` = GRSpec()[, `verbose` = 0]]]]]])

    **SynthesisProb** ([ `env_vars` = {}[, `sys_disc_vars` = {}[, `disc_props` = {}[, 
    `cont_state_space` = None[, `cont_props` = {}[, `sys_dyn` = None[, 
    `spec` = GRSpec()[, `verbose` = 0]]]]]]]])

    - `env_vars`: a dictionary {str : str} or {str : list} whose keys are the names 
      of environment variables and whose values are their possible values, e.g., 
      boolean or {0, 2, ..., 5} or [0, 2, 3, 4, 5].
    - `sys_disc_vars`: a dictionary {str : str} or {str : list} whose keys are the 
      names of discrete system variables and whose values are their possible values.
    - `disc_props`: a dictionary {str : str} whose keys are the symbols for 
      propositions on discrete variables and whose values are the actual propositions
      on discrete variables.
    - `disc_dynamics`: a PropPreservingPartition object that represents the 
      transition system obtained from the discretization procedure.
      if `disc_dynamics` is given, `cont_state_space`, `cont_props` and `sys_dyn`
      will be ignored.
    - `cont_state_space`: a Polytope object that represent the state space of the
      continuous variables
    - `cont_props`: a dictionary {str : Polytope} whose keys are the symbols for 
      propositions on continuous variables and whose values are polytopes that represent
      the region in the state space in which the corresponding proposition hold.
    - `sys_dyn`: a CtsSysDyn object that specifies the dynamics of the continuous variables
    - `spec`: a GRSpec object that specifies the specification of this synthesis problem
    - `verbose`: an integer that specifies the level of verbosity.
    """

    def __init__(self, **args):

        self.__env_vars = {}
        self.__sys_vars = {}
        self.__disc_props = {}
        self.__spec = GRSpec(env_init='', sys_init='', env_safety='', sys_safety='', \
                               env_prog='', sys_prog='')
        self.__disc_cont_var = ''
        self.__disc_dynamics = None
        self.__jtlvfile = args.get('sp_name',
                                   os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'tmpspec', 'tmp'))
        self.__realizable = None

        verbose = args.get('verbose', 0)
        env_vars = args.get('env_vars', {})
        sys_disc_vars = args.get('sys_disc_vars', {})
        disc_props = args.get('disc_props', {})

        spec = GRSpec(env_init='', sys_init='', env_safety='', sys_safety='', \
                               env_prog='', sys_prog='')
        if ('spec' in args.keys()):
            spec = args['spec']

        if ('file' in args.keys()):
            file = args['file']
            # Check that the input is valid
            if (not isinstance(file, str)):
                printError("The input file is expected to be a string", obj=self)
                raise TypeError("Invalid file.")
            if (not os.path.isfile(file)):
                printError("The rhtlp file " + file + " does not exist.", obj=self)
                raise TypeError("Invalid file.")

            (env_vars, sys_disc_vars, disc_props, sys_cont_vars, cont_state_space, \
                 cont_props, sys_dyn, spec) = parseSpec(spec_file=file)
            self.createProbFromContDynamics(env_vars=env_vars, \
                                            sys_disc_vars=sys_disc_vars, \
                                            disc_props=disc_props, \
                                            cont_state_space=cont_state_space, \
                                            cont_props=cont_props, \
                                            sys_dyn=sys_dyn, 
                                            spec=spec, \
                                            verbose=verbose)

        elif ('disc_dynamics' in args.keys()):
            disc_dynamics = args['disc_dynamics']
            self.createProbFromDiscDynamics(env_vars=env_vars, \
                                            sys_disc_vars=sys_disc_vars, \
                                            disc_props=disc_props, \
                                            disc_dynamics=disc_dynamics, \
                                            spec=spec, \
                                            verbose=verbose)
        else:       
            cont_state_space = args.get('cont_state_space', None)
            cont_props = args.get('cont_props', {})
            sys_dyn = args.get('sys_dyn', None)
            self.createProbFromContDynamics(env_vars=env_vars, \
                                            sys_disc_vars=sys_disc_vars, \
                                            disc_props=disc_props, \
                                            cont_state_space=cont_state_space, \
                                            cont_props=cont_props, \
                                            sys_dyn=sys_dyn, 
                                            spec=spec, \
                                            verbose=verbose)

    ###################################################################

    def getEnvVars(self):
        """
        Return the environment variables of this object as a dictionary 
        whose key is the name of the variable
        and whose value is the possible values that the variable can take.
        """
        return copy.deepcopy(self.__env_vars)

    ###################################################################

    def setEnvVars(self, env_vars, verbose=0):
        """Set the environment variables.
        """
        sys_disc_vars = self.getSysDiscVars()
        self.createProbFromDiscDynamics(env_vars=env_vars, \
                                            sys_disc_vars=sys_disc_vars, \
                                            disc_props=self.__disc_props, \
                                            disc_dynamics=self.__disc_dynamics, \
                                            spec=self.__spec, \
                                            verbose=verbose)

    ###################################################################

    def getSysDiscVars(self):
        """
        Return the system discrete variables of 
        this object as a dictionary whose key is the name of the variable
        and whose value is the possible values that the variable can take.
        """
        sys_disc_vars = self.getSysVars()
        if (self.getDiscretizedContVar() is not None and \
                len(self.getDiscretizedContVar()) > 0 and \
                self.getDiscretizedContVar() in sys_disc_vars):
            del sys_disc_vars[self.getDiscretizedContVar()]
        return sys_disc_vars

    ###################################################################

    def setSysDiscVars(self, sys_disc_vars, verbose=0):
        """
        Set the system discrete variables.
        """
        self.createProbFromDiscDynamics(env_vars=self.__env_vars, \
                                            sys_disc_vars=sys_disc_vars, \
                                            disc_props=self.__disc_props, \
                                            disc_dynamics=self.__disc_dynamics, \
                                            spec=self.__spec, \
                                            verbose=verbose)

    ###################################################################

    def getSysVars(self):
        """
        Return the system (discrete and discretized continuous) variables of 
        this object as a dictionary whose key is the name of the variable
        and whose value is the possible values that the variable can take.
        """
        return copy.deepcopy(self.__sys_vars)

    ###################################################################

    def getDiscProps(self):
        """
        Return the discrete propositions as a dictionary whose key is the 
        symbol of the proposition and whose value is the actual formula of
        the proposition.
        """
        return copy.deepcopy(self.__disc_props)

    ###################################################################

    def setDiscProps(self, disc_props, verbose=0):
        """
        Set the propositions on discrete variables.
        """
        sys_disc_vars = self.getSysVars()
        self.createProbFromDiscDynamics(env_vars=self.__env_vars, \
                                            sys_disc_vars=sys_disc_vars, \
                                            disc_props=disc_props, \
                                            disc_dynamics=self.__disc_dynamics, \
                                            spec=self.__spec, \
                                            verbose=verbose)

    ###################################################################

    def getSpec(self):
        """
        Return the specification of this object.
        """
        return copy.deepcopy(self.__spec)

    ###################################################################

    def setSpec(self, spec, verbose=0):
        """
        Set the specification.
        """
        sys_disc_vars = self.getSysVars()
        self.createProbFromDiscDynamics(env_vars=self.__env_vars, \
                                            sys_disc_vars=sys_disc_vars, \
                                            disc_props=self.__disc_props, \
                                            disc_dynamics=self.__disc_dynamics, \
                                            spec=spec, \
                                            verbose=verbose)

    ###################################################################

    def getDiscretizedContVar(self):
        """
        Return the name of the discretized continuous variable.
        """
        return self.__disc_cont_var

    ###################################################################

    def getDiscretizedDynamics(self):
        """
        Return the discretized dynamics.
        """
        return self.__disc_dynamics

    ###################################################################

    def setDiscretizedDynamics(self, disc_dynamics, verbose=0):
        """
        Set the discretized dynamics.
        """
        sys_disc_vars = self.getSysVars()
        self.createProbFromDiscDynamics(env_vars=self.__env_vars, \
                                            sys_disc_vars=sys_disc_vars, \
                                            disc_props=self.__disc_props, \
                                            disc_dynamics=disc_dynamics, \
                                            spec=self.__spec, \
                                            verbose=verbose)

    ###################################################################

    def getJTLVFile(self):
        """
        Return the name of JTLV files. The smv, spc and aut files are appended
        by .smv, .spc and .aut, respectively.
        """
        return self.__jtlvfile

    ###################################################################

    def setJTLVFile(self, jtlvfile):
        """
        Set the temporary jtlv files (smv, spc, aut).
        """
        if (isinstance(jtlvfile, str)):
            self.__jtlvfile = jtlvfile
            self.__realizable = None
        else:
            printError('The input jtlvfile must be a string indicating the name of the file.', \
                           obj=self)


    ###################################################################

    def createProbFromContDynamics(self, env_vars={}, sys_disc_vars={}, disc_props={}, \
                                       cont_state_space=None, cont_props={}, sys_dyn=None, \
                                       spec=GRSpec(), verbose=0):
        """
        Construct SynthesisProb from continuous dynamics.

        Input:

        - `env_vars`: a dictionary {str : str} or {str : list} whose keys are the names 
          of environment variables and whose values are their possible values, e.g., 
          boolean or {0, 2, ..., 5} or [0, 2, 3, 4, 5].
        - `sys_disc_vars`: a dictionary {str : str} or {str : list} whose keys are the 
          names of discrete system variables and whose values are their possible values.
        - `disc_props`: a dictionary {str : str} whose keys are the symbols for 
          propositions on discrete variables and whose values are the actual propositions
          on discrete variables.
        - `cont_state_space`: a Polytope object that represent the state space of the
          continuous variables
        - `cont_props`: a dictionary {str : Polytope} whose keys are the symbols for 
          propositions on continuous variables and whose values are polytopes that represent
          the region in the state space in which the corresponding proposition hold.
        - `sys_dyn`: a CtsSysDyn object that specifies the dynamics of the continuous variables
        - `spec`: a GRSpec object that specifies the specification of this synthesis problem.
        - `verbose`: an integer that specifies the level of verbosity.
        """

        if (not isinstance(cont_state_space, Polytope) and \
                cont_state_space is not None):
            printError("The type of input cont_state_space is expected to be " + \
                           "Polytope", obj=self)
            raise TypeError("Invalid cont_state_space.")
        if (not isinstance(cont_props, dict) and cont_props is not None):
            printError("The input cont_props is expected to be a dictionary " + \
                           "{str : Polytope}", obj=self)
            raise TypeError("Invalid disc_props.")
        if (not isinstance(sys_dyn, CtsSysDyn) and sys_dyn is not None):
            printError("The type of input sys_dyn is expected to be CtsSysDyn", obj=self)
            raise TypeError("Invalid sys_dyn.")

        # Process the continuous component
        if (cont_state_space is not None):
            if (cont_props is None):
                cont_props = []
            cont_partition = prop2part2(cont_state_space, cont_props)
            disc_dynamics = copy.deepcopy(cont_partition)
            disc_dynamics.trans = disc_dynamics.adj
            for fromcell in xrange(0,len(disc_dynamics.trans)):
                disc_dynamics.trans[fromcell][fromcell] = 1
            if (sys_dyn is not None):
                disc_dynamics = discretizeM(cont_partition, sys_dyn, verbose=verbose)
        else:
            if (verbose > 0):
                print("No continuous component")
            disc_dynamics = PropPreservingPartition(domain=None, num_prop=0, \
                                                        list_region=[], num_regions=0, \
                                                        adj=[], trans=[], \
                                                        list_prop_symbol=None)

        self.createProbFromDiscDynamics(env_vars=env_vars, \
                                            sys_disc_vars=sys_disc_vars, \
                                            disc_props=disc_props, \
                                            disc_dynamics=disc_dynamics, \
                                            spec=spec, \
                                            verbose=verbose)

    ###################################################################

    def createProbFromDiscDynamics(self, env_vars={}, sys_disc_vars={}, \
                       disc_props={}, disc_dynamics=PropPreservingPartition(), \
                       spec=GRSpec(), verbose=0):
        """
        Construct SynthesisProb from discretized continuous dynamics.

        Input:
        
        - `env_vars`: a dictionary {str : str} or {str : list} whose keys are the names 
          of environment variables and whose values are their possible values, e.g., 
          boolean or {0, 2, ..., 5} or [0, 2, 3, 4, 5].
        - `sys_disc_vars`: a dictionary {str : str} or {str : list} whose keys are the 
          names of discrete system variables and whose values are their possible values.
        - `disc_props`: a dictionary {str : str} whose keys are the symbols for 
          propositions on discrete variables and whose values are the actual propositions
          on discrete variables.
        - `disc_dynamics`: a PropPreservingPartition object that represents the 
          transition system obtained from the discretization procedure.
        - `spec`: a GRSpec object that specifies the specification of this synthesis problem.
        - `verbose`: an integer that specifies the level of verbosity.
        """

        # Check that the input is valid
        if (not isinstance(env_vars, dict) and env_vars is not None):
            printError("The input env_vars is expected to be a dictionary " + \
                           "{str : str} or {str : list}.", obj=self)
            raise TypeError("Invalid env_vars.")
        if (not isinstance(sys_disc_vars, dict) and sys_disc_vars is not None):
            printError("The input sys_disc_vars is expected to be a dictionary " + \
                           "{str : str} or {str : list}", obj=self)
            raise TypeError("Invalid sys_disc_vars.")
        if (not isinstance(disc_props, dict) and disc_props is not None):
            printError("The input disc_props is expected to be a dictionary " + \
                           "{str : str}", obj=self)
            raise TypeError("Invalid disc_props.")
        if (not isinstance(disc_dynamics, PropPreservingPartition) and \
                disc_dynamics is not None):
            printError("The type of input disc_dynamics is expected to be " + \
                           "PropPreservingPartition", obj=self)
            raise TypeError("Invalid disc_dynamics.")
        if (not isinstance(spec, GRSpec) and \
                (not isinstance(spec, list) or len(spec) != 2)):
            printError("The input spec is expected to be a GRSpec object", obj=self)
            raise TypeError("Invalid spec.")            

        # Check that the number of regions in disc_dynamics is correct.
        if (disc_dynamics is not None):
            if (disc_dynamics.list_region is None):
                disc_dynamics.list_region = []
            if (disc_dynamics.num_regions != len(disc_dynamics.list_region)):
                printWarning('disc_dynamics.num_regions != ' + \
                                 'len(disc_dynamics.list_regions)', obj=self)
                disc_dynamics.num_regions = len(disc_dynamics.list_region)

        # Construct this object
        self.__env_vars = copy.deepcopy(env_vars)
        self.__sys_vars = copy.deepcopy(sys_disc_vars)
        self.__disc_props = copy.deepcopy(disc_props)
        self.__realizable = None
#         self.__jtlvfile = os.path.join(os.path.abspath(os.path.dirname(__file__)), \
#                                            'tmpspec', 'tmp')

        # Replace '...' in the range of possible values of env_vars to the actual  
        # values and convert a list representation of the range of possible values 
        # to a string
        if (env_vars is not None):
            for var, reg in env_vars.iteritems():
                if ('boolean' in reg):
                    continue
                elif (isinstance(reg, str)):
                    all_values = list(set(re.findall('[-+]?\d+', reg)))
                    if (len(all_values) > 0):
                        dots_values = re.findall('([-+]?\d+)\s*,?\s*' + r'\.\.\.' + \
                                                     '\s*,?\s*([-+]?\d+)', reg)
                        for dots_pair in dots_values:
                            for val in xrange(int(dots_pair[0])+1, int(dots_pair[1])):
                                if (str(val) not in all_values):
                                    all_values.append(str(val))
                        reg = ''
                        for val in all_values:
                            if (len(reg) > 0):
                                reg += ', '
                            reg += val
                        self.__env_vars[var] = '{' + reg + '}'
                    else:
                        printWarning("Unknown possible values for environment " + \
                                         "variable " + var, obj=self)
                elif (isinstance(reg, list)):
                    all_values = ''
                    for val in reg:
                        if (len(all_values) > 0):
                            all_values += ', '
                        all_values += str(val)
                    self.__env_vars[var] = '{' + all_values + '}'
                else:
                    printWarning("Unknown possible values for environment " + \
                                     "variable "+ var, obj=self)

        # Replace '...' in the range of possible values of sys_disc_vars to the actual 
        # values and convert a list representation of the range of possible values to a 
        # string
        if (sys_disc_vars is not None):
            for var, reg in sys_disc_vars.iteritems():
                if ('boolean' in reg):
                    continue
                elif (isinstance(reg, str)):
                    all_values = list(set(re.findall('[-+]?\d+', reg)))
                    if (len(all_values) > 0):
                        dots_values = re.findall('([-+]?\d+)\s*,?\s*' + r'\.\.\.' + \
                                                     '\s*,?\s*([-+]?\d+)', reg)
                        for dots_pair in dots_values:
                            for val in xrange(int(dots_pair[0])+1, int(dots_pair[1])):
                                if (str(val) not in all_values):
                                    all_values.append(str(val))
                        reg = ''
                        for val in all_values:
                            if (len(reg) > 0):
                                reg += ', '
                            reg += val
                        self.__sys_vars[var] = '{' + reg + '}'
                    else:
                        printWarning("Unknown possible values for discrete " + \
                                         "system variable " + var, obj=self)
                elif (isinstance(reg, list)):
                    all_values = ''
                    for val in reg:
                        if (len(all_values) > 0):
                            all_values += ', '
                        all_values += str(val)
                    self.__sys_vars[var] = '{' + all_values + '}'
                else:
                    printWarning("Unknown possible values for discrete system " + \
                                     "variable " + var, obj=self)

        # New variable that identifies in which cell the continuous state is
        self.__disc_cont_var = ''
        self.__disc_dynamics = None
        cont_varname = ''
        if (disc_dynamics is not None and disc_dynamics.num_regions > 0):
            cont_varname = 'cellID' 

            # Make sure that the new variable name does not appear in env_vars 
            # or sys_disc_vars
            while (cont_varname in env_vars) | (cont_varname in sys_disc_vars):
                cont_varname = 'c' + cont_varname

            contvar_values = '{'
            for i in xrange(0,disc_dynamics.num_regions):
                if (i > 0):
                    contvar_values += ', ' + str(i)
                else:
                    contvar_values += str(i)
            contvar_values += '}'
            self.__sys_vars[cont_varname] = contvar_values
            self.__disc_cont_var = cont_varname
            self.__disc_dynamics = copy.deepcopy(disc_dynamics)

        # Process the spec
        self.__spec = copy.deepcopy(spec)

        # Replace any cont_prop XC by (s.p = P1) | (s.p = P2) | ... | (s.p = Pn) where 
        # P1, ..., Pn are cells in disc_dynamics that satisfy XC
        if (disc_dynamics is not None and disc_dynamics.list_prop_symbol is not None):
            for propInd, propSymbol in enumerate(disc_dynamics.list_prop_symbol):
                reg = [j for j in range(0,disc_dynamics.num_regions) if \
                           disc_dynamics.list_region[j].list_prop[propInd]]
                newprop = 'FALSE'
                if (len(reg) > 0):
                    newprop = ''
                    for i, regID in enumerate(reg):
                        if (i > 0):
                            newprop = newprop + ' | '
                        newprop = newprop + '(' + cont_varname + ' = ' + \
                            str(regID) + ')'
                if (isinstance(self.__spec, GRSpec)):
                    self.__spec.sym2prop(props={propSymbol:newprop}, verbose=verbose)
                else:
                    self.__spec[0] = re.sub(r'\b'+propSymbol+r'\b', '('+newprop+')', \
                                                self.__spec[0])
                    self.__spec[1] = re.sub(r'\b'+propSymbol+r'\b', '('+newprop+')', \
                                                self.__spec[1])

        # Replace symbols for propositions on discrete variables with the actual 
        # propositions
        if (isinstance(self.__spec, GRSpec)):
            self.__spec.sym2prop(props=disc_props, verbose=verbose)
        else:
            if (disc_props is not None):
                symfound = True
                while (symfound):
                    symfound = False
                    for propSymbol, prop in disc_props.iteritems():
                        if (verbose > 2):
                            print '\t' + propSymbol + ' -> ' + prop
                        if (len(re.findall(r'\b'+propSymbol+r'\b', self.__spec[0])) > 0):
                            self.__spec[0] = re.sub(r'\b'+propSymbol+r'\b', '('+prop+')', self.__spec[0])
                            symfound = True
                        if (len(re.findall(r'\b'+propSymbol+r'\b', self.__spec[1])) > 0):
                            self.__spec[1] = re.sub(r'\b'+propSymbol+r'\b', '('+prop+')', self.__spec[1])
                            symfound = True
                        

#         # Transitions for continuous dynamics
#         addAnd = False
#         if (len(self.__spec.sys_safety) > 0 and not(self.__spec.sys_safety.isspace())):
#             addAnd = True
#
#         if (disc_dynamics is not None):
#             for from_region in xrange(0,disc_dynamics.num_regions):
#                 to_regions = [j for j in range(0,disc_dynamics.num_regions) if \
#                                   disc_dynamics.trans[j][from_region]]
#                 if (addAnd):
#                     self.__spec.sys_safety += ' &\n'
#                 if (from_region == 0):
#                     self.__spec.sys_safety += '-- transition relations for continuous dynamics\n'
#                 self.__spec.sys_safety += '\t((' + cont_varname + ' = ' + \
#                                   str(from_region) + ') -> next('
#                 if (len(to_regions) == 0):
#                     self.__spec.sys_safety += 'FALSE'
#                 for i, to_region in enumerate(to_regions):
#                     if (i > 0):
#                         self.__spec.sys_safety += ' | '
#                     self.__spec.sys_safety += '(' + cont_varname + ' = ' + str(to_region) + ')'
#                 self.__spec.sys_safety += '))'
#                 addAnd = True


    ###################################################################

    def checkRealizability(self, heap_size='-Xmx128m', pick_sys_init=True, verbose=0):
        """Determine whether this SynthesisProb is realizable without 
        extracting an automaton.

        Input:

        - `heap_size`: a string that specifies java heap size. 
        - `pick_sys_init` is a boolean indicating whether the system can pick 
          its initial state (in response to the initial environment state).
        - `verbose`: an integer that specifies the level of verbosity.
        """

        smv_file = self.__jtlvfile + '.smv'
        spc_file = self.__jtlvfile + '.spc'
        aut_file = self.__jtlvfile + '.aut'
        self.toJTLVInput(smv_file=smv_file, spc_file=spc_file, \
                          file_exist_option='r', verbose=verbose)
        init_option = 1
        if (not pick_sys_init):
            init_option = 0

            
        realizable = grgameint.solveGame(smv_file=smv_file, spc_file=spc_file, \
                                         aut_file=aut_file, \
                                         heap_size=heap_size, \
                                         priority_kind=-1, \
                                         init_option=init_option, \
                                         file_exist_option='r', verbose=verbose)
        self.__realizable = realizable
        return realizable

    ###################################################################

    def getCounterExamples(self, recompute=False, heap_size='-Xmx128m', \
                               pick_sys_init=True, verbose=0):
        """Return a list of dictionary representing a state starting from
        which the system cannot satisfy the spec.

        """
        if (recompute or self.__realizable is None or self.__realizable):
            if (verbose > 0):
                print "Checking realizability"
            realizable = self.checkRealizability(heap_size=heap_size, \
                                                     pick_sys_init=pick_sys_init, \
                                                     verbose = verbose)
            self.__realizable = realizable
        ce = []
        if (not self.__realizable):
            aut_file = self.__jtlvfile + '.aut'
            ce = grgameint.getCounterExamples(aut_file=aut_file, verbose=verbose)        
        return ce


    ###################################################################

    def synthesizePlannerAut(self, heap_size='-Xmx128m', priority_kind=3, init_option=1, \
                                 verbose=0):
        """Compute a planner automaton for this SynthesisProb. 
        If this SynthesisProb is realizable, this function returns an Automaton object.
        Otherwise, it returns a list of dictionary that represents the state
        starting from which there exists no strategy for the system to satisfy the spec.

        Input:

        - `heap_size`: a string that specifies java heap size. 
        - `priority_kind`: a string of length 3 or an integer that specifies the type of 
          priority used in extracting the automaton. See the documentation of the 
          ``computeStrategy`` function for the possible values of `priority_kind`.
        - `init_option`: an integer in that specifies how to handle the initial state of 
          the system. See the documentation of the ``computeStrategy`` function for the 
          possible values of `init_option`.
        - `verbose`: an integer that specifies the level of verbosity. If verbose is set to 0,
          this function will not print anything on the screen.
        """
        smv_file = self.__jtlvfile + '.smv'
        spc_file = self.__jtlvfile + '.spc'
        aut_file = self.__jtlvfile + '.aut'
        self.toJTLVInput(smv_file=smv_file, spc_file=spc_file, \
                          file_exist_option='r', verbose=verbose)
        realizable = grgameint.solveGame(smv_file=smv_file, spc_file=spc_file, \
                                         aut_file=aut_file, \
                                         heap_size=heap_size, \
                                         priority_kind=priority_kind, \
                                         init_option=init_option, \
                                         file_exist_option='r', verbose=verbose)
        self.__realizable = realizable
        if (not realizable):
            printError('spec not realizable', obj=self)
            counter_examples = grgameint.getCounterExamples(aut_file=aut_file, verbose=verbose)
            return counter_examples
        else:
            aut = automaton.Automaton(states_or_file=aut_file, varnames=[], verbose=verbose)
            return aut


    ###################################################################

    def toJTLVInput(self, smv_file='', spc_file='', file_exist_option='r', verbose=0):
        """Generate JTLV input files: smv_file and spc_file for this SynthesisProb.

        Input:

        - `smv_file`: a string that specifies the name of the resulting smv file.
        - `spc_file`: a string that specifies the name of the resulting spc file.
        - `file_exist_option`: a string that indicate what to do when the specified smv_file 
          or spc_file exists. Possible values are: 'a' (ask whether to replace or
          create a new file), 'r' (replace the existing file), 'n' (create a new file).
        - `verbose`: an integer that specifies the level of verbosity. If verbose is set to 0,
          this function will not print anything on the screen.
        """

        # Check that the input is valid
        if (not isinstance(smv_file, str)):
            printError("The input smv_file is expected to be a string", obj=self)
            raise TypeError("Invalid smv_file.")
        if (not isinstance(spc_file, str)):
            printError("The input spc_file is expected to be a string", obj=self)
            raise TypeError("Invalid spc_file.")
        
        if (len(smv_file) == 0):
            smv_file = self.__jtlvfile + '.smv'
        if (len(spc_file) == 0):
            spc_file = self.__jtlvfile + '.spc'

        if (not os.path.exists(os.path.abspath(os.path.dirname(smv_file)))):
            if (verbose > 0):
                printWarning('Folder for smv_file ' + smv_file + \
                                 ' does not exist. Creating...', obj=self)
            os.mkdir(os.path.abspath(os.path.dirname(smv_file)))
        if (not os.path.exists(os.path.abspath(os.path.dirname(spc_file)))):
            if (verbose > 0):
                printWarning('Folder for spc_file ' + spc_file + \
                                 ' does not exist. Creating...', obj=self)
            os.mkdir(os.path.abspath(os.path.dirname(spc_file)))

        # Check whether the smv or spc file exists
        if (file_exist_option != 'r'):
            if (os.path.exists(smv_file)):
                printWarning('smv file: ' + smv_file + ' exists.', obj=self)
                smv_file_exist_option = file_exist_option
                while (smv_file_exist_option.lower() != 'r' and \
                           smv_file_exist_option.lower() != 'n'):
                    smv_file_exist_option = raw_input('Replace [r] or create a new smv file [n]: ')
                if (smv_file_exist_option.lower() == 'n'):
                    i = 1
                    smv_file_part = smv_file.partition('.')
                    smv_file = smv_file_part[0] + str(i) + smv_file_part[1] + \
                        smv_file_part[2]
                    while (os.path.exists(smv_file)):
                        i = i + 1
                        smv_file = smv_file_part[0] + str(i) + smv_file_part[1] + \
                            smv_file_part[2]
                    print('smv file: ' + smv_file)

            if (os.path.exists(spc_file)):
                printWarning('spc file: ' + spc_file + ' exists.', obj=self)
                spc_file_exist_option = file_exist_option
                while (spc_file_exist_option.lower() != 'r' and \
                           spc_file_exist_option.lower() != 'n'):
                    spc_file_exist_option = raw_input('Replace [r] or create a new spc file [n]: ')
                if (spc_file_exist_option.lower() == 'n'):
                    i = 1
                    spc_file_part = spc_file.partition('.')
                    spc_file = spc_file_part[0] + str(i) + spc_file_part[1] + \
                        spc_file_part[2]
                    while (os.path.exists(spc_file)):
                        i = i + 1
                        spc_file = spc_file_part[0] + str(i) + spc_file_part[1] + \
                            spc_file_part[2]
                    print('spc file: ' + spc_file)


        ###################################################################################
        # Generate smv file
        ###################################################################################
        env_vars = self.__env_vars
        sys_vars = self.__sys_vars

        # Write smv file
        f = open(smv_file, 'w')
    #   The use of 'with' below is a better statement but is not supported in my 
    #   version of python
    #   with open(smv_file, 'w') as f: 
        if (verbose > 0):
            print 'Generating smv file...'
        f.write('MODULE main \n')
        f.write('\tVAR\n')
        f.write('\t\te : env();\n')
        f.write('\t\ts : sys();\n\n')

        # Environment
        f.write('MODULE env \n')
        f.write('\tVAR\n')
        for var, val in env_vars.iteritems():
            f.write('\t\t' + var + ' : ' + val + ';\n')

        # System
        f.write('\nMODULE sys \n')
        f.write('\tVAR\n')
        for var, val in sys_vars.iteritems():
            f.write('\t\t' + var + ' : ' + val + ';\n')

        f.close()

        ###################################################################################
        # Generate spc file
        ###################################################################################
        if (isinstance(self.__spec, GRSpec)):
            spec = self.__spec.toJTLVSpec()
        else:
            spec = self.__spec
        assumption = spec[0]
        guarantee = spec[1]

        assumption = re.sub(r'\b'+'True'+r'\b', 'TRUE', assumption)
        guarantee = re.sub(r'\b'+'True'+r'\b', 'TRUE', guarantee)
        assumption = re.sub(r'\b'+'False'+r'\b', 'FALSE', assumption)
        guarantee = re.sub(r'\b'+'False'+r'\b', 'FALSE', guarantee)

        if (verbose > 0):
            print 'Generating spc file...'

        # Replace any environment variable var in spec with e.var and replace any 
        # system variable var with s.var
        for var in env_vars.keys():
            assumption = re.sub(r'\b'+var+r'\b', 'e.'+var, assumption)
            guarantee = re.sub(r'\b'+var+r'\b', 'e.'+var, guarantee)
        for var in sys_vars.keys():
            assumption = re.sub(r'\b'+var+r'\b', 's.'+var, assumption)
            guarantee = re.sub(r'\b'+var+r'\b', 's.'+var, guarantee)

        # Add assumption on the possible initial state of the system and all the possible 
        # values of the environment
        env_values_formula = ''
        for var, reg in env_vars.iteritems():
            all_values = re.findall('[-+]?\d+', reg)
            if (len(all_values) > 0):
                if (len(env_values_formula) > 0):
                    env_values_formula += ' & '
                current_env_values_formula = ''
                for val in all_values:
                    if (len(current_env_values_formula) > 0):
                        current_env_values_formula += ' | '
                    current_env_values_formula += 'e.' + var + '=' + val
                env_values_formula += '(' + current_env_values_formula + ')'

        sys_values_formula = ''
        for var, reg in sys_vars.iteritems():
            all_values = re.findall('[-+]?\d+', reg)
            if (len(all_values) > 0):
                if (len(sys_values_formula) > 0):
                    sys_values_formula += ' & '
                current_sys_values_formula = ''
                for val in all_values:
                    if (len(current_sys_values_formula) > 0):
                        current_sys_values_formula += ' | '
                    current_sys_values_formula += 's.' + var + '=' + val
                sys_values_formula += '(' + current_sys_values_formula + ')'

        addAnd = False
        if (len(assumption) > 0 and not(assumption.isspace())):
            assumption = assumption 
            addAnd = True
        if (len(env_values_formula) > 0):
            if (addAnd):
                assumption = assumption + ' &\n'
            assumption = assumption + '-- all initial environment states\n'
            assumption = assumption + '\t(' + env_values_formula + ')'
            assumption = assumption + ' &\n-- possible values of environment variables\n'
            assumption = assumption + '\t[](next(' + env_values_formula + '))'

        f = open(spc_file, 'w')

        # Assumption
        f.write('LTLSPEC\n')
        f.write(assumption)
        f.write('\n;\n')

        # Guarantee
        f.write('\nLTLSPEC\n')
        addAnd = False
        if (len(guarantee) > 0 and not(guarantee.isspace())):
            f.write(guarantee)
            addAnd = True
        if (len(sys_values_formula) > 0):
            if (addAnd):
                f.write(' &\n')
            f.write('-- all initial system states\n')
            f.write('\t(' + sys_values_formula + ')')
            addAnd = True        

        # Transitions for continuous dynamics
        if (self.__disc_dynamics is not None):
            for from_region in xrange(0,self.__disc_dynamics.num_regions):
                to_regions = [j for j in range(0,self.__disc_dynamics.num_regions) if \
                                  self.__disc_dynamics.trans[j][from_region]]
                if (addAnd):
                    f.write(' &\n')
                if (from_region == 0):
                    f.write('-- transition relations for continuous dynamics\n')
                f.write('\t[]((s.' + self.__disc_cont_var + ' = ' + \
                            str(from_region) + ') -> next(')
                if (len(to_regions) == 0):
                    f.write('FALSE')
                for i, to_region in enumerate(to_regions):
                    if (i > 0):
                        f.write(' | ')
                    f.write('(s.' + self.__disc_cont_var + ' = ' + str(to_region) + ')')
                f.write('))')
                addAnd = True

        # Transitions
        # For discrete transitions
        if (len(sys_values_formula) > 0):
            if (addAnd):
                f.write(' &\n')
            f.write('-- possible values of discrete system variables\n');
            f.write('\t[](next(' + sys_values_formula + '))')

        f.write('\n;')
        f.close()



###################################################################

class ShortHorizonProb(SynthesisProb):
    """
    ShortHorizonProb class for specifying a short horizon problem for receding horizon 
    temporal logic planning.
    A ShortHorizonProb object contains the following fields:

    - `W`: a proposition that specifies a set W of states.
    - `FW`: a ShortHorizonProb object or a list of ShortHorizonProb object that specifies 
      the set F(W).
    - `Phi`: a proposition that specifies the receding horizon invariant.
    - `global_prob`: a SynthesisProb object that represents the global problem.

    **Constructor**:

    **ShortHorizonProb** ([ `W` = ''[, `FW` = [][, `Phi` = ''[, `global_prob` = SynthesisProb()[, 
    `file` = '']]]]])

    **ShortHorizonProb** ([ `W` = ''[, `FW` = [][, `Phi` = ''[, `global_prob` = SynthesisProb()[, 
    `env_vars` = {}[, `sys_disc_vars` = {}[, `disc_props` = {}[, `disc_dynamics` = None]]]]]]]])

    **ShortHorizonProb** ([ `W` = ''[, `FW` = [][, `Phi` = ''[, `global_prob` = SynthesisProb()[, 
    `env_vars` = {}[, `sys_disc_vars` = {}[, `disc_props` = {}[, `cont_state_space` = None[, 
    `cont_props` = {}[, `sys_dyn` = None]]]]]]]]]])

    - `W`: a proposition that specifies a set W of states.
    - `FW`: a ShortHorizonProb object or a list of ShortHorizonProb object that specifies 
      the set F(W).
    - `Phi`: a proposition that specifies the receding horizon invariant.
    - `global_spec`: the global specification of the system.
    - `file`: the name of the rhtlp file to be parsed. If `file` is given,
      the rest of the inputs to this function will be ignored.
    - `env_vars`: a dictionary {str : str} or {str : list} whose keys are the names 
      of environment variables and whose values are their possible values, e.g., 
      boolean or {0, 2, ..., 5} or [0, 2, 3, 4, 5].
    - `sys_disc_vars`: a dictionary {str : str} or {str : list} whose keys are the 
      names of discrete system variables and whose values are their possible values.
    - `disc_props`: a dictionary {str : str} whose keys are the symbols for 
      propositions on discrete variables and whose values are the actual propositions
      on discrete variables.
    - `disc_dynamics`: a PropPreservingPartition object that represents the 
      transition system obtained from the discretization procedure.
      if `disc_dynamics` is given, `cont_state_space`, `cont_props` and `sys_dyn`
      will be ignored.
    - `cont_state_space`: a Polytope object that represent the state space of the
      continuous variables. Needed only when `discretize` is True.
    - `cont_props`: a dictionary {str : Polytope} whose keys are the symbols for 
      propositions on continuous variables and whose values are polytopes that represent
      the region in the state space in which the corresponding proposition hold.
      if `discretize` is False, `cont_props` can be just a list of symbols for 
      propositions on continuous variables.
    - `sys_dyn`: a CtsSysDyn object that specifies the dynamics of the continuous variables.
      Needed only when `discretize` is True.
    - `verbose`: an integer that specifies the level of verbosity.
    """
    def __init__(self, W='', FW=[], Phi='', global_prob=SynthesisProb(), **args):
        self.__W = ''
        self.__FW = []
        self.__Phi = ''
        self.__global_prob = SynthesisProb()

        self.setW(W=W, update=False, verbose=0)
        self.setFW(FW=FW, update=False, verbose=0)
        self.setLocalPhi(Phi=Phi, update=False, verbose=0)
        
        verbose = args.get('verbose', 0)

        if (not isinstance(global_prob, SynthesisProb)):
            printError("The input global_prob must be a SynthesisProb objects.", obj=self)
            raise TypeError("Invalid global_prob.")
        self.__global_prob = global_prob
        self.__global_spec = global_prob.getSpec()
        self.__global_spec.sym2prop(global_prob.getDiscProps(), verbose=verbose)
        args['spec'] = self.__computeLocalSpec()

        # For variables that are in the global problem but not in the local problem,
        # make its possible value being False
        global_env_vars = global_prob.getEnvVars()
        global_sys_disc_vars = global_prob.getSysVars()
        env_vars = args.get('env_vars', {})
        sys_disc_vars = args.get('sys_disc_vars', {})
        disc_props = args.get('disc_props', {})
        for var in global_env_vars.keys():
            if (not var in env_vars.keys()):
                disc_props[var] = 'False'
        for var in global_sys_disc_vars.keys():
            if (not var in sys_disc_vars.keys()):
                disc_props[var] = 'False'
        args['disc_props'] = disc_props

        global_disc_props = global_prob.getDiscProps()
        for prop in global_disc_props.keys():
            if (not prop in disc_props.keys()):
                disc_props[prop] = global_disc_props[prop]
        
        SynthesisProb.__init__(self, **args)

    def getGlobalSpec(self):
        """ Return the global specification of this short horizon problem.
        """
        return copy.deepcopy(self.__global_spec)

    def setW(self, W, update=True, verbose=0):
        """ Set the set W of this short horizon problem.
        """
        if (not isinstance(W, str)):
            printError("The input W must be a string.", obj=self)
            raise TypeError("Invalid W.")
        self.__W = W
        if (update):
            self.updateLocalSpec(verbose=verbose)

    def getW(self):
        """ Return the set W of this short horizon problem.
        """
        return self.__W

    def setFW(self, FW, update=True, verbose=0):
        """ Set the set F(W) of this short horizon problem.
        """
        if (FW is None):
            FW = []
        if (not isinstance(FW, list) and not isinstance(FW, ShortHorizonProb)):
            printError("The input FW must be a ShortHorizonProb object or " + \
                           "a list of ShortHorizonProb object.", obj=self)
            raise TypeError("Invalid W.")
        if (isinstance(FW, list)):
            self.__FW = copy.copy(FW)
        else:
            self.__FW = FW
        if (update):
            self.updateLocalSpec(verbose=verbose)

    def getFW(self):
        """ Return the set F(W) of this short horizon problem.
        """
        return self.__FW

    def setLocalPhi(self, Phi, update=True, verbose=0):
        """ Set the local invariant Phi of this short horizon problem.
        """
        if (not isinstance(Phi, str)):
            printError("The input Phi must be a string.", obj=self)
            raise TypeError("Invalid Phi.")
        self.__Phi = Phi
        if (update):
            self.updateLocalSpec(verbose=verbose)

    def getLocalPhi(self, allow_disc_cont_var=True):
        """ Return the local invariant Phi of this short horizon problem.
        """
        disc_dynamics = self.getDiscretizedDynamics()
        if (allow_disc_cont_var or self.getDiscretizedContVar() is None or \
                len(self.getDiscretizedContVar()) == 0 or \
                self.getDiscretizedContVar() not in self.getSysVars() or \
                disc_dynamics is None or \
                disc_dynamics.list_prop_symbol is None):
            return self.__Phi

        # Replace disc_cont_var in Phi with cont_prop
        ret = self.__Phi
        cont_props = disc_dynamics.list_prop_symbol
        for val in xrange(0,disc_dynamics.num_regions):
            prop_id = disc_dynamics.list_region[val].list_prop
            prop_str = ''
            for ind in xrange(0, len(prop_id)):
                if (len(prop_str) > 0):
                    prop_str += ' & '
                if (not prop_id[ind]):
                    prop_str += ' !'
                prop_str += cont_props[ind]
            ret = re.sub(r'\b'+self.getDiscretizedContVar()+r'\b'+'\s*=\s*'+str(val), \
                             '('+prop_str+')', ret)
        return ret

    def updateLocalSpec(self, verbose=0):
        """
        Update the short horizon specification based on the current W, FW and Phi.
        """
        local_spec = self.__computeLocalSpec()
        sys_disc_vars = self.getSysDiscVars()
            
        self.createProbFromDiscDynamics(env_vars=self.getEnvVars(), sys_disc_vars=sys_disc_vars, \
                                            disc_props=self.getDiscProps(), \
                                            disc_dynamics=self.getDiscretizedDynamics(), \
                                            spec=local_spec, verbose=verbose)

    def __computeLocalSpec(self, verbose=0):
        local_spec = copy.deepcopy(self.__global_spec)
        local_spec.sys_init = ''
        local_spec.env_init = ''
        if (len(self.__W) > 0):
            if (len(local_spec.env_init) > 0):
                local_spec.env_init += ' & \n'
            local_spec.env_init += '-- W\n'
            local_spec.env_init += '\t(' + self.__W + ')'

        if (len(self.__Phi) > 0):
            if (len(local_spec.env_init) > 0):
                local_spec.env_init += ' & \n'
            local_spec.env_init += '-- Phi\n'
            local_spec.env_init += '\t(' + self.__Phi + ')'
            if (len(local_spec.sys_safety) > 0):
                local_spec.sys_safety += ' & \n'
            local_spec.sys_safety += '-- Phi\n'
            local_spec.sys_safety += '\t(' + self.__Phi + ')'

        local_spec.sys_prog = []
        if (self.__FW is not None and isinstance(self.__FW, list) and len(self.__FW) > 0):
            for fw in self.__FW:
                if (len(fw.getW()) == 0 or fw.getW().isspace()):
                    continue
                if (len(local_spec.sys_prog) == 0):
                    local_spec.sys_prog += [fw.getW()]
                else:
                    if (len(local_spec.sys_prog[0]) > 0):
                        local_spec.sys_prog[0] += ' & '
                    local_spec.sys_prog[0] += fw.getW()
        elif (isinstance(self.__FW, ShortHorizonProb)):
            if(len(self.__FW.getW()) > 0 and not self.__FW.getW().isspace()):
                if (len(local_spec.sys_prog) == 0):
                    local_spec.sys_prog += [self.__FW.getW()]
                else:
                    if (len(local_spec.sys_prog[0]) > 0):
                        local_spec.sys_prog[0] += ' & '
                    local_spec.sys_prog[0] += self.__FW.getW()
                
        
#         local_spec.sym2prop(self.__disc_props, verbose=verbose)
        return local_spec


    def computeLocalPhi(self, verbose=0):
        """
        Compute the local Phi for this ShortHorizonProb object.
        Return a boolean that indicates whether the local Phi gets updated.
        If the current prob is realizable, then the local Phi is not updated and this
        function will return False.
        Otherwise, the local Phi will get updated and this function wil return True.
        """
        self.updateLocalSpec(verbose=verbose)
        counter_examples = self.getCounterExamples(recompute=True, pick_sys_init=False, \
                                                       verbose=verbose)
        if (len(counter_examples) == 0):
            return False
        else:
            if (len(self.__Phi) > 0 and not self.__Phi.isspace()):
                self.__Phi = '(' + self.__Phi + ')'
            for ce in counter_examples:
                ce_formula = ''
                for var, val in ce.iteritems():
                    if (len(ce_formula) > 0):
                        ce_formula += ' & '
                    ce_formula += var + ' = ' + str(val)
                if (len(self.__Phi) > 0):
                    self.__Phi += ' & '
                self.__Phi += '!(' + ce_formula + ')'
            return True



###################################################################


class RHTLPProb(SynthesisProb):
    """
    RHTLPProb class for specifying a receding horizon temporal logic planning problem.
    A RHTLPProb object contains the following fields:

    - `shprobs`: a list of ShortHorizonProb objects
    - `Phi`: the invariant for the RHTLP problem

    **Constructor**:

    **RHTLPProb** ([ `shprobs` = [][, `Phi` = 'True'[, `discretize` = False[, `file` = '']]]]): 
    construct this SynthesisProb object from `file`.

    **RHTLPProb** ([ `shprobs` = [][, `Phi` = 'True'[, `discretize` = False[, `env_vars` = {}[, `
    sys_disc_vars` = {}[, `disc_props` = {}[, `disc_dynamics` = None[, 
    `spec` = GRSpec()]]]]]]]])

    **RHTLPProb** ([ `shprobs` = [][, `Phi` = 'True'[, `discretize` = False[, `env_vars` = {}[, 
    `sys_disc_vars` = {}[, `disc_props` = {}[, `cont_state_space` = None[, 
    `cont_props` = {}[, `sys_dyn` = None[, `spec` = GRSpec()]]]]]]])

    - `shprobs`: a list of ShortHorizonProb objects.
    - `Phi`: a string specifying the invariant for the RHTLP problem.
    - `discretize`:  a boolean indicating whether to discretize the global problem.
    - `file`: the name of the rhtlp file to be parsed. If `file` is given,
      the rest of the inputs to this function will be ignored.
    - `env_vars`: a dictionary {str : str} or {str : list} whose keys are the names 
      of environment variables and whose values are their possible values, e.g., 
      boolean or {0, 2, ..., 5} or [0, 2, 3, 4, 5].
    - `sys_disc_vars`: a dictionary {str : str} or {str : list} whose keys are the 
      names of discrete system variables and whose values are their possible values.
    - `disc_props`: a dictionary {str : str} whose keys are the symbols for 
      propositions on discrete variables and whose values are the actual propositions
      on discrete variables.
    - `disc_dynamics`: a PropPreservingPartition object that represents the 
      transition system obtained from the discretization procedure.
      if `disc_dynamics` is given, `cont_state_space`, `cont_props` and `sys_dyn`
      will be ignored.
    - `cont_state_space`: a Polytope object that represent the state space of the
      continuous variables. Needed only when `discretize` is True.
    - `cont_props`: a dictionary {str : Polytope} whose keys are the symbols for 
      propositions on continuous variables and whose values are polytopes that represent
      the region in the state space in which the corresponding proposition hold.
      if `discretize` is False, `cont_props` can be just a list of symbols for 
      propositions on continuous variables.
    - `sys_dyn`: a CtsSysDyn object that specifies the dynamics of the continuous variables.
      Needed only when `discretize` is True.
    - `spec`: a GRSpec object that specifies the specification of this synthesis problem
    - `verbose`: an integer that specifies the level of verbosity.
    """
    def __init__(self, shprobs=[], Phi='True', discretize=False, **args):
        self.shprobs = []
        self.__Phi = 'True'
        self.__disc_props = {}
        self.__cont_props = []
        self.__sys_prog = 'True'
        self.__all_init = 'True'
        self.setJTLVFile(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), \
                                           'tmpspec', 'tmp'))
#        self.setJTLVFile(os.path.join(os.path.abspath(os.path.dirname(__file__)), \
#                                           'tmpspec', 'tmp'))

        if (isinstance(shprobs, list)):
            for shprob in shprobs:
                if (isinstance(shprob, ShortHorizonProb)):
                    self.shprobs.append(shprob)
                else:
                    printError("The input shprobs must be " + \
                                   "a list of ShortHorizonProb objects.", obj=self)
        elif (shprobs is not None):
            printError("The input shprobs must be " + \
                           "a list of ShortHorizonProb objects.", obj=self)

        if (isinstance(Phi, str)):
            self.__Phi = Phi
        else:
            printError("The input Phi must be a string.", obj=self)
        
        
        verbose = args.get('verbose', 0)
        if ('disc_props' in args.keys()):
            self.__disc_props = copy.deepcopy(args['disc_props'])

        cont_props = args.get('cont_props', {})
        if (isinstance(cont_props, dict)):
            self.__cont_props = copy.deepcopy(cont_props.keys())
        elif (isinstance(cont_props, list)):
            self.__cont_props = copy.deepcopy(cont_props)
        else:
            printError("The input cont_props is expected to be a dictionary " + \
                           "{str : Polytope}", obj=self)
            raise TypeError("Invalid cont_props.")
                
        spec = GRSpec(env_init='', sys_init='', env_safety='', sys_safety='', \
                          env_prog='', sys_prog='')
        if ('spec' in args.keys()):
            if (not isinstance(args['spec'], GRSpec)):
                printError("The input spec must be a GRSpec objects.", obj=self)
                raise TypeError("Invalid spec.")
            spec = copy.deepcopy(args['spec'])
            if (isinstance(spec.sys_prog, list) and len(spec.sys_prog) > 1):
                printError("The input spec can have at most one system progress formula.", \
                               obj=self)
                raise TypeError("Invalid spec.")
            if (isinstance(spec.sys_prog, str)):
                self.__sys_prog = spec.sys_prog
            elif (isinstance(spec.sys_prog, list) and len(spec.sys_prog) == 1 and \
                      len(spec.sys_prog[0]) > 0 and not spec.sys_prog[0].isspace()):
                self.__sys_prog = spec.sys_prog[0]
            if (isinstance(spec.sys_init, str) and len(spec.sys_init) > 0):
                self.__all_init += ' & ' + spec.sys_init
            elif (isinstance(spec.sys_init, list)):
                for init_cond in spec.sys_init:
                    if (len(init_cond) > 0):
                        self.__all_init += ' & ' + init_cond
            if (isinstance(spec.env_init, str) and len(spec.env_init) > 0):
                self.__all_init += ' & ' + spec.env_init
            elif (isinstance(spec.env_init, list)):
                for init_cond in spec.env_init:
                    if (len(init_cond) > 0):
                        self.__all_init += ' & ' + init_cond

        env_vars = args.get('env_vars', {})
        sys_disc_vars = args.get('sys_disc_vars', {})
        cont_state_space = None
        sys_dyn = None
        disc_dynamics = None

        if ('file' in args.keys()):
            file = args['file']
            # Check that the input is valid
            if (not isinstance(file, str)):
                printError("The input file is expected to be a string", obj=self)
                raise TypeError("Invalid file.")
            if (not os.path.isfile(file)):
                printError("The rhtlp file " + file + " does not exist.", obj=self)
                raise TypeError("Invalid file.")

            (env_vars, sys_disc_vars, self.__disc_props, sys_cont_vars, cont_state_space, \
                 cont_props, sys_dyn, spec) = parseSpec(spec_file=file)
            self.__cont_props = cont_props.keys()

        elif ('disc_dynamics' in args.keys()):
            if (discretize):
                printWarning('Discretized dynamics is already given.', obj=self)
            discretize = False
            disc_dynamics = args['disc_dynamics']
            if (len(self.__cont_props) == 0):
                if (disc_dynamics is not None and disc_dynamics.list_prop_symbol is not None):
                    self.__cont_props = copy.deepcopy(disc_dynamics.list_prop_symbol)
            else:
                if (disc_dynamics is None or disc_dynamics.list_prop_symbol is None or \
                        not (set(self.__cont_props) == set(disc_dynamics.list_prop_symbol))):
                    printWarning("The given cont_prop does not match the propositions" + \
                                     " in the given disc_dynamics", obj=self)
        else:        
            cont_state_space = args.get('cont_state_space', None)
            sys_dyn = args.get('sys_dyn', None)

        if (discretize):
            self.createProbFromContDynamics(env_vars=env_vars, \
                                                sys_disc_vars=sys_disc_vars, \
                                                disc_props=self.__disc_props, \
                                                cont_state_space=cont_state_space, \
                                                cont_props=cont_props, \
                                                sys_dyn=sys_dyn, \
                                                spec=spec, \
                                                verbose=verbose)
        else:
            self.createProbFromDiscDynamics(env_vars=env_vars, \
                                                sys_disc_vars=sys_disc_vars, \
                                                disc_props=self.__disc_props, \
                                                disc_dynamics=disc_dynamics, \
                                                spec=spec, \
                                                verbose=verbose)


    def addSHProb(self, shprob):
        """ Add a short horizon problem to this RHTLP problem.
        """
        if (isinstance(shprob, ShortHorizonProb)):
            self.shprobs.append(shprob)
        else:
            printError("The input shprob must be a ShortHorizonProb object.", obj=self)


    def getPhi(self):
        """ Return the global invariant Phi for this RHTLP problem.
        """
        return self.__Phi

    def __replacePropSymbols(self, formula = '', verbose=0):
        newformula = copy.deepcopy(formula)
        # disc_prop
        if (self.__disc_props is not None):
            symfound = True
            while (symfound):
                symfound = False
                for propSymbol, prop in self.__disc_props.iteritems():
                    if (verbose > 2):
                        print '\t' + propSymbol + ' -> ' + prop
                    if (isinstance(newformula, list)):
                        for ind in xrange(0, len(newformula)):
                            if (len(re.findall(r'\b'+propSymbol+r'\b', newformula[ind])) > 0):
                                newformula[ind] = re.sub(r'\b'+propSymbol+r'\b', '('+prop+')', \
                                                             newformula[ind])
                                symfound = True
                    elif (isinstance(newformula, str)):
                        if (len(re.findall(r'\b'+propSymbol+r'\b', newformula)) > 0):
                            newformula = re.sub(r'\b'+propSymbol+r'\b', '('+prop+')', newformula)
                            symfound = True

        # Replace any cont_prop XC by (s.p = P1) | (s.p = P2) | ... | (s.p = Pn) where 
        # P1, ..., Pn are cells in disc_dynamics that satisfy XC
        if (self.getDiscretizedContVar() is not None and \
                len(self.getDiscretizedContVar()) > 0 and \
                self.getDiscretizedContVar() in self.getSysVars() and \
                self.getDiscretizedDynamics() is not None and \
                self.getDiscretizedDynamics().list_prop_symbol is not None):
            disc_dynamics = self.getDiscretizedDynamics()
            for propInd, propSymbol in enumerate(disc_dynamics.list_prop_symbol):
                reg = [j for j in range(0,disc_dynamics.num_regions) if \
                           disc_dynamics.list_region[j].list_prop[propInd]]
                prop = 'FALSE'
                if (len(reg) > 0):
                    prop = ''
                    for i, regID in enumerate(reg):
                        if (i > 0):
                            prop += ' | '
                        prop += '(' + self.getDiscretizedContVar() + ' = ' + \
                            str(regID) + ')'
                if (verbose > 1):
                    print '\t' + propSymbol + ' -> ' + prop
                if (isinstance(newformula, list)):
                    for ind in xrange(0, len(newformula)):
                        newformula[ind] = re.sub(r'\b'+propSymbol+r'\b', '('+prop+')', \
                                                     newformula[ind])
                elif (isinstance(newformula, str)):
                    newformula = re.sub(r'\b'+propSymbol+r'\b', '('+prop+')', newformula)
        return newformula

    def __getAllVarsRaw(self, verbose=0):
        allvars = self.getEnvVars()
        sys_disc_vars = self.getSysVars()
        allvars.update(sys_disc_vars)
        if ((self.getDiscretizedContVar() is None or \
                len(self.getDiscretizedContVar()) == 0 or \
                not self.getDiscretizedContVar() in sys_disc_vars) and \
                self.__cont_props is not None):
            for var in self.__cont_props:
                allvars[var] = 'boolean'
        return allvars


    def __getAllVars(self, verbose=0): 
#         allvars = self.getEnvVars()
#         sys_disc_vars = self.getSysVars()
#         allvars.update(sys_disc_vars)
#         if ((self.getDiscretizedContVar() is None or \
#                 len(self.getDiscretizedContVar()) == 0 or \
#                 not self.getDiscretizedContVar() in sys_disc_vars) and \
#                 self.__cont_props is not None):
#             for var in self.__cont_props:
#                 allvars[var] = 'boolean'
        allvars = self.__getAllVarsRaw()
        allvars_values = ()
        allvars_variables = []
        for var, val in allvars.iteritems():
            tmp = [0,1]
            if (not 'boolean' in val):
                tmp = re.findall('[-+]?\d+', val)
                tmp = [int(i) for i in tmp]
            allvars_values += tmp,
            allvars_variables.append(var)

        return allvars_variables, allvars_values


    def checkcovering(self, excluded_state=[], verbose=0):
        """
        Check whether the disjunction of all the W's covers the entire state space.
        """
        allW_formula = 'False'
        for shprob in self.shprobs:
            allW_formula += ' | (' + shprob.getW() + ')'

        if (verbose > 0):
            print 'W = ' + allW_formula

        es_formula = 'False'
        if (isinstance(excluded_state, list)):
            for es in excluded_state:
                curr_es_formula = ''
                for var, val in es.iteritems():
                    if (len(curr_es_formula) > 0):
                        curr_es_formula += ' & '
                    curr_es_formula += var + ' = ' + str(val)
                es_formula += ' | (' + curr_es_formula + ')'
        elif (isinstance(excluded_state, dict)):
            curr_es_formula = ''
            for var, val in excluded_state.iteritems():
                if (len(curr_es_formula) > 0):
                    curr_es_formula += ' & '
                curr_es_formula += var + ' = ' + str(val)
            es_formula += ' | (' + curr_es_formula + ')'
        else:
            printError('excluded_state has to be a list or a dictionary.')
            raise TypeError("Invalid excluded_state.")
            
        use_yices = True
        try:
            if (verbose > 0):
                print("Trying yices")
            allvars = self.__getAllVarsRaw(verbose=verbose)
            expr = '!(' + allW_formula + ') & !(' + es_formula + ')'
            ysfile = os.path.dirname(self.getJTLVFile())
            ysfile = os.path.join(ysfile, 'tmp.ys')
            ret = rhtlputil.yicesSolveSat(expr=expr, allvars=allvars, ysfile=ysfile, \
                                              verbose=verbose)
            if (ret is None):
                use_yices = False
            elif (ret[0]):
                return ret[1]
            else:
                return True
        except:
            printError("yices failed!")
            print sys.exc_info()[0], sys.exc_info()[1]
            use_yices = False
        
        if (not use_yices):
            print("yices failed. Enumerating states.")

            (allvars_variables, allvars_values) = self.__getAllVars(verbose=verbose)

            allvars_values_iter = rhtlputil.product(*allvars_values)
            vardict = {}
            for val in allvars_values_iter:
                vardict = dict(zip(allvars_variables, val))
                try:
                    ret = rhtlputil.evalExpr(allW_formula, vardict, verbose)
                except:
                    printError('Invalid W', obj=self)
                    print sys.exc_info()[0], sys.exc_info()[1]
                    return vardict
                if (not ret):
    #                 state = dict(zip(allvars_variables, val))
    #                 state = ''
    #                 for i in xrange(0, len(allvars_variables)):
    #                     if (len(state) > 0):
    #                         state += ', '
    #                     state += allvars_variables[i] + ':' + str(val[i])
    #                 printError('state <' + state + '> is not in any W', obj=self)
                    return vardict
            return True


    def constructWGraph(self, verbose=0):
        """
        Construct the graph for W's. There is an edge in this graph from Wi to Wj
        if F(Wi) = Wj.
        """
        graph = []
        for wind, shprob in enumerate(self.shprobs):
            if (isinstance(shprob.getFW(), list)):
                fw_ind = []
                for fw in shprob.getFW():
                    if (isinstance(fw, int)):
                        fw_ind.append(fw)
                    elif (isinstance(fw, ShortHorizonProb)):
                        tmpind = self.__findWInd(fw, verbose=verbose)
                        if (tmpind >= 0):
                            fw_ind.append(tmpind)
                        else:
                            printError("FW for shprobs[" + str(wind) + "]" + \
                                           " is not in this RHTLPProb.", obj=self)
                            raise Exception("Invalid FW.")
                    else:
                        printError("Invalid FW", obj=self)
                        raise TypeError("Invalid FW.")
                graph.append(fw_ind)
            elif (isinstance(shprob.getFW(), int)):
                graph.append([shprob.getFW()])
            elif (isinstance(shprob.getFW(), ShortHorizonProb)):
                fw_ind = self.__findWInd(shprob.getFW(), verbose=verbose)
                if (fw_ind >= 0):
                    graph.append([fw_ind])
                else:
                    printError("FW for shprobs[" + str(wind) + "]" + \
                                   " is not in this RHTLPProb.", obj=self)
                    raise Exception("Invalid FW.")
            else:
                printError("Invalid FW for shprobs[" + str(wind) + "].", obj=self)
                raise TypeError("Invalid FW.")
        return graph


    def __findWInd(self, W, verbose=0):
        ind = 0
        while (ind < len(self.shprobs)):
            if (W == self.shprobs[ind]):
                return ind
            ind += 1
        return -1


    def findW0Ind(self, verbose=0):
        """
        Find the indices of W0 in shprobs
        """
        W0ind = range(0, len(self.shprobs))

        if (self.__sys_prog == True):
            return W0ind

        use_yices = True
        try:
            if (verbose > 0):
                print("Trying yices")
            allvars = self.__getAllVarsRaw(verbose=verbose)
            ysfile = os.path.dirname(self.getJTLVFile())
            ysfile = os.path.join(ysfile, 'tmp.ys')
            newW0ind = []
            for ind in W0ind:
                expr = '!((' + self.shprobs[ind].getW() + ') -> (' + self.__sys_prog + '))'
                ret = rhtlputil.yicesSolveSat(expr=expr, allvars=allvars, ysfile=ysfile, \
                                                  verbose=verbose)
                if (ret is None):
                    use_yices = False
                    break
                elif (not ret[0]):
                    newW0ind.append(ind)
                elif (verbose > 0):
                    print 'W[' + str(ind) + '] does not satisfy spec.sys_prog'
                    print 'counter example: \n' + ret[1]
            if (use_yices):
                W0ind = newW0ind
        except:
            printError("yices failed!")
            print sys.exc_info()[0], sys.exc_info()[1]
            use_yices = False
        
        if (not use_yices):
            print("yices failed. Enumerating states.")

            (allvars_variables, allvars_values) = self.__getAllVars(verbose=verbose)

            allvars_values_iter = rhtlputil.product(*allvars_values)
            vardict = {}
            for val in allvars_values_iter:
                vardict = dict(zip(allvars_variables, val))
                try:
                    ret = rhtlputil.evalExpr(self.__sys_prog, vardict, verbose)
                except:
                    printError('Invalid W', obj=self)
                    print sys.exc_info()[0], sys.exc_info()[1]
                    print vardict
                    ret = False
                if (ret):
                    newW0ind = []
                    for ind in W0ind:
                        ret = rhtlputil.evalExpr(self.shprobs[ind].getW(), vardict, verbose)
                        if (ret):
                            newW0ind.append(ind)
                        elif (verbose > 0):
                            print 'W[' + str(ind) + '] does not satisfy spec.sys_prog'
                            print 'counter example: ', vardict
                    W0ind = newW0ind
                    if (len(W0ind) == 0):
                        return W0ind

        return W0ind


    def checkTautologyPhi(self, verbose=0):
        """
        Check whether sys_init -> Phi is a tautology.
        """
        self.__all_init = self.__replacePropSymbols(formula = self.__all_init, verbose=verbose)
        self.__Phi = self.__replacePropSymbols(formula = self.__Phi, verbose=verbose)

        # Check whether self.__all_init -> Phi is a tautology
        use_yices = True
        try:
            if (verbose > 0):
                print("Trying yices")
            allvars = self.__getAllVarsRaw(verbose=verbose)
            expr = '!((' + self.__all_init + ') -> (' + self.__Phi + '))'
            ysfile = os.path.dirname(self.getJTLVFile())
            ysfile = os.path.join(ysfile, 'tmp.ys')
            ret = rhtlputil.yicesSolveSat(expr=expr, allvars=allvars, ysfile=ysfile, \
                                              verbose=verbose)
            if (ret is None):
                use_yices = False
            elif (not ret[0]):
                return True
            elif (verbose > 0):
                printInfo('sys_init -> Phi is not a tautology')
                print 'counter example: \n' + ret[1]
        except:
            printError("yices failed!")
            print sys.exc_info()[0], sys.exc_info()[1]
            use_yices = False
        
        if (not use_yices):
            print("yices failed. Enumerating states.")

            (allvars_variables, allvars_values) = self.__getAllVars(verbose=verbose)
            allvars_values_iter = rhtlputil.product(*allvars_values)
            vardict = {}
            for val in allvars_values_iter:
                vardict = dict(zip(allvars_variables, val))
                try:
                    ret = rhtlputil.evalExpr(self.__all_init, vardict, verbose)
                except:
                    printError('Invalid initial condition', obj=self)
                    print sys.exc_info()[0], sys.exc_info()[1]
                    return False
                if (ret):
                    try:
                        ret = rhtlputil.evalExpr(self.__Phi, vardict, verbose)
                    except:
                        printError('Invalid Phi', obj=self)
                        print sys.exc_info()[0], sys.exc_info()[1]
                        return False
                    if (not ret):
                        if (verbose > 0):
                            printInfo('sys_init -> Phi is not a tautology')
                            print 'counter example: ', vardict
                        return False
            return True

    def updatePhi(self, verbose=0):
        """
        Update Phi for this RHTLPProb object based on the local Phi
        in the short horizon problems.
        """
        if (len(self.__Phi) == 0 or self.__Phi.isspace()):
            self.__Phi = 'True'

        allow_disc_cont_var = True
        if (self.getDiscretizedContVar() is None or \
                len(self.getDiscretizedContVar()) == 0 or \
                not self.getDiscretizedContVar() in self.getSysVars() or \
                self.getDiscretizedDynamics() is None or \
                self.getDiscretizedDynamics().list_prop_symbol is None):
            allow_disc_cont_var = False

        for shprob in self.shprobs:
            localPhi = shprob.getLocalPhi(allow_disc_cont_var=allow_disc_cont_var)
            if (len(localPhi) > 0 and not localPhi.isspace()):
                self.__Phi += ' & (' + localPhi + ')'
        
        self.__Phi = self.__replacePropSymbols(formula = self.__Phi, verbose=verbose)


    def computePhi(self, checktautology=True, verbose=0):
        """
        Compute Phi for this RHTLPProb object.
        Return a boolean that indicates whether a valid Phi exists.
        """
        self.updatePhi(verbose = verbose)
        allow_disc_cont_var = True
        if (self.getDiscretizedContVar() is None or \
                len(self.getDiscretizedContVar()) == 0 or \
                not self.getDiscretizedContVar() in self.getSysVars() or \
                self.getDiscretizedDynamics() is None or \
                self.getDiscretizedDynamics().list_prop_symbol is None):
            allow_disc_cont_var = False

        done = False
        while (not done):
            done = True
            for i, shprob in enumerate(self.shprobs):
                if (checktautology):
                    if (verbose > 1):
                        print "Checking tautology of init -> Phi for shprob[" + str(i) + '].'
                    tautology = self.checkTautologyPhi(verbose = verbose)
                    if (not tautology):
                        return False

                if (verbose > 1):
                    print "Setting local Phi of shprob[" + str(i) + '].'
                shprob.setLocalPhi(Phi=self.__Phi, update=True, verbose=verbose)
                if (verbose > 1):
                    print "Computing local Phi of shprob[" + str(i) + '].'
                phi_updated = shprob.computeLocalPhi(verbose=verbose)
                if (phi_updated):
                    if (verbose > 0):
                        print "Updating Phi."
                    done = False
                    self.__Phi = shprob.getLocalPhi(allow_disc_cont_var=allow_disc_cont_var)
        return True


    def validate(self, checkcovering=True, excluded_state=[],
                 checkpartial_order=True, checktautology=True,
                 checkrealizable=True,
                 heap_size='-Xmx128m', verbose=0):
        """Check whether the list of ShortHorizonProb objects satisfies the sufficient
        conditions for receding horizon temporal logic planning.
        """
        self.__sys_prog = self.__replacePropSymbols(formula = self.__sys_prog, verbose=verbose)
        for shprob in self.shprobs:
            shprob.setW(W=self.__replacePropSymbols(formula = shprob.getW(), verbose=verbose), \
                            update=False, verbose=verbose)


        # First, make sure that the union of W's covers the entire state space
        if (checkcovering):
            if (verbose > 0):
                print 'Checking that the union of W covers the entire state space...'
            vardict = self.checkcovering(excluded_state=excluded_state, verbose=verbose)
            if (isinstance(vardict, dict)):                
                printInfo('state ' + str(vardict) + ' is not in any W')
                return False
            elif (isinstance(vardict, str)):               
                printInfo('state \n' + vardict + ' is not in any W')
                return False
                

        # Check the partial order condition
        W0ind = self.findW0Ind(verbose=verbose)
        if (verbose > 1):
            print('W0ind = ' + str(W0ind))
        if (checkpartial_order):
            # No cycle
            if (verbose > 0):
                print 'Checking that the partial order condition is satisfied...'
                if (verbose > 1):
                    print 'Checking that there is no cycle...'
            wgraph = self.constructWGraph(verbose=verbose)
            cycle = rhtlputil.findCycle(wgraph, W0ind, verbose=verbose)
            if (len(cycle) != 0):
                cycleStr = ''
                for i in cycle:
                    if (len(cycleStr) > 0):
                        cycleStr += ' -> '
                    cycleStr += 'W[' + str(i) + ']'
                printInfo('Partial order condition is violated due to the cycle ' + cycleStr)
                return False

            # Path to W0
            if (verbose > 1):
                print 'Checking that there is a path to W0...'
            if (len(W0ind) == 0):
                printInfo('Partial order condition violated. ' + \
                              'No W0 since all W do not satisfy spec.sys_prog')
                return False
            if (verbose > 0):
                if (len(W0ind) > 0):
                    W0indStr = ''
                    for ind in W0ind:
                        if (ind != W0ind[0]):
                            W0indStr += ', '
                        W0indStr += 'W[' + str(ind) + ']'
                    print W0indStr + ' satisfy spec.sys_prog.'

            for wind in xrange(0, len(self.shprobs)):
                if (not wind in W0ind):
                    path_found = False
                    for w0ind in W0ind:
                        path = rhtlputil.findPath(wgraph, wind, w0ind, verbose=verbose)
                        if (len(path) > 0):
                            path_found = True
                            break
                    if (not path_found):
                        printInfo('Partial order condition violated. ' + \
                                      'No path from W[' + str(wind) + ' to W0')
                        return False
                
        # Check that all_init -> Phi is a tautology
        if (checktautology):
            if (verbose > 0):
                print 'Checking that init -> Phi is a tautology...'
            self.updatePhi(verbose = verbose)
            tautology = self.checkTautologyPhi(verbose=verbose)
            if (not tautology):
                printInfo('sys_init -> Phi is not a tautology.')
                return False

        # Check that all the short horizon specs are realizable
        if (checkrealizable):
            if (verbose > 0):
                print 'Checking that all the short horizon specs are realizable...'

            self.updatePhi(verbose = verbose)
            for i, shprob in enumerate(self.shprobs):
                shprob.setLocalPhi(Phi=self.__Phi, update=True, verbose=verbose)
                realizable = shprob.checkRealizability(heap_size=heap_size, pick_sys_init=False, \
                                                           verbose=verbose)
                if (not realizable):
                    printInfo('shprob[' + str(i) + '] is not realizable')
                    return False
            return True



###################################################################

# Test case
#  * 1: load from rhtlp file
#  * 2: with dynamics, start from continuous dynamics
#  * 3: with dynamics, start from discretized continuous dynamics
#  * 4: no dynamics, start from continuous dynamics
#  * 5: no dynamics, start from discretized continuous dynamics
if __name__ == "__main__":
    from numpy import array

    print('Testing createProb')
    if (not '2' in sys.argv and not '3' in sys.argv and not '4' in sys.argv and \
            not '5' in sys.argv):
        file = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'examples', \
                                'example.rhtlp')
        prob = SynthesisProb(file=file, verbose=3)
    else:
        env_vars = {'park' : 'boolean', 'cellID' : '{0,...,3,4,5}'}
        sys_disc_vars = {'gear' : '{-1...1}'}
        disc_props = {'Park' : 'park', \
                          'X0d' : 'cellID=0', \
                          'X1d' : 'cellID=1', \
                          'X2d' : 'cellID=2', \
                          'X3d' : 'cellID=3', \
                          'X4d' : 'cellID=4', \
                          'X5d' : 'gear = 1'}
        assumption = '[]<>(!park) & []<>(!X4d)'
        guarantee = '[]<>(X4d -> (X4 | X5)) & []<>X1 & [](Park -> (X0 | X2 | X5))'
        spec = GRSpec()
        spec.env_prog = ['!park', '!X4d']
        spec.sys_prog = ['X4d -> (X4 | X5)', 'X1']
        spec.sys_safety = 'Park -> (X0 | X2 | X5)'

        if ('4' in sys.argv or '5' in sys.argv): # For spec with no dynamics
            spec.sys_prog = '[]<>(X0d -> X5d)'  
            spec.sys_init = ''
            spec.sys_safety = ''
            if ('4' in sys.argv):
                prob = SynthesisProb(env_vars=env_vars, sys_disc_vars=sys_disc_vars, \
                                         disc_props=disc_props, \
                                         sys_cont_vars=[], cont_state_space=None, \
                                         cont_props={}, sys_dyn=None, spec=spec, verbose=3)
            else:
                disc_dynamics=PropPreservingPartition()
                prob = SynthesisProb()
                prob.createProbFromDiscDynamics(env_vars=env_vars, \
                                                    sys_disc_vars=sys_disc_vars, \
                                                    disc_props=disc_props, \
                                                    disc_dynamics=PropPreservingPartition(), \
                                                    spec=spec, verbose=3)
        else:
            cont_state_space = Polytope(array([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]]), \
                                            array([[0.], [3.], [0.], [2.]]))
            cont_props = {}
            cont_props['X0'] = Polytope(array([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]]), \
                                            array([[0.], [1.], [0.], [1.]]))
            cont_props['X1'] = Polytope(array([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]]), \
                                            array([[-1.], [2.], [0.], [1.]]))
            cont_props['X2'] = Polytope(array([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]]), \
                                            array([[-2.], [3.], [0.], [1.]]))
            cont_props['X3'] = Polytope(array([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]]), \
                                            array([[0.], [1.], [-1.], [2.]]))
            cont_props['X4'] = Polytope(array([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]]), \
                                            array([[-1.], [2.], [-1.], [2.]]))
            cont_props['X5'] = Polytope(array([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]]), \
                                            array([[-2.], [3.], [-1.], [2.]]))

            if ('2' in sys.argv):
                prob = SynthesisProb(env_vars=env_vars, sys_disc_vars=sys_disc_vars, \
                                         disc_props=disc_props, \
                                         sys_cont_vars=[], cont_state_space=cont_state_space, \
                                         cont_props=cont_props, sys_dyn=None, spec=spec, verbose=3)
            else:
                disc_dynamics = PropPreservingPartition()
                region0 = Region([cont_props['X0']], [1, 0, 0, 0, 0, 0])
                region1 = Region([cont_props['X1']], [0, 1, 0, 0, 0, 0])
                region2 = Region([cont_props['X2']], [1, 0, 1, 0, 0, 0])
                region3 = Region([cont_props['X3']], [0, 0, 0, 1, 0, 0])
                region4 = Region([cont_props['X4']], [0, 0, 0, 0, 1, 0])
                region5 = Region([cont_props['X5']], [1, 0, 0, 1, 1, 1])
                disc_dynamics.list_region = [region0, region1, region2, region3, region4, region5]
                disc_dynamics.num_regions = len(disc_dynamics.list_region)
                disc_dynamics.trans = [[1, 1, 0, 1, 0, 0], \
                                     [1, 1, 1, 0, 1, 0], \
                                     [0, 1, 1, 0, 0, 1], \
                                     [1, 0, 0, 1, 1, 0], \
                                     [0, 1, 0, 1, 1, 1], \
                                     [0, 0, 1, 0, 1, 1]]
                disc_dynamics.list_prop_symbol = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5']
                disc_dynamics.num_prop = len(disc_dynamics.list_prop_symbol)
                prob = SynthesisProb()
                prob.createProbFromDiscDynamics(env_vars=env_vars, \
                                                    sys_disc_vars=sys_disc_vars, \
                                                    disc_props=disc_props, \
                                                    disc_dynamics=disc_dynamics, \
                                                    spec=spec, verbose=3)

    print('DONE')
    print('================================\n')

    ####################################

    print('Testing checkRealizability')
    realizable = prob.checkRealizability()
    print realizable
    print('DONE')
    print('================================\n')

    ####################################

    print('Testing synthesizePlannerAut')
    aut = prob.synthesizePlannerAut()
    print('DONE')
    print('================================\n')

    ####################################

    if ('2' in sys.argv or '3' in sys.argv):
        print('Testing ShortHorizonProb')
        spec.sys_prog = 'X1'
        rhtlpprob = RHTLPProb(shprobs=[], discretize=False, \
                                  env_vars=prob.getEnvVars(), sys_disc_vars=sys_disc_vars, \
                                  disc_props=disc_props, \
                                  disc_dynamics=prob.getDiscretizedDynamics(), \
                                  spec=spec, verbose=3)
        grspec = GRSpec()
        grspec.env_prog=['!park', '!X4d']
        grspec.sys_prog=['X4d -> (X4 | X5)', 'X1']
        grspec.sys_safety='Park -> (X0 | X2 | X5)'
        shprob = ShortHorizonProb(W='', FW=[], Phi='',
                                      global_prob=rhtlpprob, \
                                      env_vars=prob.getEnvVars(), \
                                      sys_disc_vars=sys_disc_vars, \
                                      disc_props=disc_props, \
                                      disc_dynamics=prob.getDiscretizedDynamics(), \
                                      verbose=3)
        shprob.setW('(X2 -> X3 | X1 -> X4) & X2d')
        shprob2 = copy.deepcopy(shprob)
        shprob2.setW('X1')
        shprob.setFW([shprob2])
        shprob2.setFW([shprob])
        shprob.setLocalPhi('X2 | X4')
        shprob.updateLocalSpec()
        shprob.computeLocalPhi()
        shprob2.updateLocalSpec()
        print shprob.getLocalPhi()
        print('DONE')
        print('================================\n')

        print('Testing RHTLPProb')
        rhtlpprob.addSHProb(shprob)
        rhtlpprob.addSHProb(shprob2)
#         spec.sys_prog = 'X1'
#         rhtlpprob = RHTLPProb(shprobs=[shprob, shprob2], discretize=False, \
#                                   env_vars=prob.getEnvVars(), sys_disc_vars=sys_disc_vars, \
#                                   disc_props=disc_props, \
#                                   disc_dynamics=prob.getDiscretizedDynamics(), \
#                                   spec=spec, verbose=3)
        rhtlpprob.validate(checkcovering=False, checkpartial_order=True, checktautology=True, \
                     checkrealizable=True, verbose=3)
        print('DONE')
        print('================================\n')

    elif ('4' in sys.argv or '5' in sys.argv):
        print('Testing ShortHorizonProb')
        spec.sys_prog = 'X1'
        rhtlpprob = RHTLPProb(shprobs=[], discretize=False, \
                                  env_vars=prob.getEnvVars(), sys_disc_vars=sys_disc_vars, \
                                  disc_props=disc_props, \
                                  disc_dynamics=PropPreservingPartition(), \
                                  spec=spec, verbose=3)
        W='X2d'
        if ('4' in sys.argv):
            shprob = ShortHorizonProb(W=W, FW=[], Phi='', \
                                          global_prob=rhtlpprob, \
                                          env_vars=prob.getEnvVars(), \
                                          sys_disc_vars=sys_disc_vars, \
                                          disc_props=disc_props, \
                                          sys_cont_vars=[], cont_state_space=None, \
                                          cont_props={}, sys_dyn=None, spec=spec, verbose=3)
        else:
            shprob = ShortHorizonProb(W=W, FW=[], Phi='', \
                                          global_prob=rhtlpprob, \
                                          env_vars=prob.getEnvVars(), \
                                          sys_disc_vars=sys_disc_vars, \
                                          disc_props=disc_props, \
                                          disc_dynamics=PropPreservingPartition(), \
                                          spec=spec, verbose=3)
            
        shprob2 = copy.deepcopy(shprob)
        shprob2.setW('X1d')
        shprob.setFW([shprob2])
        shprob2.setFW([shprob])
        shprob.computeLocalPhi()
        print shprob.getLocalPhi()
        print('DONE')
        print('================================\n')

        print('Testing RHTLPProb')
        rhtlpprob.addSHProb(shprob)
        rhtlpprob.addSHProb(shprob2)
#         spec.sys_prog = 'X1'
#         rhtlpprob = RHTLPProb(shprobs=[shprob, shprob2], discretize=False, \
#                                   env_vars=prob.getEnvVars(), sys_disc_vars=sys_disc_vars, \
#                                   disc_props=disc_props, \
#                                   disc_dynamics=PropPreservingPartition(), \
#                                   spec=spec, verbose=3)
        rhtlpprob.validate(checkcovering=True, checkpartial_order=True, checktautology=True, \
                     checkrealizable=True, verbose=3)
        print('DONE')
        print('================================\n')
