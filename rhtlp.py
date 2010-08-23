#!/usr/bin/env python

""" 
-----------------------------------------------
Receding horizon Temporal Logic Planning Module
-----------------------------------------------

Nok Wongpiromsarn (nok@cds.caltech.edu)

:Date: August 20, 2010
:Version: 0.1.0
"""

import sys, os, re, subprocess, copy
from errorprint import printWarning, printError
from parsespec import parseSpec
from polytope_computations import Polytope, Region
from ufm import CtsSysDyn
from prop2part import PropPreservingPartition, prop2part2
from automaton import *
import grgameint


class GRSpec:
    def __init__(self, env_init='', sys_init='', env_safety='', sys_safety='', \
                     env_prog=[], sys_prog=[]):
        self.env_init = env_init
        self.sys_init = sys_init
        self.env_safety = env_safety
        self.sys_safety = sys_safety
        self.env_prog = env_prog
        self.sys_prog = sys_prog
    
    def toSpec(self):
        spec = ['', '']
        spec[0] += self.env_init
        if (len(self.env_safety) > 0):
            if (len(spec[0]) > 0):
                spec[0] += ' & '
            spec[0] += '[](' + self.env_safety + ')'
        for prog in self.env_prog:
            if (len(prog) > 0):
                if (len(spec[0]) > 0):
                    spec[0] += ' & '
                spec[0] += '[]<>(' + prog + ')'

        spec[1] += self.sys_init
        if (len(self.sys_safety) > 0):
            if (len(spec[1]) > 0):
                spec[1] += ' & '
            spec[1] +='[](' + self.sys_safety + ')'
        for prog in self.sys_prog:
            if (len(prog) > 0):
                if (len(spec[1]) > 0):
                    spec[1] += ' & '
                spec[1] += '[]<>(' + prog + ')'
        return spec


###################################################################

class SynthesisProb:
    """
    SynthesisProb class for specifying the receding horizon temporal logic planning
    problem.
    An SynthesisProb object contains the following fields:

    - `env_vars`: a dictionary {str : str} whose keys are the names 
      of environment variables and whose values are their possible values, e.g., 
      boolean or {0, 2, 3, 4, 5}
    - `sys_vars`: a dictionary {str : str} whose keys are the 
      names of system variables and whose values are their possible values.
    - `spec`: a list of two strings that represents system specification of the form
      assumption -> guarantee; the first string is the assumption and the second 
      string is the guarantee.
    - `disc_cont_var`: the name of the continuous variable after the discretization
    - `disc_dynamics`: a list of Region objects corresponding to the partition of the
      continuous state space
    """


    def __init__(self, file='', env_vars={}, sys_disc_vars={}, disc_props={}, \
                     sys_cont_vars=[], cont_state_space=None, \
                     cont_props={}, sys_dyn=None, \
                     compute_disc_dynamics=True, disc_dynamics=None, \
                     spec=['',''], verbose=0):
        self.env_vars = {}
        self.sys_vars = {}
        self.spec = ['','']
        self.disc_cont_var = ''
        self.disc_dynamics = None
        self.__realizable = None
        self.jtlvfile = os.path.join(os.path.abspath(os.path.dirname(__file__)), \
                                         'tmpspec', 'tmp')
        if (file is not None and isinstance(file,str) and \
            len(file) > 0 and not file.isspace()):
            if (not compute_disc_dynamics):
                printWarning("WARNING rhtlp.SynthesisProb(): Discretized dynamics will be " + \
                                 "computed since a spec file is given.")
                compute_disc_dynamics = True

        if (compute_disc_dynamics):
            self.createProbFromContDynamics(file=file, env_vars=env_vars, \
                                            sys_disc_vars=sys_disc_vars, \
                                            disc_props=disc_props, \
                                            sys_cont_vars=sys_cont_vars, \
                                            cont_state_space=cont_state_space, \
                                            cont_props=cont_props, \
                                            sys_dyn=sys_dyn, spec=spec, \
                                            verbose=verbose)
        else:
            self.createProbFromDiscDynamics(env_vars=env_vars, \
                                            sys_disc_vars=sys_disc_vars, \
                                            disc_props=disc_props, \
                                            disc_dynamics=disc_dynamics, \
                                            spec=spec, \
                                            verbose=verbose)

    ###################################################################

    def createProbFromContDynamics(self, file, env_vars={}, sys_disc_vars={}, disc_props={}, \
                       sys_cont_vars=[], cont_state_space=None, \
                       cont_props={}, sys_dyn=None, spec=['',''], verbose=0):
        """
        Construct SynthesisProb from continuous dynamics.

        Input:

        - `file`: the name of the rhtlp file to be parsed. If `file` is not an empty string,
          the rest of the inputs to this function will be ignored.
        - `env_vars`: a dictionary {str : str} or {str : list} whose keys are the names 
          of environment variables and whose values are their possible values, e.g., 
          boolean or {0, 2, ..., 5} or [0, 2, 3, 4, 5].
        - `sys_disc_vars`: a dictionary {str : str} or {str : list} whose keys are the 
          names of discrete system variables and whose values are their possible values.
        - `disc_props`: a dictionary {str : str} whose keys are the symbols for 
          propositions on discrete variables and whose values are the actual propositions
          on discrete variables.
        - `sys_cont_vars`: a list of strings representing the name of continuous variables
        - `cont_state_space`: a Polytope object that represent the state space of the
          continuous variables
        - `cont_props`: a dictionary {str : Polytope} whose keys are the symbols for 
          propositions on continuous variables and whose values are polytopes that represent
          the region in the state space in which the corresponding proposition hold.
        - `sys_dyn`: a CtsSysDyn object that specifies the dynamics of the continuous variables
        - `spec`: a list of two strings that represents system specification of the form
          assumption -> guarantee; the first string is the assumption and the second 
          string is the guarantee.
        - `verbose`: an integer that specifies the verbose level.
        """

        # Check that the input is valid
        if (not isinstance(file, str) and file is not None):
            printError("ERROR rhtlp.SynthesisProb.createProbFromContDynamics: " + \
                           "The input file is expected to be a string")
            raise TypeError("Invalid file.")
        if (not isinstance(sys_cont_vars, list) and sys_cont_vars is not None):
            printError("ERROR rhtlp.SynthesisProb.createProbFromContDynamics: " + \
                           "The input sys_cont_vars is expected to be a list")
            raise TypeError("Invalid disc_props.")
        if (not isinstance(cont_state_space, Polytope) and \
                cont_state_space is not None):
            printError("ERROR rhtlp.SynthesisProb.createProbFromContDynamics: " + \
                           "The type of input cont_state_space is expected to be " + \
                           "Polytope")
            raise TypeError("Invalid cont_state_space.")
        if (not isinstance(cont_props, dict) and cont_props is not None):
            printError("ERROR rhtlp.SynthesisProb.createProbFromContDynamics: " + \
                           "The input cont_props is expected to be a dictionary " + \
                           "{str : Polytope}")
            raise TypeError("Invalid disc_props.")
        if (not isinstance(sys_dyn, CtsSysDyn) and sys_dyn is not None):
            printError("ERROR rhtlp.SynthesisProb.createProbFromContDynamics: " + \
                           "The type of input sys_dyn is expected to be CtsSysDyn")
            raise TypeError("Invalid sys_dyn.")

        # Parse spec file
        if (file is not None and isinstance(file,str) and \
                len(file) > 0 and not file.isspace()):
            (env_vars, sys_disc_vars, disc_props, sys_cont_vars, cont_state_space, \
                 cont_props, sys_dyn, spec) = parseSpec(spec_file=file)

        # Process the continuous component
        if (cont_state_space is not None and cont_props is not None and \
                len(cont_props) > 0):
            cont_partition = prop2part2(cont_state_space, cont_props)
            disc_dynamics = copy.deepcopy(cont_partition)
            disc_dynamics.trans = disc_dynamics.adj
            for fromcell in xrange(0,len(disc_dynamics.trans)):
                disc_dynamics.trans[fromcell][fromcell] = 1
            if (sys_dyn is not None):
                # TODO: Use Ufuk's function instead of below
                pass
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
                       spec=['',''], verbose=0):
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
        - `spec`: a list of two strings that represents system specification of the form
          assumption -> guarantee; the first string is the assumption and the second 
          string is the guarantee.
        - `verbose`: an integer that specifies the verbose level.
        """

        if (isinstance(spec, GRSpec)):
            spec = spec.toSpec()

        # Check that the input is valid
        if (not isinstance(env_vars, dict) and env_vars is not None):
            printError("ERROR rhtlp.SynthesisProb.createProbFromDiscDynamics: " + \
                           "The input env_vars is expected to be a dictionary " + \
                           "{str : str} or {str : list}.")
            raise TypeError("Invalid env_vars.")
        if (not isinstance(sys_disc_vars, dict) and sys_disc_vars is not None):
            printError("ERROR rhtlp.SynthesisProb.createProbFromDiscDynamics: " + \
                           "The input sys_disc_vars is expected to be a dictionary " + \
                           "{str : str} or {str : list}")
            raise TypeError("Invalid sys_disc_vars.")
        if (not isinstance(disc_props, dict) and disc_props is not None):
            printError("ERROR rhtlp.SynthesisProb.createProbFromDiscDynamics: " + \
                           "The input disc_props is expected to be a dictionary " + \
                           "{str : str}")
            raise TypeError("Invalid disc_props.")
        if (not isinstance(disc_dynamics, PropPreservingPartition) and \
                disc_dynamics is not None):
            printError("ERROR rhtlp.SynthesisProb.createProbFromDiscDynamics: " + \
                           "The type of input disc_dynamics is expected to be " + \
                           "PropPreservingPartition")
            raise TypeError("Invalid disc_dynamics.")
        if (not isinstance(spec, list) or len(spec) != 2):
            printError("ERROR rhtlp.SynthesisProb.createProbFromDiscDynamics: " + \
                           "The input spec is expected to be a list of two strings " + \
                           "[assumption, guarantee]")
            raise TypeError("Invalid spec.")

        # Check that the number of regions in disc_dynamics is correct.
        if (disc_dynamics is not None):
            if (disc_dynamics.list_region is None):
                disc_dynamics.list_region = []
            if (disc_dynamics.num_regions != len(disc_dynamics.list_region)):
                printWarning('WARNING rhtlp.SynthesisProb.createProbFromDiscDynamics: ' + \
                                 'disc_dynamics.num_regions != ' + \
                                 'len(disc_dynamics.list_regions)')
                disc_dynamics.num_regions = len(disc_dynamics.list_region)

        # Construct this object
        self.env_vars = copy.deepcopy(env_vars)
        self.sys_vars = copy.deepcopy(sys_disc_vars)

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
                        dots_values = re.findall('([-+]?\d+)\s*?,?\s*?' + r'\.\.\.' + \
                                                     '\s*?,?\s*?([-+]?\d+)', reg)
                        for dots_pair in dots_values:
                            for val in xrange(int(dots_pair[0])+1, int(dots_pair[1])):
                                if (str(val) not in all_values):
                                    all_values.append(str(val))
                        reg = ''
                        for val in all_values:
                            if (len(reg) > 0):
                                reg += ', '
                            reg += val
                        self.env_vars[var] = '{' + reg + '}'
                    else:
                        printWarning('WARNING rhtlp.SynthesisProb.createProbFromDiscDynamics: ' + \
                                         "Unknown possible values for environment " + \
                                         "variable " + var)
                elif (isinstance(reg, list)):
                    all_values = ''
                    for val in reg:
                        if (len(all_values) > 0):
                            all_values += ', '
                        all_values += str(val)
                    self.env_vars[var] = '{' + all_values + '}'
                else:
                    printWarning('WARNING rhtlp.SynthesisProb.createProbFromDiscDynamics: ' + \
                                     "Unknown possible values for environment " + \
                                     "variable "+ var)

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
                        self.sys_vars[var] = '{' + reg + '}'
                    else:
                        printWarning('WARNING rhtlp.SynthesisProb.createProbFromDiscDynamics: ' + \
                                         "Unknown possible values for discrete " + \
                                         "system variable " + var)
                elif (isinstance(reg, list)):
                    all_values = ''
                    for val in reg:
                        if (len(all_values) > 0):
                            all_values += ', '
                        all_values += str(val)
                    self.sys_vars[var] = '{' + all_values + '}'
                else:
                    printWarning('WARNING rhtlp.SynthesisProb.createProbFromDiscDynamics: ' + \
                                     "Unknown possible values for discrete system " + \
                                     "variable " + var)

        # New variable that identifies in which cell the continuous state is
        self.disc_cont_var = ''
        self.disc_dynamics = None
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
            self.sys_vars[cont_varname] = contvar_values
            self.disc_cont_var = cont_varname
            self.disc_dynamics = disc_dynamics

        # Process the spec
        assumption = spec[0]
        guarantee = spec[1]

        # Replace any cont_prop XC by (s.p = P1) | (s.p = P2) | ... | (s.p = Pn) where 
        # P1, ..., Pn are cells in disc_dynamics that satisfy XC
        if (disc_dynamics is not None and disc_dynamics.list_prop_symbol is not None):
            for propInd, propSymbol in enumerate(disc_dynamics.list_prop_symbol):
                reg = [j for j in range(0,disc_dynamics.num_regions) if \
                           disc_dynamics.list_region[j].list_prop[propInd]]
                newprop = 'FALSE'
                if (len(reg) > 0):
                    newprop = '('
                    for i, regID in enumerate(reg):
                        if (i > 0):
                            newprop = newprop + ' | '
                        newprop = newprop + '(' + cont_varname + ' = ' + \
                            str(regID) + ')'
                    newprop = newprop + ')'
                if (verbose > 1):
                    print '\t' + propSymbol + ' -> ' + newprop
                assumption = re.sub(r'\b'+propSymbol+r'\b', newprop, assumption)
                guarantee = re.sub(r'\b'+propSymbol+r'\b', newprop, guarantee)

        # Replace symbols for propositions on discrete variables with the actual 
        # propositions
        if (disc_props is not None):
            for propSymbol, prop in disc_props.iteritems():
                if (verbose > 1):
                    print '\t' + propSymbol + ' -> ' + prop
                assumption = re.sub(r'\b'+propSymbol+r'\b', '('+prop+')', assumption)
                guarantee = re.sub(r'\b'+propSymbol+r'\b', '('+prop+')', guarantee)

        # Transitions for continuous dynamics
        addAnd = False
        if (len(guarantee) > 0 and not(guarantee.isspace())):
            addAnd = True

        if (disc_dynamics is not None):
            for from_region in xrange(0,disc_dynamics.num_regions):
                to_regions = [j for j in range(0,disc_dynamics.num_regions) if \
                                  disc_dynamics.trans[j][from_region]]
                if (addAnd):
                    guarantee += ' &\n'
                if (from_region == 0):
                    guarantee += '-- transition relations for continuous dynamics\n'
                guarantee += '\t[]((' + cont_varname + ' = ' + \
                                  str(from_region) + ') -> next('
                if (len(to_regions) == 0):
                    guarantee += 'FALSE'
                for i, to_region in enumerate(to_regions):
                    if (i > 0):
                        guarantee += ' | '
                    guarantee += '(' + cont_varname + ' = ' + str(to_region) + ')'
                guarantee += '))'
                addAnd = True

        self.spec = [assumption, guarantee]

    ###################################################################

    def checkRealizability(self, heap_size='-Xmx128m', pick_sys_init=True, verbose=0):
        """Determine whether this SynthesisProb is realizable without 
        extracting an automaton.

        Input:

        - `heap_size`: a string that specifies java heap size. 
        - `pick_sys_init` is a boolean indicating whether the system can pick 
          its initial state (in response to the initial environment state).
        - `verbose`: an integer that specifies the verbose level.
        """

        smv_file = self.jtlvfile + '.smv'
        spc_file = self.jtlvfile + '.spc'
        aut_file = self.jtlvfile + '.aut'
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
            aut_file = self.jtlvfile + '.aut'
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
        - `verbose`: an integer that specifies the verbose level. If verbose is set to 0,
          this function will not print anything on the screen.
        """
        smv_file = self.jtlvfile + '.smv'
        spc_file = self.jtlvfile + '.spc'
        aut_file = self.jtlvfile + '.aut'
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
            printError('ERROR rhtlp.SynthesisProb.synthesizePlannerAut: spec not realizable')
            counter_examples = grgameint.getCounterExamples(aut_file=aut_file, verbose=verbose)
            return counter_examples
        else:
            aut = Automaton(states_or_file=aut_file, varnames=[], verbose=verbose)
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
        - `verbose`: an integer that specifies the verbose level. If verbose is set to 0,
          this function will not print anything on the screen.
        """

        # Check that the input is valid
        if (not isinstance(smv_file, str)):
            printError("ERROR rhtlp.SynthesisProb.toJTLVInput: " + \
                           "The input smv_file is expected to be a string")
            raise TypeError("Invalid smv_file.")
        if (not isinstance(spc_file, str)):
            printError("ERROR rhtlp.SynthesisProb.toJTLVInput: " + \
                           "The input spc_file is expected to be a string")
            raise TypeError("Invalid spc_file.")
        
        if (len(smv_file) == 0):
            smv_file = self.jtlvfile + '.smv'
        if (len(spc_file) == 0):
            spc_file = self.jtlvfile + '.spc'

        if (not os.path.exists(os.path.abspath(os.path.dirname(smv_file)))):
            if (verbose > 0):
                printWarning('WARNING rhtlp.SynthesisProb.toJTLVInput: ' + \
                                 'Folder for smv_file ' + smv_file + \
                                 ' does not exist. Creating...')
            os.mkdir(os.path.abspath(os.path.dirname(smv_file)))
        if (not os.path.exists(os.path.abspath(os.path.dirname(spc_file)))):
            if (verbose > 0):
                printWarning('WARNING rhtlp.SynthesisProb.toJTLVInput: ' + \
                                 'Folder for spc_file ' + spc_file + \
                                 ' does not exist. Creating...')
            os.mkdir(os.path.abspath(os.path.dirname(spc_file)))

        # Check whether the smv or spc file exists
        if (file_exist_option != 'r'):
            if (os.path.exists(smv_file)):
                printWarning('WARNING rhtlp.SynthesisProb.toJTLVInput: ' + \
                                 'smv file: ' + smv_file + ' exists.')
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
                printWarning('WARNING rhtlp.SynthesisProb.toJTLVInput: ' + \
                                 'spc file: ' + spc_file + ' exists.')
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
        env_vars = self.env_vars
        sys_vars = self.sys_vars
        spec = self.spec

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
        assumption = spec[0]
        guarantee = spec[1]

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
            assumption = '-- original assumptions\n\t' + assumption 
            addAnd = True
        if (len(env_values_formula) > 0):
            if (addAnd):
                assumption = assumption + ' &\n'
            assumption = assumption + '-- initial env states\n'
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
            f.write('-- original requirements\n\t' + guarantee)
            addAnd = True
        if (len(sys_values_formula) > 0):
            if (addAnd):
                f.write(' &\n')
            f.write('-- initial sys states\n')
            f.write('\t(' + sys_values_formula + ')')
            addAnd = True

        # Transitions
        # For discrete transitions
        if (len(sys_values_formula) > 0):
            if (addAnd):
                f.write(' &\n')
            f.write('-- transition relations for discrete sys states\n');
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
    - `FW`: a list of ShortHorizonProb object that specifies a set F(W).
    - `Phi`: a proposition that specifies the receding horizon invariant.
    """
    def __init__(self, W='', FW=[], Phi='', file='', \
                     env_vars={}, sys_disc_vars={}, disc_props={}, \
                     sys_cont_vars=[], cont_state_space=None, \
                     cont_props={}, sys_dyn=None, \
                     compute_disc_dynamics=True, disc_dynamics=None, \
                     spec=GRSpec(), verbose=0):
        self.W = W
        self.FW = FW
        self.Phi = Phi
        self.__global_spec = copy.deepcopy(spec)
        self.__disc_props = disc_props
        spec = self.__computeLocalSpec()
        SynthesisProb.__init__(self, file=file, env_vars=env_vars, \
                                   sys_disc_vars=sys_disc_vars, \
                                   disc_props=disc_props, \
                                   sys_cont_vars=sys_cont_vars, \
                                   cont_state_space=cont_state_space, \
                                   cont_props=cont_props, sys_dyn=sys_dyn, \
                                   compute_disc_dynamics=compute_disc_dynamics, \
                                   disc_dynamics=disc_dynamics, \
                                   spec=spec, verbose=verbose)

    def getPhi(self):
        """
        Return the local Phi for this ShortHorizonProb object.
        """
        return self.Phi

    def updateLocalSpec(self, verbose=0):
        """
        Update the short horizon specification based on the current W, FW and Phi.
        """
        localspec = self.__computeLocalSpec()
        sys_disc_vars = copy.deepcopy(self.sys_vars)
        if (self.disc_cont_var is not None and len(self.disc_cont_var) > 0 and \
                self.disc_cont_var in sys_disc_vars):
            del sys_disc_vars[self.disc_cont_var]
            
        self.createProbFromDiscDynamics(env_vars=self.env_vars, sys_disc_vars=sys_disc_vars, \
                       disc_props=self.__disc_props, disc_dynamics=self.disc_dynamics, \
                       spec=localspec, verbose=verbose)

    def __computeLocalSpec(self):
        spec = copy.deepcopy(self.__global_spec)
        if (len(self.W) > 0):
            if (len(spec.env_init) > 0):
                spec.env_init += ' & '
            spec.env_init += '(' + self.W + ')'

        if (len(self.Phi) > 0):
            if (len(spec.env_init) > 0):
                spec.env_init += ' & '
            spec.env_init += '(' + self.Phi + ')'
            if (len(spec.sys_safety) > 0):
                spec.sys_safety += ' & '
            spec.sys_safety += '(' + self.Phi + ')'

        spec.sys_prog = []
        if (self.FW is not None and len(self.FW) > 0):
            for fw in self.FW:
                if (len(fw.W) == 0):
                    continue
                if (len(spec.sys_prog) == 0):
                    spec.sys_prog += [fw.W]
                else:
                    if (len(spec.sys_prog[0]) > 0):
                        spec.sys_prog[0] += ' & '
                    spec.sys_prog[0] += fw.W
        specStr = spec.toSpec()
        return specStr

    def computeLocalPhi(self, verbose=0):
        """
        Compute the local Phi for this ShortHorizonProb object.
        Return a boolean that indicates whether the local Phi gets updated.
        If the current prob is realizable, then the local Phi is not updated and this
        function will return True.
        Otherwise, the local Phi will get updated and this function wil return False.
        """
        self.updateLocalSpec(verbose=verbose)
        aut_file = self.jtlvfile + '.aut'
        counter_examples = self.getCounterExamples(recompute=True, pick_sys_init=False, \
                                                       verbose=verbose)
        if (len(counter_examples) == 0):
            return True
        else:
            self.Phi = '(' + self.Phi + ')'
            for ce in counter_examples:
                ce_formula = ''
                for var, val in ce.iteritems():
                    if (len(ce_formula) > 0):
                        ce_formula += ' & '
                    ce_formula += var + ' = ' + str(val)
                if (len(self.Phi) > 0):
                    self.Phi += ' & '
                self.Phi += '!(' + ce_formula + ')'
            return False



###################################################################

class RHTLPProb:
    """
    RHTLPProb class for specifying a receding horizon temporal logic planning problem.
    A RHTLPProb object contains the following fields:

    - `shprobs`: a list of ShortHorizonProb objects
    - `global_spec`: a GRSpec object that specifies the global specification
    """
    def __init__(self, global_spec=GRSpec(), shprobs=[], verbose=0):
        if (isinstance(shprobs, list)):
            self.shprobs = shprobs
        elif (shprobs is None):
            self.shprobs = []
        else:
            self.shprobs = []
            printError("ERROR rhtlp.RHTLPProb(shprobs): the input shprobs must be " + \
                           "a list of ShortHorizonProb objects.")

        if (isinstance(global_spec, GRSpec)):
            self.global_spec = global_spec
        else:
            printError("ERROR rhtlp.RHTLPProb(shprobs): the input global_spec must be " + \
                           "a GRSpec objects.")
        

    def addSHProb(self, shprob):
        if (isinstance(shprob), ShortHorizonProb):
            shprobs.append(shprob)
        else:
            printError("ERROR rhtlp.RHTLPProb.addSHProb: the input shprob must be " + \
                           "a ShortHorizonProb object.")

    def validate(self):
        """
        Check whether the list of ShortHorizonProb objects satisfies the sufficient
        conditions for receding horizon temporal logic planning
        """
        pass


###################################################################

# Test case
#  * 1: load from rhtlp file
#  * 2: with dynamics, start from continuous dynamics
#  * 3: with dynamics, start from discretized continuous dynamics
#  * 4: no dynamics, start from continuous dynamics
#  * 5: no dynamics, start from discretized continuous dynamics
if __name__ == "__main__":
    from polytope_computations import Polytope
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
        spec = [assumption, guarantee]

        if ('4' in sys.argv or '5' in sys.argv): # For spec with no dynamics
            spec[1] = '[]<>(X0d -> X5d)'  
            if ('4' in sys.argv):
                prob = SynthesisProb(file='', env_vars=env_vars, sys_disc_vars=sys_disc_vars, \
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
                prob = SynthesisProb(file='', env_vars=env_vars, sys_disc_vars=sys_disc_vars, \
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

    if ('2' in sys.argv or '3' in sys.argv or '4' in sys.argv or '5' in sys.argv):
        print('Testing ShortHorizonProb')
        grspec = GRSpec()
        grspec.env_prog=['!park', '!X4d']
        grspec.sys_prog=['X4d -> (X4 | X5)', 'X1']
        grspec.sys_safety='Park -> (X0 | X2 | X5)'
        shprob = ShortHorizonProb(W='', FW=[], Phi='', file='', \
                                      env_vars=prob.env_vars, \
                                      sys_disc_vars=sys_disc_vars, \
                                      disc_props=disc_props, \
                                      sys_cont_vars=[], \
                                      cont_state_space=None, \
                                      cont_props={}, sys_dyn=None, \
                                      compute_disc_dynamics=False, \
                                      disc_dynamics=prob.disc_dynamics, \
                                      spec=grspec)
        shprob.W = 'X2'
        shprob2 = copy.deepcopy(shprob)
        shprob2.W = 'X4'
        shprob.FW = [shprob2]
        shprob.Phi = 'X2 | X4'
        shprob.updateLocalSpec()
        shprob.computeLocalPhi()
        print shprob.Phi
        print('DONE')
        print('================================\n')
    

