#!/usr/bin/env python

""" 
-----------------------------------------------
Receding horizon Temporal Logic Planning Module
-----------------------------------------------

Nok Wongpiromsarn (nok@cds.caltech.edu)

:Date: August 25, 2010
:Version: 0.1.0
"""

import sys, os, re, subprocess, copy
from errorprint import printWarning, printError, printInfo
from parsespec import parseSpec
from polytope_computations import Polytope, Region
from ufm import CtsSysDyn
from prop2part import PropPreservingPartition, prop2part2
from automaton import Automaton
from spec import GRSpec
import rhtlputil
import grgameint


class SynthesisProb:
    """
    SynthesisProb class for specifying the receding horizon temporal logic planning
    problem.
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
    """

    def __init__(self, **args):
        """
        SynthesisProb(`file`=''): construct this SynthesisProb object from `file`

        - `file`: the name of the rhtlp file to be parsed. If `file` is given,
          the rest of the inputs to this function will be ignored.

        SynthesisProb(`env_vars`={}, `sys_disc_vars`={}, `disc_props`={}, 
        `disc_dynamics`=None, `spec`=GRSpec())

        SynthesisProb(`env_vars`={}, `sys_disc_vars`={}, `disc_props`={}, 
        `cont_state_space`=None, `cont_props`={}, `sys_dyn`=None, `spec`=GRSpec())

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
        - `verbose`: an integer that specifies the verbose level.
        """

        self.env_vars = {}
        self.sys_vars = {}
        self.spec = GRSpec(env_init='', sys_init='', env_safety='', sys_safety='', \
                               env_prog='', sys_prog='')
        self.disc_cont_var = ''
        self.disc_dynamics = None
        self.jtlvfile = os.path.join(os.path.abspath(os.path.dirname(__file__)), \
                                         'tmpspec', 'tmp')
        self.__realizable = None

        verbose = args.get('verbose', 0)
        env_vars = args.get('env_vars', {})
        sys_disc_vars = args.get('sys_disc_vars', {})
        disc_props = args.get('disc_props', {})

        spec = GRSpec(env_init='', sys_init='', env_safety='', sys_safety='', \
                               env_prog='', sys_prog='')
        if ('spec' in args.keys()):
            spec = args['spec']

        cont_state_space = None
        cont_props = {}
        sys_dyn = None

        if ('file' in args.keys()):
            file = args['file']
            # Check that the input is valid
            if (not isinstance(file, str)):
                printError("ERROR rhtlp.SynthesisProb: " + \
                               "The input file is expected to be a string")
                raise TypeError("Invalid file.")
            if (not os.path.isfile(file)):
                printError("ERROR rhtlp.SynthesisProb: " + \
                               "The rhtlp file " + file + " does not exist.")
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
            if ('cont_state_space' in args.keys()):
                cont_state_space = args['cont_state_space']
            if ('cont_props' in args.keys()):
                cont_props = args['cont_props']
            if ('sys_dyn' in args.keys()):
                sys_dyn = args['sys_dyn']
            self.createProbFromContDynamics(env_vars=env_vars, \
                                            sys_disc_vars=sys_disc_vars, \
                                            disc_props=disc_props, \
                                            cont_state_space=cont_state_space, \
                                            cont_props=cont_props, \
                                            sys_dyn=sys_dyn, 
                                            spec=spec, \
                                            verbose=verbose)


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
        - `verbose`: an integer that specifies the verbose level.
        """

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
        - `verbose`: an integer that specifies the verbose level.
        """

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
        if (not isinstance(spec, GRSpec) and \
                (not isinstance(spec, list) or len(spec) != 2)):
            printError("ERROR rhtlp.SynthesisProb.createProbFromDiscDynamics: " + \
                           "The input spec is expected to be a GRSpec object")
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
            self.disc_dynamics = copy.deepcopy(disc_dynamics)

        # Process the spec
        self.spec = copy.deepcopy(spec)

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
                if (isinstance(self.spec, GRSpec)):
                    self.spec.sym2prop(props={propSymbol:newprop}, verbose=verbose)
                else:
                    self.spec[0] = re.sub(r'\b'+propSymbol+r'\b', '('+newprop+')', self.spec[0])
                    self.spec[1] = re.sub(r'\b'+propSymbol+r'\b', '('+newprop+')', self.spec[1])

        # Replace symbols for propositions on discrete variables with the actual 
        # propositions
        if (isinstance(self.spec, GRSpec)):
            self.spec.sym2prop(props=disc_props, verbose=verbose)
        else:
            if (disc_props is not None):
                for propSymbol, prop in disc_props.iteritems():
                    if (verbose > 1):
                        print '\t' + propSymbol + ' -> ' + prop
                    self.spec[0] = re.sub(r'\b'+propSymbol+r'\b', '('+prop+')', self.spec[0])
                    self.spec[1] = re.sub(r'\b'+propSymbol+r'\b', '('+prop+')', self.spec[1])

#         # Transitions for continuous dynamics
#         addAnd = False
#         if (len(self.spec.sys_safety) > 0 and not(self.spec.sys_safety.isspace())):
#             addAnd = True
#
#         if (disc_dynamics is not None):
#             for from_region in xrange(0,disc_dynamics.num_regions):
#                 to_regions = [j for j in range(0,disc_dynamics.num_regions) if \
#                                   disc_dynamics.trans[j][from_region]]
#                 if (addAnd):
#                     self.spec.sys_safety += ' &\n'
#                 if (from_region == 0):
#                     self.spec.sys_safety += '-- transition relations for continuous dynamics\n'
#                 self.spec.sys_safety += '\t((' + cont_varname + ' = ' + \
#                                   str(from_region) + ') -> next('
#                 if (len(to_regions) == 0):
#                     self.spec.sys_safety += 'FALSE'
#                 for i, to_region in enumerate(to_regions):
#                     if (i > 0):
#                         self.spec.sys_safety += ' | '
#                     self.spec.sys_safety += '(' + cont_varname + ' = ' + str(to_region) + ')'
#                 self.spec.sys_safety += '))'
#                 addAnd = True


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
        if (isinstance(self.spec, GRSpec)):
            spec = self.spec.toJTLVSpec()
        else:
            spec = self.spec
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
        if (self.disc_dynamics is not None):
            for from_region in xrange(0,self.disc_dynamics.num_regions):
                to_regions = [j for j in range(0,self.disc_dynamics.num_regions) if \
                                  self.disc_dynamics.trans[j][from_region]]
                if (addAnd):
                    f.write(' &\n')
                if (from_region == 0):
                    f.write('-- transition relations for continuous dynamics\n')
                f.write('\t[]((s.' + self.disc_cont_var + ' = ' + \
                            str(from_region) + ') -> next(')
                if (len(to_regions) == 0):
                    f.write('FALSE')
                for i, to_region in enumerate(to_regions):
                    if (i > 0):
                        f.write(' | ')
                    f.write('(s.' + self.disc_cont_var + ' = ' + str(to_region) + ')')
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
    """
    def __init__(self, W='', FW=[], Phi='', global_spec=GRSpec(), **args):
        self.setW(W=W, update=False, verbose=0)
        self.setFW(FW=FW, update=False, verbose=0)
        self.setPhi(Phi=Phi, update=False, verbose=0)
        
        self.__disc_props = {}
        if ('disc_props' in args.keys()):
            self.__disc_props = copy.deepcopy(args['disc_props'])
        
        verbose = args.get('verbose', 0)

        if (not isinstance(global_spec, GRSpec)):
            printError("ERROR rhtlp.ShortHorizonProb: the input global_spec must be " + \
                           "a GRSpec objects.")
            raise TypeError("Invalid global_spec.")
        self.global_spec = copy.deepcopy(global_spec)
        self.global_spec.sym2prop(self.__disc_props, verbose=verbose)

        args['spec'] = self.__computeLocalSpec()
        SynthesisProb.__init__(self, **args)

    def setW(self, W, update=True, verbose=0):
        if (not isinstance(W, str)):
            printError("ERROR rhtlp.ShortHorizonProb.setW: the input W must be " + \
                           "a string.")
            raise TypeError("Invalid W.")
        self.W = W
        if (update):
            self.updateLocalSpec(verbose=verbose)

    def setFW(self, FW, update=True, verbose=0):
        if (not isinstance(FW, list) and not isinstance(FW, ShortHorizonProb)):
            printError("ERROR rhtlp.ShortHorizonProb: the input FW must be " + \
                           "a ShortHorizonProb object or a list of ShortHorizonProb object.")
            raise TypeError("Invalid W.")
        if (isinstance(FW, list)):
            self.FW = copy.copy(FW)
        else:
            self.FW = FW
        if (update):
            self.updateLocalSpec(verbose=verbose)

    def setPhi(self, Phi, update=True, verbose=0):
        if (not isinstance(Phi, str)):
            printError("ERROR rhtlp.ShortHorizonProb: the input Phi must be " + \
                           "a string.")
            raise TypeError("Invalid Phi.")
        self.Phi = Phi
        if (update):
            self.updateLocalSpec(verbose=verbose)

    def getPhi(self):
        """
        Return the local Phi for this ShortHorizonProb object.
        """
        return self.Phi

    def updateLocalSpec(self, verbose=0):
        """
        Update the short horizon specification based on the current W, FW and Phi.
        """
        local_spec = self.__computeLocalSpec()
        sys_disc_vars = copy.deepcopy(self.sys_vars)
        if (self.disc_cont_var is not None and len(self.disc_cont_var) > 0 and \
                self.disc_cont_var in sys_disc_vars):
            del sys_disc_vars[self.disc_cont_var]
            
        self.createProbFromDiscDynamics(env_vars=self.env_vars, sys_disc_vars=sys_disc_vars, \
                       disc_props=self.__disc_props, disc_dynamics=self.disc_dynamics, \
                       spec=local_spec, verbose=verbose)

    def __computeLocalSpec(self, verbose=0):
        local_spec = copy.deepcopy(self.global_spec)
        if (len(self.W) > 0):
            if (len(local_spec.env_init) > 0):
                local_spec.env_init += ' & \n'
            local_spec.env_init += '-- W\n'
            local_spec.env_init += '\t(' + self.W + ')'

        if (len(self.Phi) > 0):
            if (len(local_spec.env_init) > 0):
                local_spec.env_init += ' & \n'
            local_spec.env_init += '-- Phi\n'
            local_spec.env_init += '\t(' + self.Phi + ')'
            if (len(local_spec.sys_safety) > 0):
                local_spec.sys_safety += ' & \n'
            local_spec.sys_safety += '-- Phi\n'
            local_spec.sys_safety += '\t(' + self.Phi + ')'

        local_spec.sys_prog = []
        if (self.FW is not None and len(self.FW) > 0):
            for fw in self.FW:
                if (len(fw.W) == 0):
                    continue
                if (len(local_spec.sys_prog) == 0):
                    local_spec.sys_prog += [fw.W]
                else:
                    if (len(local_spec.sys_prog[0]) > 0):
                        local_spec.sys_prog[0] += ' & '
                    local_spec.sys_prog[0] += fw.W
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
        aut_file = self.jtlvfile + '.aut'
        counter_examples = self.getCounterExamples(recompute=True, pick_sys_init=False, \
                                                       verbose=verbose)
        if (len(counter_examples) == 0):
            return False
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
            return True



###################################################################


class RHTLPProb(SynthesisProb):
    """
    RHTLPProb class for specifying a receding horizon temporal logic planning problem.
    A RHTLPProb object contains the following fields:

    - `shprobs`: a list of ShortHorizonProb objects
    - `spec`: a GRSpec object that specifies the global specification
    """
    def __init__(self, shprobs=[], discretize=False, **args):
        self.shprobs = []
        self.__disc_props = {}
        self.__cont_props = []
        self.env_vars = {}
        self.sys_vars = {}
        self.spec = GRSpec(env_init='', sys_init='', env_safety='', sys_safety='', \
                               env_prog=[], sys_prog=[])
        self.disc_cont_var = ''
        self.disc_dynamics = None
        self.jtlvfile = os.path.join(os.path.abspath(os.path.dirname(__file__)), \
                                         'tmpspec', 'tmp')
        self.__realizable = None
        self.__sys_prog = True
        
        verbose = args.get('verbose', 0)
        if ('disc_props' in args.keys()):
            self.__disc_props = copy.deepcopy(args['disc_props'])
        cont_props = args.get('cont_props', {})
        if (isinstance(cont_props, dict)):
            self.__cont_props = copy.deepcopy(cont_props.keys())
        elif (isinstance(cont_props, list)):
            self.__cont_props = copy.deepcopy(cont_props)
        else:
            printError("ERROR rhtlp.RHTLPProb: " + \
                           "The input cont_props is expected to be a dictionary " + \
                           "{str : Polytope}")
            raise TypeError("Invalid disc_props.")
                
        if ('spec' in args.keys()):
            if (not isinstance(args['spec'], GRSpec)):
                printError("ERROR rhtlp.RHTLPProb: The input spec must be " + \
                           "a GRSpec objects.")
                raise TypeError("Invalid spec.")
            if (isinstance(spec.sys_prog, list) and len(spec.sys_prog) > 1):
                printError("ERROR rhtlp.RHTLPProb: The input spec can have " + \
                               "at most one system progress formula.")
                raise TypeError("Invalid spec.")
            self.spec = copy.deepcopy(args['spec'])
            if (isinstance(self.spec.sys_prog, str)):
                self.__sys_prog = self.spec.sys_prog
            elif (isinstance(self.spec.sys_prog, list) and len(self.spec.sys_prog) == 1 and \
                      len(self.spec.sys_prog[0]) > 0 and not self.spec.sys_prog[0].isspace()):
                self.__sys_prog = self.spec.sys_prog[0]

        if (isinstance(shprobs, list)):
            for shprob in shprobs:
                if (isinstance(shprob, ShortHorizonProb)):
                    self.shprobs.append(shprob)
                else:
                    printError("ERROR rhtlp.RHTLPProb: the input shprobs must be " + \
                                   "a list of ShortHorizonProb objects.")
        elif (shprobs is not None):
            printError("ERROR rhtlp.RHTLPProb: the input shprobs must be " + \
                           "a list of ShortHorizonProb objects.")

        env_vars = args.get('env_vars', {})
        sys_disc_vars = args.get('sys_disc_vars', {})
        cont_state_space = None
        sys_dyn = None
        disc_dynamics = None

        if ('file' in args.keys()):
            file = args['file']
            # Check that the input is valid
            if (not isinstance(file, str)):
                printError("ERROR rhtlp.RHTLPProb: " + \
                               "The input file is expected to be a string")
                raise TypeError("Invalid file.")
            if (not os.path.isfile(file)):
                printError("ERROR rhtlp.RHTLPProb: " + \
                               "The rhtlp file " + file + " does not exist.")
                raise TypeError("Invalid file.")

            (env_vars, sys_disc_vars, self.__disc_props, sys_cont_vars, cont_state_space, \
                 cont_props, sys_dyn, self.spec) = parseSpec(spec_file=file)
            self.__cont_props = cont_props.keys()

        elif ('disc_dynamics' in args.keys()):
            if (discretize):
                printWarning('WARNING rhtlp.RHTLPProb ' + \
                                 'Discretized dynamics is already given.')
            discretize = False
            disc_dynamics = args['disc_dynamics']
            if (len(self.__cont_props) == 0):
                self.__cont_props = copy.deepcopy(disc_dynamics.list_prop_symbol)
            else:
                if (disc_dynamics is None or \
                        not (set(self.__cont_props) == set(disc_dynamics.list_prop_symbol))):
                    printWarning("WARNING rhtlp.RHTLPProb: " + \
                                     "The given cont_prop does not match the propositions" + \
                                     " in the given disc_dynamics")
        else:        
            if ('cont_state_space' in args.keys()):
                cont_state_space = args['cont_state_space']
            if ('sys_dyn' in args.keys()):
                sys_dyn = args['sys_dyn']

        if (discretize):
            self.createProbFromContDynamics(env_vars=env_vars, \
                                                sys_disc_vars=sys_disc_vars, \
                                                disc_props=self.__disc_props, \
                                                cont_state_space=cont_state_space, \
                                                cont_props=cont_props, \
                                                sys_dyn=sys_dyn, \
                                                spec=self.spec, \
                                                verbose=verbose)
        else:
            self.createProbFromDiscDynamics(env_vars=env_vars, \
                                                sys_disc_vars=sys_disc_vars, \
                                                disc_props=self.__disc_props, \
                                                disc_dynamics=disc_dynamics, \
                                                spec=self.spec, \
                                                verbose=verbose)


    def addSHProb(self, shprob):
        if (isinstance(shprob), ShortHorizonProb):
            self.shprobs.append(shprob)
        else:
            printError("ERROR rhtlp.RHTLPProb.addSHProb: the input shprob must be " + \
                           "a ShortHorizonProb object.")


    def __checkcovering(self, verbose=0):
        allW_formula = 'False'
        for shprob in self.shprobs:
            allW_formula += ' | (' + shprob.W + ')'

#         if (self.__disc_props is not None):
#             for propSymbol, prop in self.__disc_props.iteritems():
#                 if (verbose > 1):
#                     print '\t' + propSymbol + ' -> ' + prop
#                 allW_formula = re.sub(r'\b'+propSymbol+r'\b', '('+prop+')', allW_formula)

        if (verbose > 0):
            print 'W = ' + allW_formula
        
        allvars = copy.deepcopy(self.env_vars)
        sys_disc_vars = copy.deepcopy(self.sys_vars)
        if (self.disc_cont_var is not None and len(self.disc_cont_var) > 0 and \
                self.disc_cont_var in sys_disc_vars):
            del sys_disc_vars[self.disc_cont_var]
        allvars.update(sys_disc_vars)
        for var in self.__cont_props:
            allvars[var] = 'boolean'

        allvars_values = ()
        allvars_variables = []
        for var, val in allvars.iteritems():
            tmp = [0,1]
            if (not 'boolean' in val):
                tmp = re.findall('[-+]?\d+', val)
                tmp = [int(i) for i in tmp]
            allvars_values += tmp,
            allvars_variables.append(var)

        allvars_values_iter = rhtlputil.product(*allvars_values)
        vardict = {}
        for val in allvars_values_iter:
            vardict = dict(zip(allvars_variables, val))
#             for i in xrange(0, len(allvars_variables)):
#                 vardict[allvars_variables[i]] = int(val[i])
            
            try:
                ret = rhtlputil.evalExpr(allW_formula, vardict, verbose)
            except:
                printError('ERROR rhtlp.RHTLPProb.validate: ' + \
                               'invalid W')
                raise Exception("Invalid W")
            if (not ret):
#                 state = dict(zip(allvars_variables, val))
#                 state = ''
#                 for i in xrange(0, len(allvars_variables)):
#                     if (len(state) > 0):
#                         state += ', '
#                     state += allvars_variables[i] + ':' + str(val[i])
#                 printError('ERROR rhtlp.RHTLPProb.validate: ' + \
#                                'state <' + state + '> is not in any W')
                return vardict
        return True


    def __constructWGraph(self, verbose=0):
        graph = []
        for wind, shprob in enumerate(self.shprobs):
            if (isinstance(shprob.FW, list)):
                fw_ind = []
                for fw in shprob.FW:
                    if (isinstance(fw, int)):
                        fw_ind.append(fw)
                    elif (isinstance(fw, ShortHorizonProb)):
                        tmpind = self.__findWInd(fw, verbose=verbose)
                        if (tmpind >= 0):
                            fw_ind.append(tmpind)
                        else:
                            printError("ERROR rhtlp.RHTLPProb.__constructWGraph " + \
                                           "FW for shprobs[" + str(wind) + "]" + \
                                           " is not in this RHTLPProb.")
                            raise Exception("Invalid FW.")
                        graph.append(fw_ind)
                    else:
                        printError("ERROR rhtlp.RHTLPProb.__constructWGraph " + \
                                       "Invalid FW")
                        raise TypeError("Invalid FW.")
            elif (isinstance(shprob.FW, int)):
                graph.append([shprob.FW])
            elif (isinstance(shprob.FW, ShortHorizonProb)):
                fw_ind = self.__findWInd(fw, verbose=verbose)
                if (tmpind >= 0):
                    graph.append([fw_ind])
                else:
                    printError("ERROR rhtlp.RHTLPProb.__constructWGraph " + \
                                   "FW for shprobs[" + str(wind) + "]" + \
                                   " is not in this RHTLPProb.")
                    raise Exception("Invalid FW.")
            else:
                printError("ERROR rhtlp.RHTLPProb.__constructWGraph " + \
                               "Invalid FW for shprobs[" + str(wind) + "].")
                raise TypeError("Invalid FW.")
        return graph


    def __findWInd(self, W, verbose=0):
        ind = 0
        while (ind < len(self.shprobs)):
            if (W == self.shprobs[ind]):
                return ind
            ind += 1
        return -1

    def __findW0Ind(self, verbose=0):
        W0ind = range(0, len(self.shprobs))

        if (self.__sys_prog == True):
            return W0ind
        
        allvars = copy.deepcopy(self.env_vars)
        sys_disc_vars = copy.deepcopy(self.sys_vars)
        if (self.disc_cont_var is not None and len(self.disc_cont_var) > 0 and \
                self.disc_cont_var in sys_disc_vars):
            del sys_disc_vars[self.disc_cont_var]
        allvars.update(sys_disc_vars)
        for var in self.__cont_props:
            allvars[var] = 'boolean'

        allvars_values = ()
        allvars_variables = []
        for var, val in allvars.iteritems():
            tmp = [0,1]
            if (not 'boolean' in val):
                tmp = re.findall('[-+]?\d+', val)
                tmp = [int(i) for i in tmp]
            allvars_values += tmp,
            allvars_variables.append(var)

        allvars_values_iter = rhtlputil.product(*allvars_values)
        vardict = {}
        for val in allvars_values_iter:
            vardict = dict(zip(allvars_variables, val))
            try:
                ret = rhtlputil.evalExpr(sys_prog, vardict, verbose)
            except:
                printError('ERROR rhtlp.RHTLPProb.validate: ' + \
                               'invalid W')
                raise Exception("Invalid W")
            if (ret):
                newW0ind = []
                for ind in W0ind:
                    ret = rhtlputil.evalExpr(self.shprobs[ind].W, vardict, verbose)
                    if (ret):
                        newW0ind.append(ind)
                    elif (verbose > 0):
                        print 'W[' + str(ind) + '] does not satisfy spec.sys_prog'
                        print 'counter example: ', vardict
                W0ind = newW0ind
                if (len(W0ind) == 0):
                    return W0ind
        return W0ind


    def validate(self, verbose=0):
        """
        Check whether the list of ShortHorizonProb objects satisfies the sufficient
        conditions for receding horizon temporal logic planning
        """

        if (self.__disc_props is not None):
            for propSymbol, prop in self.__disc_props.iteritems():
                if (verbose > 1):
                    print '\t' + propSymbol + ' -> ' + prop
                self.__sys_prog = re.sub(r'\b'+propSymbol+r'\b', '('+prop+')', self.__sys_prog)
                for shprob in self.shprobs:
                    shprob.W = re.sub(r'\b'+propSymbol+r'\b', '('+prop+')', shprob.W)
#                     if (isinstance(shprob.FW, ShortHorizonProb)):
#                         shprob.FW.W = re.sub(r'\b'+propSymbol+r'\b', '('+prop+')', shprob.FW.W)
#                     elif (isinstance(shprob.FW, list)):
#                         for fw in shprob.FW:
#                             if (isinstance(fw, ShortHorizonProb)):
#                                 fw.W = re.sub(r'\b'+propSymbol+r'\b', '('+prop+')', fw.W)
                                

        # First, make sure that the union of W's covers the entire state space
        vardict = self.__checkcovering(verbose=verbose)
        if (isinstance(vardict, dict)):                
            printInfo('state ' + str(vardict) + ' is not in any W')
            return False

        # Check the partial order condition
        # No cycle
        wgraph = self.__constructWGraph(verbose=verbose)
        cycle = rhtlputil.findCycle(wgraph, verbose=verbose)
        if (len(cycle) != 0):
            cycleStr = ''
            for i in cycle:
                if (len(cycleStr) > 0):
                    cycleStr += ' -> '
                cycleStr += 'W[' + str(i) + ']'
            printInfo('Partial order condition is violated due to the cycle ' + cycleStr)
            return False

        # Path to W0
        W0ind = self.__findW0Ind(verbose=verbose)
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
                
        # Check that all the short horizon specs are realizable
        return True







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

    if ('2' in sys.argv or '3' in sys.argv or '4' in sys.argv or '5' in sys.argv):
        print('Testing ShortHorizonProb')
        grspec = GRSpec()
        grspec.env_prog=['!park', '!X4d']
        grspec.sys_prog=['X4d -> (X4 | X5)', 'X1']
        grspec.sys_safety='Park -> (X0 | X2 | X5)'
        shprob = ShortHorizonProb(W='', FW=[], Phi='',
                                      global_spec=grspec, \
                                      env_vars=prob.env_vars, \
                                      sys_disc_vars=sys_disc_vars, \
                                      disc_props=disc_props, \
                                      disc_dynamics=prob.disc_dynamics, \
                                      verbose=3)
        shprob.W = '(X2=1 -> X3 | X1 -> X4) & X2d'
        shprob2 = copy.deepcopy(shprob)
        shprob2.W = 'X1'
        shprob.FW = [shprob2]
        shprob2.FW = [shprob]
        shprob.Phi = 'X2 | X4'
        shprob.updateLocalSpec()
        shprob.computeLocalPhi()
        shprob2.updateLocalSpec()
        print shprob.Phi
        print('DONE')
        print('================================\n')

    ####################################

        print('Testing ShortHorizonProb')
        spec.sys_prog = 'X1'
        rhtlpprob = RHTLPProb(shprobs=[shprob, shprob2], discretize=False, \
                                  env_vars=prob.env_vars, sys_disc_vars=sys_disc_vars, \
                                  disc_props=disc_props, disc_dynamics=prob.disc_dynamics, \
                                  spec=spec, verbose=3)
        rhtlpprob.validate(3)
        print('DONE')
        print('================================\n')
