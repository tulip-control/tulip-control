#!/usr/bin/env python

""" 
-------------------------------------------------------------------
Jtlvint Module --- Interface to the JTLV implementation of GR1 game
-------------------------------------------------------------------

About JTLV, see http://jtlv.ysaar.net/

:Date: August 3, 2010
:Version: 0.1.0
"""

import re, os, subprocess, sys
from polyhedron import Vrep, Hrep
from prop2part import Region, PropPreservingPartition
from errorprint import printWarning, printError

# Get jtlv_path
JTLV_PATH = os.path.abspath(os.path.dirname(__file__))
JTLV_EXE = 'jtlv_grgame.jar'

def setJTLVPath(jtlv_path):
    """Set path to jtlv_grgame.jar.

    Input:

    - `jtlv_path`: a string indicating the full path to the JTLV folder.
    """
    globals()["JTLV_PATH"] = jtlv_path

def setJTLVExe(jtlv_exe):
    """Set the name of the jtlv executable.

    Input:

    - `jtlv_exe`: a string indicating the name of the executable jar containing the 
      jtlv GR1 game implementation.
    """
    globals()["JTLV_EXE"] = jtlv_exe

def generateJTLVInput(env_vars={}, disc_sys_vars={}, spec='', disc_props={}, \
                          disc_dynamics=PropPreservingPartition(), \
                          smv_file='specs/spec.smv', spc_file='specs/spec.spc', \
                          file_exist_option='a', verbose=0):
    """Generate JTLV input files: smv_file and spc_file.

    Input:

    - `env_vars`: a dictionary {str : str} or {str : list} whose keys are the names 
      of environment variables and whose values are their possible values, e.g., 
      boolean or {0, 2, ..., 5} or [0, 2, 3, 4, 5].
    - `disc_sys_vars`: a dictionary {str : str} or {str : list} whose keys are the 
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
    - `verbose`: an integer that specifies the verbose level. If verbose is set to 0,
      this function will not print anything on the screen.
    """

    # Check that the input is valid
    if (not isinstance(env_vars, dict)):
        printError("The input env_vars is expected to be a dictionary {str : str} " +
                   "or {str : list}.")
        raise TypeError("Invalid env_vars.")
    if (not isinstance(disc_sys_vars, dict)):
        printError("The input disc_sys_vars is expected to be a dictionary " + \
                       "{str : str} or {str : list}")
        raise TypeError("Invalid disc_sys_vars.")
    if (not isinstance(spec, list) and len(spec) != 2):
        printError("The input spec is expected to be a list of two strings " + \
                       "[assumption, guarantee]")
        raise TypeError("Invalid spec.")
    if (not isinstance(disc_dynamics, PropPreservingPartition)):
        printError("The type of input spec is expected to be PropPreservingPartition")
        raise TypeError("Invalid disc_dynamics.")
    if (not isinstance(smv_file, str)):
        printError("The input smv_file is expected to be a string")
        raise TypeError("Invalid smv_file.")
    if (not isinstance(spc_file, str)):
        printError("The input spc_file is expected to be a string")
        raise TypeError("Invalid spc_file.")

#    # Figure out the names of the smv and spc files
#     if (smv_file[-4:] != '.smv'):
#         smv_file = smv_file + '.smv'
#     if (spc_file[-4:] != '.spc'):
#         spc_file = spc_file + '.spc'

    if (not os.path.exists(os.path.abspath(os.path.dirname(smv_file)))):
        printWarning('Folder for smv_file ' + smv_file + ' does not exist. Creating...')
        os.mkdir(os.path.abspath(os.path.dirname(smv_file)))
    if (not os.path.exists(os.path.abspath(os.path.dirname(spc_file)))):
        printWarning('Folder for spc_file ' + spc_file + ' does not exist. Creating...')
        os.mkdir(os.path.abspath(os.path.dirname(spc_file)))

    # Check whether the smv or spc file exists
    if (file_exist_option != 'r'):
        if (os.path.exists(smv_file)):
            printWarning('smv file: ' + smv_file + ' exists.')
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
            printWarning('spc file: ' + spc_file + ' exists.')
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
    # Check that the number of regions in disc_dynamics is correct.
    if (disc_dynamics.num_regions != len(disc_dynamics.list_region)):
        printWarning('WARNING: disc_dynamics.num_regions != ' + \
                         "len(disc_dynamics.list_regions)")
        disc_dynamics.num_regions = len(disc_dynamics.list_region)

    # Replace '...' in the range of possible values of env_vars to the actual values 
    # and convert a list representation of the range of possible values to a string
    for var, reg in env_vars.iteritems():
        if ('boolean' in reg):
            continue
        elif (isinstance(reg, str)):
            dots_values = re.findall('([-+]?\d+)\s*?,?\s*?' + r'\.\.\.' + \
                                         '\s*?,?\s*?([-+]?\d+)', reg)
            all_values = list(set(re.findall('[-+]?\d+', reg)))
            if (len(all_values) > 0):
                for dots_pair in dots_values:
                    for val in range(int(dots_pair[0])+1, int(dots_pair[1])):
                        if (str(val) not in all_values):
                            all_values.append(str(val))
                reg = ''
                for val in all_values:
                    if (len(reg) > 0):
                        reg = reg + ', '
                    reg = reg + val
                env_vars[var] = '{' + reg + '}'
            else:
                printWarning("WARNING: Unknown possible values for environment " + \
                                 "variable " + var)
        elif (isinstance(reg, list)):
            all_values = ''
            for val in reg:
                if (len(all_values) > 0):
                    all_values = all_values + ', '
                all_values = all_values + str(val)
            env_vars[var] = '{' + all_values + '}'
        else:
            printWarning("WARNING: Unknown possible values for environment " + \
                             "variable "+ var)
                
    # Replace '...' in the range of possible values of disc_sys_vars to the actual 
    # values and convert a list representation of the range of possible values to a 
    # string
    for var, reg in disc_sys_vars.iteritems():
        if ('boolean' in reg):
            continue
        elif (isinstance(reg, str)):
            dots_values = re.findall('([-+]?\d+)\s*,?\s*' + r'\.\.\.' + \
                                         '\s*,?\s*([-+]?\d+)', reg)
            all_values = list(set(re.findall('[-+]?\d+', reg)))
            if (len(all_values) > 0):
                for dots_pair in dots_values:
                    for val in range(int(dots_pair[0])+1, int(dots_pair[1])):
                        if (str(val) not in all_values):
                            all_values.append(str(val))
                reg = ''
                for val in all_values:
                    if (len(reg) > 0):
                        reg = reg + ', '
                    reg = reg + val
                disc_sys_vars[var] = '{' + reg + '}'
            else:
                printWarning("WARNING: Unknown possible values for discrete system " + \
                                 "variable " + var)
        elif (isinstance(reg, list)):
            all_values = ''
            for val in reg:
                if (len(all_values) > 0):
                    all_values = all_values + ', '
                all_values = all_values + str(val)
            disc_sys_vars[var] = '{' + all_values + '}'
        else:
            printWarning("WARNING: Unknown possible values for discrete system " + \
                             "variable " + var)

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

    # New variable that identifies in which cell the continuous state is
    newvarname = ''
    newvar_values = []
    if (disc_dynamics.num_regions > 0):
        newvarname = 'cellID' 

        # Make sure that the new variable name does not appear in env_vars or disc_sys_vars
        while (newvarname in env_vars) | (newvarname in disc_sys_vars):
            newvarname = 'c' + newvarname
    
        newvar_values = range(0,disc_dynamics.num_regions)
        newvar = '{'
        for i in newvar_values:
            if (i > 0):
                newvar = newvar + ', ' + str(i)
            else:
                newvar = newvar + str(i)
        newvar = newvarname + ' : ' + newvar + '}'
        f.write('\t\t' + newvar + ';\n')

    # Discrete system variables
    for var, reg in disc_sys_vars.iteritems():
        f.write('\t\t' + var + ' : ' + reg + ';\n')

    f.close()

    ###################################################################################
    # Generate spc file
    ###################################################################################
    assumption = spec[0]
    guarantee = spec[1]

    if (verbose > 0):
        print 'Generating spc file...'

    # Replace any environment variable var in spec with e.var and replace any discrete 
    # system variable var with s.var
    for var in env_vars.keys():
        assumption = re.sub(r'\b'+var+r'\b', 'e.'+var, assumption)
        guarantee = re.sub(r'\b'+var+r'\b', 'e.'+var, guarantee)
    for var in disc_sys_vars.keys():
        assumption = re.sub(r'\b'+var+r'\b', 's.'+var, assumption)
        guarantee = re.sub(r'\b'+var+r'\b', 's.'+var, guarantee)

    # Replace any cont_prop XC by (s.p = P1) | (s.p = P2) | ... | (s.p = Pn) where 
    # P1, ..., Pn are cells in disc_dynamics that satisfy XC
    if (disc_dynamics.list_prop_symbol is not None):
        for propInd, propSymbol in enumerate(disc_dynamics.list_prop_symbol):
            reg = [j for j in range(0,disc_dynamics.num_regions) if \
                       disc_dynamics.list_region[j].list_prop[propInd]]
            newprop = 'FALSE'
            if (len(reg) > 0):
                newprop = '('
                for i, regID in enumerate(reg):
                    if (i > 0):
                        newprop = newprop + ' | '
                    newprop = newprop + '(s.' + newvarname + ' = ' + \
                        str(newvar_values[regID]) + ')'
                newprop = newprop + ')'
            if (verbose > 1):
                print '\t' + propSymbol + ' -> ' + newprop
            assumption = re.sub(r'\b'+propSymbol+r'\b', newprop, assumption)
            guarantee = re.sub(r'\b'+propSymbol+r'\b', newprop, guarantee)

    # Replace symbols for propositions on discrete variables with the actual 
    # propositions
    for propSymbol, prop in disc_props.iteritems():
        for var in env_vars.keys():
            prop = re.sub(r'\b'+var+r'\b', 'e.'+var, prop)
        for var in disc_sys_vars.keys():
            prop = re.sub(r'\b'+var+r'\b', 's.'+var, prop)
        if (verbose > 1):
            print '\t' + propSymbol + ' -> ' + prop
        assumption = re.sub(r'\b'+propSymbol+r'\b', '('+prop+')', assumption)
        guarantee = re.sub(r'\b'+propSymbol+r'\b', '('+prop+')', guarantee)

    # Add assumption on the possible initial state of the system and all the possible 
    # values of the environment
    env_values_formula = ''
    for var, reg in env_vars.iteritems():
        all_values = re.findall('[-+]?\d+', reg)
        if (len(all_values) > 0):
            if (len(env_values_formula) > 0):
                env_values_formula = env_values_formula + ' & '
            current_env_values_formula = ''
            for val in all_values:
                if (len(current_env_values_formula) > 0):
                    current_env_values_formula = current_env_values_formula + ' | '
                current_env_values_formula = current_env_values_formula + 'e.' + \
                    var + '=' + val
            env_values_formula = env_values_formula + '(' + \
                current_env_values_formula + ')'

    disc_sys_values_formula = ''
    for var, reg in disc_sys_vars.iteritems():
        all_values = re.findall('[-+]?\d+', reg)
        if (len(all_values) > 0):
            if (len(disc_sys_values_formula) > 0):
                disc_sys_values_formula = disc_sys_values_formula + ' & '
            current_disc_sys_values_formula = ''
            for val in all_values:
                if (len(current_disc_sys_values_formula) > 0):
                    current_disc_sys_values_formula = current_disc_sys_values_formula + \
                        ' | '
                current_disc_sys_values_formula = current_disc_sys_values_formula + \
                    's.' + var + '=' + val
            disc_sys_values_formula = disc_sys_values_formula + '(' + \
                current_disc_sys_values_formula + ')'

    newvar_values_formula = ''
    for val in newvar_values:
        if (len(newvar_values_formula) > 0):
            newvar_values_formula = newvar_values_formula + ' | '
        newvar_values_formula = newvar_values_formula + '(s.' + newvarname + ' = ' + \
            str(val) + ')'

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
    if (len(newvar_values_formula) > 0):
        if (addAnd):
            f.write(' &\n')
        f.write('-- initial continuous sys states\n')
        f.write('\t(' + newvar_values_formula + ')')
        addAnd = True
    if (len(disc_sys_values_formula) > 0):
        if (addAnd):
            f.write(' &\n')
        f.write('-- initial discrete sys states\n')
        f.write('\t(' + disc_sys_values_formula + ')')
        addAnd = True

    # Transitions
    # For continuous dynamics
    for from_region in range(0,disc_dynamics.num_regions):
        to_regions = [j for j in range(0,disc_dynamics.num_regions) if \
                          disc_dynamics.trans[j][from_region]]
        if (addAnd):
            f.write(' &\n')
        if (from_region == 0):
            f.write('-- transition relations for continuous dynamics\n')
        f.write('\t[]((' + 's.' + newvarname + ' = ' + \
                    str(newvar_values[from_region]) + ') -> next(')
        if (len(to_regions) == 0):
            f.write('FALSE')
        for i, to_region in enumerate(to_regions):
            if (i > 0):
                f.write(' | ')
            f.write('(s.' + newvarname + ' = ' + str(newvar_values[to_region]) + ')')
        f.write('))')
        addAnd = True

    # For discrete transitions
    if (len(disc_sys_values_formula) > 0):
        if (addAnd):
            f.write(' &\n')
        f.write('-- transition relations for discrete sys states\n');
        f.write('\t[](next(' + disc_sys_values_formula + '))')
    
    f.write('\n;')
    f.close()

    return newvarname


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
    - `jtlv_path`: a string containing the full path to the JTLV folder.
    - `heap_size`: a string that specifies java heap size. 
    - `verbose`: an integer that specifies the verbose level. If verbose is set to 0, 
      this function will not print anything on the screen.
    - `pick_sys_init` is a boolean indicating whether the system can pick 
      its initial state (in response to the initial environment state).
    - `file_exist_option`: a string that indicate what to do when the specified aut_file 
      exists. Possible values are: 'a' (ask whether to replace or create a new file), 
      'r' (replace the existing file), 'n' (create a new file).
    - `verbose`: an integer that specifies the verbose level.
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

def synthesize(env_vars={}, disc_sys_vars={}, spec='', disc_props={}, \
                   disc_dynamics=PropPreservingPartition(), \
                   smv_file='specs/spec.smv', spc_file='specs/spec.spc', \
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
    - `disc_sys_vars`: a dictionary {str : str} or {str : list} whose keys are the 
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
    - `verbose`: an integer that specifies the verbose level. If verbose is set to 0,
      this function will not print anything on the screen.
    """
    generateJTLVInput(env_vars=env_vars, disc_sys_vars=disc_sys_vars, spec=spec, \
                          disc_props=disc_props, disc_dynamics=disc_dynamics, \
                          smv_file=smv_file, spc_file=spc_file, \
                          file_exist_option=file_exist_option, verbose=verbose)
    realizability = computeStrategy(smv_file=smv_file, spc_file=spc_file, \
                                        aut_file=aut_file, heap_size=heap_size, \
                                        priority_kind=priority_kind, \
                                        init_option=init_option, \
                                        file_exist_option=file_exist_option, \
                                        verbose=verbose)
    return realizability


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
    - `verbose`: an integer that specifies the verbose level.
    """

    # Check that the input is valid
    if (not os.path.isfile(smv_file)):
        printError("The smv file " + smv_file + " does not exist.")
    if (not os.path.isfile(spc_file)):
        printError("The spc file " + spc_file + " does not exist.")

    if (verbose > 0):
        print 'Creating automaton...\n'

    # Get the right aut_file in case it's not specified.
    if (len(aut_file) == 0 or aut_file.isspace()):
        aut_file = re.sub(r'\.'+'[^'+r'\.'+']+$', '', spc_file)
        aut_file = aut_file + '.aut'
        print('aut file: ' + aut_file)
    if (not os.path.exists(os.path.abspath(os.path.dirname(aut_file)))):
        printWarning('Folder for aut_file ' + aut_file + ' does not exist. Creating...')
        os.mkdir(os.path.abspath(os.path.dirname(aut_file)))

    # Check whether the aut file exists
    if (file_exist_option != 'r'):
        if (os.path.exists(aut_file)):
            printWarning('aut file: ' + aut_file + ' exists.')
            aut_file_exist_option = file_exist_option
            while (aut_file_exist_option.lower() != 'r' and \
                       aut_file_exist_option.lower() != 'n'):
                aut_file_exist_option = raw_input('Replace [r] or create a new aut file [n]: ')
            if (aut_file_exist_option.lower() == 'n'):
                i = 1
                aut_file_part = aut_file.partition('.')
                aut_file = aut_file_part[0] + str(i) + aut_file_part[1] + \
                    aut_file_part[2]
                while (os.path.exists(aut_file)):
                    i = i + 1
                    aut_file = aut_file_part[0] + str(i) + aut_file_part[1] + \
                        aut_file_part[2]
                print('aut file: ' + aut_file)

    # Convert the priority_kind to the corresponding integer
    if (isinstance(priority_kind, str)):
        if (priority_kind == 'ZYX'):
            priority_kind = 3
        elif (priority_kind == 'ZXY'):
            priority_kind = 7
        elif (priority_kind == 'YZX'):
            priority_kind = 11
        elif (priority_kind == 'YXZ'):
            priority_kind = 15
        elif (priority_kind == 'XZY'):
            priority_kind = 19
        elif (priority_kind == 'XYZ'):
            priority_kind = 23
        else:
            printWarning("Unknown priority_kind. Setting it to the default (ZYX)")
            priority_kind = 3
    elif (isinstance(priority_kind, int)):
        if (priority_kind > 0 and priority_kind != 3 and priority_kind != 7 and \
                priority_kind != 11 and priority_kind != 15 and priority_kind != 19 and \
                priority_kind != 23):
            printWarning("Unknown priority_kind. Setting it to the default (ZYX)")
            priority_kind = 3
    else:
        printWarning("Unknown priority_kind. Setting it to the default (ZYX)")
        priority_kind = 3

    # init_option
    if (isinstance(init_option, int)):
        if (init_option < 0 or init_option > 2):
            printWarning("Unknown init_option. Setting it to the default (1)")
            init_option = 1
    else:
        printWarning("Unknown init_option. Setting it to the default (1)")
        init_option = 1

    if (verbose > 0):
        print 'Calling jtlv with the following arguments:'
        print '  heap size: ' + heap_size
        print '  smv file: ' + smv_file
        print '  spc file: ' + spc_file
        print '  aut file: ' + aut_file
        print '  jtlv path: ' + JTLV_PATH
        print '  priority_kind: ' + str(priority_kind) + '\n'

    if (len(JTLV_EXE) > 0):
        jtlv_grgame = os.path.join(JTLV_PATH, JTLV_EXE)
        cmd = subprocess.Popen( \
            ["java", heap_size, "-jar", jtlv_grgame, smv_file, spc_file, aut_file, \
                 str(priority_kind), str(init_option)], \
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)
        if (verbose > 1):
            print "  java", heap_size, "-jar", jtlv_grgame, smv_file, spc_file, \
                aut_file, str(priority_kind), str(init_option)
    else: # For debugging purpose
        classpath = os.path.join(JTLV_PATH, "JTLV") + ":" + \
            os.path.join(JTLV_PATH, "JTLV", "jtlv-prompt1.4.1.jar")
        if (verbose > 1):
            print "  java", heap_size, "-cp", classpath, "GRMain", smv_file, \
                spc_file, aut_file, str(priority_kind), str(init_option)
        cmd = subprocess.Popen( \
            ["java", heap_size, "-cp", classpath, "GRMain", smv_file, spc_file, \
                 aut_file, str(priority_kind), str(init_option)], \
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)

    realizable = False
    for line in cmd.stdout:
        print "\t" + line,
        if "Specification is realizable" in line:
            realizable = True

    cmd.stdout.close()

    if (realizable and priority_kind > 0):
        print("\nAutomaton successfully synthesized.\n")
    elif (priority_kind > 0):
        print("\nERROR: Specification was unrealizable.\n")

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    return realizable


###################################################################

def getCounterExamples(aut_file, verbose=0):
    """Return a list of dictionary, each representing a counter example.

    Input:

    - `aut_file`: a string containing the name of the file containing the
      counter examples generated by JTLV.
    """

    counter_examples = []
    line_found = False
    f = open(aut_file, 'r')
    for line in f:
        if (line.find('The env player can win from states') >= 0):
            line_found = True
            continue
        if (line_found and (len(line) == 0 or line.isspace())):
            line_found = False
        if (line_found):
            counter_ex = dict(re.findall('(\w+):([-+]?\d+)', line))
            for var, val in counter_ex.iteritems():
                counter_ex[var] = int(val)
            counter_examples.append(counter_ex)
            if (verbose > 0):
                print counter_ex
    return counter_examples

###################################################################

# Test case
#  * Default: Use init_option=1 with dynamics
#  * 1: Use init_option=2 with dynamics
#  * 2: Use init_option=0 with dynamics. This makes the spec unrealizable.
#  * 3: Use init_option=1 with no dynamics
if __name__ == "__main__":
    testfile = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'specs', \
                                'test')
    smvfile = testfile + '.smv'
    spcfile = testfile + '.spc'
    autfile = testfile + '.aut'
    print('Testing generateJTLVInput')
#    env_vars = {'park' : 'boolean', 'cellID' : '{0,...,3,4,5}'}
    env_vars = {'park' : 'boolean', 'cellID' : [0,1,2,3,4,5]}
    disc_sys_vars = {'gear' : '{-1...1}'}
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
        newvarname = generateJTLVInput(env_vars=env_vars, disc_sys_vars=disc_sys_vars, \
                                           spec=spec, disc_props=disc_props, \
                                           disc_dynamics=disc_dynamics, \
                                           smv_file=smvfile, spc_file=spcfile, verbose=2)
    else:
        newvarname = generateJTLVInput(env_vars=env_vars, disc_sys_vars=disc_sys_vars, \
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

    realizability = synthesize(env_vars=env_vars, disc_sys_vars=disc_sys_vars, \
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
