#!/usr/bin/env python

""" jtlvint.py --- Interface to the JTLV implementation of GR1 game

About JTLV, see http://jtlv.ysaar.net/

Nok Wongpiromsarn (nok@cds.caltech.edu)
August 3, 2010
"""

import re, os, subprocess, sys
from polyhedron import Vrep, Hrep
from prop2part_ex1 import Region, PropPreservingPartition
from errorprint import printWarning, printError

# Get jtlv_path
JTLV_PATH = os.path.abspath(os.path.dirname(sys.argv[0]))
JTLV_EXE = 'jtlv_grgame.jar'

def setJTLVPath(jtlv_path):
    """Set path to jtlv_grgame.jar.

    - jtlv_path is a string indicating the full path to the JTLV folder
    """
    globals()["JTLV_PATH"] = jtlv_path

def setJTLVExe(jtlv_exe):
    """Set the name of the jtlv executable.

    - jtlv_exe is a string indicating the name of the fatjar containing the jtlv GR1 game implementation
    """
    globals()["JTLV_EXE"] = jtlv_exe

def generateJTLVInput(env_vars={}, disc_sys_vars={}, spec='', cont_props=[], disc_props={}, disc_dynamics=PropPreservingPartition(), smv_file='specs/spec.smv', spc_file='specs/spec.spc', verbose=0):
    """Generate JTLV input files: smv_file and spc_file.

    - env_vars is a dictionary {str : str} or {str : list} whose keys are the names of environment variables 
      and whose values are their possible values, e.g., boolean or {0, 2, ..., 5} or [0, 2, 3, 4, 5].
    - disc_sys_vars is a dictionary {str : str} or {str : list} whose keys are the names of discrete system  
      variables and whose values are their possible values.
    - spec is a list of two strings that represents system specification of the form: assumption -> guarantee; 
      the first string is the assumption and the second string is the guarantee.
    - cont_props is a list of string representing symbols for propositions on continuous variables.
    - disc_props is a dictionary {str : str} whose keys are the symbols for propositions on discrete variables
      and whose values are the actual propositions on discrete variables.
    - disc_dynamics is of type PropPreservingPartition and represents the transition system obtained 
      from the discretization procedure.
    - verbose is an integer that specifies the verbose level. If verbose is set to 0, this function will not
      print anything on the screen.
    """

    # Figure out the names of the smv and spc files
    if (smv_file[-4:] != '.smv'):
        smv_file = smv_file + '.smv'
    if (spc_file[-4:] != '.spc'):
        spc_file = spc_file + '.spc'

    ##################################################################################################
    # Generate smv file
    ##################################################################################################
    # Replace '...' in the range of possible values of env_vars to the actual values and
    # convert a list representation of the range of possible values to a string
    for var, reg in env_vars.iteritems():
        if ('boolean' in reg):
            continue
        elif (isinstance(reg, str)):
            dots_values = re.findall('([-+]?\d+)\s*?,?\s*?'+r'\.\.\.'+'\s*?,?\s*?([-+]?\d+)', reg)
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
                printWarning("WARNING: Unknown possible values for environment variable " + var)
        elif (isinstance(reg, list)):
            all_values = ''
            for val in reg:
                if (len(all_values) > 0):
                    all_values = all_values + ', '
                all_values = all_values + str(val)
            env_vars[var] = '{' + all_values + '}'
        else:
            printWarning("WARNING: Unknown possible values for environment variable " + var)
                

    # Replace '...' in the range of possible values of disc_sys_vars to the actual values and
    # convert a list representation of the range of possible values to a string
    for var, reg in disc_sys_vars.iteritems():
        if ('boolean' in reg):
            continue
        elif (isinstance(reg, str)):
            dots_values = re.findall('([-+]?\d+)\s*?,?\s*?'+r'\.\.\.'+'\s*?,?\s*?([-+]?\d+)', reg)
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
                printWarning("WARNING: Unknown possible values for discrete system variable " + var)
        elif (isinstance(reg, list)):
            all_values = ''
            for val in reg:
                if (len(all_values) > 0):
                    all_values = all_values + ', '
                all_values = all_values + str(val)
            disc_sys_vars[var] = '{' + all_values + '}'
        else:
            printWarning("WARNING: Unknown possible values for discrete system variable " + var)

    # Write smv file
    f = open(smv_file, 'w')
#   The use of 'with' below is a better statement but is not supported in my version of python
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
    
    newvarname = 'cellID' # New variable that identifies in which cell the continuous state is

    # Make sure that the new variable name does not appear in env_vars or disc_sys_vars
    while (newvarname in env_vars) | (newvarname in disc_sys_vars):
        newvarname = 'c' + newvarname

    if (disc_dynamics.num_regions != len(disc_dynamics.list_region)):
        printWarning("WARNING: disc_dynamics.num_regions != len(disc_dynamics.list_regions)")
        disc_dynamics.num_regions = len(disc_dynamics.list_region)
    
    newvar_values = range(0,disc_dynamics.num_regions)
    if (len(newvar_values) > 0):
        newvar = '{'
        for i in newvar_values:
            if (i > 0):
                newvar = newvar + ', ' + str(i)
            else:
                newvar = newvar + str(i)
        newvar = newvarname + ' : ' + newvar + '}'
        f.write('\t\t' + newvar + ';\n')

    for var, reg in disc_sys_vars.iteritems():
        f.write('\t\t' + var + ' : ' + reg + ';\n')

    f.close()

    ##################################################################################################
    # Generate spc file
    ##################################################################################################
    assumption = spec[0]
    guarantee = spec[1]

    if (verbose > 0):
        print 'Generating spc file...'

    # Replace any environment variable var in spec with e.var and replace any discrete system
    # variable var with s.var
    for var in env_vars.keys():
        assumption = re.sub(r'\b'+var+r'\b', 'e.'+var, assumption)
        guarantee = re.sub(r'\b'+var+r'\b', 'e.'+var, guarantee)
    for var in disc_sys_vars.keys():
        assumption = re.sub(r'\b'+var+r'\b', 's.'+var, assumption)
        guarantee = re.sub(r'\b'+var+r'\b', 's.'+var, guarantee)

    # Replace any cont_prop XC by (s.p = P1) | (s.p = P2) | ... | (s.p = Pn) where P1, ..., Pn are cells 
    # in disc_dynamics that satisfy XC
    for propSymbol in cont_props:
        # Determine which regions satisfy this prop
        propInd = [j for j in range(0,disc_dynamics.num_prop) if disc_dynamics.list_prop_symbol[j]==propSymbol]
        if (len(propInd) == 0):
            printWarning('WARNING: Proposition' + propSymbol + ' is not defined in disc_dynamics.list_prop_symbol!')
            continue
        elif (len(propInd) > 1):
            printWarning('WARNING: Proposition' + propSymbol + ' is not unique in disc_dynamics.list_prop_symbol!')

        propInd = propInd[0]
        reg = [j for j in range(0,disc_dynamics.num_regions) if disc_dynamics.list_region[j].list_prop[propInd]]
        newprop = 'FALSE'
        if (len(reg) > 0):
            newprop = '('
            for i, regID in enumerate(reg):
                if (i > 0):
                    newprop = newprop + ' | '
                newprop = newprop + 's.' + newvarname + '=' + str(newvar_values[regID])
            newprop = newprop + ')'
        if (verbose > 1):
            print '\t' + propSymbol + ' -> ' + newprop
        assumption = re.sub(r'\b'+propSymbol+r'\b', newprop, assumption)
        guarantee = re.sub(r'\b'+propSymbol+r'\b', newprop, guarantee)

    # Replace symbols for propositions on discrete variables with the actual propositions
    for propSymbol, prop in disc_props.iteritems():
        for var in env_vars.keys():
            prop = re.sub(r'\b'+var+r'\b', 'e.'+var, prop)
        for var in disc_sys_vars.keys():
            prop = re.sub(r'\b'+var+r'\b', 's.'+var, prop)
        if (verbose > 1):
            print '\t' + propSymbol + ' -> ' + prop
        assumption = re.sub(r'\b'+propSymbol+r'\b', '('+prop+')', assumption)
        guarantee = re.sub(r'\b'+propSymbol+r'\b', '('+prop+')', guarantee)

    # Add assumption on the possible initial state of the system and all the possible values of
    # the environment
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
                current_env_values_formula = current_env_values_formula + 'e.' + var + '=' + val
            env_values_formula = env_values_formula + '(' + current_env_values_formula + ')'

    disc_sys_values_formula = ''
    for var, reg in disc_sys_vars.iteritems():
        all_values = re.findall('[-+]?\d+', reg)
        if (len(all_values) > 0):
            if (len(disc_sys_values_formula) > 0):
                disc_sys_values_formula = disc_sys_values_formula + ' & '
            current_disc_sys_values_formula = ''
            for val in all_values:
                if (len(current_disc_sys_values_formula) > 0):
                    current_disc_sys_values_formula = current_disc_sys_values_formula + ' | '
                current_disc_sys_values_formula = current_disc_sys_values_formula + 's.' + var + '=' + val
            disc_sys_values_formula = disc_sys_values_formula + '(' + current_disc_sys_values_formula + ')'

    newvar_values_formula = ''
    for val in newvar_values:
        if (len(newvar_values_formula) > 0):
            newvar_values_formula = newvar_values_formula + ' | '
        newvar_values_formula = newvar_values_formula + 's.' + newvarname + '=' + str(val)

    addAnd = False
    if (len(env_values_formula) > 0 or len(disc_sys_values_formula) > 0 or len(newvar_values_formula) > 0):
        if (len(assumption) > 0):
            assumption = '-- original assumptions\n\t' + assumption + ' &\n'
        assumption = assumption + '-- initial states\n'
    if (len(newvar_values_formula) > 0):
        assumption = assumption + '\t(' + newvar_values_formula + ')'
        addAnd = True
    if (len(disc_sys_values_formula) > 0):
        if (addAnd):
            assumption = assumption + ' &\n'
        assumption = assumption + '\t(' + disc_sys_values_formula + ')'
        addAnd = True
    if (len(env_values_formula) > 0):
        if (addAnd):
            assumption = assumption + ' &\n'
        assumption = assumption + '\t(' + env_values_formula + ')'
        assumption = assumption + ' &\n-- possible values of environment variables\n'
        assumption = assumption + '\t[](next(' + env_values_formula + '))'
#        assumption = assumption + '\t[]((' + env_values_formula + ') -> next(' + env_values_formula + '))'

    f = open(spc_file, 'w')
#   The use of 'with' below is a better statement but is not supported in my version of python (2.5)
#   with open(spc_file, 'w') as f:

    # Assumption
    f.write('LTLSPEC\n')
    f.write(assumption)
    f.write('\n;\n')

    # Guarantee
    f.write('\nLTLSPEC\n')
    formula_added = False
    if (len(guarantee) > 0 and not(guarantee.isspace())):
        f.write('-- original requirements\n\t' + guarantee)
        formula_added = True

    # Transitions
    # For continuous dynamics
    for from_region in range(0,disc_dynamics.num_regions):
        to_regions = [j for j in range(0,disc_dynamics.num_regions) if disc_dynamics.adj[j][from_region]]
        if (formula_added):
            f.write(' &\n')
        if (from_region == 0):
            f.write('-- transition relations for continuous dynamics\n')
        f.write('\t[]((' + 's.' + newvarname + '=' + str(newvar_values[from_region]) + ') -> next(')
        if (len(to_regions) == 0):
            f.write('FALSE')
        for i, to_region in enumerate(to_regions):
            if (i > 0):
                f.write(' | ')
            f.write('s.' + newvarname + '=' + str(newvar_values[to_region]))
        f.write('))')
        formula_added = True

    # For discrete transitions
    if (len(disc_sys_values_formula) > 0):
        if (formula_added):
            f.write(' &\n')
        f.write('-- transition relations for discrete system state\n');
        f.write('\t[](next(' + disc_sys_values_formula + '))')
        
#    add_desc = True
#    disc_sys_values_formula
#    for prop in new_disc_props.values():
#        if (formula_added):
#            f.write(' &\n')
#        if (add_desc):
#            f.write('-- transition relations for discrete system state\n');
#            add_desc = False
#        f.write('\t[](next((' + prop + ') | !(' + prop + ')))')
#        formula_added = True
    
    f.write('\n;')
    f.close()

    return newvarname


###################################################################

# Part of the following function is extracted and modified from the ltlmop toolbox
def checkRealizability(smv_file='', spc_file='', aut_file='', heap_size='-Xmx128m', verbose=0):
    """Determine whether the spec in smv_file and spc_file is realizable without extracting an automaton.

    - smv_file is a string containing the name of the smv file.
    - spc_file is a string containing the name of the spc file.
    - aut_file is a string containing the name of the file containing the output of JTLV
      (e.g. an initial state starting from which the spec is cannot be satisfied).
    - jtlv_path is a string containing the full path to the JTLV folder.
    - heap_size is a string that specifies java heap size. 
    - verbose is an integer that specifies the verbose level. If verbose is set to 0, this function will not
      print anything on the screen.
    """
    realizable = computeStrategy(smv_file=smv_file, spc_file=spc_file, aut_file=aut_file, heap_size=heap_size, priority_kind=-1, verbose=verbose)
    return realizable


###################################################################

# Part of the following function is extracted and modified from the ltlmop toolbox
def computeStrategy(smv_file='', spc_file='', aut_file='', heap_size='-Xmx128m', priority_kind=3, verbose=0):
    """Compute an automaton satisfying the spec in smv_file and spc_file and store in aut_file, return the realizability of the spec.

    - smv_file is a string containing the name of the smv file.
    - spc_file is a string containing the name of the spc file.
    - aut_file is a string containing the name of the file containing the resulting automaton.
    - heap_size is a string that specifies java heap size. 
    - verbose is an integer that specifies the verbose level. If verbose is set to 0, this function will not
      print anything on the screen.
    - priority_kind is an integer that specifies the type of priority used in extracting the automaton. 
      Possible priorities are: 
        * 3 - Z Y X
        * 7 - Z X Y
        * 11 - Y Z X
        * 15 - Y X Z
        * 19 - X Z Y
        * 23 - X Y Z
      Here X means that the controller tries to disqualify one of the environment assumptions, 
      Y means that the controller tries to advance with a finite path to somewhere, and
      Z means that the controller tries to satisfy one of his guarantees.
    """
    if (verbose > 0):
        print 'Creating automaton...\n'
    # Get the right aut_file in case it's not specified.
    if (len(aut_file) == 0 or aut_file.isspace()):
        aut_file = re.sub(r'\.'+'[^'+r'\.'+']+$', '', spc_file)
        aut_file = aut_file + '.aut'
    if (verbose > 0):
        print 'Calling jtlv with the following arguments:'
        print '  heap size: ' + heap_size
        print '  smv file: ' + smv_file
        print '  spc file: ' + spc_file
        print '  aut file: ' + aut_file
        print '  jtlv path: ' + JTLV_PATH
        print '  priority_kind: ' + str(priority_kind)

    if (len(JTLV_EXE) > 0): # Use fatjar
        if (verbose > 0):
            print "Using fatjar"
        jtlv_grgame = JTLV_PATH + '/' + JTLV_EXE
        cmd = subprocess.Popen(["java", heap_size, "-jar", jtlv_grgame, smv_file, spc_file, aut_file, str(priority_kind)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)
        if (verbose > 1):
            print "  java", heap_size, "-jar", jtlv_grgame, smv_file, spc_file, aut_file, str(priority_kind)
    else:
        if (verbose > 0):
            print "NOT using fatjar"
        classpath = os.path.join(JTLV_PATH, "JTLV") + ":" + os.path.join(JTLV_PATH, "JTLV", "jtlv-prompt1.4.1.jar")
        if (verbose > 1):
            print "  java", heap_size, "-cp", classpath, "GRMain", smv_file, spc_file, aut_file, str(priority_kind)
        cmd = subprocess.Popen(["java", heap_size, "-cp", classpath, "GRMain", smv_file, spc_file, aut_file, str(priority_kind)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)

    realizable = False
    for line in cmd.stdout:
        print("\t"+line)
        if "Specification is realizable" in line:
            realizable = True

    cmd.stdout.close()

    if (realizable and priority_kind > 0):
        print("Automaton successfully synthesized.\n")
    elif (priority_kind > 0):
        print("ERROR: Specification was unrealizable.\n")

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    return realizable


###################################################################

# Test case
if __name__ == "__main__":
    testfile = 'specs/test'
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
    region2 = Region('p2', [0, 0, 1, 0, 0])
    region3 = Region('p3', [0, 0, 0, 1, 0])
    region4 = Region('p4', [0, 0, 0, 0, 1])
    region5 = Region('p5', [1, 1, 1, 1, 1])
    disc_dynamics.list_region = [region0, region1, region2, region3, region4, region5]
    disc_dynamics.num_regions = len(disc_dynamics.list_region)
    disc_dynamics.adj = [[1, 1, 0, 1, 0, 0], [1, 1, 1, 0, 1, 0], [0, 1, 1, 0, 0, 1], [1, 0, 0, 1, 1, 0], [0, 1, 0, 1, 1, 1], [0, 0, 1, 0, 1, 1]]
    disc_dynamics.list_prop_symbol = cont_props
    disc_dynamics.num_prop = len(disc_dynamics.list_prop_symbol)
    disc_props = {'Park' : 'park', 'X0d' : 'cellID=0', 'X1d' : 'cellID=1', 'X2d' : 'cellID=2', 'X3d' : 'cellID=3', 'X4d' : 'gear = 0', 'X5d' : 'gear = 1'}
    assumption = '[]<>(!park) & []<>(!X0d) & []<>(Park -> !X0d)'  # For realizable spec (default case)

    if ('2' in sys.argv): # For unrealizable spec
        assumption = ''  

    guarantee = '[]<>(X0d -> X0) & []<>X1 & []<>(Park -> X4)'
    spec = [assumption, guarantee]

    if ('3' in sys.argv): # For spec with no dynamics
        spec[1] = '[]<>(X0d -> X5d)'  
        newvarname = generateJTLVInput(env_vars=env_vars, disc_sys_vars=disc_sys_vars, spec=spec, cont_props=[], disc_props=disc_props, disc_dynamics=PropPreservingPartition(), smv_file=smvfile, spc_file=spcfile, verbose=2)
    else:
        newvarname = generateJTLVInput(env_vars=env_vars, disc_sys_vars=disc_sys_vars, spec=spec, cont_props=cont_props, disc_props=disc_props, disc_dynamics=disc_dynamics, smv_file=smvfile, spc_file=spcfile, verbose=2)
    print('DONE')
    print('================================\n')

    ####################################

    print('Testing checkRealizability')
    realizability = checkRealizability(smv_file=smvfile, spc_file=spcfile, aut_file='', heap_size='-Xmx128m', verbose=3)
    print realizability
    print('DONE')
    print('================================\n')

    ####################################

    print('Testing computeStrategy')
    realizability = computeStrategy(smv_file=smvfile, spc_file=spcfile, aut_file='', heap_size='-Xmx128m', priority_kind=3, verbose=3)
    print realizability
    print('DONE')
    print('================================\n')

