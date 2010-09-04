#!/usr/bin/env python

""" 
---------------------------------------------------------------------
Grgameint Module --- Interface to the JTLV implementation of GR1 game
---------------------------------------------------------------------

About JTLV, see http://jtlv.ysaar.net/

Nok Wongpiromsarn (nok@cds.caltech.edu)

:Date: August 20, 2010
:Version: 0.1.0
"""

import re, os, subprocess, sys
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

###################################################################

def setJTLVExe(jtlv_exe):
    """Set the name of the jtlv executable.

    Input:

    - `jtlv_exe`: a string indicating the name of the executable jar containing the 
      jtlv GR1 game implementation.
    """
    globals()["JTLV_EXE"] = jtlv_exe

###################################################################

def solveGame(smv_file, spc_file, aut_file='', heap_size='-Xmx128m', \
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
        if (verbose > 1):
            print "  java", heap_size, "-jar", jtlv_grgame, smv_file, spc_file, \
                aut_file, str(priority_kind), str(init_option)
        cmd = subprocess.call( \
            ["java", heap_size, "-jar", jtlv_grgame, smv_file, spc_file, aut_file, \
                 str(priority_kind), str(init_option)])
    else: # For debugging purpose
        classpath = os.path.join(JTLV_PATH, "JTLV") + ":" + \
            os.path.join(JTLV_PATH, "JTLV", "jtlv-prompt1.4.1.jar")
        if (verbose > 1):
            print "  java", heap_size, "-cp", classpath, "GRMain", smv_file, \
                spc_file, aut_file, str(priority_kind), str(init_option)
        cmd = subprocess.call( \
            ["java", heap_size, "-cp", classpath, "GRMain", smv_file, spc_file, \
                 aut_file, str(priority_kind), str(init_option)])
#         cmd = subprocess.Popen( \
#             ["java", heap_size, "-cp", classpath, "GRMain", smv_file, spc_file, \
#                  aut_file, str(priority_kind), str(init_option)], \
#                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)

    realizable = False
    if (os.path.isfile(aut_file)):
        f = open(aut_file, 'r')
        for line in f:
            if ("Specification is realizable" in line):
                realizable = True
                break
            elif ("Specification is unrealizable" in line):
                realizable = False
                break
    else:
        printError("Cannot write to aut file. Running synthesis again...")
        cmd = subprocess.Popen( \
            ["java", heap_size, "-jar", jtlv_grgame, smv_file, spc_file, aut_file, \
                 str(priority_kind), str(init_option)], \
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)
        for line in cmd.stdout:
            print "\t" + line,
            if "Specification is realizable" in line:
                realizable = True
        cmd.stdout.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    if (realizable and priority_kind > 0):
        print("\nAutomaton successfully synthesized.\n")
    elif (priority_kind > 0):
        print("\nERROR: Specification was unrealizable.\n")

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
