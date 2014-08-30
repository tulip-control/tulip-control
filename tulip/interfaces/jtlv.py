# Copyright (c) 2012-2014 by California Institute of Technology
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
Interface to the JTLV implementation of GR(1) synthesis

Relevant links:
  - U{JTLV<http://jtlv.ysaar.net/>}
"""
import logging
logger = logging.getLogger(__name__)

import os, re, subprocess, tempfile, textwrap
import warnings

import networkx as nx

from tulip.spec.parser import parse

JTLV_PATH = os.path.abspath(os.path.dirname(__file__))
JTLV_EXE = 'jtlv_grgame.jar'

DEBUG_SMV_FILE = 'smv.txt'
DEBUG_LTL_FILE = 'ltl.txt'
DEBUG_AUT_FILE = 'aut.txt'

def check_realizable(spec, heap_size='-Xmx128m', priority_kind=-1,
                     init_option=1):
    """Decide realizability of specification defined by given GRSpec object.

    ...for standalone use

    @return: True if realizable, False if not, or an error occurs.
    """
    fSMV, fLTL, fAUT = create_files(spec)
    realizable = solve_game(spec, fSMV, fLTL, fAUT, heap_size,
                            priority_kind, init_option)
    os.unlink(fSMV)
    os.unlink(fLTL)
    os.unlink(fAUT)
    return realizable

def solve_game(
    spec, fSMV, fLTL, fAUT, heap_size='-Xmx128m',
    priority_kind=3, init_option=1
):
    """Decide realizability of specification.

    Return True if realizable, False if not, or an error occurs.
    Automaton is extracted for file names by fAUT, unless priority_kind == -1
    

    @type spec: L{GRSpec}
    @param fSMV, fLTL, fAUT: file name for use with JTLV.  This
        enables synthesize() to access them later on.
    @param heap_size: a string that specifies java heap size.

    @param priority_kind: a string of length 3 or an integer that specifies
        the type of priority used in extracting the
        automaton. Possible values of C{priority_kind} are:

            - -1 - No Automaton
            - 3 - 'ZYX'
            - 7 - 'ZXY'
            - 11 - 'YZX'
            - 15 - 'YXZ'
            - 19 - 'XZY'
            - 23 - 'XYZ'

        Here X means that the controller tries to disqualify one of
        the environment assumptions, Y means that the controller tries
        to advance with a finite path to somewhere, and Z means that
        the controller tries to satisfy one of his guarantees.

    @param init_option: an integer that specifies how to handle the
        initial state of the system. Possible values of C{init_option}
        are:

            - 0 - The system has to be able to handle all
                the possible initial system
                states specified on the guarantee side of the specification.
                
            - 1 (default) - The system can choose its initial state,
                in response to the initial
                environment state. For each initial environment state,
                the resulting automaton contains
                exactly one initial system state,
                starting from which the system can satisfy the specification.
                
            - 2 - The system can choose its initial state,
                in response to the initial environment state.
                For each initial environment state, the resulting
                automaton contain all the possible initial system states,
                starting from which the system can satisfy the specification.
    """
    priority_kind = get_priority(priority_kind)
    
    # init_option
    if (isinstance(init_option, int)):
        if (init_option < 0 or init_option > 2):
            warnings.warn("Unknown init_option. Setting it to the default (1)")
            init_option = 1
    else:
        warnings.warn("Unknown init_option. Setting it to the default (1)")
        init_option = 1

    
    call_JTLV(heap_size, fSMV, fLTL, fAUT, priority_kind, init_option)
    
    realizable = False
    
    f = open(fAUT, 'r')
    for line in f:
        if ("Specification is realizable" in line):
            realizable = True
            break
        elif ("Specification is unrealizable" in line):
            realizable = False
            break
        
    if (realizable and priority_kind > 0):
        print("\nAutomaton successfully synthesized.\n")
    elif (priority_kind > 0):
        print("\nERROR: Specification was unrealizable.\n")

    return realizable

def synthesize(
    spec, heap_size='-Xmx128m', priority_kind = 3,
    init_option = 1
):
    """Synthesize a strategy satisfying the specification.

    Arguments are described in documentation for L{solve_game}.
    
    @return: Return strategy as instance of L{MealyMachine}, or a list
        of counter-examples as returned by L{get_counterexamples}.
    """
    fSMV, fLTL, fAUT = create_files(spec)

    realizable = solve_game(spec, fSMV, fLTL, fAUT, heap_size,
                            priority_kind, init_option)

    # Build Automaton
    if (not realizable):
        counter_examples = get_counterexamples(fAUT)
        os.unlink(fSMV)
        os.unlink(fLTL)
        os.unlink(fAUT)
        return counter_examples
    else: 
        aut = load_file(fAUT, spec)
        os.unlink(fSMV)
        os.unlink(fLTL)
        os.unlink(fAUT)
        return aut

def create_files(spec):
    """Create temporary files for read/write by JTLV."""
    fSMV = tempfile.NamedTemporaryFile(delete=False,suffix="smv")
    fSMV.write(generate_JTLV_SMV(spec))
    fSMV.close()
    
    fLTL = tempfile.NamedTemporaryFile(delete=False,suffix="ltl")
    fLTL.write(generate_JTLV_LTL(spec))
    fLTL.close()
    
    
    fAUT = tempfile.NamedTemporaryFile(delete=False)
    fAUT.close()
    return fSMV.name, fLTL.name, fAUT.name

def get_priority(priority_kind):
    """Validate and convert priority_kind to the corresponding integer.

    @type priority_kind: str or int
    @param priority_kind: a string of length 3 or integer as may be
        used when invoking L{solve_game}.  Check documentation there
        for possible values.

    @rtype: int
    @return: if given priority_kind is permissible, then return
        integer representation of it.  Else, return default ("ZYX").
    """
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
            warnings.warn('Unknown priority_kind.' +
                'Setting it to the default (ZYX)')
            priority_kind = 3
    elif (isinstance(priority_kind, int)):
        if (priority_kind > 0 and priority_kind != 3 and \
            priority_kind != 7 and priority_kind != 11 and \
            priority_kind != 15 and priority_kind != 19 and \
            priority_kind != 23
        ):
            warnings.warn("Unknown priority_kind." +
                " Setting it to the default (ZYX)")
            priority_kind = 3
    else:
        warnings.warn("Unknown priority_kind. Setting it to the default (ZYX)")
        priority_kind = 3
    return priority_kind

def call_JTLV(heap_size, fSMV, fLTL, fAUT, priority_kind, init_option):
    """Subprocess calls to JTLV.
    """
    logger.info('Calling jtlv with the following arguments:')
    logger.info('  heap size: ' + heap_size)
    logger.info('  jtlv path: ' + JTLV_PATH)
    logger.info('  priority_kind: ' + str(priority_kind) + '\n')

    if (len(JTLV_EXE) > 0):
        jtlv_grgame = os.path.join(JTLV_PATH, JTLV_EXE)
        
        # debugging log
        if logger.getEffectiveLevel() <= logging.DEBUG:
            logger.debug(jtlv_grgame)
            logger.debug(
                '  java ' + str(heap_size) +
                ' -jar ' + str(jtlv_grgame) +
                ' ' + str(fSMV) +
                ' ' + str(fLTL) +
                ' ' + str(fAUT) +
                ' ' + str(priority_kind) +
                ' ' + str(init_option)
            )
            
            # besides dumping to debug logging stream,
            # also copy files to ease manual debugging
            import shutil
            
            shutil.copyfile(fSMV, DEBUG_SMV_FILE)
            shutil.copyfile(fLTL, DEBUG_LTL_FILE)
            shutil.copyfile(fAUT, DEBUG_AUT_FILE)
        
        subprocess.call( \
            ["java", heap_size, "-jar", jtlv_grgame, fSMV, fLTL, fAUT, \
                 str(priority_kind), str(init_option)])
    else: # For debugging purpose
        classpath = os.path.join(JTLV_PATH, "JTLV") + ":" + \
            os.path.join(JTLV_PATH, "JTLV", "jtlv-prompt1.4.1.jar")
        logger.debug(
            '  java ' + str(heap_size) +
            ' -cp ' + str(classpath) +
            ' GRMain ' + str(fSMV) +
            ' ' + str(fLTL) +
            ' ' + str(fAUT) +
            ' ' + str(priority_kind) +
            ' ' + str(init_option)
        )
        subprocess.call([
            "java", heap_size, "-cp", classpath, "GRMain", fSMV, fLTL,
            fAUT, str(priority_kind), str(init_option)
        ])


def canon_to_jtlv_domain(dom):
    """Convert an LTL or GRSpec variable domain to JTLV style.

    @param dom: a variable domain, as would be provided by the values
    in the output_variables attribute of L{spec.form.LTL} or sys_vars
    of L{spec.form.GRSpec}.

    @rtype: str
    @return: variable domain string, ready for use in an SMV file, as
        expected by the JTLV solver.
    """
    if dom == "boolean":
        return dom
    elif isinstance(dom, tuple) and len(dom) == 2:
        return "{"+", ".join([str(i) for i in range(dom[0], dom[1]+1)])+"}"
    else:
        raise ValueError("Unrecognized domain type: "+str(dom))

def generate_JTLV_SMV(spec):
    """Return the SMV module definitions needed by JTLV.

    Raises exception if malformed GRSpec object is detected.

    @type spec: L{GRSpec}

    @rtype: str
    @return: string conforming to the SMV file format that JTLV expects.
    """
    smv = ""

    # Write the header
    smv+=textwrap.dedent("""

    MODULE main
        VAR
            e : env();
            s : sys();
    """);

    # Define env vars
    smv+=(textwrap.dedent("""
    MODULE env -- inputs
        VAR
    """))
    for var, dom in spec.env_vars.items():
        smv+= '\t\t'
        smv+= var
        smv+= ' : '+canon_to_jtlv_domain(dom)+';\n'

    
    # Define sys vars
    smv+=(textwrap.dedent("""
    MODULE sys -- outputs
        VAR
    """))
    for var, dom in spec.sys_vars.items():
        smv+= '\t\t'
        smv+= var
        smv+= ' : '+canon_to_jtlv_domain(dom)+';\n'
    
    logger.debug(smv)
    return smv

def generate_JTLV_LTL(spec):
    """Return the LTLSPEC for JTLV.

    It takes as input a GRSpec object.
    """
    formula = spec.to_canon()
    parse(formula)  # Raises exception if syntax error

    specLTL = spec.to_jtlv()
    logger.debug(''.join([str(x) for x in specLTL]) )
    
    assumption = specLTL[0]
    guarantee = specLTL[1]
    
    # Replace any environment variable var in spec with e.var and replace any 
    # system variable var with s.var
    for var in spec.env_vars.keys():
        assumption = re.sub(r'\b'+var+r'\b', 'e.'+var, assumption)
        guarantee = re.sub(r'\b'+var+r'\b', 'e.'+var, guarantee)
    for var in spec.sys_vars.keys():
        assumption = re.sub(r'\b'+var+r'\b', 's.'+var, assumption)
        guarantee = re.sub(r'\b'+var+r'\b', 's.'+var, guarantee)

    # Assumption
    ltl = 'LTLSPEC\n(\n'
    if assumption:
        ltl += assumption
    else:
        ltl += "TRUE"
    ltl += '\n);\n'

    # Guarantee
    ltl += '\nLTLSPEC\n(\n'
    if guarantee:
        ltl += guarantee
    else:
        ltl += "TRUE"
    ltl += '\n);'    

    return ltl

def load_file(aut_file, spec):
    """Construct a Mealy Machine from an aut_file.

    @param aut_file: the name of the text file containing the
        automaton, or an (open) file-like object.
    @type spec: L{GRSpec}

    @rtype: L{MealyMachine}
    """
    if isinstance(aut_file, str):
        f = open(aut_file, 'r')
    else:
        # assume aut_file behaves as file object
        f = aut_file
    
    g = nx.DiGraph()
    
    varnames = set(spec.sys_vars)
    varnames.update(spec.env_vars)
    
    for line in f:
        # parse states
        if line.find('State ') >= 0:
            state_id = re.search('State (\d+)', line)
            state_id = int(state_id.group(1))
            
            state = dict(re.findall('(\w+):(\w+)', line))
            for var, val in state.iteritems():
                try:
                    state[var] = int(val)
                except:
                    state[var] = val
                
                if varnames:
                    if not var in varnames:
                        logger.error('Unknown variable ' + var)
            
            for var in varnames:
                if not var in state.keys():
                    logger.error('Variable ' + var + ' not assigned')

        # parse transitions
        if line.find('successors') >= 0:
            succ = [int(x) for x in re.findall(' (\d+)', line)]
            
            g.add_node(state_id, state=state)
            g.add_edges_from([(state_id, v) for v in succ])
    
    return g
            
def get_counterexamples(aut_file):
    """Return a list of dictionaries, each representing a counter example.

    @param aut_file: a string containing the name of the file
        containing the counter examples generated by JTLV.
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
            logger.info(counter_ex)
    return counter_examples


def remove_comments(spec):
    """Remove comment lines from string."""
    speclines = spec.split('\n')
    newspec = ''
    for line in speclines:
        if not '--' in line:
            newspec+=line+'\n'
    return newspec

def check_vars(varNames):
    """Complain if any variable name is a number or not a string.
    """
    for item in varNames:
        # Check that the vars are strings
        if type(item) != str:
            logger.error("Prop " + str(item) + " is invalid")
            return False

        # Check that the keys are not numbers
        try:
            int(item)
            float(item)
            logger.errror("Prop " + str(item) + " is invalid")
            return False
        except ValueError:
            continue
    return True

def check_spec(spec, varNames):
    """Verify that all non-operators in "spec" are in the list of vars.
    """
    # Replace all special characters with whitespace
    special_characters = ["next(", "[]", "<>", "->", "&", "|", "!",  \
      "(", ")", "\n", "<", ">", "<=", ">=", "<->", "\t", "="]
    for word in special_characters:
        spec = spec.replace(word, "")
        
    # Now, replace all variable names and values with whitespace as well.
    for value in varNames:
        if isinstance(value, (list, tuple)):
            for individual_value in value:
                spec = spec.replace(str(individual_value), "")
        else:
            spec = spec.replace(value, "")

    # Remove all instances of "true" and "false"
    spec = spec.lower()
        
    spec = spec.replace("true", "")
    spec = spec.replace("false", "")

    # Make sure that the resulting string is empty
    spec = spec.split()
        
    return not spec
