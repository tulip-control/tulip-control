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
Interface to the slugs implementation of GR(1) synthesis
Author: Vasu Raman
Date: August 8, 2014

Relevant links:
  - U{slugs<https://github.com/LTLMoP/slugs>}
"""
import sys

import math,copy

import logging
logger = logging.getLogger(__name__)

import itertools, os, re, subprocess, tempfile, textwrap
import warnings
from collections import OrderedDict

from tulip.transys.machines import create_machine_ports
from tulip import transys
from tulip.spec.parser import parse

SLUGS_PATH = os.path.abspath(os.path.dirname(__file__))
SLUGS_EXE = 'slugs/src/slugs'

def check_realizable(spec, options=''):
    """Decide realizability of specification defined by given GRSpec object.

    ...for standalone use

    @return: True if realizable, False if not, or an error occurs.
    """
    fSlugs, fAUT = create_files(spec)

    realizable = solve_game(spec, fSlugs, fAUT, '--onlyRealizability'+options)
    os.unlink(fSlugs)
    os.unlink(fAUT)

    return realizable

def solve_game(
    spec, fSlugs, fAUT, options
):
    """Decide realizability of specification.

    Return True if realizable, False if not, or an error occurs.
    Automaton is extracted for file names by fAUT.


    @type spec: L{GRSpec}
    @param fSlugs, fAUT: file name for use with slugs format. This
        enables synthesize() to access them later on.

    @param options: a string containing options for syntheiss with slugs.
        Possible values of C{priority_kind} are:


    """

    call_slugs(fSlugs, fAUT, options)

    realizable = False

    f = open(fAUT, 'r')

    for line in f:
        if ("Specification is realizable" in line):
            realizable = True
            break
        elif ("Specification is unrealizable" in line):
            realizable = False
            break

    if '--onlyRealizability' not in options:
        if (realizable):
            print("\nAutomaton successfully synthesized.\n")
        else:
            print("\nERROR: Specification was unrealizable.\n")

    return realizable

def synthesize(
    spec, options=''
):
    """Synthesize a strategy satisfying the specification.

    Arguments are described in documentation for L{solve_game}.

    @return: Return strategy as instance of L{MealyMachine}, or a list
        of counter-examples as returned by L{get_counterexamples}.
    """
    realizable = check_realizable(spec, options)

    fSlugs, fAUT = create_files(spec)

    # Build Automaton
    if (not realizable):
        solve_game(spec, fSlugs, fAUT, "--counterStrategy"+options)
        #counter_examples = get_counterexamples(fAUT)
        counter_aut = load_file(fAUT, spec, False)
        #os.unlink(fAUT)
        os.unlink(fSlugs)
        return fAUT
    else:
        solve_game(spec, fSlugs, fAUT, options)
        aut = load_file(fAUT, spec)
        #os.unlink(fAUT)
        os.unlink(fSlugs)
        return aut

def create_files(spec):
    """Create temporary files for read/write by slugs."""
    fStructuredSlugs = tempfile.NamedTemporaryFile(delete=False,suffix="structuredslugs")
    fStructuredSlugs.write(generate_StructuredSLUGS(spec))
    fStructuredSlugs.close()

    structured_slugs_compiler_path = os.path.join(SLUGS_PATH, "slugs", "tools", "StructuredSlugsParser")
    sys.path.insert(0, structured_slugs_compiler_path)
    from compiler import performConversion

    # Call the conversion script
    fSlugs = tempfile.NamedTemporaryFile(delete=False,suffix="slugsin")
    sys.stdout = fSlugs
    performConversion(fStructuredSlugs.name, True)
    fSlugs.close()
    sys.stdout = sys.__stdout__


    os.unlink(fStructuredSlugs.name)

    #fAUT = tempfile.NamedTemporaryFile(delete=False,suffix="aut")
    fAUT = open(os.path.join(os.path.abspath(os.path.dirname(__file__)),"counter.aut"),'w+')
    fAUT.close()

    return fSlugs.name, fAUT.name


def call_slugs(fSlugs, fAUT, options):
    """Subprocess calls to slugs
    """
    logger.info('Calling slugs with the following arguments:')
    logger.info('  slugs path: ' + SLUGS_PATH)
    logger.info('  options: ' + options)


    if (len(SLUGS_EXE) > 0):
        slugs_grgame = os.path.join(SLUGS_PATH, SLUGS_EXE)
        logger.debug(slugs_grgame)
        slugs_cmd = [str(slugs_grgame) ,\
            ' ' , options ,\
            ' \"' , str(fSlugs),\
            '\" > ' , str(fAUT), ' 2>&1']
        logger.debug(
            "".join(slugs_cmd)
        )
        os.system("".join(slugs_cmd))
    else: # For debugging purpose
        slugs_grgame = os.path.join(SLUGS_PATH, "slugs/src/slugs")
        slugs_cmd = [str(slugs_grgame),\
            ' ' , options, \
            ' ' , str(fSlugs) ,\
            ' > ' , str(fAUT)]
        logger.debug(
            "".join(slugs_cmd)
        )
        os.system("".join(slugs_cmd))



def format_For_slugs(spec):
    spec = re.sub(r'X\((\w+)\)', r"\1'", spec)
    return spec

def get_vars_for_slugs(vardict):
    output = ""
    for variable, domain in vardict.items():
        if domain == "boolean":
            output += "\n"+variable
        elif isinstance(domain, tuple) and len(domain) == 2:
            output += "\n"+variable+": "+str(domain[0])+"..."+str(domain[1])
        else:
            raise ValueError("Domain type unsupported by slugs: " +
                str(domain))
    return output

def generate_StructuredSLUGS(spec):
    """Return the slugs spec

    It takes as input a GRSpec object.
    """
    specStr = ['[INPUT]', '[OUTPUT]', '[ENV_TRANS]', '[ENV_LIVENESS]', '[ENV_INIT]', '[SYS_TRANS]', '[SYS_LIVENESS]', '[SYS_INIT]']

    specStr[0] += get_vars_for_slugs(spec.env_vars)
    specStr[1] +=get_vars_for_slugs(spec.sys_vars)

    first = True
    for env_init in spec.env_init:
        env_init = format_For_slugs(env_init)
        if (len(env_init) > 0):
            if first:
                specStr[4] += ' \n ' + env_init
                first = False
            else:
                specStr[4] += ' & ' + env_init
    for env_safety in spec.env_safety:
        env_safety = format_For_slugs(env_safety)
        if (len(env_safety) > 0):
            specStr[2] += '\n'+env_safety
    for env_prog in spec.env_prog:
        env_prog = format_For_slugs(env_prog)
        if (len(env_prog) > 0):
            specStr[3] += '\n'+env_prog

    first = True
    for sys_init in spec.sys_init:
        sys_init = format_For_slugs(sys_init)
        if (len(sys_init) > 0):
            if first:
                specStr[7] += ' \n ' + sys_init
                first = False
            else:
                specStr[7] += ' & ' + sys_init
    for sys_safety in spec.sys_safety:
        sys_safety = format_For_slugs(sys_safety)
        if (len(sys_safety) > 0):
            specStr[5] += '\n'+sys_safety
    for sys_prog in spec.sys_prog:
        sys_prog = format_For_slugs(sys_prog)
        if (len(sys_prog) > 0):
            specStr[6] += '\n'+sys_prog
    return '\n\n'.join(specStr)


def load_file(aut_file, spec, realizable=True):
    if not realizable:
        with open(aut_file) as f:
            content = f.readlines()
        return content
    if realizable:
        return load_strategy(aut_file, spec)
    #else:
    #    return load_counterStrategy(aut_file, spec)


def load_strategy(aut_file, spec):
    """Construct a Mealy Machine from an aut_file.

    @param aut_file: the name of the text file containing the
        automaton, or an (open) file-like object.
    @type spec: L{GRSpec}

    @rtype: L{MealyMachine}
    """
    if isinstance(aut_file, str):
        f = open(aut_file, 'r')
        closable = True
    else:
        f = aut_file  # Else, assume aut_file behaves as file object.
        closable = False

    #build Mealy Machine
    m = transys.MealyMachine()

    # show port only when true
    mask_func = lambda x: bool(x)

    # input defs
    inputs = create_machine_ports(spec.env_vars)
    m.add_inputs(inputs)

    # outputs def
    outputs = create_machine_ports(spec.sys_vars)
    masks = {k:mask_func for k in spec.sys_vars.keys()}
    m.add_outputs(outputs, masks)

    # state variables def
    state_vars = outputs
    m.add_state_vars(state_vars)

    varnames = spec.sys_vars.keys()+spec.env_vars.keys()

    stateDict = {}

    for line in f:
        # parse states
        line = bitwise_to_int_domain(line,spec)
        if (line.find('State ') >= 0):
            stateID = re.search('State (\d+)', line)
            stateID = int(stateID.group(1))


            state = dict(re.findall('(\w+):(\w+)', line))
            for var, val in state.iteritems():
                try:
                    state[var] = int(val)
                except:
                    state[var] = val
                if (len(varnames) > 0):
                    if not var in varnames:
                        logger.error('Unknown variable ' + var)


            for var in varnames:
                if not var in state.keys():
                    logger.error('Variable ' + var + ' not assigned')

        # parse transitions
        if (line.find('successors') >= 0):
            transition = re.findall(' (\d+)', line)
            for i in xrange(0,len(transition)):
                transition[i] = int(transition[i])

            m.states.add(stateID)

            # mark initial states (states that
            # do not appear in previous transitions)
            #seenSoFar = [t for (s,trans) in stateDict.values() for t in trans]
            #if stateID not in seenSoFar:
                #m.states.initial.add(stateID)

            stateDict[stateID] = (state,transition)

    # add transitions with guards to the Mealy Machine
    for from_state in stateDict.keys():
        state, transitions = stateDict[from_state]

        for to_state in transitions:
            guard = stateDict[to_state][0]
            try:
                m.transitions.add(from_state, to_state, **guard)
            except Exception, e:
                raise Exception('Failed to add transition:\n' +str(e) )

    initial_state = 'Sinit'
    m.states.add(initial_state)
    m.states.initial |= [initial_state]

    # Mealy reaction to initial env input
    for v in m.states:
        if v is 'Sinit':
            continue

        var_values = stateDict[v][0]
        bool_values = {k:str(bool(v) ) for k, v in var_values.iteritems() }

        t = spec.evaluate(bool_values)

        if t['env_init'] and t['sys_init']:
            m.transitions.add(initial_state, v, **var_values)
    """
    # label states with variable valuations
    # TODO: consider adding typed states to Mealy machines
    for to_state in m.states:
        predecessors = m.states.pre(to_state)

        # any incoming edges ?
        if predecessors:
            # pick a predecessor
            from_state = predecessors[0]
            trans = m.transitions.find([from_state], [to_state] )
            (from_, to_, trans_label) = trans[0]

            state_label = {k:trans_label[k] for k in m.state_vars}
            m.states.add(to_state, **state_label)
    """
    return m


def bool_to_int_val(var, dom, boolValDict):
    for boolVar, boolVal in boolValDict.iteritems():
        m = re.search(r'('+var+r'@\w+\.'+str(dom[0])+r'\.'+str(dom[1])+r')', boolVar)
        if m:
           min_int = dom[0]
           max_int = dom[1]
           boolValDict[boolVar.split('.')[0]] = boolValDict.pop(boolVar)
           if len(boolValDict) != max_int - min_int:
                logger.error('Error in boolean representation of ' + var)

    assert(min_int>=0)
    assert(max_int>=0)

    val = 0

    for i in range(min_int, int(math.ceil(math.log(max_int)))+1):
        current_key = var+"@"+str(i)
        val += 2**int(boolValDict[current_key])


    return val



def load_counterStrategy(aut_file, spec):
    """Construct a Moore Machine from an aut_file.

    @param aut_file: the name of the text file containing the
        automaton, or an (open) file-like object.
    @type spec: L{GRSpec}

    @rtype: L{MooreMachine}
    """
    if isinstance(aut_file, str):
        f = open(aut_file, 'r')
        closable = True
    else:
        f = aut_file  # Else, assume aut_file behaves as file object.
        closable = False

    #build Moore Machine
    m = transys.MooreMachine()

    # show port only when true
    mask_func = lambda x: bool(x)

    # input defs
    inputs = create_machine_ports(spec.env_vars)
    m.add_inputs(inputs)

    # outputs def
    outputs = create_machine_ports(spec.sys_vars)
    masks = {k:mask_func for k in spec.sys_vars.keys()}
    m.add_outputs(outputs, masks)

    # state variables def
    state_vars = outputs
    m.add_state_vars(state_vars)

    varnames = spec.sys_vars.keys()+spec.env_vars.keys()

    stateDict = {}
    for line in f:
        # parse states
        if (line.find('State ') >= 0):
            stateID = re.search('State (\d+)', line)
            stateID = int(stateID.group(1))


            state = dict(re.findall('(\w+):(\w+)', line))
            for var, val in state.iteritems():
                try:
                    state[var] = int(val)
                except:
                    state[var] = val
                if (len(varnames) > 0):
                    if not var in varnames:
                        logger.error('Unknown variable ' + var)


            for var in varnames:
                if not var in state.keys():
                    logger.error('Variable ' + var + ' not assigned')

        # parse transitions
        if (line.find('successors') >= 0):
            transition = re.findall(' (\d+)', line)
            for i in xrange(0,len(transition)):
                transition[i] = int(transition[i])

            m.states.add(stateID)

            # mark initial states (states that
            # do not appear in previous transitions)
            #seenSoFar = [t for (s,trans) in stateDict.values() for t in trans]
            #if stateID not in seenSoFar:
                #m.states.initial.add(stateID)

            stateDict[stateID] = (state,transition)

    # add transitions with guards to the Moore Machine
    for from_state in stateDict.keys():
        state, transitions = stateDict[from_state]

        for to_state in transitions:
            guard = stateDict[to_state][0]
            try:
                m.transitions.add(from_state, to_state, **guard)
            except Exception, e:
                raise Exception('Failed to add transition:\n' +str(e) )

    initial_state = 'Sinit'
    m.states.add(initial_state)
    m.states.initial |= [initial_state]

    # Moore reaction to initial env input
    for v in m.states:
        if v is 'Sinit':
            continue

        var_values = stateDict[v][0]
        bool_values = {k:str(bool(v) ) for k, v in var_values.iteritems() }

        t = spec.evaluate(bool_values)

        if t['env_init'] and t['sys_init']:
            m.transitions.add(initial_state, v, **var_values)
    """
    # label states with variable valuations
    # TODO: consider adding typed states to Moore machines
    for to_state in m.states:
        predecessors = m.states.pre(to_state)

        # any incoming edges ?
        if predecessors:
            # pick a predecessor
            from_state = predecessors[0]
            trans = m.transitions.find([from_state], [to_state] )
            (from_, to_, trans_label) = trans[0]

            state_label = {k:trans_label[k] for k in m.state_vars}
            m.states.add(to_state, **state_label)
    """
    return m

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
