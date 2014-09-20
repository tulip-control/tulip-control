# Copyright (c) 2014 by California Institute of Technology
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

import os, re, subprocess, tempfile


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
    fStructuredSlugs.write(generate_structured_slugs(spec))
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


def solve_game(spec, fSlugs, fAUT, options):
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


def call_slugs(fSlugs, fAUT, options):
    """Subprocess calls to slugs."""
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


def generate_structured_slugs(spec):
    """Return the slugs spec.

    @type spec: L{GRSpec}.
    """
    specStr = ['[INPUT]', '[OUTPUT]', '[ENV_TRANS]', '[ENV_LIVENESS]',
               '[ENV_INIT]', '[SYS_TRANS]', '[SYS_LIVENESS]', '[SYS_INIT]']
    
    specStr[0] += get_vars_for_slugs(spec.env_vars)
    specStr[1] += get_vars_for_slugs(spec.sys_vars)
        
    first = True
    for env_init in spec.env_init:
        env_init = format_for_slugs(env_init)
        if env_init:
            if first:
                specStr[4] += ' \n ' + env_init
                first = False
            else:
                specStr[4] += ' & ' + env_init
    
    for env_safety in spec.env_safety:
        env_safety = format_for_slugs(env_safety)
        if env_safety:
            specStr[2] += '\n' + env_safety
    
    for env_prog in spec.env_prog:
        env_prog = format_for_slugs(env_prog)
        if env_prog:
            specStr[3] += '\n' + env_prog        

    first = True
    for sys_init in spec.sys_init:
        sys_init = format_for_slugs(sys_init)
        if sys_init:
            if first:
                specStr[7] += ' \n ' + sys_init
                first = False
            else:
                specStr[7] += ' & ' + sys_init
    
    for sys_safety in spec.sys_safety:
        sys_safety = format_for_slugs(sys_safety)
        if sys_safety:
            specStr[5] += '\n' + sys_safety
    
    for sys_prog in spec.sys_prog:
        sys_prog = format_for_slugs(sys_prog)
        if sys_prog:
            specStr[6] += '\n' + sys_prog 
    return '\n\n'.join(specStr)


def get_vars_for_slugs(vardict):
    output = ''
    for variable, domain in vardict.items():
        if domain == 'boolean':
            output += "\n" + variable
        elif isinstance(domain, tuple) and len(domain) == 2:
            output += (
                "\n" + variable + ": " +
                str(domain[0]) + "..." + str(domain[1])
            )
        else:
            raise ValueError("Domain type unsupported by slugs: " +
                str(domain))
    return output


def format_for_slugs(spec):
    spec = re.sub(r'X\((\w+)\)', r"\1'", spec)
    return spec

    
def bool_to_int_val(var, dom, boolValDict):
    for boolVar, boolVal in boolValDict.iteritems():
        m = re.search(
            r'(' + var + r'@\w+\.' +
            str(dom[0]) + r'\.' + str(dom[1]) + r')',
            boolVar
        )
        if m:
           min_int = dom[0]
           max_int = dom[1]
           boolValDict[boolVar.split('.')[0]] = boolValDict.pop(boolVar)
           if len(boolValDict) != max_int - min_int:
                logger.error('Error in boolean representation of ' + var)
    
    assert(min_int >= 0)
    assert(max_int >= 0)
    
    val = 0
    
    for i in range(min_int, int(math.ceil(math.log(max_int))) + 1):
        current_key = var + "@" + str(i)
        val += 2 ** int(boolValDict[current_key])
    
    return val


def bitwise_to_int_domain(line, spec):
    """Convert bitwise representation to integer domain defined in spec."""
    allVars = dict(spec.sys_vars.items() + spec.env_vars.items())
    for var, dom in allVars.iteritems():
        if isinstance(dom, tuple) and len(dom) == 2:
            
            boolValDict = dict(re.findall(
                r'(' + var + r'@\w+|' + var + r'@\w+\.' + 
                str(dom[0]) + r'\.' + str(dom[1]) + r'):(\w+)', line
            ))
            
            if boolValDict:
                intVal = bool_to_int_val(var, dom,
                                         copy.deepcopy(boolValDict))
                
                first = True
                for key,val in boolValDict.iteritems():
                    if first:
                        line = re.sub(
                            r'(' + re.escape(key) + r'\w*:' +
                            str(val) + r')',
                            var + ":" + str(intVal),
                            line
                        )
                        first=False
                    else:
                        line = re.sub(
                            r'(' + re.escape(key) + r'\w*:' +
                            str(val) + r'[,]*)',
                            "",
                            line
                        )

    return line
