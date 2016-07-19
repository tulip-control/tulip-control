# Copyright (c) 2012-2015 by California Institute of Technology
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
import os
import re
import subprocess
import tempfile
import warnings
import networkx as nx
from tulip.spec import translation


logger = logging.getLogger(__name__)
DEBUG_SMV_FILE = 'smv.txt'
DEBUG_LTL_FILE = 'ltl.txt'
DEBUG_AUT_FILE = 'aut.txt'


def check_realizable(spec, heap_size='-Xmx128m', priority_kind=-1,
                     init_option=1):
    """Decide realizability of specification defined by given GRSpec object.

    @return: True if realizable, False if not, or an error occurs.
    """
    assert not spec.moore
    assert not spec.plus_one
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

    call_jtlv(heap_size, fSMV, fLTL, fAUT, priority_kind, init_option)

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
    spec, heap_size='-Xmx128m', priority_kind=3,
    init_option=1
):
    """Synthesize a strategy satisfying the specification.

    Arguments are described in documentation for L{solve_game}.

    @return: Return strategy as instance of C{networkx.DiGraph}, or a
        list of counter-examples as returned by L{get_counterexamples}.
    """
    assert not spec.moore
    assert not spec.plus_one
    fSMV, fLTL, fAUT = create_files(spec)

    realizable = solve_game(spec, fSMV, fLTL, fAUT, heap_size,
                            priority_kind, init_option)

    if realizable:
        logger.info('loading JTLV output...')

        with open(fAUT, 'r') as f:
            lines = f.readlines()

        logger.info('JTLV returned:\n' + '\n'.join(lines))

        aut = jtlv_output_to_networkx(lines, spec)

        os.unlink(fSMV)
        os.unlink(fLTL)
        os.unlink(fAUT)
        return aut
    else:
        counter_examples = get_counterexamples(fAUT)
        os.unlink(fSMV)
        os.unlink(fLTL)
        os.unlink(fAUT)
        return counter_examples


def create_files(spec):
    """Create temporary files for read/write by JTLV."""
    fSMV = tempfile.NamedTemporaryFile(delete=False, suffix='smv')
    fSMV.write(generate_jtlv_smv(spec))
    fSMV.close()

    fLTL = tempfile.NamedTemporaryFile(delete=False, suffix="ltl")
    fLTL.write(generate_jtlv_ltl(spec))
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
    if isinstance(priority_kind, str):
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
    elif isinstance(priority_kind, int):
        if (
            priority_kind > 0 and priority_kind != 3 and
            priority_kind != 7 and priority_kind != 11 and
            priority_kind != 15 and priority_kind != 19 and
            priority_kind != 23
        ):
            warnings.warn("Unknown priority_kind." +
                          " Setting it to the default (ZYX)")
            priority_kind = 3
    else:
        warnings.warn("Unknown priority_kind. Setting it to the default (ZYX)")
        priority_kind = 3
    return priority_kind


def call_jtlv(heap_size, fSMV, fLTL, fAUT, priority_kind, init_option):
    """Subprocess calls to JTLV."""
    JTLV_PATH = os.path.abspath(os.path.dirname(__file__))
    JTLV_EXE = 'jtlv_grgame.jar'
    logger.info(
        ('Calling jtlv with arguments:\n'
         '  heap size: {heap}\n'
         '  jtlv path: {path}\n'
         '  priority_kind: {priority}\n').format(
            heap=heap_size,
            path=JTLV_PATH,
            priority=priority_kind
        )
    )
    jtlv_grgame = os.path.join(JTLV_PATH, JTLV_EXE)
    cmd = ['java', heap_size, '-jar', jtlv_grgame, fSMV, fLTL, fAUT,
           str(priority_kind), str(init_option)]
    # debugging log
    if logger.getEffectiveLevel() <= logging.DEBUG:
        logger.debug('Expected: ' + jtlv_grgame)
        logger.debug(' '.join(cmd))
        # besides dumping to debug logging stream,
        # also copy files to ease manual debugging
        import shutil
        shutil.copyfile(fSMV, DEBUG_SMV_FILE)
        shutil.copyfile(fLTL, DEBUG_LTL_FILE)
        shutil.copyfile(fAUT, DEBUG_AUT_FILE)
    try:
        subprocess.call(cmd)
    except OSError as e:
        if e.errno == os.errno.ENOENT:
            raise Exception('Java not found in path: cannot run jtlv.')
        else:
            raise


def canon_to_jtlv_domain(dom):
    """Convert an LTL or GRSpec variable domain to JTLV style.

    @param dom: a variable domain, as would be provided by the values
    in the output_variables attribute of L{spec.form.LTL} or sys_vars
    of L{spec.form.GRSpec}.

    @rtype: str
    @return: variable domain string, ready for use in an SMV file, as
        expected by the JTLV solver.
    """
    if dom == 'boolean':
        return dom
    elif isinstance(dom, tuple) and len(dom) == 2:
        return '{%s}' % ', '.join([str(i) for i in xrange(dom[0], dom[1]+1)])
    else:
        raise ValueError("Unrecognized domain type: "+str(dom))


def generate_jtlv_smv(spec):
    """Return the SMV module definitions needed by JTLV.

    Raises exception if malformed GRSpec object is detected.

    @type spec: L{GRSpec}

    @rtype: str
    @return: string conforming to SMV file format that JTLV expects.
    """
    # Define env vars
    env = []
    for var, dom in spec.env_vars.iteritems():
        int_dom = translation.convert_domain(dom)

        env.append('\t\t{v} : {c};'.format(
            v=var, c=canon_to_jtlv_domain(int_dom)
        ))

    # Define sys vars
    sys = []
    for var, dom in spec.sys_vars.iteritems():
        int_dom = translation.convert_domain(dom)

        sys.append('\t\t{v} : {c};'.format(
            v=var, c=canon_to_jtlv_domain(int_dom)
        ))

    env_vars = '\n'.join(env)
    sys_vars = '\n'.join(sys)

    smv = smv_template.format(env_vars=env_vars, sys_vars=sys_vars)

    logger.debug('SMV file:\n' + smv)
    return smv


smv_template = '''
MODULE main
    VAR
        e : env();
        s : sys();

MODULE env -- inputs
    VAR
{env_vars}


MODULE sys -- outputs
    VAR
{sys_vars}
'''


def generate_jtlv_ltl(spec):
    """Return LTL specification for JTLV.

    @type spec: L{GRSpec}
    """
    a, g = translation.translate(spec, 'jtlv')

    if not a:
        a = 'TRUE'

    if not g:
        g = 'TRUE'

    ltl = ltl_template.format(a=a, g=g)

    logger.debug('JTLV input formula:\n' + ltl)

    return ltl


ltl_template = '''
LTLSPEC
(
{a}
);

LTLSPEC
(
{g}
);
'''


def jtlv_output_to_networkx(lines, spec):
    """Return a graph after parsing JTLV's output.

    @param lines: as loaded from file
    @type lines: C{list} of C{str}

    @type spec: L{GRSpec}

    @rtype: C{networkx.DiGraph}
    """
    logger.info('parsing jtlv output to create a graph...')

    g = nx.DiGraph()

    varnames = dict(spec.sys_vars)
    varnames.update(spec.env_vars)

    # g.env_vars = spec.env_vars.copy()
    # g.sys_vars = spec.sys_vars.copy()

    for line in lines:
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
                    if var not in varnames:
                        logger.error('Unknown variable "{var}"'.format(
                                     var=var))

            for var, dom in varnames.iteritems():
                # singleton domain ?
                if isinstance(dom, tuple):
                    if dom[0] == dom[1]:
                        state[var] = dom[0]
                if isinstance(dom, list):
                    if len(dom) == 1:
                        state[var] = dom[0]
                # still missing ?
                if var not in state:
                    raise ValueError(
                        'Variable "{var}" not assigned in "{line}"'.format(
                            var=var, line=line))

        # parse transitions
        if line.find('successors') >= 0:
            succ = [int(x) for x in re.findall(' (\d+)', line)]

            g.add_node(state_id, state=state)
            g.add_edges_from([(state_id, v) for v in succ])

    logger.info('done parsing jtlv output.\n')
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
        if '--' not in line:
            newspec += line + '\n'
    return newspec


def check_vars(varNames):
    """Complain if any variable name is a number or not a string."""
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
    """Verify that all non-operators in "spec" are in the list of vars."""
    # Replace all special characters with whitespace
    special_characters = [
        "next(", "[]", "<>", "->", "&", "|", "!",
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
