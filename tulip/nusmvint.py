#!/usr/bin/env python
#
# Copyright (c) 2011, 2012 by California Institute of Technology
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
# $Id:$
# NuSMV interface
import rhtlp, os, ltl_parse, time
from subprocess import Popen, PIPE, STDOUT
from prop2part import PropPreservingPartition
from errorprint import printWarning, printError

class NuSMVError(Exception):
    pass
    
class RealizabilityError(Exception):
    pass

class NuSMVInstance:
    NUSMV_PATH = '/home/nick/SURF/NuSMV-2.5.4/nusmv/NuSMV'
    def __init__(self, model='tmp.smv', out='tmp.aut', path=NUSMV_PATH, verbose=0):
        self.model = model
        self.out = out
        self.verbose = verbose
        try:
            self.instance = Popen([path, '-int', model], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        except OSError:
            printError("Could not execute " + path)
            return
    def command(self, cmd):
        self.instance.stdin.write(cmd + '\n')
    def generateTrace(self):
        self.command('go; check_ltlspec; show_traces -o ' + self.out)
    def quit(self):
        self.command('quit')
        output = self.instance.communicate()
        if self.verbose >= 2:
            print output[0]
            print output[1]
        if "is true" in output[0]:
            # !spec valid => spec unsatisfiable
            raise RealizabilityError()
        if output[1]: # some errors?
            raise NuSMVError(output[1])

def generateNuSMVInput(*args, **kwargs):
    generateSolverInput(*args, solver='NuSMV', **kwargs)

def generateSolverInput(sys_disc_vars={}, spec=[],
        disc_props = {}, disc_dynamics=PropPreservingPartition(),
        outfile='tmp.smv', initials={}, solver='NuSMV'):
    # Generate input for a general LTL solver from a JTLV
    # spec and discrete dynamics
    prob = rhtlp.SynthesisProb(env_vars={}, sys_disc_vars={}, disc_props={},
                       sys_cont_vars=[], cont_state_space=None,
                       cont_props={}, sys_dyn=None, spec=['',''])
    prob.createProbFromDiscDynamics(sys_disc_vars=sys_disc_vars,
                                         disc_props=disc_props,
                                         disc_dynamics=disc_dynamics, spec=spec)
    disc_cont_var = prob.getDiscretizedContVar()
    # Convert initial propositions to real values
    for prop in initials:
        if prop in disc_dynamics.list_prop_symbol and initials[prop] is True:
            propInd = disc_dynamics.list_prop_symbol.index(prop)
            reg = [j for j in range(0,disc_dynamics.num_regions) if \
               disc_dynamics.list_region[j].list_prop[propInd]]
            if len(reg) >= 1:
                initials[disc_cont_var] = reg[0]
                del(initials[prop])
                if len(reg) > 1:
                    print "WARNING: Selected region %d covered by %s" % (reg[0], prop)
            elif len(reg) == 0:
                print "WARNING: No corresponding region found for proposition " + prop

    trans = disc_dynamics.trans
    sys_vars = prob.getSysVars()
    
    spec = prob.getSpec()
    assumption = guarantee = None
    if len(spec) >= 2: # assumption-guarantee
        if spec[0]: assumption = ltl_parse.parse(spec[0])
        if spec[1]: guarantee = ltl_parse.parse(spec[1])
    if assumption and guarantee:
        # assumption -> guarantee
        # N.B. This does not work like JTLV's assumption-guarantee.
        # NuSMV can choose to refute the assumption (if possible) which
        # allows it to form a plan that does not satisfy the guarantee.
        spec = ltl_parse.ASTImp.new(assumption, guarantee)
    elif guarantee:
        # only guarantee
        spec = guarantee
    else:
        # empty guarantee; X -> T is always true, probably error
        printWarning("No guarantee provided")
    
    if solver == "NuSMV":
        writeSMV(outfile, spec, (trans, disc_cont_var), sys_vars)
    elif solver == "SPIN":
        writePromela(outfile, spec, (trans, disc_cont_var), sys_vars, initials)
    return prob
    
def writePromela(pml_file, spec, dynamics=None, sys_vars=None, initials={}):
    if (not os.path.exists(os.path.abspath(os.path.dirname(pml_file)))):
        if (verbose > 0):
            printWarning('Folder for pml_file ' + pml_file + \
                             ' does not exist. Creating...', obj=self)
        os.mkdir(os.path.abspath(os.path.dirname(pml_file)))
    f = open(pml_file, 'w')
    
    # Model
    if sys_vars:
        for var, val in sys_vars.iteritems():
            if val == "boolean":
                if var in initials:
                    f.write("bool %s = %s\n" % (var, str(initials[var])))
                else:
                    # default initial bool = false
                    f.write("bool %s = false\n" % var)
            else:
                # assume integer
                if var in initials:
                    f.write("int %s = %s\n" % (var, str(initials[var])))
                else:
                    # default initial int = 0
                    f.write("int %s = 0\n" % var)
    
    if dynamics:
        (trans, disc_cont_var) = dynamics
        f.write("active proctype Dynamics() {\n\tdo\n")
        for from_region in xrange(0, len(trans)):
            to_regions = [j for j in range(0, len(trans)) if \
                              trans[j][from_region]]
            for to_region in to_regions:
                f.write("\t\t:: %s == %d -> %s = %d\n" % (disc_cont_var, from_region,
                            disc_cont_var, to_region))
        f.write("\tod\n}\n")
        
    # Negate spec & write to file
    spec = ltl_parse.ASTNot.new(spec)
    f.write("ltl {" + spec.toPromela() + "}\n")
    f.close()

def writeSMV(smv_file, spec, dynamics=None, sys_vars=None):
    if (not os.path.exists(os.path.abspath(os.path.dirname(smv_file)))):
        if (verbose > 0):
            printWarning('Folder for smv_file ' + smv_file + \
                             ' does not exist. Creating...', obj=self)
        os.mkdir(os.path.abspath(os.path.dirname(smv_file)))
        
    f = open(smv_file, 'w')
    # Model
    f.write("MODULE main\n")
    
    if sys_vars:
        f.write("\tVAR\n")
        for var, val in sys_vars.iteritems():
            f.write('\t\t' + var + ' : ' + val + ';\n')

    if dynamics:
        # Discrete dynamics - explicit transition system
        (trans, disc_cont_var) = dynamics
        f.write("\tASSIGN\n")
        f.write("\t\tnext(" + disc_cont_var + ") := case\n")
        for from_region in xrange(0, len(trans)):
            to_regions = [j for j in range(0, len(trans)) if \
                              trans[j][from_region]]
            f.write("\t\t\t" + disc_cont_var + " = " + str(from_region) + " : " + \
                        "{" + ', '.join(map(str, to_regions)) + "};\n")
        f.write("\t\tesac;\n")

    # Negate spec
    spec = ltl_parse.ASTNot.new(spec)

    # Write spec to file
    f.write("\tLTLSPEC\n")
    f.write("\t\t" + spec.toSMV())

    f.close()

def computeStrategy(smv_file, aut_file, verbose=0):
    start = time.time()
    nusmv = NuSMVInstance(smv_file, aut_file, verbose=verbose)
    nusmv.generateTrace()
    try:
        nusmv.quit()
        result = True
    except NuSMVError as e:
        printError("NuSMV error: " + e.message)
    except RealizabilityError:
        result = False
    if verbose >= 1:
        print "NuSMV ran in " + str(time.time()-start) + "s"
    return result
