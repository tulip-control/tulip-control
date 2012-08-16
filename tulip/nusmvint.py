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
# $Id$
# NuSMV interface
import rhtlp, os, ltl_parse, time, copy, threading
from subprocess import Popen, PIPE, STDOUT
from prop2part import PropPreservingPartition
from errorprint import printWarning, printError
from solver_common import SolverException, memoryMonitor

# total (OS + user) CPU time of children
chcputime = (lambda: (lambda x: x[2] + x[3])(os.times()))

class NuSMVError(SolverException):
    pass

class NuSMVInstance:
    NUSMV_PATH = os.path.join(os.path.dirname(__file__), "solvers/NuSMV")
    def __init__(self, model='tmp.smv', out='tmp.aut', path=NUSMV_PATH, verbose=0):
        self.model = model
        self.out = out
        self.verbose = verbose
        self.t_start = chcputime()
        self.t_time = None
        try:
            self.instance = Popen([path, '-int', model], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        except OSError:
            printError("Could not execute " + path)
            return
        def mmthread_run():
            self.max_mem = memoryMonitor(self.instance.pid)
        self.mmthread = threading.Thread(target=mmthread_run)
        self.mmthread.start()
    def command(self, cmd):
        self.instance.stdin.write(cmd + '\n')
    def generateTrace(self):
        self.command('go; check_ltlspec; show_traces -o ' + self.out)
    def quit(self):
        self.command('quit')
        self.mmthread.join()
        output = self.instance.communicate()
        self.t_time = chcputime() - self.t_start
        if self.verbose >= 2:
            print output[0]
            print output[1]
        if "is true" in output[0]:
            # !spec valid => spec unsatisfiable
            return False
        if output[1]: # some errors?
            raise NuSMVError(output[1])
        return True
    def time(self):
        return self.t_time
    def memory(self):
        return self.max_mem
            
def modularize(spec, name):
    def f(t):
        if isinstance(t, ltl_parse.ASTVar):
            t.val = name + "." + t.val
        return t
    spec = spec.map(f)
    return spec
    
def SMV_transitions(trans, var, turns=False):
    s = "\t\tnext(" + var + ") := case\n"
    for from_region in xrange(0, len(trans)):
        to_regions = [j for j in range(0, len(trans)) if \
                          trans[j][from_region]]
        if turns:
            s += "\t\t\t%s = %d & turn = mine : {%s};\n" % (var, from_region, ', '.join(map(str, to_regions)))
        else:
            s += "\t\t\t%s = %d : {%s};\n" % (var, from_region, ', '.join(map(str, to_regions)))
    if turns: s += "\t\t\tturn != mine : %s;\n" % var
    s += "\t\tesac;\n"
    return s

def writeSMV(smv_file, spec, modules, turns=False):
    if (not os.path.exists(os.path.abspath(os.path.dirname(smv_file)))):
        if (verbose > 0):
            printWarning('Folder for smv_file ' + smv_file + \
                             ' does not exist. Creating...', obj=self)
        os.mkdir(os.path.abspath(os.path.dirname(smv_file)))
    spec = spec[:]
    
    f = open(smv_file, 'w')
    turn = 0
    main_vars = {}
    for m in modules:
        for n in range(m["instances"]):
            if m["instances"] > 1:
                instance_name = "%s_%d" % (m["name"], n)
            else:
                instance_name = m["name"]
            if turns:
                main_vars[instance_name] = m["name"] + "(turn,%d)" % turn
                turn += 1
            else:
                main_vars[instance_name] = m["name"] + "()"
            modspec = copy.deepcopy(m["spec"])
            spec.append(modularize(modspec, instance_name))
    if turns:
        main_vars["turn"] = "0 .. %d" % (turn-1)
        main_trans = [ [ int((n+1) % turn == m) for m in range(turn) ] for n in range(turn) ]
        main_dynamics = (main_trans, "turn")
        main_initials = {"turn" : 0}
    else:
        main_dynamics = None
        main_initials = None
    modules = modules + [{"name" : "main", "vars" : main_vars, "dynamics" : main_dynamics,
                    "initials" : main_initials }]
    for m in modules:
        if turns and not m["name"] == "main":
            f.write("MODULE %s(turn,mine)\n" % m["name"])
        else:
            f.write("MODULE %s\n" % m["name"])
        
        if m["vars"]:
            f.write("\tVAR\n")
            for var, val in m["vars"].iteritems():
                f.write('\t\t' + var + ' : ' + val + ';\n')
        
        f.write("\tASSIGN\n")
        
        if m["initials"]:
            for var, val in m["initials"].iteritems():
                if isinstance(val, list):
                    vl = [ str(v) for v in val ]
                    f.write("\t\tinit(" + var + ") := {" + ", ".join(vl) + "};\n")
                else:
                    f.write("\t\tinit(" + var + ") := " + str(val) + ";\n")
                
        if m["dynamics"]:
            # Discrete dynamics - explicit transition system
            f.write(SMV_transitions(*m["dynamics"], turns=(turns and not m["name"] == "main")))
        
    spec = reduce(ltl_parse.ASTAnd.new, spec)
    # Negate spec
    spec = ltl_parse.ASTNot.new(spec)

    # Write spec to file
    f.write("\tLTLSPEC\n")
    f.write("\t\t" + spec.toSMV() + "\n")

    f.close()
    
# Raises NuSMVError
def check(smv_file, aut_file, verbose=0, **opts):
    nusmv = NuSMVInstance(smv_file, aut_file, verbose=verbose)
    nusmv.generateTrace()
    result = nusmv.quit()
    return (nusmv, result)

def computeStrategy(smv_file, aut_file, verbose=0):
    start = time.time()
    (nusmv, result) = check(smv_file, aut_file, verbose)
    if verbose >= 1:
        print "NuSMV ran in " + str(time.time()-start) + "s"
    return result
