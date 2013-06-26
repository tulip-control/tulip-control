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
# SPIN interface
import os, copy, re
from subprocess import Popen, PIPE, call
from contextlib import contextmanager
from errorprint import printWarning, printError
from solver_common import SolverException, memoryMonitor
import ltl_parse

class SPINError(SolverException):
    pass

# total (OS + user) CPU time of children
chcputime = (lambda: (lambda x: x[2] + x[3])(os.times()))

@contextmanager
def cd(path):
    old = os.getcwd()
    if path:
        os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)

prv = lambda x: os.path.join("..", x)
class SPINInstance:
    SPIN_PATH = os.path.join(os.path.dirname(__file__), "solvers/spin")
    def __init__(self, model="tmp.pml", out="tmp.aut", path=SPIN_PATH,
             preduce=True, verbose=0, safety=False, pandir="pan"):
        """Create new SPIN instance, generate pan verifier.
        
        @param model: Path of Promela model file.
        @param out: Output path for resulting automaton.
        @param path: Path to the spin executable (default TULIP_DIR/solvers/spin).
        @param preduce: If True, use partial-order reduction to speed up solving.
        @param verbose: Verbosity level, 0 = quiet, > 0 = print messages.
        @param safety: Set to True to use optimisations for safety specifications.
        
        @raise SPINError: When any SPIN subcomponent raises an error.
        """
        (self.model, self.out, self.path, self.pandir) = (model, out, path, pandir)
        # Time everything, including generation of verifier
        (self.t_start, self.t_time) = (chcputime(), None)
        try:
            os.mkdir(self.pandir)
        except OSError:
            # PAN directory already exists
            pass
        with cd(self.pandir):
            spin = Popen([self.path, "-O", "-a", prv(self.model)], stdout=PIPE, stderr=PIPE)
            self.max_mem = memoryMonitor(spin.pid)
            (out, err) = spin.communicate()
            if verbose > 0:
                print out
            if err:
                raise SPINError(err)
            elif spin.returncode != 0:
                # SPIN may put error messages on stdout
                raise SPINError(out)
            opts = []
            if not preduce:
                opts.append("-DNOREDUCE")
            if safety:
                opts.append("-DSAFETY")
            shstr = "gcc %s -o pan pan.c" % " ".join(opts)
            gcc = Popen(shstr, shell=True)
            self.max_mem = max(self.max_mem, memoryMonitor(gcc.pid))
            retcode = gcc.wait()
            if not retcode == 0:
                raise SPINError("Could not compile verifier")
    def generateTrace(self, verbose=0):
        """Run verifier and generate automaton file.
        
        @param verbose: Verbosity level, 0 = quiet, > 0 = print messages.
        
        @raise SPINError: When any SPIN subcomponent raises an error.
        """
        with open(self.out, 'w') as f:
            (model_dn, model_fn) = os.path.split(self.model)
            if not model_dn: model_dn = "."
            pan = Popen([os.path.join(self.pandir, "pan"), "-a"], stdout=PIPE,
                         stderr=PIPE, cwd=model_dn)
            self.max_mem = max(self.max_mem, memoryMonitor(pan.pid))
            (out, err) = pan.communicate()
            if verbose > 0:
                print out
            if err:
                raise SPINError(err)
            if not "acceptance cycle" in out:
                return False
            spin = Popen([self.path, "-t", "-p", "-g", "-w", "-l", model_fn],
                        stdout=f, stderr=PIPE, cwd=model_dn)
            (out, err) = spin.communicate()
            self.t_time = chcputime() - self.t_start
            if err:
                raise SPINError(err)
            return True
    def time(self):
        return self.t_time
    def memory(self):
        return self.max_mem

def check(pml_file, aut_file, verbose=0, **opts):
    """Generate trace from model and store in automaton file.
    
    @param pml_file: Path of Promela model file.
    @param aut_file: Output path for resulting automaton.
    @param verbose: Verbosity level, 0 = quiet, > 0 = print messages.
    @param preduce: If True, use partial-order reduction to speed up solving.
    @param safety: Set to True to use optimisations for safety specifications.
    
    @rtype: (SPINInstance, boolean)
    @returns: Tuple of created SPIN instance and realizability result.
    """
    kwargs = dict([ (k, opts[k]) for k in ["preduce", "safety"] if k in opts ])
    spin = SPINInstance(pml_file, aut_file, verbose=verbose, **kwargs)
    result = spin.generateTrace(verbose)
    return (spin, result)
       
def computeStrategy(pml_file, aut_file, verbose=0):
    (spin, result) = check(pml_file, aut_file, verbose)
    return result

def modularize(spec, name, v=None):
    def f(t):
        if isinstance(t, ltl_parse.ASTVar) and (v is None or t.val in v):
            t.val = name + ":" + t.val
        return t
    return spec.map(f)
    
def decanonize(spec, slvi):
    def f(t):
        if isinstance(t, ltl_parse.ASTVar):
            t.val = slvi.varName(t.val)
        return t
    return spec.map(f)
    
def promelaVar(var, val, initial=None):
    if val == "boolean":
        if initial:
            if isinstance(initial, list):
                raise TypeError("SPIN interface does not accept multiple initials")
            return "\tbool %s = %s;\n" % (var, str(initial))
        else:
            # default initial bool = false
            return "\tbool %s = false;\n" % var
    else:
        # assume integer
        if initial:
            if isinstance(initial, list):
                raise TypeError("SPIN interface does not accept multiple initials")
            return "\tint %s = %s;\n" % (var, str(initial))
        else:
            # default initial int = 0
            return "\tint %s = 0;\n" % var
            
def local_refs(spec):
    """Test if the specification has local references."""
    has_lrefs = False
    def f(t):
        if isinstance(t, ltl_parse.ASTVar):
            #if re.match("(\w+)\[(\d+)\]:(\w+)", t.val):
            if t.val.find(":") != -1:
                has_lrefs = True
        return t
    for s in spec:
        s.map(f)
    return has_lrefs

def writePromela(slvi, turns=False):
    """Write a Promela model file from the provided SolverInput.
    
    @param slvi: SolverInput instance to generate model from.
    @type turns: boolean
    @param turns: Require that modules execute in-turn.
    """
    (pml_file, spec, modules, globalv) = (slvi.out_file, slvi.spec, slvi.modules, slvi.globals)
    if (not os.path.exists(os.path.abspath(os.path.dirname(pml_file)))):
        if (verbose > 0):
            printWarning('Folder for pml_file ' + pml_file + \
                             ' does not exist. Creating...', obj=self)
        os.mkdir(os.path.abspath(os.path.dirname(pml_file)))
    
    f = open(pml_file, 'w')
    for var, (val, initial) in globalv.iteritems():
        f.write(promelaVar(var, val, initial))
    spec = [ decanonize(s, slvi) for s in spec ]
    
    if len(modules) == 1 and modules[0]["instances"] == 1:
        # 'turns' has no effect if there's only one module
        turns = False
    
    for m in modules:
        if m["instances"] > 1:
            for n in range(m["instances"]):
                if m["spec"]:
                    modspec = modularize(m["spec"], "%s[%d]" % (m["name"], n), m["vars"].keys())
                    spec.append(modspec)
        else:
            # Seems to be a bug in SPIN: for process P with local variable x and
            # exactly one instance, P:x works but P[0]:x does not
            for n in range(m["instances"]):
                if m["spec"]:
                    modspec = modularize(m["spec"], m["name"], m["vars"].keys())
                    spec.append(modspec)
    
    if turns: f.write(promelaVar("SYNC_VAR", "int", 0))
    for m in modules:
        # Model
        f.write("active [%d] proctype %s() {\n" % (m["instances"], m["name"]))
        if m["vars"]:
            for var, val in m["vars"].iteritems():
                if var in m["initials"]:
                    f.write(promelaVar(var, val, m["initials"][var]))
                else:
                    f.write(promelaVar(var, val))
        # Dynamics
        if m["dynamics"]:
            turn = modules.index(m)
            f.write("\tdo\n")
            if turns: f.write("\t\t:: SYNC_VAR == %d -> if\n" % turn)
            (trans, disc_cont_var) = m["dynamics"]
            for from_region in xrange(0, len(trans)):
                to_regions = [j for j in range(0, len(trans)) if \
                                  trans[j][from_region]]
                for to_region in to_regions:
                    f.write("\t\t\t:: %s == %d -> %s = %d\n" % (disc_cont_var, from_region,
                                disc_cont_var, to_region))
            if turns: f.write("\t\tfi; SYNC_VAR = %d\n" % ((turn+1) % len(modules)))
            f.write("\tod\n")
        f.write("}\n")
    
    if spec:
        if all(ltl_parse.issafety(s) for s in spec):
            slvi.opts["safety"] = True
        slvi.opts["preduce"] = local_refs(spec)
        # Spec
        spec = reduce(ltl_parse.ASTAnd.new, spec)
        # Negate & write to file
        spec = ltl_parse.ASTNot.new(spec)
        f.write("ltl {" + spec.toPromela() + "}\n")
    
    f.close()

def compatible(modules):
    if any(any(isinstance(initial, list) for initial in m["initials"]) for m in modules):
        return False
    return True
