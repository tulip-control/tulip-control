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
# Generic solver interface

from tulip import rhtlp, ltl_parse, nusmvint, spinint, automaton
from tulip.prop2part import PropPreservingPartition
from tulip.errorprint import printWarning, printError
import copy, re, resource, time, threading, os
from solver_common import SolverException

class SolverInput:
    def __init__(self, solver='NuSMV'):
        self.modules = []
        self.spec = []
        self.globals = {}
        self.setSolver(solver)
        self.out_file = None
        self.aut_file = None
        self.opts = {}
        self.si = None
        self.realized = False
        
    def __repr__(self):
        return "Modules:\n" + "\n".join([str(m) for m in self.modules])
        
    def __iter__(self):
        return iter([ m["name"] for m in self.modules ])
        
    def __getitem__(self, index):
        items = [m for m in self.modules if m["name"] == index]
        if len(items) == 1:
            return items[0]
        elif len(items) == 0:
            raise KeyError("No module with name " + index + " found")
        elif len(items) > 1:
            raise KeyError("Duplicate keys found!")
            
    def moduleInstances(self):
        return [ "%s_%d" % (m["name"], n)
            for m in self.modules for n in range(m["instances"]) ]
            
    def addModule(self, name, spec=None, dynamics=None, sys_vars=None,
                        initials=None, instances=1):
        try:
            if self[name]:
                self.delModule(name)
        except KeyError:
            pass
        if sys_vars is None:
            sys_vars = {}
        if initials is None:
            initials = {}
        self.modules.append({"name" : name, "spec" : spec, "dynamics" : dynamics,
                    "vars" : sys_vars, "initials" : initials, "instances" : instances})
                    
    def delModule(self, name):
        self.modules.remove(self[name])

    def decompose(self, name, globalize=False):
        # Decompose a module with n instances and n initial values into n modules,
        # each with a single instance and a single initial. Useful for SPIN.
        m = self[name]
        for n in range(m["instances"]):
            # Split initials
            init_inst = { k : v[n] for k,v in m["initials"].iteritems() if isinstance(v, list) }
            nm = copy.deepcopy(m)
            nm["name"] = "%s_%d" % (m["name"], n)
            nm["initials"] = dict(m["initials"].items() + init_inst.items())
            nm["instances"] = 1
            self.modules.append(nm)
            if globalize:
                self.globalize(nm["name"])
        self.delModule(name)
        
    def globalize(self, name):
        nm = self[name]
        for var, val in nm["vars"].iteritems():
            if var in nm["initials"]:
                self.globals["%s__%s" % (nm["name"], var)] = (val, nm["initials"][var])
            else:
                self.globals["%s__%s" % (nm["name"], var)] = (val, None)
            def f(t):
                if isinstance(t, ltl_parse.ASTVar) and t.val == var:
                    t.val = "%s__%s" % (nm["name"], var)
                return t
            if nm["spec"] is not None:
                nm["spec"] = nm["spec"].map(f)
        if nm["dynamics"] is not None:
            (trans, dcv) = nm["dynamics"]
            nm["dynamics"] = (trans, "%s__%s" % (nm["name"], dcv))
        nm["vars"] = {}
        nm["initials"] = {}
        
    def setSolver(self, solver):
        self.solver = solver
        if solver == "NuSMV":
            self.write = self.writeSMV
            self._solve = nusmvint.check
        elif solver == "SPIN":
            self.write = self.writePromela
            self._solve = spinint.check
        else:
            raise ValueError(solver + " is not a recognised solver.")
            
    def varName(self, name):
        # Given a canonical variable name, return a solver-compatible name
        components = name.split(".")
        if "__".join(components) in self.globals:
            # Global variable
            return "__".join(components)
        elif self.solver == "NuSMV":
            return name
        elif self.solver == "SPIN":
            (modname, sep, inst) = components[0].rpartition("_")
            if modname in self:
                try:
                    i = int(inst)
                except TypeError:
                    modname = components[0]
                    i = 0
                return "%s[%d]:%s" % (modname, i, ".".join(components[1:]))
            else:
                # Decomposed
                return components[0] + ":" + ".".join(components[1:])
            
    def canonical(self, name):
        # Given a solver variable name, return its canonical form
        if name in self.globals:
            (n, s, v) = name.rpartition("__")
            # Should be true, but allow to fall through to another scheme
            if n in self:
                return n + "." + v
        if self.solver == "NuSMV":
            return name
        elif self.solver == "SPIN":
            match = re.match("(\w+)(?:\[|\()(\d+)(?:\)|\]):([\w.]+)", name)
            if match:
                (a, b, c) = match.groups()
                if a in self and self[a]["instances"] > 1:
                    return a + "_" + b + "." + c
                else:
                    return a + "." + c
            else:
                return ".".join(name.split(":"))
        
    def addSpec(self, spec):
        if isinstance(spec, str):
            spec = ltl_parse.parse(spec)
        self.spec.append(spec)
    def clearSpec(self):
        self.spec = []
    def getSpec(self):
        return self.spec
    # Enforce 'turns' by default. This creates a 'turn' variable that determines
    # which module can run at each step, and which is incremented when a module
    # finishes. This makes NuSMV and SPIN behave in the same way.
    def writeSMV(self, filename, turns=True):
        self.out_file = filename
        nusmvint.writeSMV(filename, self.spec, self.modules, turns=turns)
    def writePromela(self, filename, turns=True):
        self.out_file = filename
        self.opts["preduce"] = self.local_refs()
        spinint.writePromela(self, turns=turns)
        
    def local_refs(self):
        has_lrefs = False
        def f(t):
            if isinstance(t, ltl_parse.ASTVar):
                #if re.match("(\w+)\[(\d+)\]:(\w+)", t.val):
                if t.val.find(".") != -1:
                    has_lrefs = True
            return t
        for s in self.spec:
            s.map(f)
        return has_lrefs
    
    def solve(self, aut_file, verbose=0):
        if self.out_file:
            (self.si, result) = self._solve(self.out_file, aut_file, verbose, **self.opts)
            self.realized = result
            if result:
                self.aut_file = aut_file
            return result
        else:
            raise SolverException("No file to solve")
            
    def automaton(self):
        aut = automaton.Automaton('', [])
        if not self.aut_file:
            raise SolverException("Automaton file has not been generated")
        if self.solver == "NuSMV":
            aut.loadSMVAut(self.aut_file, [])
        elif self.solver == "SPIN":
            aut.loadSPINAut(self.aut_file, [])
        for state in aut.states:
            # Canonical representation
            for k in state.state.keys():
                val = state.state[k]
                del(state.state[k])
                state.state[self.canonical(k)] = val
        return aut
    
    # Metrics
    def solveTime(self):
        if self.si:
            return self.si.time()
        else:
            return None
            
    def autSize(self):
        try:
            return len(self.automaton())
        except SolverException:
            return None
            
    def specNodes(self):
        return sum(len(s) for s in self.spec) + len(self.spec)
        
    def numTransitions(self, module):
        m = self[module]
        trans = m["dynamics"][0]
        return sum(sum(t) for t in trans)
    
    def memoryUsage(self):
        # Memory usage in kilobytes
        if self.si:
            return self.si.memory()
        else:
            return None

def generateSolverInput(sys_disc_vars={}, spec=[],
        disc_props = {}, disc_dynamics=PropPreservingPartition(),
        outfile='tmp.smv', initials={}, solver='NuSMV'):
    solverin = SolverInput()
    ddmodel = discDynamicsModel(sys_disc_vars, spec, disc_props,
                     disc_dynamics, initials)
    solverin.addModule("module", *ddmodel)
    if solver == "NuSMV":
        solverin.writeSMV(outfile)
    elif solver == "SPIN":
        solverin.writePromela(outfile)
    return solverin
    
def prop2reg(prop, regions, props):
    propInd = props.index(prop)
    reg = [j for j in range(0,len(regions)) if \
        regions[j].list_prop[propInd]]
    if len(reg) >= 1:
        if len(reg) > 1:
            print "WARNING: Selected region %d covered by %s" % (reg[0], prop)
        return reg[0]
    else:
        return 
    
def discDynamicsModel(sys_disc_vars, spec, disc_props, disc_dynamics, initials):
    # Generate model for a general LTL solver from a
    # spec and discrete dynamics
    prob = rhtlp.SynthesisProb(env_vars={}, sys_disc_vars={}, disc_props={},
                       sys_cont_vars=[], cont_state_space=None,
                       cont_props={}, sys_dyn=None, spec=['',''])
    prob.createProbFromDiscDynamics(sys_disc_vars=sys_disc_vars,
                                         disc_props=disc_props,
                                         disc_dynamics=disc_dynamics, spec=spec)
    disc_cont_var = prob.getDiscretizedContVar()
    # Convert initial propositions to real values
    initials_iter = initials.copy()
    for prop in initials_iter:
        if prop in disc_dynamics.list_prop_symbol and initials[prop] is True:
            reg = prop2reg(prop, disc_dynamics.list_region, disc_dynamics.list_prop_symbol)
            if reg is None:
                print "WARNING: No corresponding region found for proposition " + prop
                continue
            if disc_cont_var in initials:
                if isinstance(initials[disc_cont_var], list):
                    initials[disc_cont_var].append(reg)
                else:
                    initials[disc_cont_var] = [initials[disc_cont_var], reg]
            else:
                initials[disc_cont_var] = reg
            del(initials[prop])

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
        # empty guarantee; X -> T is always true, assume no spec
        spec = None
    return (spec, (trans, disc_cont_var), sys_vars, initials)

def generateNuSMVInput(*args, **kwargs):
    return generateSolverInput(*args, solver='NuSMV', **kwargs)
    
def generateSPINInput(*args, **kwargs):
    generateSolverInput(*args, solver='SPIN', **kwargs)

def restore_propositions(aut, pp):
    for state in aut.states:
        # translate cellID -> proposition
        for k in state.state.keys():
            var = k.rsplit(".")
            if var[-1] == "cellID":
                props = pp.reg2props(state.state[k])
                if props:
                    for p in props:
                        var[-1] = p
                        state.state[".".join(var)] = True
                    del(state.state[k])
