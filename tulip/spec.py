# Copyright (c) 2011-2013 by California Institute of Technology
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
Specification module
"""

import re, copy


class GRSpec(object):
    """GR(1) specification

    The canonical form is::

      (env_init & []env_safety & []<>env_prog_1 & []<>env_prog_2 & ...)
        -> (sys_init & []sys_safety & []<>sys_prog_1 & []<>sys_prog_2 & ...)

    A GRSpec object contains the following attributes:

      - C{env_vars}: a list of variables (names given as strings) that
        are determined by the environment.

      - C{env_init}: a string or a list of string that specifies the
        assumption about the initial state of the environment.

      - C{env_safety}: a string or a list of string that specifies the
        assumption about the evolution of the environment state.

      - C{env_prog}: a string or a list of string that specifies the
        justice assumption on the environment.

      - C{sys_vars}: a list of variables (names given as strings) that
        are controlled by the system.

      - C{sys_init}: a string or a list of string that specifies the
        requirement on the initial state of the system.

      - C{sys_safety}: a string or a list of string that specifies the
        safety requirement.

      - C{sys_prog}: a string or a list of string that specifies the
        progress requirement.

    An empty list for any formula (e.g., if env_init = []) is marked
    as "True" in the specification. This corresponds to the constant
    Boolean function, which usually means that subformula has no
    effect (is non-restrictive) on the spec.
    """
    def __init__(self, env_vars=[], sys_vars=[],
                 env_init='', sys_init='',
                 env_safety='', sys_safety='',
                 env_prog='', sys_prog=''):
        self.env_vars = copy.copy(env_vars)
        self.sys_vars = copy.copy(sys_vars)
        self.env_init = copy.deepcopy(env_init)
        self.sys_init = copy.deepcopy(sys_init)
        self.env_safety = copy.deepcopy(env_safety)
        self.sys_safety = copy.deepcopy(sys_safety)
        self.env_prog = copy.deepcopy(env_prog)
        self.sys_prog = copy.deepcopy(sys_prog)

        # It is easier to work with lists...
        if isinstance(self.sys_safety, str):
            if len(self.sys_safety) == 0:
                self.sys_safety = []
            else:
                self.sys_safety = [self.sys_safety]
        if isinstance(self.sys_init, str):
            if len(self.sys_init) == 0:
                self.sys_init = []
            else:
                self.sys_init = [self.sys_init]
        if isinstance(self.sys_prog, str):
            if len(self.sys_prog) == 0:
                self.sys_prog = []
            else:
                self.sys_prog = [self.sys_prog]
        if isinstance(self.env_safety, str):
            if len(self.env_safety) == 0:
                self.env_safety = []
            else:
                self.env_safety = [self.env_safety]
        if isinstance(self.env_init, str):
            if len(self.env_init) == 0:
                self.env_init = []
            else:
                self.env_init = [self.env_init]
        if isinstance(self.env_prog, str):
            if len(self.env_prog) == 0:
                self.env_prog = []
            else:
                self.env_prog = [self.env_prog]


    def import_GridWorld(self, gworld, offset=(0,0), controlled_dyn=True):
        """Append specification describing a gridworld.

        Basically, call the spec method of the given GridWorld object
        and merge with its output.  See documentation about the
        L{spec<gridworld.GridWorld.spec>} method of
        L{GridWorld<gridworld.GridWorld>} class for details.

        @type gworld: L{GridWorld}
        """
        s = gworld.spec(offset=offset, controlled_dyn=controlled_dyn)
        for evar in s.env_vars:
            if evar not in self.env_vars:
                self.env_vars.append(evar)
        for svar in s.sys_vars:
            if svar not in self.sys_vars:
                self.sys_vars.append(svar)
        self.env_init.extend(s.env_init)
        self.env_safety.extend(s.env_safety)
        self.env_prog.extend(s.env_prog)
        self.sys_init.extend(s.sys_init)
        self.sys_safety.extend(s.sys_safety)
        self.sys_prog.extend(s.sys_prog)


    def import_PropPreservingPartition(self, disc_dynamics, cont_varname="cellID"):
        """Append results of discretization (abstraction) to specification.

        disc_dynamics is an instance of PropPreservingPartition, such
        as returned by the function discretize in module discretize.

        Notes
        =====
          - The cell names are *not* mangled, in contrast to the
            approach taken in the createProbFromDiscDynamics method of
            the SynthesisProb class.

          - Any name in disc_dynamics.list_prop_symbol matching a system
            variable is removed from sys_vars, and its occurrences in
            the specification are replaced by a disjunction of
            corresponding cells.

          - gr1c does not (yet) support variable domains beyond Boolean,
            so we treat each cell as a separate Boolean variable and
            explicitly enforce mutual exclusion.
        """
        if len(disc_dynamics.list_region) == 0:  # Vacuous call?
            return
        cont_varname += "_"  # ...to make cell number easier to read
        for i in range(len(disc_dynamics.list_region)):
            if (cont_varname+str(i)) not in self.sys_vars:
                self.sys_vars.append(cont_varname+str(i))

        # The substitution code and transition code below are mostly
        # from createProbFromDiscDynamics and toJTLVInput,
        # respectively, in the rhtlp module, with some style updates.
        for prop_ind, prop_sym in enumerate(disc_dynamics.list_prop_symbol):
            reg = [j for j in range(len(disc_dynamics.list_region))
                   if disc_dynamics.list_region[j].list_prop[prop_ind] != 0]
            if len(reg) == 0:
                subformula = "False"
                subformula_next = "False"
            else:
                subformula = " | ".join([cont_varname+str(regID) for regID in reg])
                subformula_next = " | ".join([cont_varname+str(regID)+"'" for regID in reg])
            prop_sym_next = prop_sym+"'"
            self.sym_to_prop(props={prop_sym_next:subformula_next})
            self.sym_to_prop(props={prop_sym:subformula})

        # It is easier to work with lists...
        if isinstance(self.sys_safety, str):
            if len(self.sys_safety) == 0:
                self.sys_safety = []
            else:
                self.sys_safety = [self.sys_safety]
        if isinstance(self.sys_init, str):
            if len(self.sys_init) == 0:
                self.sys_init = []
            else:
                self.sys_init = [self.sys_init]

        # Transitions
        for from_region in range(len(disc_dynamics.list_region)):
            to_regions = [i for i in range(len(disc_dynamics.list_region))
                          if disc_dynamics.trans[i][from_region] != 0]
            self.sys_safety.append(cont_varname+str(from_region) + " -> (" + " | ".join([cont_varname+str(i)+"'" for i in to_regions]) + ")")

        # Mutex
        self.sys_init.append("")
        self.sys_safety.append("")
        for regID in range(len(disc_dynamics.list_region)):
            if len(self.sys_safety[-1]) > 0:
                self.sys_init[-1] += "\n| "
                self.sys_safety[-1] += "\n| "
            self.sys_init[-1] += "(" + cont_varname+str(regID)
            if len(disc_dynamics.list_region) > 1:
                self.sys_init[-1] += " & " + " & ".join(["(!"+cont_varname+str(i)+")" for i in range(len(disc_dynamics.list_region)) if i != regID])
            self.sys_init[-1] += ")"
            self.sys_safety[-1] += "(" + cont_varname+str(regID)+"'"
            if len(disc_dynamics.list_region) > 1:
                self.sys_safety[-1] += " & " + " & ".join(["(!"+cont_varname+str(i)+"')" for i in range(len(disc_dynamics.list_region)) if i != regID])
            self.sys_safety[-1] += ")"


    def sym_to_prop(self, props, verbose=0):
        """Replace the symbols of propositions with the actual propositions.
        """
        if (props is not None):
            symfound = True
            while (symfound):
                symfound = False
                for propSymbol, prop in props.iteritems():
                    if propSymbol[-1] != "'":  # To handle gr1c primed variables
                        propSymbol += r"\b"
                    if (verbose > 2):
                        print '\t' + propSymbol + ' -> ' + prop
                    if (isinstance(self.env_init, str)):
                        if (len(re.findall(r'\b'+propSymbol, self.env_init)) > 0):
                            self.env_init = re.sub(r'\b'+propSymbol, '('+prop+')', self.env_init)
                            symfound = True
                    else:
                        for i in xrange(0, len(self.env_init)):
                            if (len(re.findall(r'\b'+propSymbol, self.env_init[i])) > 0):
                                self.env_init[i] = re.sub(r'\b'+propSymbol, '('+prop+')', \
                                                              self.env_init[i])
                                symfound = True

                    if (isinstance(self.sys_init, str)):
                        if (len(re.findall(r'\b'+propSymbol, self.sys_init)) > 0):
                            self.sys_init = re.sub(r'\b'+propSymbol, '('+prop+')', self.sys_init)
                            symfound = True
                    else:
                        for i in xrange(0, len(self.sys_init)):
                            if (len(re.findall(r'\b'+propSymbol, self.sys_init[i])) > 0):
                                self.sys_init[i] = re.sub(r'\b'+propSymbol, '('+prop+')', \
                                                              self.sys_init[i])
                                symfound = True

                    if (isinstance(self.env_safety, str)):
                        if (len(re.findall(r'\b'+propSymbol, self.env_safety)) > 0):
                            self.env_safety = re.sub(r'\b'+propSymbol, '('+prop+')', self.env_safety)
                            symfound = True
                    else:
                        for i in xrange(0, len(self.env_safety)):
                            if (len(re.findall(r'\b'+propSymbol, self.env_safety[i])) > 0):
                                self.env_safety[i] = re.sub(r'\b'+propSymbol, '('+prop+')', \
                                                                self.env_safety[i])
                                symfound = True

                    if (isinstance(self.sys_safety, str)):
                        if (len(re.findall(r'\b'+propSymbol, self.sys_safety)) > 0):
                            self.sys_safety = re.sub(r'\b'+propSymbol, '('+prop+')', self.sys_safety)
                            symfound = True
                    else:
                        for i in xrange(0, len(self.sys_safety)):
                            if (len(re.findall(r'\b'+propSymbol, self.sys_safety[i])) > 0):
                                self.sys_safety[i] = re.sub(r'\b'+propSymbol, '('+prop+')', \
                                                                self.sys_safety[i])
                                symfound = True

                    if (isinstance(self.env_prog, str)):
                        if (len(re.findall(r'\b'+propSymbol, self.env_prog)) > 0):
                            self.env_prog = re.sub(r'\b'+propSymbol, '('+prop+')', self.env_prog)
                            symfound = True
                    else:
                        for i in xrange(0, len(self.env_prog)):
                            if (len(re.findall(r'\b'+propSymbol, self.env_prog[i])) > 0):
                                self.env_prog[i] = re.sub(r'\b'+propSymbol, '('+prop+')', \
                                                              self.env_prog[i])
                                symfound = True

                    if (isinstance(self.sys_prog, str)):
                        if (len(re.findall(r'\b'+propSymbol, self.sys_prog)) > 0):
                            self.sys_prog = re.sub(r'\b'+propSymbol, '('+prop+')', self.sys_prog)
                            symfound = True
                    else:
                        for i in xrange(0, len(self.sys_prog)):
                            if (len(re.findall(r'\b'+propSymbol, self.sys_prog[i])) > 0):
                                self.sys_prog[i] = re.sub(r'\b'+propSymbol, '('+prop+')', \
                                                              self.sys_prog[i])
                                symfound = True

    def to_smv(self):
        raise Exception("GRSpec.to_smv is defunct, possibly temporarily")
        # trees = []
        # # NuSMV can only handle system states
        # for s, ops in [(self.sys_init, []), (self.sys_safety, ['[]']),
        #                 (self.sys_prog, ['[]', '<>'])]:
        #     if s:
        #         if isinstance(s, str):
        #             s = [s]
        #         subtrees = []
        #         ops.reverse() # so we apply operators outwards
        #         for f in s:
        #             t = ltl_parse.parse(f)
        #             # assign appropriate temporal operators
        #             for op in ops:
        #                 t = ltl_parse.ASTUnTempOp.new(t, op)
        #             subtrees.append(t)
        #         # & together expressions
        #         t = reduce(lambda x, y: ltl_parse.ASTAnd.new(x, y), subtrees)
        #         trees.append(t)
        # # & together converted subformulae
        # return reduce(lambda x, y: ltl_parse.ASTAnd.new(x, y), trees)

    def to_jtlv(self):
        """Return specification as list of two strings [assumption, guarantee].
        """
        spec = ['', '']
        # env init
        if (isinstance(self.env_init, str)):
            if (len(self.env_init) > 0 and not self.env_init.isspace()):
                spec[0] += '-- valid initial env states\n'
                spec[0] += '\t' + self.env_init
        else:
            desc_added = False
            for env_init in self.env_init:
                if (len(env_init) > 0):
                    if (len(spec[0]) > 0):
                        spec[0] += ' & \n'
                    if (not desc_added):
                        spec[0] += '-- valid initial env states\n'
                        desc_added = True
                    spec[0] += '\t' + env_init
        
        # env safety
        if (isinstance(self.env_safety, str)):
            if (len(self.env_safety) > 0):
                if (len(spec[0]) > 0):
                    spec[0] += ' & \n'
                spec[0] += '-- safety assumption on environment\n'
                spec[0] += '\t[](' + self.env_safety + ')'
        else:
            desc_added = False
            for env_safety in self.env_safety:
                if (len(env_safety) > 0):
                    if (len(spec[0]) > 0):
                        spec[0] += ' & \n'
                    if (not desc_added):
                        spec[0] += '-- safety assumption on environment\n'
                        desc_added = True
                    spec[0] += '\t[](' + env_safety + ')'

        # env progress
        if (isinstance(self.env_prog, str)):
            if (len(self.env_prog) > 0):
                if (len(spec[0]) > 0):
                    spec[0] += ' & \n'
                spec[0] += '-- justice assumption on environment\n'
                spec[0] += '\t[]<>(' + self.env_prog + ')'
        else:
            desc_added = False
            for prog in self.env_prog:
                if (len(prog) > 0):
                    if (len(spec[0]) > 0):
                        spec[0] += ' & \n'
                    if (not desc_added):
                        spec[0] += '-- justice assumption on environment\n'
                        desc_added = True
                    spec[0] += '\t[]<>(' + prog + ')'

        # sys init
        if (isinstance(self.sys_init, str)):
            if (len(self.sys_init) > 0 and not self.sys_init.isspace()):
                spec[1] += '-- valid initial system states\n'
                spec[1] += '\t' + self.sys_init
        else:
            desc_added = False
            for sys_init in self.sys_init:
                if (len(sys_init) > 0):
                    if (len(spec[1]) > 0):
                        spec[1] += ' & \n'
                    if (not desc_added):
                        spec[1] += '-- valid initial system states\n'
                        desc_added = True
                    spec[1] += '\t' + sys_init

        # sys safety
        if (isinstance(self.sys_safety, str)):
            if (len(self.sys_safety) > 0):
                if (len(spec[1]) > 0):
                    spec[1] += ' & '
                spec[1] += '-- safety requirement on system\n'
                spec[1] +='\t[](' + self.sys_safety + ')'
        else:
            desc_added = False
            for sys_safety in self.sys_safety:
                if (len(sys_safety) > 0):
                    if (len(spec[1]) > 0):
                        spec[1] += ' & \n'
                    if (not desc_added):
                        spec[1] += '-- safety requirement on system\n'
                        desc_added = True
                    spec[1] += '\t[](' + sys_safety + ')'

        # sys progress
        if (isinstance(self.sys_prog, str)):
            if (len(self.sys_prog) > 0):
                if (len(spec[1]) > 0):
                    spec[1] += ' & \n'
                spec[1] += '-- progress requirement on system\n'
                spec[1] += '\t[]<>(' + self.sys_prog + ')'
        else:
            desc_added = False
            for prog in self.sys_prog:
                if (len(prog) > 0):
                    if (len(spec[1]) > 0):
                        spec[1] += ' & \n'
                    if (not desc_added):
                        spec[1] += '-- progress requirement on system\n'
                        desc_added = True
                    spec[1] += '\t[]<>(' + prog + ')'
        return spec


    def to_gr1c(self):
        """Dump to gr1c specification string."""
        output = "ENV: "+" ".join(self.env_vars)+";\n"
        output += "SYS: "+" ".join(self.sys_vars)+";\n\n"

        if isinstance(self.env_init, str):
            output += "ENVINIT: "+self.env_init+";\n"
        else:
            output += "ENVINIT: "+"\n& ".join(["("+s+")" for s in self.env_init])+";\n"
        if len(self.env_safety) == 0:
            output += "ENVTRANS:;\n"
        elif isinstance(self.env_safety, str):
            output += "ENVTRANS: []"+self.env_safety+";\n"
        else:
            output += "ENVTRANS: "+"\n& ".join(["[]("+s+")" for s in self.env_safety])+";\n"
        if len(self.env_prog) == 0:
            output += "ENVGOAL:;\n\n"
        elif isinstance(self.env_prog, str):
            output += "ENVGOAL: []<>"+self.env_prog+";\n\n"
        else:
            output += "ENVGOAL: "+"\n& ".join(["[]<>("+s+")" for s in self.env_prog])+";\n\n"
        
        if isinstance(self.sys_init, str):
            output += "SYSINIT: "+self.sys_init+";\n"
        else:
            output += "SYSINIT: "+"\n& ".join(["("+s+")" for s in self.sys_init])+";\n"
        if len(self.sys_safety) == 0:
            output += "SYSTRANS:;\n"
        elif isinstance(self.sys_safety, str):
            output += "SYSTRANS: []"+self.sys_safety+";\n"
        else:
            output += "SYSTRANS: "+"\n& ".join(["[]("+s+")" for s in self.sys_safety])+";\n"
        if len(self.sys_prog) == 0:
            output += "SYSGOAL:;\n"
        elif isinstance(self.sys_prog, str):
            output += "SYSGOAL: []<>"+self.sys_prog+";\n"
        else:
            output += "SYSGOAL: "+"\n& ".join(["[]<>("+s+")" for s in self.sys_prog])+";\n"
        return output
