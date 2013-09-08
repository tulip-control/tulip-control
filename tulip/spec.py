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

import time
import re, copy


class LTL(object):
    """LTL formula (specification)

    Minimal class that describes an LTL formula in the canonical TuLiP
    syntax.  It contains three attributes:

      - C{formula}: a C{str} of the formula.  Syntax is only enforced
        if the user requests it, e.g., using the L{check_form} method.

      - C{input_variables}: a C{dict} of variables (names given as
        strings) and their domains; each key is a variable name and
        its value (in the dictionary) is its domain.  See notes below.
        Semantically, these variables are considered to be inputs
        (i.e., uncontrolled, externally determined).

      - C{output_variables}: similar to C{input_variables}, but
        considered to be outputs, i.e., controlled, the strategy for
        setting of which we seek in formal synthesis.

    N.B., domains are specified in multiple datatypes.  The type is
    indicated below in parenthesis.  Recognized domains, along with
    examples, are:

      - boolean (C{str}); this domain is specified by C{"boolean"};
      - finite_set (C{set}); e.g., C{{1,3,5}};
      - range (C{tuple} of length 2); e.g., C{(0,15)}.

    As with the C{formula} attribute, type-checking is only performed
    if requested by the user.  E.g., any iterable can act as a
    finite_set.  However, a range domain must be a C{tuple} of length
    2; otherwise it is ambiguous with finite_set.
    """
    def __init__(self, formula=None, input_variables=None, output_variables=None):
        """Instantiate an LTL object.

        Any non-None arguments are saved to the corresponding
        attribute by reference.
        """
        if formula is None:
            formula = ""
        if input_variables is None:
            input_variables = dict()
        if output_variables is None:
            output_variables = dict()
        self.formula = formula
        self.input_variables = input_variables
        self.output_variables = output_variables

    def __str__(self):
        return str(self.formula)

    def dumps(self, timestamp=False):
        """Dump TuLiP LTL file string.

        @param timestamp: If True, then add comment to file with
            current time in UTC.
        """
        if timestamp:
            output = "# Generated at "+time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())+"\n"
        else:
            output = ""
        output += "0  # Version\n%%\n"
        if len(self.input_variables) > 0:
            output += "INPUT:\n"
            for (k,v) in self.input_variables.items():
                output += str(k) + " : " + str(v) + ";\n"
        if len(self.output_variables) > 0:
            output += "\nOUTPUT:\n"
            for (k,v) in self.output_variables.items():
                output += str(k) + " : " + str(v) + ";\n"
        return output + "\n%%\n\n"+self.formula

    def check_form(self, check_undeclared_identifiers=False):
        """Verify formula syntax and type-check variable domains.

        Return True iff OK.
        """
        raise NotImplementedError


class GRSpec(LTL):
    """GR(1) specification

    The basic form is::

      (env_init & []env_safety & []<>env_prog_1 & []<>env_prog_2 & ...)
        -> (sys_init & []sys_safety & []<>sys_prog_1 & []<>sys_prog_2 & ...)

    A GRSpec object contains the following attributes:

      - C{env_vars}: alias for C{input_variables} of L{LTL},
        concerning variables that are determined by the environment

      - C{env_init}: a list of string that specifies the assumption
        about the initial state of the environment.

      - C{env_safety}: a list of string that specifies the assumption
        about the evolution of the environment state.

      - C{env_prog}: a list of string that specifies the justice
        assumption on the environment.

      - C{sys_vars}: alias for C{output_variables} of L{LTL},
        concerning variables that are controlled by the system.

      - C{sys_init}: a list of string that specifies the requirement
        on the initial state of the system.

      - C{sys_safety}: a list of string that specifies the safety
        requirement.

      - C{sys_prog}: a list of string that specifies the progress
        requirement.

    An empty list for any formula (e.g., if env_init = []) is marked
    as "True" in the specification. This corresponds to the constant
    Boolean function, which usually means that subformula has no
    effect (is non-restrictive) on the spec.

    Consult L{GRSpec.__init__} concerning arguments at the time of
    instantiation.
    """
    def __init__(self, env_vars=None, sys_vars=None, env_init='', sys_init='',
                 env_safety='', sys_safety='', env_prog='', sys_prog=''):
        """Instantiate a GRSpec object.

        Instantiating GRSpec without arguments results in an empty
        formula.  The default domain of a variable is "boolean".

        @type env_vars: dict or iterable
        @param env_vars: If env_vars is a dictionary, then its keys
            should be variable names, and values are domains of the
            corresponding variable.  Else, if env_vars is an iterable,
            then assume all environment variables are C{boolean} (or
            "atomic propositions").  Cf. L{GRSpec} for details.

        @type sys_vars: set or dict
        @param sys_vars: Mutatis mutandis, env_vars.

        @param env_init, env_safety, env_prog, sys_init, sys_safety, sys_prog:
            A string or iterable of strings.  An empty string is
            converted to an empty list.  A string is placed in a list.
            iterables are converted to lists.  Cf. L{GRSpec}.
        """
        if env_vars is None:
            env_vars = dict()
        elif not isinstance(env_vars, dict):
            env_vars = dict([(v,"boolean") for v in env_vars])
        if sys_vars is None:
            sys_vars = dict()
        elif not isinstance(sys_vars, dict):
            sys_vars = dict([(v,"boolean") for v in sys_vars])

        self.env_vars = copy.deepcopy(env_vars)
        self.sys_vars = copy.deepcopy(sys_vars)

        self.env_init = env_init
        self.sys_init = sys_init
        self.env_safety = env_safety
        self.sys_safety = sys_safety
        self.env_prog = env_prog
        self.sys_prog = sys_prog

        for formula_component in ["env_init", "env_safety", "env_prog",
                                  "sys_init", "sys_safety", "sys_prog"]:
            if isinstance(getattr(self, formula_component), str):
                if len(getattr(self, formula_component)) == 0:
                    setattr(self, formula_component, [])
                else:
                    setattr(self, formula_component,
                            [getattr(self, formula_component)])
            elif not isinstance(getattr(self, formula_component), list):
                setattr(self, formula_component,
                        list(getattr(self, formula_component)))
            setattr(self, formula_component,
                    copy.deepcopy(getattr(self, formula_component)))

        LTL.__init__(self, formula=self.to_canon(),
                     input_variables=self.env_vars,
                     output_variables=self.sys_vars)

    def __str__(self):
        return self.to_canon()

    def dumps(self, timestamp=False):
        self.formula = self.to_canon()
        return LTL.dumps(self, timestamp=timestamp)

    def pretty(self):
        output = "ENVIRONMENT VARIABLES:\n"
        if len(self.env_vars) > 0:
            for (k,v) in self.env_vars.items():
                output += "\t"+str(k)+"\t"+str(v)+"\n"
        else:
            output += "\t(none)\n"
        output += "\nSYSTEM VARIABLES:\n"
        if len(self.sys_vars) > 0:
            for (k,v) in self.sys_vars.items():
                output += "\t"+str(k)+"\t"+str(v)+"\n"
        else:
            output += "\t(none)\n"
        output += "\nFORMULA:\n"
        output += "ASSUMPTION:\n"
        if len(self.env_init) > 0:
            output += "    INITIAL\n\t  "
            output += "\n\t& ".join(["("+f+")" for f in self.env_init])+"\n"
        if len(self.env_safety) > 0:
            output += "    SAFETY\n\t  []"
            output += "\n\t& []".join(["("+f+")" for f in self.env_safety])+"\n"
        if len(self.env_prog) > 0:
            output += "    LIVENESS\n\t  []<>"
            output += "\n\t& []<>".join(["("+f+")" for f in self.env_prog])+"\n"

        output += "GUARANTEE:\n"
        if len(self.sys_init) > 0:
            output += "    INITIAL\n\t  "
            output += "\n\t& ".join(["("+f+")" for f in self.sys_init])+"\n"
        if len(self.sys_safety) > 0:
            output += "    SAFETY\n\t  []"
            output += "\n\t& []".join(["("+f+")" for f in self.sys_safety])+"\n"
        if len(self.sys_prog) > 0:
            output += "    LIVENESS\n\t  []<>"
            output += "\n\t& []<>".join(["("+f+")" for f in self.sys_prog])+"\n"
        return output

    def check_form(self):
        self.formula = self.to_canon()
        return LTL.check_form(self)

    def copy(self):
        return GRSpec(env_vars=dict(self.env_vars),
                      sys_vars=dict(self.sys_vars),
                      env_init=copy.copy(self.env_init),
                      env_safety=copy.copy(self.env_safety),
                      env_prog=copy.copy(self.env_prog),
                      sys_init=copy.copy(self.sys_init),
                      sys_safety=copy.copy(self.sys_safety),
                      sys_prog=copy.copy(self.sys_prog))

    def __or__(self, other):
        """Create union of two specifications."""
        result = self.copy()
        for varname in other.env_vars.keys():
            if result.env_vars.has_key(varname) and other[varname] != result.env_vars[varname]:
                raise ValueError("Mismatched variable domains")
        for varname in other.sys_vars.keys():
            if result.sys_vars.has_key(varname) and other[varname] != result.sys_vars[varname]:
                raise ValueError("Mismatched variable domains")
        result.env_vars.update(other.env_vars)
        result.sys_vars.update(other.sys_vars)
        for formula_component in ["env_init", "env_safety", "env_prog",
                                  "sys_init", "sys_safety", "sys_prog"]:
            getattr(result, formula_component).extend(getattr(other, formula_component))
        return result

    def to_canon(self):
        """Output formula in TuLiP LTL syntax.

        Beware!  The canonical TuLiP LTL syntax is under active
        development and may change without notice.  Consult
        U{https://sourceforge.net/p/tulip-control/wiki/TL_formula_syntax/}
        """
        conj_cstr = lambda s: " && " if len(s) > 0 else ""
        assumption = ""
        if len(self.env_init) > 0:
            assumption += " && ".join(["("+s+")" for s in self.env_init])
        if len(self.env_safety) > 0:
            assumption += conj_cstr(assumption)+" && ".join(["[]("+s+")" for s in self.env_safety])
        if len(self.env_prog) > 0:
            assumption += conj_cstr(assumption)+" && ".join(["[]<>("+s+")" for s in self.env_prog])
        guarantee = ""
        if len(self.sys_init) > 0:
            guarantee += conj_cstr(guarantee)+" && ".join(["("+s+")" for s in self.sys_init])
        if len(self.sys_safety) > 0:
            guarantee += conj_cstr(guarantee)+" && ".join(["[]("+s+")" for s in self.sys_safety])
        if len(self.sys_prog) > 0:
            guarantee += conj_cstr(guarantee)+" && ".join(["[]<>("+s+")" for s in self.sys_prog])

        # Put the parts together, simplifying in special cases
        if len(guarantee) > 0:
            if len(assumption) > 0:
                return "("+assumption+") -> ("+guarantee+")"
            else:
                return guarantee
        else:
            return "True"


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
            if not self.env_vars.has_key(evar):
                self.env_vars[evar] = "boolean"
        for svar in s.sys_vars:
            if not self.sys_vars.has_key(svar):
                self.sys_vars[svar] = "boolean"
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

        @type props: dict
        @param props: a dictionary describing subformula (e.g.,
            variable name) substitutions, where for each key-value
            pair, all occurrences of key are replaced with value in
            all components of this GRSpec object.  However, env_vars
            and sys_vars are not changed.
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
                    for i in xrange(0, len(self.env_init)):
                        if (len(re.findall(r'\b'+propSymbol, self.env_init[i])) > 0):
                            self.env_init[i] = re.sub(r'\b'+propSymbol, '('+prop+')',
                                                      self.env_init[i])
                            symfound = True

                    for i in xrange(0, len(self.sys_init)):
                        if (len(re.findall(r'\b'+propSymbol, self.sys_init[i])) > 0):
                            self.sys_init[i] = re.sub(r'\b'+propSymbol, '('+prop+')',
                                                      self.sys_init[i])
                            symfound = True

                    for i in xrange(0, len(self.env_safety)):
                        if (len(re.findall(r'\b'+propSymbol, self.env_safety[i])) > 0):
                            self.env_safety[i] = re.sub(r'\b'+propSymbol, '('+prop+')',
                                                        self.env_safety[i])
                            symfound = True

                    for i in xrange(0, len(self.sys_safety)):
                        if (len(re.findall(r'\b'+propSymbol, self.sys_safety[i])) > 0):
                            self.sys_safety[i] = re.sub(r'\b'+propSymbol, '('+prop+')',
                                                        self.sys_safety[i])
                            symfound = True

                    for i in xrange(0, len(self.env_prog)):
                        if (len(re.findall(r'\b'+propSymbol, self.env_prog[i])) > 0):
                            self.env_prog[i] = re.sub(r'\b'+propSymbol, '('+prop+')',
                                                      self.env_prog[i])
                            symfound = True

                    for i in xrange(0, len(self.sys_prog)):
                        if (len(re.findall(r'\b'+propSymbol, self.sys_prog[i])) > 0):
                            self.sys_prog[i] = re.sub(r'\b'+propSymbol, '('+prop+')',
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

        Format is that of JTLV.  Cf. L{jtlvint}.
        """
        spec = ['', '']
        desc_added = False
        for env_init in self.env_init:
            if (len(env_init) > 0):
                if (len(spec[0]) > 0):
                    spec[0] += ' & \n'
                if (not desc_added):
                    spec[0] += '-- valid initial env states\n'
                    desc_added = True
                spec[0] += '\t' + env_init

        desc_added = False
        for env_safety in self.env_safety:
            if (len(env_safety) > 0):
                if (len(spec[0]) > 0):
                    spec[0] += ' & \n'
                if (not desc_added):
                    spec[0] += '-- safety assumption on environment\n'
                    desc_added = True
                spec[0] += '\t[](' + env_safety + ')'

        desc_added = False
        for prog in self.env_prog:
            if (len(prog) > 0):
                if (len(spec[0]) > 0):
                    spec[0] += ' & \n'
                if (not desc_added):
                    spec[0] += '-- justice assumption on environment\n'
                    desc_added = True
                spec[0] += '\t[]<>(' + prog + ')'

        desc_added = False
        for sys_init in self.sys_init:
            if (len(sys_init) > 0):
                if (len(spec[1]) > 0):
                    spec[1] += ' & \n'
                if (not desc_added):
                    spec[1] += '-- valid initial system states\n'
                    desc_added = True
                spec[1] += '\t' + sys_init

        desc_added = False
        for sys_safety in self.sys_safety:
            if (len(sys_safety) > 0):
                if (len(spec[1]) > 0):
                    spec[1] += ' & \n'
                if (not desc_added):
                    spec[1] += '-- safety requirement on system\n'
                    desc_added = True
                spec[1] += '\t[](' + sys_safety + ')'

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
        """Dump to gr1c specification string.

        Cf. L{gr1cint}.
        """
        output = "ENV: "+" ".join(self.env_vars)+";\n"
        output += "SYS: "+" ".join(self.sys_vars)+";\n\n"

        output += "ENVINIT: "+"\n& ".join(["("+s+")" for s in self.env_init])+";\n"
        if len(self.env_safety) == 0:
            output += "ENVTRANS:;\n"
        else:
            output += "ENVTRANS: "+"\n& ".join(["[]("+s+")" for s in self.env_safety])+";\n"
        if len(self.env_prog) == 0:
            output += "ENVGOAL:;\n\n"
        else:
            output += "ENVGOAL: "+"\n& ".join(["[]<>("+s+")" for s in self.env_prog])+";\n\n"
        
        output += "SYSINIT: "+"\n& ".join(["("+s+")" for s in self.sys_init])+";\n"
        if len(self.sys_safety) == 0:
            output += "SYSTRANS:;\n"
        else:
            output += "SYSTRANS: "+"\n& ".join(["[]("+s+")" for s in self.sys_safety])+";\n"
        if len(self.sys_prog) == 0:
            output += "SYSGOAL:;\n"
        else:
            output += "SYSGOAL: "+"\n& ".join(["[]<>("+s+")" for s in self.sys_prog])+";\n"
        return output
