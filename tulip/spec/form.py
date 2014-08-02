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
Formulae constituting specifications
"""
import logging
logger = logging.getLogger(__name__)

import time, re, copy

from tulip.spec import parser

def mutex(varnames):
    """Create mutual exclusion formulae from iterable of variables.

    E.g., given a set of variable names {"a", "b", "c"}, return a set
    of formulae {"a -> ! (c || b)", "c -> ! (b)"}.
    """
    mutex = set()
    numVars = len(varnames)
    varnames = list(varnames)
    for i in range(0,numVars-1):
        mut_str = varnames[i]+' -> ! ('+varnames[i+1]
        for j in range(i+2,numVars):
            mut_str += ' || '+varnames[j]
        mut_str += ')'
        mutex |= {mut_str}
    return mutex

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
    def __init__(self, formula=None, input_variables=None,
                 output_variables=None):
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

    def _domain_str(self, d):
        if d == "boolean":
            return d
        elif isinstance(d, tuple) and len(d) == 2:
            return "["+str(d[0])+", "+str(d[1])+"]"
        elif hasattr(d, "__iter__"):
            return "{"+", ".join([str(e) for e in d])+"}"
        else:
            raise ValueError("Unrecognized variable domain type.")

    def dumps(self, timestamp=False):
        """Dump TuLiP LTL file string.

        @param timestamp: If True, then add comment to file with
            current time in UTC.
        """
        if timestamp:
            output = "# Generated at "+time.strftime(
                "%Y-%m-%d %H:%M:%S UTC", time.gmtime() )+"\n"
        else:
            output = ""
        output += "0  # Version\n%%\n"
        if len(self.input_variables) > 0:
            output += "INPUT:\n"
            for (k,v) in self.input_variables.items():
                output += str(k) + " : " + self._domain_str(v) + ";\n"
        if len(self.output_variables) > 0:
            output += "\nOUTPUT:\n"
            for (k,v) in self.output_variables.items():
                output += str(k) + " : " + self._domain_str(v) + ";\n"
        return output + "\n%%\n"+self.formula

    def check_form(self, check_undeclared_identifiers=False):
        """Verify formula syntax and type-check variable domains.

        Return True iff OK.

        UNDER DEVELOPMENT; function signature may change without
        notice.  Calling will result in NotImplementedError.
        """
        raise NotImplementedError

    @staticmethod
    def loads(s):
        """Create LTL object from TuLiP LTL file string."""
        for line in s.splitlines():
            line = line.strip()  # Ignore leading and trailing whitespace
            comment_ind = line.find("#")
            if comment_ind > -1:
                line = line[:comment_ind]
            if len(line) > 0:  # Version number is the first nonblank line
                try:
                    version = int(line)
                except ValueError:
                    raise ValueError("Malformed version number.")
                if version != 0:
                    raise ValueError("Unrecognized version number: " +
                        str(version))
                break
        try:
            s = re.sub(r"#.*(\n|$)", "", s)  # Strip comments
            preamble, declar, formula = s.split("%%\n")
            input_ind = declar.find("INPUT:")
            output_ind = declar.find("OUTPUT:")
            if output_ind == -1:
                output_ind = 0  # Default is OUTPUT
            if input_ind == -1:
                input_section = ""
                output_section = declar[output_ind:]
            elif input_ind < output_ind:
                input_section = declar[input_ind:output_ind]
                output_section = declar[output_ind:]
            else:
                output_section = declar[output_ind:input_ind]
                input_section = declar[input_ind:]
            input_section = input_section.replace("INPUT:", "")
            output_section = output_section.replace("OUTPUT:", "")

            variables = [dict(), dict()]
            sections = [input_section, output_section]
            for i in range(2):
                for var_declar in sections[i].split(";"):
                    if len(var_declar.strip()) == 0:
                        continue
                    name, domain = var_declar.split(":")
                    name = name.strip()
                    domain = domain.lstrip().rstrip(";")
                    if domain[0] == "[":  # Range-type domain
                        domain = domain.split(",")
                        variables[i][name] = (int(domain[0][1:]),
                            int(domain[1][:domain[1].index("]")]))
                    elif domain[0] == "{":  # Finite set domain
                        domain.strip()
                        assert domain[-1] == "}"
                        domain = domain[1:-1]  # Remove opening, closing braces
                        variables[i][name] = list()
                        for elem in domain.split(","):
                            try:
                                variables[i][name].append(int(elem))
                            except ValueError:
                                variables[i][name].append(str(elem))
                        variables[i][name] = set(variables[i][name])
                    elif domain == "boolean":
                        variables[i][name] = domain
                    else:
                        raise TypeError

        except (ValueError, TypeError):
            raise ValueError("Malformed TuLiP LTL specification string.")

        return LTL(
            formula=formula,
            input_variables=variables[0],
            output_variables=variables[1]
        )

    @staticmethod
    def load(f):
        """Wrap L{loads} for reading from files.

        @param f: file or str.  In the latter case, attempt to open a
            file named "f" read-only.
        """
        if isinstance(f, str):
            f = open(f, "rU")
        return LTL.loads(f.read())

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

        @type sys_vars: dict or iterable
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

    @staticmethod
    def loads(s):
        """Create GRSpec object from TuLiP LTL file string.

        UNDER DEVELOPMENT; function signature may change without
        notice.  Calling will result in NotImplementedError.
        """
        raise NotImplementedError

    @staticmethod
    def load(f):
        """Wrap L{loads} for reading from files.

        UNDER DEVELOPMENT; function signature may change without
        notice.  Calling will result in NotImplementedError.
        """
        raise NotImplementedError

    def pretty(self):
        """Return pretty printing string."""
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
            if result.env_vars.has_key(varname) and \
                other.env_vars[varname] != result.env_vars[varname]:
                raise ValueError("Mismatched variable domains")
        for varname in other.sys_vars.keys():
            if result.sys_vars.has_key(varname) and \
                other.sys_vars[varname] != result.sys_vars[varname]:
                raise ValueError("Mismatched variable domains")
        result.env_vars.update(other.env_vars)
        result.sys_vars.update(other.sys_vars)
        for formula_component in ["env_init", "env_safety", "env_prog",
                                  "sys_init", "sys_safety", "sys_prog"]:
            getattr(result, formula_component).extend(
                getattr(other, formula_component))
        return result

    def to_canon(self):
        """Output formula in TuLiP LTL syntax.

        The format is described in the U{Specifications section
        <http://tulip-control.sourceforge.net/doc/specifications.html>}
        of the TuLiP User's Guide.
        """
        conj_cstr = lambda s: " && " if len(s) > 0 else ""
        assumption = ""
        if len(self.env_init) > 0:
            assumption += _conj(self.env_init)
        if len(self.env_safety) > 0:
            assumption += conj_cstr(assumption) + _conj(self.env_safety, '[]')
        if len(self.env_prog) > 0:
            assumption += conj_cstr(assumption) + _conj(self.env_prog, '[]<>')
        guarantee = ""
        if len(self.sys_init) > 0:
            guarantee += conj_cstr(guarantee) + _conj(self.sys_init)
        if len(self.sys_safety) > 0:
            guarantee += conj_cstr(guarantee) + _conj(self.sys_safety, '[]')
        if len(self.sys_prog) > 0:
            guarantee += conj_cstr(guarantee) + _conj(self.sys_prog, '[]<>')

        # Put the parts together, simplifying in special cases
        if len(guarantee) > 0:
            if len(assumption) > 0:
                return "("+assumption+") -> ("+guarantee+")"
            else:
                return guarantee
        else:
            return "True"
    
    def sym_to_prop(self, props):
        """Replace the symbols of propositions with the actual propositions.

        @type props: dict
        @param props: a dictionary describing subformula (e.g.,
            variable name) substitutions, where for each key-value
            pair, all occurrences of key are replaced with value in
            all components of this GRSpec object.  However, env_vars
            and sys_vars are not changed.
        """
        if props is None:
            return
        
        symfound = True
        while (symfound):
            for propSymbol, prop in props.iteritems():
                logger.debug('propSymbol: ' + str(propSymbol))
                logger.debug('prop: ' + str(prop))
                
                if not isinstance(propSymbol, str):
                    raise TypeError('propSymbol: ' + str(propSymbol) +
                                    'is not a string.')
                
                if propSymbol[-1] != "'":  # To handle gr1c primed variables
                    propSymbol += r"\b"
                logger.debug('\t' + propSymbol + ' -> ' + prop)
                
                symfound  = _sub_all(self.env_init, propSymbol, prop)
                symfound |= _sub_all(self.env_safety, propSymbol, prop)
                symfound |= _sub_all(self.env_prog, propSymbol, prop)
                
                symfound |= _sub_all(self.sys_init, propSymbol, prop)
                symfound |= _sub_all(self.sys_safety, propSymbol, prop)
                symfound |= _sub_all(self.sys_prog, propSymbol, prop)
    
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

        Format is that of JTLV.  Cf. L{interfaces.jtlv}.
        """
        spec = ['', '']
        desc_added = False
        for env_init in self.env_init:
            if (len(env_init) > 0):
                if (len(spec[0]) > 0):
                    spec[0] += ' && \n'
                if (not desc_added):
                    spec[0] += '-- valid initial env states\n'
                    desc_added = True
                spec[0] += '\t' + parser.parse(env_init).to_jtlv()

        desc_added = False
        for env_safety in self.env_safety:
            if (len(env_safety) > 0):
                if (len(spec[0]) > 0):
                    spec[0] += ' && \n'
                if (not desc_added):
                    spec[0] += '-- safety assumption on environment\n'
                    desc_added = True
                env_safety = parser.parse(env_safety).to_jtlv()
                spec[0] += '\t[](' +re.sub(r"next\s*\(", "next(", env_safety) +')'

        desc_added = False
        for prog in self.env_prog:
            if (len(prog) > 0):
                if (len(spec[0]) > 0):
                    spec[0] += ' && \n'
                if (not desc_added):
                    spec[0] += '-- justice assumption on environment\n'
                    desc_added = True
                spec[0] += '\t[]<>(' + parser.parse(prog).to_jtlv() + ')'

        desc_added = False
        for sys_init in self.sys_init:
            if (len(sys_init) > 0):
                if (len(spec[1]) > 0):
                    spec[1] += ' && \n'
                if (not desc_added):
                    spec[1] += '-- valid initial system states\n'
                    desc_added = True
                spec[1] += '\t' + parser.parse(sys_init).to_jtlv()

        desc_added = False
        for sys_safety in self.sys_safety:
            if (len(sys_safety) > 0):
                if (len(spec[1]) > 0):
                    spec[1] += ' && \n'
                if (not desc_added):
                    spec[1] += '-- safety requirement on system\n'
                    desc_added = True
                sys_safety = parser.parse(sys_safety).to_jtlv()
                spec[1] += '\t[](' +re.sub(r"next\s*\(", "next(", sys_safety) +')'

        desc_added = False
        for prog in self.sys_prog:
            if (len(prog) > 0):
                if (len(spec[1]) > 0):
                    spec[1] += ' && \n'
                if (not desc_added):
                    spec[1] += '-- progress requirement on system\n'
                    desc_added = True
                spec[1] += '\t[]<>(' + parser.parse(prog).to_jtlv() + ')'
        return spec

    def to_gr1c(self):
        """Dump to gr1c specification string.

        Cf. L{interfaces.gr1c}.
        """
        def _to_gr1c_print_vars(vardict):
            output = ""
            for variable, domain in vardict.items():
                if domain == "boolean":
                    output += " "+variable
                elif isinstance(domain, tuple) and len(domain) == 2:
                    output += " "+variable+" ["+str(domain[0]) +\
                        ", "+str(domain[1])+"]"
                else:
                    raise ValueError("Domain type unsupported by gr1c: " +
                        str(domain))
            return output
        
        tmp = finite_domain2ints(self)
        if tmp is not None:
            return tmp.to_gr1c()
        
        output = "ENV:"+_to_gr1c_print_vars(self.env_vars)+";\n"
        output += "SYS:"+_to_gr1c_print_vars(self.sys_vars)+";\n"

        output += "ENVINIT: "+"\n& ".join([
            "("+parser.parse(s).to_gr1c()+")"
            for s in self.env_init
        ]) + ";\n"
        if len(self.env_safety) == 0:
            output += "ENVTRANS:;\n"
        else:
            output += "ENVTRANS: "+"\n& ".join([
                "[]("+parser.parse(s).to_gr1c()+")"
                for s in self.env_safety
            ]) + ";\n"
        if len(self.env_prog) == 0:
            output += "ENVGOAL:;\n\n"
        else:
            output += "ENVGOAL: "+"\n& ".join([
                "[]<>("+parser.parse(s).to_gr1c()+")"
                for s in self.env_prog
            ]) + ";\n\n"
        
        output += "SYSINIT: "+"\n& ".join([
            "("+parser.parse(s).to_gr1c()+")"
            for s in self.sys_init
        ]) + ";\n"
        if len(self.sys_safety) == 0:
            output += "SYSTRANS:;\n"
        else:
            output += "SYSTRANS: "+"\n& ".join([
                "[]("+parser.parse(s).to_gr1c()+")"
                for s in self.sys_safety
            ]) + ";\n"
        if len(self.sys_prog) == 0:
            output += "SYSGOAL:;\n"
        else:
            output += "SYSGOAL: "+"\n& ".join([
                "[]<>("+parser.parse(s).to_gr1c()+")"
                for s in self.sys_prog
            ]) + ";\n"
        return output
    
    def evaluate(self, var_values):
        """Evaluate env_init, sys_init, given a valuation of variables.
        
        Returns the Boolean value of the subformulas env_init, sys_init,
        resulting from substitution of env_vars and sys_vars symbols
        by the given assignment var_values of values to them.
        
        Note: variable value type checking not implemented yet.
        
        @param var_values: valuation of env_vars and sys_vars
        @type var_values: {'var_name':'var_value', ...}
        
        @return: truth values of spec parts::
        
            {'env_init' : env_init[var_values],
             'sys_init' : sys_init[var_values] }
        
        @rtype: dict
        """
        cp = self.copy()
        cp.sym_to_prop(var_values)
        
        env_init = _eval_formula(_conj(cp.env_init) )
        sys_init = _eval_formula(_conj(cp.sys_init) )
        
        return {'env_init':env_init, 'sys_init':sys_init}

def _eval_formula(f):
    f = re.sub(r'\|\|', ' or ', f)
    f = re.sub(r'&&', ' and ', f)
    f = re.sub(r'!', ' not ', f)
    f = re.sub(r'=', ' == ', f)
    
    if re.findall(r'->', f) or re.findall(r'<->', f):
            raise NotImplementedError('todo: Eval of -> and <->')
    
    if len(f) > 0:
        return eval(f)
    else:
        return True

def _sub_all(formula, propSymbol, prop):
    symfound = False
    for i in xrange(0, len(formula)):
        if (len(re.findall(r'\b'+propSymbol, formula[i])) > 0):
            formula[i] = re.sub(r'\b'+propSymbol, '('+prop+')',
                                      formula[i])
            symfound = True
    return symfound

def _conj(iterable, unary=''):
    return ' && '.join([unary + '(' + s + ')' for s in iterable])

def finite_domain2ints(spec):
    """Replace arbitrary finite vars with int vars.
    
    Returns spec itself if it contains only int vars.
    Otherwise it returns a copy of spec with all arbitrary
    finite vars replaced by int-valued vars.
    """
    ints_only = True
    for domain in spec.env_vars.itervalues():
        if isinstance(domain, list):
            ints_only = False
            break
    if ints_only:
        for domain in spec.sys_vars.itervalues():
            if isinstance(domain, list):
                ints_only = False
                break
    
    # nothing todo ?
    if ints_only:
        return None
    
    spec0 = spec.copy()
    
    _sub_var(spec0, spec0.env_vars)
    _sub_var(spec0, spec0.sys_vars)
    
    return spec0
    
def _sub_var(spec, vars_dict):
    for variable, domain in vars_dict.items():
        if not isinstance(domain, list):
            continue
        
        if len(domain) == 0:
            raise Exception('variable: ' + str(variable) +
                            'has empty domain: ' + str(domain))
        
        logger.debug('mapping arbitrary finite domain to integers...')
        
        # integers cannot be an arbitrary finite domain,
        # neither as integers (like 1), nor as strings (like '1')
        # because they will be indistinguishable from
        # from values of gr1c integer variables        
        if any([not isinstance(x, str) for x in domain]):
            msg = 'Found non-string elements in domain: ' + str(domain) + '\n'
            msg += 'only string elements allowed in arbitrary finite domains.'
            raise TypeError(msg)
        
        # the order provided will be the map to ints
        vars_dict[variable] = (0, len(domain)-1)
        values2ints = {var:str(i) for i, var in enumerate(domain)}
        
        # replace symbols by ints
        spec.sym_to_prop(values2ints)
