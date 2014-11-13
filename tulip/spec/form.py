# Copyright (c) 2011-2014 by California Institute of Technology
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
from __future__ import absolute_import
import logging
logger = logging.getLogger(__name__)
import warnings
import pprint
import time
import re
import copy
from tulip.spec import parser
from . import ast


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
            formula = ''
        if input_variables is None:
            input_variables = dict()
        if output_variables is None:
            output_variables = dict()

        self.formula = formula
        self.input_variables = input_variables
        self.output_variables = output_variables

        self.check_vars()

    def __str__(self):
        return str(self.formula)

    def _domain_str(self, d):
        if d == 'boolean':
            return d
        elif isinstance(d, tuple) and len(d) == 2:
            return '[' + str(d[0]) + ', ' + str(d[1]) + ']'
        elif hasattr(d, '__iter__'):
            return '{' + ', '.join([str(e) for e in d]) + '}'
        else:
            raise ValueError('Unrecognized variable domain type.')

    def dumps(self, timestamp=False):
        """Dump TuLiP LTL file string.

        @param timestamp: If True, then add comment to file with
            current time in UTC.
        """
        if timestamp:
            output = '# Generated at ' + time.strftime(
                '%Y-%m-%d %H:%M:%S UTC', time.gmtime()) + '\n'
        else:
            output = ''
        output += '0  # Version\n%%\n'
        if self.input_variables:
            output += 'INPUT:\n'
            for k, v in self.input_variables.items():
                output += str(k) + ' : ' + self._domain_str(v) + ';\n'
        if self.output_variables:
            output += '\nOUTPUT:\n'
            for k, v in self.output_variables.items():
                output += str(k) + ' : ' + self._domain_str(v) + ';\n'
        return output + '\n%%\n' + self.formula

    def check_vars(self):
        """Raise Exception if variabe definitions are invalid.

        Checks:

          - env and sys have common vars
          - some var is also a possible value of some var (including itself)
            (of arbitrary finite data type)
        """
        common_vars = {x for x in self.input_variables
                       if x in self.output_variables}
        if common_vars:
            raise Exception('Env and sys have variables in common: ' +
                            str(common_vars))

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
            comment_ind = line.find('#')
            if comment_ind > -1:
                line = line[:comment_ind]
            if line:  # Version number is the first nonblank line
                try:
                    version = int(line)
                except ValueError:
                    raise ValueError("Malformed version number.")
                if version != 0:
                    raise ValueError("Unrecognized version number: " +
                                     str(version))
                break
        try:
            s = re.sub(r'#.*(\n|$)', '', s)  # Strip comments
            preamble, declar, formula = s.split('%%\n')
            input_ind = declar.find('INPUT:')
            output_ind = declar.find('OUTPUT:')
            if output_ind == -1:
                output_ind = 0  # Default is OUTPUT
            if input_ind == -1:
                input_section = ''
                output_section = declar[output_ind:]
            elif input_ind < output_ind:
                input_section = declar[input_ind:output_ind]
                output_section = declar[output_ind:]
            else:
                output_section = declar[output_ind:input_ind]
                input_section = declar[input_ind:]
            input_section = input_section.replace('INPUT:', '')
            output_section = output_section.replace('OUTPUT:', '')

            variables = [dict(), dict()]
            sections = [input_section, output_section]
            for i in range(2):
                for var_declar in sections[i].split(';'):
                    if not var_declar.strip():
                        continue
                    name, domain = var_declar.split(':')
                    name = name.strip()
                    domain = domain.lstrip().rstrip(';')
                    if domain[0] == '[':  # Range-type domain
                        domain = domain.split(',')
                        variables[i][name] = (
                            int(domain[0][1:]),
                            int(domain[1][:domain[1].index(']')])
                        )
                    elif domain[0] == '{':  # Finite set domain
                        domain.strip()
                        assert domain[-1] == '}'
                        domain = domain[1:-1]  # Remove opening, closing braces
                        variables[i][name] = list()
                        for elem in domain.split(','):
                            try:
                                variables[i][name].append(int(elem))
                            except ValueError:
                                variables[i][name].append(str(elem))
                        variables[i][name] = set(variables[i][name])
                    elif domain == 'boolean':
                        variables[i][name] = domain
                    else:
                        raise TypeError

        except (ValueError, TypeError):
            raise ValueError('Malformed TuLiP LTL specification string.')

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
            f = open(f, 'rU')
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
        self._ast = dict()
        self._cache = {
            'string': dict(),
            'jtlv': dict(),
            'gr1c': dict(),
            'slugs': dict()
        }
        self._bool_int = dict()
        self._parts = {
            x + y
            for x in {'env_', 'sys_'}
            for y in {'init', 'safety', 'prog'}
        }

        if env_vars is None:
            env_vars = dict()
        elif not isinstance(env_vars, dict):
            env_vars = dict([(v, 'boolean') for v in env_vars])
        if sys_vars is None:
            sys_vars = dict()
        elif not isinstance(sys_vars, dict):
            sys_vars = dict([(v, 'boolean') for v in sys_vars])

        self.env_vars = copy.deepcopy(env_vars)
        self.sys_vars = copy.deepcopy(sys_vars)

        self.env_init = env_init
        self.sys_init = sys_init
        self.env_safety = env_safety
        self.sys_safety = sys_safety
        self.env_prog = env_prog
        self.sys_prog = sys_prog

        for formula_component in self._parts:
            x = getattr(self, formula_component)

            if isinstance(x, str):
                if not x:
                    setattr(self, formula_component, [])
                else:
                    setattr(self, formula_component, [x])
            elif not isinstance(x, list):
                setattr(self, formula_component, list(x))

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

        output = 'ENVIRONMENT VARIABLES:\n'
        if self.env_vars:
            for k, v in self.env_vars.iteritems():
                output += '\t' + str(k) + '\t' + str(v) + '\n'
        else:
            output += '\t(none)\n'

        output += '\nSYSTEM VARIABLES:\n'
        if self.sys_vars:
            for k, v in self.sys_vars.iteritems():
                output += '\t' + str(k) + '\t' + str(v) + '\n'
        else:
            output += '\t(none)\n'

        output += '\nFORMULA:\n'

        output += 'ASSUMPTION:\n'
        if self.env_init:
            output += (
                '    INITIAL\n\t  ' +
                '\n\t& '.join([
                    '(' + f + ')' for f in self.env_init
                ]) + '\n'
            )
        if self.env_safety:
            output += (
                '    SAFETY\n\t  []' +
                '\n\t& []'.join([
                    '(' + f + ')' for f in self.env_safety
                ]) + '\n'
            )
        if self.env_prog:
            output += (
                '    LIVENESS\n\t  []<>' +
                '\n\t& []<>'.join([
                    '(' + f + ')' for f in self.env_prog
                ]) + '\n'
            )

        output += 'GUARANTEE:\n'
        if self.sys_init:
            output += (
                '    INITIAL\n\t  '
                '\n\t& '.join([
                    '(' + f + ')' for f in self.sys_init
                ]) + '\n'
            )
        if self.sys_safety:
            output += (
                '    SAFETY\n\t  []'
                '\n\t& []'.join([
                    '(' + f + ')' for f in self.sys_safety
                ]) + '\n'
            )
        if self.sys_prog:
            output += (
                '    LIVENESS\n\t  []<>'
                '\n\t& []<>'.join([
                    '(' + f + ')' for f in self.sys_prog
                ]) + '\n'
            )
        return output

    def check_form(self):
        self.formula = self.to_canon()
        return LTL.check_form(self)

    def copy(self):
        return GRSpec(
            env_vars=dict(self.env_vars),
            sys_vars=dict(self.sys_vars),
            env_init=copy.copy(self.env_init),
            env_safety=copy.copy(self.env_safety),
            env_prog=copy.copy(self.env_prog),
            sys_init=copy.copy(self.sys_init),
            sys_safety=copy.copy(self.sys_safety),
            sys_prog=copy.copy(self.sys_prog)
        )

    def __or__(self, other):
        """Create union of two specifications."""
        result = self.copy()

        if not isinstance(other, GRSpec):
            raise TypeError('type(other) must be GRSpec')

        # common vars have same types ?
        for varname in set(other.env_vars) & set(result.env_vars):
            if other.env_vars[varname] != result.env_vars[varname]:
                raise ValueError('Mismatched variable domains')

        for varname in set(other.sys_vars) & set(result.sys_vars):
            if other.sys_vars[varname] != result.sys_vars[varname]:
                raise ValueError('Mismatched variable domains')

        result.env_vars.update(other.env_vars)
        result.sys_vars.update(other.sys_vars)

        for x in self._parts:
            getattr(result, x).extend(getattr(other, x))

        return result

    def to_canon(self):
        """Output formula in TuLiP LTL syntax.

        The format is described in the U{Specifications section
        <http://tulip-control.sourceforge.net/doc/specifications.html>}
        of the TuLiP User's Guide.
        """
        conj_cstr = lambda s: ' && ' if s else ''

        assumption = ''
        if self.env_init:
            assumption += _conj(self.env_init)
        if self.env_safety:
            assumption += conj_cstr(assumption) + _conj(self.env_safety, '[]')
        if self.env_prog:
            assumption += conj_cstr(assumption) + _conj(self.env_prog, '[]<>')

        guarantee = ''
        if self.sys_init:
            guarantee += conj_cstr(guarantee) + _conj(self.sys_init)
        if self.sys_safety:
            guarantee += conj_cstr(guarantee) + _conj(self.sys_safety, '[]')
        if self.sys_prog:
            guarantee += conj_cstr(guarantee) + _conj(self.sys_prog, '[]<>')

        # Put the parts together, simplifying in special cases
        if guarantee:
            if assumption:
                return '(' + assumption + ') -> (' + guarantee + ')'
            else:
                return guarantee
        else:
            return 'True'


    def to_jtlv(self):
        """Return specification as list of two strings [assumption, guarantee].

        Format is that of JTLV.  Cf. L{interfaces.jtlv}.
        """
        logger.info('translate to jtlv...')
        _finite_domain2ints(self)

        f = self._jtlv_str

        parts = [f(self.env_init, 'valid initial env states', ''),
                 f(self.env_safety, 'safety assumption on environment', '[]'),
                 f(self.env_prog, 'justice assumption on environment', '[]<>')]

        assumption = ' & \n'.join(x for x in parts if x)

        parts = [f(self.sys_init, 'valid initial system states', ''),
                 f(self.sys_safety, 'safety requirement on system', '[]'),
                 f(self.sys_prog, 'progress requirement on system', '[]<>')]

        guarantee = ' & \n'.join(x for x in parts if x)

        return (assumption, guarantee)

    def _jtlv_str(self, m, comment, prefix='[]<>'):
        # no clauses ?
        if not m:
            return ''

        w = []
        for x in m:
            logger.debug('translate clause: ' + str(x))

            if not x:
                continue

            c = _to_lang(self, x, 'jtlv')

            # collapse any whitespace between any
            # "next" operator that precedes parenthesis
            if prefix == '[]':
                c = re.sub(r'next\s*\(', 'next(', c)

            w.append('\t{prefix}({formula})'.format(prefix=prefix, formula=c))

        return '-- {comment}\n{formula}'.format(
            comment=comment, formula=' & \n'.join(w)
        )

    def to_gr1c(self):
        """Dump to gr1c specification string.

        Cf. L{interfaces.gr1c}.
        """
        logger.info('translate to gr1c...')

        def _to_gr1c_print_vars(vardict):
            output = ''
            for var, dom in vardict.iteritems():
                if dom == 'boolean':
                    output += ' ' + var
                elif isinstance(dom, tuple) and len(dom) == 2:
                    output += ' %s [%d, %d]' % (var, dom[0], dom[1])
                elif isinstance(dom, list) and len(dom) > 0:
                    int_dom = convert_domain(dom)
                    output += ' %s [%d, %d]' % (var, int_dom[0], int_dom[1])
                else:
                    raise ValueError('Domain not supported by gr1c: ' +
                                     str(dom))
            return output

        _finite_domain2ints(self)

        output = (
            'ENV:' + _to_gr1c_print_vars(self.env_vars) + ';\n' +
            'SYS:' + _to_gr1c_print_vars(self.sys_vars) + ';\n' +

            self._gr1c_str(self.env_init, 'ENVINIT', '') +
            self._gr1c_str(self.env_safety, 'ENVTRANS', '[]') +
            self._gr1c_str(self.env_prog, 'ENVGOAL', '[]<>') + '\n' +

            self._gr1c_str(self.sys_init, 'SYSINIT', '') +
            self._gr1c_str(self.sys_safety, 'SYSTRANS', '[]') +
            self._gr1c_str(self.sys_prog, 'SYSGOAL', '[]<>')
        )
        return output

    def _gr1c_str(self, s, name='SYSGOAL', prefix='[]<>'):
        if not s:
            return '{name}:;\n'.format(name=name)

        f = '\n& '.join([
            prefix + '({u})'.format(u=_to_lang(self, x, 'gr1c'))
            for x in s
        ])
        return '{name}: {f};\n'.format(name=name, f=f)

    def to_slugs(self):
        """Return structured slugs spec.

        @type spec: L{GRSpec}.
        """
        _finite_domain2ints(self)

        f = self._slugs_str
        return (
            self._format_slugs_vars(self.env_vars, 'INPUT') +
            self._format_slugs_vars(self.sys_vars, 'OUTPUT') +

            f(self.env_safety, 'ENV_TRANS') +
            f(self.env_prog, 'ENV_LIVENESS') +
            f(self.env_init, 'ENV_INIT', sep='&') +

            f(self.sys_safety, 'SYS_TRANS') +
            f(self.sys_prog, 'SYS_LIVENESS') +
            f(self.sys_init, 'SYS_INIT', sep='&')
        )

    def _slugs_str(self, r, name, sep='\n'):
        if not r:
            return '[{name}]\n'.format(name=name)

        sep = ' {sep} '.format(sep=sep)
        f = sep.join(_to_lang(self, x, 'slugs') for x in r if x)
        return '[{name}]\n{f}\n\n'.format(name=name, f=f)

    def _format_slugs_vars(self, vardict, name):
        a = []
        for var, dom in vardict.iteritems():
            if dom == 'boolean':
                a.append(var)
            elif isinstance(dom, tuple) and len(dom) == 2:
                a.append('{var}: {min}...{max}'.format(
                    var=var, min=dom[0], max=dom[1])
                )
            else:
                raise ValueError('unknown domain type: {dom}'.format(dom=dom))
        return '[{name}]\n{vars}\n\n'.format(name=name, vars='\n'.join(a))

    def ast(self, x):
        """Return AST corresponding to formula x.

        If AST for formula C{x} has already been computed earlier,
        then return cached result.
        """
        if logger.getEffectiveLevel() <= logging.DEBUG:
            logger.debug('current cache of ASTs:\n' +
                         pprint.pformat(self._ast) + 3 * '\n')
            logger.debug('check if: ' + str(x) + ', is in cache.')
        if x in self._ast:
            logger.debug(str(x) + ' is already in cache')
        else:
            logger.info('AST cache does not contain:\n\t' + str(x) +
                        '\nNeed to parse.')
            self.parse()
        return self._ast[x]

    def parse(self):
        """Parse each clause and store it.

        The AST resulting from each clause is stored
        in the C{dict} attribute C{ast}.
        """
        logger.info('parsing ASTs to cache them...')
        vardoms = dict(self.env_vars)
        vardoms.update(self.sys_vars)
        # parse new clauses and cache the resulting ASTs
        for p in self._parts:
            s = getattr(self, p)
            for x in s:
                if x in self._ast:
                    logger.debug(str(x) + ' is already in cache')
                    continue
                logger.debug('parse: ' + str(x))
                tree = parser.parse(x)

                ast.check_for_undefined_identifiers(tree, vardoms)

                self._ast[x] = tree
        # rm cached ASTs that correspond to deleted clauses
        self._collect_cache_garbage(self._ast)
        logger.info('done parsing ASTs.\n')

    def _collect_cache_garbage(self, cache):
        logger.info('collecting garbage from GRSpec cache...')
        # rm cached ASTs that correspond to deleted clauses
        s = set(cache)
        for p in self._parts:
            # emptied earlier ?
            if not s:
                break
            w = getattr(self, p)
            # exclude given formulas
            s.difference_update(w)
            # exclude int/bool-only forms of formulas
            s.difference_update({self._bool_int.get(x) for x in w})
        for x in s:
            cache.pop(x)
        logger.info('cleaned ' + str(len(s)) + ' cached elements.\n')


    def sub_values(self, var_values):
        """Substitute given values for variables.

        Note that there are three ways to substitute values for variables:

          - syntactic using this function

          - no substitution by user code, instead flatten to python and
            use C{eval} together with a C{dict} defining
            the values of variables, as done in L{eval_init}.

        For converting non-integer finite types to
        integer types, use L{replace_finite_by_int}.

        @return: C{dict} of ASTs after the substitutions,
            keyed by original clause (before substitution).
        """
        logger.info('substitute values for variables...')

        a = copy.deepcopy(self._ast)

        for formula, tree in a.iteritems():
            a[formula] = ast.sub_values(tree, var_values)

        logger.info('done with substitutions.\n')
        return a

    def compile_init(self, no_str):
        """Compile python expression for initial conditions.

        The returned bytecode can be used with C{eval}
        and a C{dict} assigning values to variables.
        Its value is the conjunction of C{env_init} and C{sys_init}.

        Use the corresponding python data types
        for the C{dict} values:

              - C{bool} for Boolean variables
              - C{int} for integers
              - C{str} for arbitrary finite types

        @param no_str: if True, then compile the formula
            where all string variables have been replaced by integers.
            Otherwise compile the original formula containing strings.

        @return: python expression compiled for C{eval}
        @rtype: C{code}
        """
        clauses = self.env_init + self.sys_init

        if no_str:
            clauses = [self._bool_int[x] for x in clauses]

        logger.info('clauses to compile: ' + str(clauses))

        c = [self.ast(x).to_python() for x in clauses]
        logger.info('after to_python: ' + str(c))

        s = _conj(c, op='and')

        if not s:
            s = 'True'

        return compile(s, '<string>', 'eval')


def _conj(iterable, unary='', op='&&'):
    return (' ' + op + ' ').join([unary + '(' + s + ')' for s in iterable])

def _finite_domain2ints(spec):
    """Replace arbitrary finite vars with int vars.

    Returns spec itself if it contains only int vars.
    Otherwise it returns a copy of spec with all arbitrary
    finite vars replaced by int-valued vars.
    """
    logger.info('convert string variables to integers...')

    vars_dict = dict(spec.env_vars)
    vars_dict.update(spec.sys_vars)

    fvars = {v: d for v, d in vars_dict.iteritems() if isinstance(d, list)}

    # replace symbols by ints
    for p in spec._parts:
        for x in getattr(spec, p):
            if spec._bool_int.get(x) in spec._ast:
                logger.debug(str(x) + ' is in _bool_int cache')
                continue
            else:
                logger.debug(str(x) + ' is not in _bool_int cache')

            # get AST
            a = spec.ast(x)

            # create AST copy with int and bool vars only
            a = copy.deepcopy(a)
            a.sub_constants(fvars)

            # formula of int/bool AST
            f = str(a)
            spec._ast[f] = a  # cache

            # remember map from clauses to int/bool clauses
            spec._bool_int[x] = f

    logger.info('done converting to integer variables.\n')

def _to_lang(spec, s, lang):
        """Get cached jtlv string.

        If not found, then it translates all clauses to
        jtlv syntax and caches the results.

        It also collects garbage from the jtlv cache.
        """
        if spec._bool_int.get(s) in spec._cache[lang]:
            logger.info('{s} is in {lang} cache.'.format(s=s, lang=lang))
        else:
            logger.info(
                ('{s} not found in {lang} cache, '
                'have to flatten...').format(s=s, lang=lang)
            )

            for p in spec._parts:
                for x in getattr(spec, p):
                    z = spec._bool_int[x]

                    if z in spec._cache[lang]:
                        continue

                    if lang == 'gr1c':
                        w = spec.ast(z).to_gr1c()
                    elif lang == 'slugs':
                        w = spec.ast(z).to_slugs()
                    elif lang == 'jtlv':
                        w = spec.ast(z).to_jtlv(spec.env_vars,
                                                spec.sys_vars)
                    else:
                        raise Exception('Unknown language')
                    spec._cache[lang][z] = w

            logger.info('collect garbage from {0} cache.\n'.format(lang))
            spec._collect_cache_garbage(spec._cache[lang])

        return spec._cache[lang][spec._bool_int[s]]

def replace_dependent_vars(spec, bool2form):
    logger.debug('replacing dependent variables using map:\n\t' +
                 str(bool2form))
    vs = dict(spec.env_vars)
    vs.update(spec.sys_vars)
    logger.debug('variables:\n\t' + str(vs))
    bool2subtree = dict()
    for boolvar, formula in bool2form.iteritems():
        logger.debug('checking var: ' + str(boolvar))
        if boolvar in vs:
            assert(vs[boolvar] == 'boolean')
            logger.debug(str(boolvar) + ' is indeed Boolean')
        else:
            logger.debug('spec does not contain var: ' + str(boolvar))
        tree = parser.parse(formula)
        bool2subtree[boolvar] = tree

    for s in {'env_init', 'env_safety', 'env_prog',
              'sys_init', 'sys_safety', 'sys_prog'}:
        part = getattr(spec, s)
        new = []
        for clause in part:
            logger.debug('replacing in clause:\n\t' + clause)
            tree = spec.ast(clause)
            ast.sub_bool_with_subtree(tree, bool2subtree)

            f = str(tree)
            new.append(f)
            logger.debug('caluse tree after replacement:\n\t' + f)
        setattr(spec, s, new)


def _check_var_name_conflict(f, varname):
    t = parser.parse(f)
    v = {x.val for x in t.get_vars()}

    if varname in v:
        raise ValueError('var name "' + varname + '" already used')
    return v


def convert_domain(dom):
    """Return equivalent integer domain if C{dom} contais strings.

    @type dom: C{list} of C{str}
    @rtype: C{'boolean'} or C{(min_int, max_int)}
    """
    # not a string variable ?
    if not isinstance(dom, list):
        return dom

    return (0, len(dom) - 1)

def infer_constants(formula, variables):
    """Enclose all non-variable names in quotes.

    @param formula: well-formed LTL formula
    @type formula: C{str} or L{LTL_AST}

    @param variables: domains of variables, or only their names.
        If the domains are given, then they are checked
        for ambiguities as for example a variable name
        duplicated as a possible value in the domain of
        a string variable (the same or another).

        If the names are given only, then a warning is raised,
        because ambiguities cannot be checked in that case,
        since they depend on what domains will be used.
    @type variables: C{dict} as accepted by L{GRSpec} or
        container of C{str}

    @return: C{formula} with all string literals not in C{variables}
        enclosed in double quotes
    @rtype: C{str}
    """
    import networkx as nx

    if isinstance(variables, dict):
        for var in variables:
            other_vars = dict(variables)
            other_vars.pop(var)

            check_var_conflicts({var}, other_vars)
    else:
        logger.error('infer constants does not know the variable domains.')
        warnings.warn(
            'infer_constants can give an incorrect result '
            'depending on the variable domains.\n'
            'If you give the variable domain definitions as dict, '
            'then infer_constants will check for ambiguities.'
        )

    tree = parser.parse(formula)
    old2new = dict()

    for u in tree:
        if not isinstance(u, ast.Var):
            continue

        if str(u) in variables:
            continue

        # Var (so NAME token) but not a variable
        # turn it into a string constant
        old2new[u] = ast.Const(str(u))

    nx.relabel_nodes(tree, old2new, copy=False)
    return str(tree)

def check_var_conflicts(s, variables):
    """Raise exception if set intersects existing variable name, or values.

    Values refers to arbitrary finite data types.

    @param s: set

    @param variables: definitions of variable types
    @type variables: C{dict}
    """
    # check conflicts with variable names
    vars_redefined = {x for x in s if x in variables}

    if vars_redefined:
        raise Exception('Variables redefined: ' + str(vars_redefined))

    # check conflicts with values of arbitrary finite data types
    for var, domain in variables.iteritems():
        # not arbitrary finite type ?
        if not isinstance(domain, list):
            continue

        # var has arbitrary finite type
        conflicting_values = {x for x in s if x in domain}

        if conflicting_values:
            raise Exception('Values redefined: ' + str(conflicting_values))
