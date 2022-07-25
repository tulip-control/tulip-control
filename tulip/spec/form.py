# Copyright (c) 2011-2015 by California Institute of Technology
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
"""Formulae constituting specifications.

what this module adds to the pure syntactic manipulations of
`transformation` is a variable table, with game semantics in particular.

Also, those modules are for manipulating expressions.
This module knows about program structure,
namely the sections of a specification file.
"""
import collections.abc as _abc
import copy
import logging
import pprint
import time
import types
import re

import tulip.spec.parser as parser
import tulip.spec.transformation as tx
import tulip.spec.translation as ts


__all__ = [
    'LTL',
    'GRSpec',
    'replace_dependent_vars']


logger = logging.getLogger(__name__)


class LTL:
    """LTL formula (specification).

    Minimal class that describes an LTL formula in the canonical TuLiP
    syntax.  It contains three attributes:

      - `formula`: a `str` of the formula.  Syntax is only enforced
        if the user requests it, e.g., using the `check_form` method.

      - `input_variables`: a `dict` of variables (names given as
        strings) and their domains; each key is a variable name and
        its value (in the dictionary) is its domain.  See notes below.
        Semantically, these variables are considered to be inputs
        (i.e., uncontrolled, externally determined).

      - `output_variables`: similar to `input_variables`, but
        considered to be outputs, i.e., controlled, the strategy for
        setting of which we seek in formal synthesis.

    N.B., domains are specified in multiple datatypes.  The type is
    indicated below in parenthesis.  Recognized domains, along with
    examples, are:

      - boolean (`str`); this domain is specified by `"boolean"`;
      - finite_set (`set`); e.g., `{1, 3, 5}`;
      - range (`tuple` of length 2); e.g., `(0, 15)`.

    As with the `formula` attribute, type-checking is only performed
    if requested by the user.  E.g., any iterable can act as a
    finite_set.  However, a range domain must be a `tuple` of length
    2; otherwise it is ambiguous with finite_set.
    """

    def __init__(
            self,
            formula=None,
            input_variables=None,
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

    def __repr__(self):
        return (
            f"LTL('{self.formula}', "
            f"input_variables={self.input_variables}, "
            f"output_variables={self.output_variables})")

    def __str__(self):
        return str(self.formula)

    def _domain_str(self, d):
        if d == 'boolean':
            return d
        elif isinstance(d, tuple) and len(d) == 2:
            return f'[{d[0]}, {d[1]}]'
        elif hasattr(d, '__iter__'):
            return '{' + ', '.join([str(e) for e in d]) + '}'
        else:
            raise ValueError('Unrecognized variable domain type.')

    def dumps(self, timestamp=False):
        """Dump TuLiP LTL file string.

        @param timestamp:
            If True, then add comment to file with
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
                output += f'{k} : {self._domain_str(v)};\n'
        if self.output_variables:
            output += '\nOUTPUT:\n'
            for k, v in self.output_variables.items():
                output += f'{k} : {self._domain_str(v)};\n'
        return f'{output}\n%%\n{self.formula}'

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
            raise Exception(
                'Env and sys have variables in '
                f'common: {common_vars}')

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
            s = re.sub(
                r' \# .* (\n | $) ',
                '',
                s,
                flags=re.VERBOSE)
                    # Strip comments
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
                            int(domain[1][:domain[1].index(']')]))
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
            output_variables=variables[1])

    @staticmethod
    def load(f):
        """Wrap `loads` for reading from files.

        @param f:
            file or str.  In the latter case, attempt to open a
            file named "f" read-only.
        """
        if isinstance(f, str):
            f = open(f, 'rU')
        return LTL.loads(f.read())


class GRSpec(LTL):
    r"""GR(1) specification.

    The basic form is:

    ```
    (env_init & []env_safety & []<>env_prog_1 & []<>env_prog_2 & ...)
    -> (sys_init & []sys_safety & []<>sys_prog_1 & []<>sys_prog_2 & ...)
    ```

    Attributes:

    - `moore`: select whether a strategy can see primed
      environment variables.

    - `plus_one`: select causal implication between
      assumptions and guarantees.

    - `qinit`: select quantification of initial values
      for variables:

      `win` below describes the set of winning states.
      `internal_init` is the initial condition for the
      internal strategy variables.
      `Op == expr` means operator `Op` is defined as
      the expression `expr`.

      - `'\A \A'`:
        `forall env_vars:  forall sys_vars:  env_init -> win`.
        `sys_init` must be empty or contain true.
        The strategy enumeration iterates through
        all assignments that satisfy
        `env_init & internal_init`.

      - `'\A \E'`:
        `forall env_vars:  exist sys_vars:  form`,
        where:
        - `form == sys_init & (env_init -> win)`
          (`plus_one is True`)
        - `form == env_init -> (sys_init & win)`
          (`plus_one is False`)
        The strategy enumeration iterates
        through all assignments that
        satisfy `\E sys_vars:  env_init`, and
        picks assignments that satisfy
        `form & internal_init`.

      - `'\E \A'`:

        ```
        exist sys_vars:  forall env_vars:  form
        ```

        where:
        - `form == sys_init & (env_init -> win)`
          (`plus_one is True`)
        - `form == env_init -> (sys_init & win)`
          (`plus_one is False`)

        The strategy enumeration picks
        an assignment that satisfies:

        ```
        internal_init & \A env_vars:  form
        ```

        and iterates through all assignments
        that satisfy `env_init`.

      - `'\E \E'`:

        ```
        exist env_vars:  exist sys_vars:  sys_init & win
        ```

        `env_init` must be empty or contain true.
        The strategy enumeration picks
        an assignment that satisfies:

        ```
        sys_init & win & internal_init
        ```

    - `env_vars`: alias for `input_variables` of `LTL`,
      concerning variables that are determined
      by the environment.

    - `env_init`: `list` of `str` that
      specifies the assumption about the
      initial state of the environment.

    - `env_safety`: `list` of `str` that
      specifies the assumption about the
      evolution of the environment state.

    - `env_prog`: `list` of `str` that
      specifies the justice assumption
      on the environment.

    - `sys_vars`: alias for `output_variables` of `LTL`,
      concerning variables that are controlled
      by the system.

    - `sys_init`: `list` of `str` that
      specifies the requirement
      on the initial state of the system.

    - `sys_safety`: `list` of `str` that
      specifies the safety requirement.

    - `sys_prog`: `list` of `str` that
      specifies the progress requirement.

    An empty list for any formula
    (e.g., if `env_init = list()`)
    is marked as `True` in the specification.

    This corresponds to the constant Boolean function,
    which usually means that subformula has no
    effect (is non-restrictive) on the spec.

    Consult `GRSpec.__init__` concerning
    arguments at the time of instantiation.
    """

    def __init__(
            self,
            env_vars:
                dict |
                _abc.Iterable |
                None=None,
            sys_vars:
                dict |
                _abc.Iterable |
                None=None,
            env_init='',
            sys_init='',
            env_safety='',
            sys_safety='',
            env_prog='',
            sys_prog='',
            moore:
                bool=True,
            plus_one:
                bool=True,
            qinit=r'\A \A',
            parser=parser):
        """Instantiate a GRSpec object.

        Instantiating GRSpec without arguments results in an empty
        formula.  The default domain of a variable is "boolean".

        @param env_vars:
            If env_vars is a dictionary, then its keys
            should be variable names, and values are domains of the
            corresponding variable.  Else, if env_vars is an iterable,
            then assume all environment variables are `boolean` (or
            "atomic propositions").  Cf. `GRSpec` for details.
        @param sys_vars:
            Mutatis mutandis, env_vars.
        @param env_init, env_safety, env_prog,
                sys_init, sys_safety, sys_prog:
            A string or iterable of strings.  An empty string is
            converted to an empty list.  A string is placed in a list.
            iterables are converted to lists.  Cf. `GRSpec`.
        @param qinit:
            see class docstring
        """
        self.parser = parser
        self._ast = dict()
        self._cache = {
            'string':
                dict(),
            'jtlv':
                dict(),
            'gr1c':
                dict(),
            'slugs':
                dict()}
        self._bool_int = dict()
        self._parts = {
            x + y
            for x in {'env_', 'sys_'}
            for y in {'init', 'safety', 'prog'}}
        self.moore = moore
        self.plus_one = plus_one
        self.qinit = qinit
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
        LTL.__init__(
            self,
            formula=self.to_canon(),
            input_variables=self.env_vars,
            output_variables=self.sys_vars)

    def declare(self, *arg, **kw):
        """Declare flexible variables.

        Positional arguments are names of Boolean-valued variables.
        Keyword arguments are names of integer or string valued
        variables, declared by a pair of integers, or a sequence of
        strings respectively.

        By default declare as system variables.
        If `env=True` then declare as environment variables.
        To declare a variable named `env`, use the constructor
        or modify directly the attributes `env_vars`, `sys_vars`.

        If already declared, the given type hints should
        match existing ones.
        """
        env = kw.pop('env', False)
        assert isinstance(env, bool), env
            # "env" not a var name
        d = dict()
        for k, v in kw.items():
            is_int = len(v) == 2 and all(
                isinstance(q, int) for q in v)
            if is_int:
                d[k] = tuple(v)
                continue
            # duck check by appending ''
            d[k] = list(s + '' for s in v)
        d.update((v, 'boolean') for v in arg)
        # `LTL.*_variables` equal these (see `self.__init__`)
        target, other = self.sys_vars, self.env_vars
        if env:
            target, other = other, target
        # redeclarations must match
        for k, v in d.items():
            assert k not in other, (k, other)
            if k in target:
                assert target[k] == v, (target[k], v)
        target.update(d)

    def __repr__(self):
        args = ',\n\n'.join([
                f'env_vars={self.env_vars!r}',
                f'sys_vars={self.sys_vars!r}',
                f'env_init={self.env_init!r}',
                f'sys_init={self.sys_init!r}',
                f'env_safety={self.env_safety!r}',
                f'sys_safety={self.sys_safety!r}',
                f'env_prog={self.env_prog!r}',
                f'sys_prog={self.sys_prog!r}'])
        return f'{type(self).__name__}({args})'

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
        """Wrap `loads` for reading from files.

        UNDER DEVELOPMENT; function signature may change without
        notice.  Calling will result in NotImplementedError.
        """
        raise NotImplementedError

    def pretty(self):
        """Return pretty printing string."""
        output = 'ENVIRONMENT VARIABLES:\n'
        if self.env_vars:
            for k, v in self.env_vars.items():
                output += f'\t{k}\t{v}\n'
        else:
            output += '\t(none)\n'
        output += '\nSYSTEM VARIABLES:\n'
        if self.sys_vars:
            for k, v in self.sys_vars.items():
                output += f'\t{k}\t{v}\n'
        else:
            output += '\t(none)\n'
        output += '\nFORMULA:\n'
        output += 'ASSUMPTION:\n'
        if self.env_init:
            output += (
                '    INITIAL\n\t  ' +
                '\n\t& '.join([
                    f'({f})' for f in self.env_init
                ]) + '\n')
        if self.env_safety:
            output += (
                '    SAFETY\n\t  []' +
                '\n\t& []'.join([
                    f'({f})' for f in self.env_safety
                ]) + '\n')
        if self.env_prog:
            output += (
                '    LIVENESS\n\t  []<>' +
                '\n\t& []<>'.join([
                    f'({f})' for f in self.env_prog
                ]) + '\n')
        output += 'GUARANTEE:\n'
        if self.sys_init:
            output += (
                '    INITIAL\n\t  ' +
                '\n\t& '.join([
                    f'({f})' for f in self.sys_init
                ]) + '\n')
        if self.sys_safety:
            output += (
                '    SAFETY\n\t  []' +
                '\n\t& []'.join([
                    f'({f})' for f in self.sys_safety
                ]) + '\n')
        if self.sys_prog:
            output += (
                '    LIVENESS\n\t  []<>' +
                '\n\t& []<>'.join([
                    f'({f})' for f in self.sys_prog
                ]) + '\n')
        return output

    def check_syntax(self):
        """Raise `AssertionError` for misplaced primed variables."""
        self._assert_no_primed(
            self.env_init,
            'assumed initial condition')
        self._assert_no_primed(
            self.sys_init,
            'guaranteed initial condition')
        self._assert_no_primed(
            self.env_prog,
            'liveness assumption')
        self._assert_no_primed(
            self.env_prog,
            'liveness guarantee')
        for f in self.env_safety:
            a = self.ast(f)
            primed = tx.collect_primed_vars(a)
            for var in primed:
                if var in self.sys_vars:
                    raise AssertionError(
                        'Syntax error: '
                        f'primed system variable "{var}"'
                        f' found in env safety: {f}')

    def _assert_no_primed(self, formulae, name):
        """Raise `AssertionError` if primed vars in `formulae`."""
        for f in formulae:
            a = self.ast(f)
            primed = tx.collect_primed_vars(a)
            if primed:
                raise AssertionError(
                    'Syntax error: '
                    f'primed variables: {primed}'
                    f' found in {name}: {f}')

    def copy(self):
        """Return a copy of `self`."""
        r = GRSpec(
            env_vars=dict(self.env_vars),
            sys_vars=dict(self.sys_vars),
            env_init=copy.copy(self.env_init),
            env_safety=copy.copy(self.env_safety),
            env_prog=copy.copy(self.env_prog),
            sys_init=copy.copy(self.sys_init),
            sys_safety=copy.copy(self.sys_safety),
            sys_prog=copy.copy(self.sys_prog))
        r.moore = self.moore
        r.plus_one = self.plus_one
        r.qinit = self.qinit
        return r

    def __or__(self, other):
        """Create union of two specifications."""
        result = self.copy()
        if not isinstance(other, GRSpec):
            raise TypeError('type(other) must be GRSpec')
        assert self.moore == other.moore, (
            self.moore, other.moore)
        assert self.plus_one == other.plus_one, (
            self.plus_one, other.plus_one)
        assert self.qinit == other.qinit, (
            self.qinit, other.qinit)
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
        <https://tulip-control.sourceforge.io/doc/specifications.html>}
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
                return f'({assumption}) -> ({guarantee})'
            else:
                return guarantee
        else:
            return 'True'

    def sub_values(self, var_values):
        """Substitute given values for variables.

        Note that there are three ways to substitute values for variables:

          - syntactic using this function

          - no substitution by user code, instead flatten to python and
            use `eval` together with a `dict` defining
            the values of variables, as done in `eval_init`.

        For converting non-integer finite types to
        integer types, use `replace_finite_by_int`.

        @return:
            `dict` of ASTs after the substitutions,
            keyed by original clause (before substitution).
        """
        logger.info('substitute values for variables...')
        a = copy.deepcopy(self._ast)
        for formula, tree in a.items():
            g = tx.Tree.from_recursive_ast(tree)
            tx.sub_values(g, var_values)
            a[formula] = g.to_recursive_ast()
        logger.info('done with substitutions.\n')
        return a

    def compile_init(
            self,
            no_str
            ) -> types.CodeType:
        """Compile python expression for initial conditions.

        The returned bytecode can be used with `eval`
        and a `dict` assigning values to variables.
        Its value is the implication

        ```tla
        env_init => sys_init
        ```

        Use the corresponding python data types
        for the `dict` values:

              - `bool` for Boolean variables
              - `int` for integers
              - `str` for arbitrary finite types

        @param no_str:
            if True, then compile the formula
            where all string variables have been replaced by integers.
            Otherwise compile the original formula containing strings.
        @return:
            python expression compiled for `eval`
        """
        self.str_to_int()
        init = {
            'env':
                self.env_init,
            'sys':
                self.sys_init}
        pyinit = dict()
        for side, clauses in init.items():
            if no_str:
                clauses = [self._bool_int[x] for x in clauses]
            logger.info(f'clauses to compile: {clauses}')
            c = [ts.translate_ast(self.ast(x), 'python').flatten()
                 for x in clauses]
            logger.info(f'after translation to python: {c}')
            s = _conj(c, op='and')
            if not s:
                s = 'True'
            pyinit[side] = s
        assumption = pyinit['env']
        guarantee = pyinit['sys']
        s = f'not ({assumption}) or ({guarantee})'
        return compile(s, '<string>', 'eval')

    def str_to_int(self):
        """Replace arbitrary finite vars with int vars.

        Returns spec itself if it contains only int vars.
        Otherwise it returns a copy of spec with all arbitrary
        finite vars replaced by int-valued vars.
        """
        logger.info('convert string variables to integers...')
        vars_dict = dict(self.env_vars)
        vars_dict.update(self.sys_vars)
        fvars = {v: d for v, d in vars_dict.items() if isinstance(d, list)}
        # replace symbols by ints
        for p in self._parts:
            for x in getattr(self, p):
                if self._bool_int.get(x) in self._ast:
                    logger.debug(f'{x} is in _bool_int cache')
                    continue
                else:
                    logger.debug(f'{x} is not in _bool_int cache')
                # get AST
                a = self.ast(x)
                # create AST copy with int and bool vars only
                g = tx.Tree.from_recursive_ast(a)
                tx.sub_constants(g, fvars)
                b = g.to_recursive_ast()
                # formula of int/bool AST
                f = b.flatten()
                self._ast[f] = b  # cache
                # remember map from clauses to int/bool clauses
                self._bool_int[x] = f
        logger.info('done converting to integer variables.\n')

    def ast(self, x):
        """Return AST corresponding to formula x.

        If AST for formula `x` has already been computed earlier,
        then return cached result.
        """
        if logger.getEffectiveLevel() <= logging.DEBUG:
            logger.debug('current cache of ASTs:\n' +
                         pprint.pformat(self._ast) + 3 * '\n')
            logger.debug(f'check if: {x}, is in cache.')
        if x in self._ast:
            logger.debug(f'{x} is already in cache')
        else:
            logger.info(f'AST cache does not contain:\n\t{x}'
                        '\nNeed to parse.')
            self.parse()
        return self._ast[x]

    def parse(self):
        """Parse each clause and store it.

        The AST resulting from each clause is stored
        in the `dict` attribute `ast`.
        """
        logger.info('parsing ASTs to cache them...')
        vardoms = dict(self.env_vars)
        vardoms.update(self.sys_vars)
        # parse new clauses and cache the resulting ASTs
        for p in self._parts:
            s = getattr(self, p)
            for x in s:
                if x in self._ast:
                    logger.debug(f'{x} is already in cache')
                    continue
                logger.debug(f'parse: {x}')
                tree = self.parser.parse(x)
                g = tx.Tree.from_recursive_ast(tree)
                tx.check_for_undefined_identifiers(g, vardoms)
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
        logger.info(f'cleaned {len(s)} cached elements.\n')


def replace_dependent_vars(spec, bool2form):
    logger.debug(
        'replacing dependent variables '
        f'using the map:\n\t{bool2form}')
    vs = dict(spec.env_vars)
    vs.update(spec.sys_vars)
    logger.debug(f'variables:\n\t{vs}')
    bool2subtree = dict()
    for boolvar, formula in bool2form.items():
        logger.debug(f'checking var: {boolvar}')
        if boolvar in vs:
            assert vs[boolvar] == 'boolean'
            logger.debug(
                f'{boolvar} is indeed Boolean')
        else:
            logger.debug(
                f'spec does not contain var: {boolvar}')
        tree = parser.parse(formula)
        bool2subtree[boolvar] = tx.Tree.from_recursive_ast(tree)
    for s in {'env_init', 'env_safety', 'env_prog',
              'sys_init', 'sys_safety', 'sys_prog'}:
        part = getattr(spec, s)
        new = list()
        for clause in part:
            logger.debug(
                f'replacing in clause:\n\t{clause}')
            tree = spec.ast(clause)
            g = tx.Tree.from_recursive_ast(tree)
            tx.sub_bool_with_subtree(g, bool2subtree)
            f = g.to_recursive_ast().flatten()
            new.append(f)
            logger.debug(
                f'caluse tree after replacement:\n\t{f}')
        setattr(spec, s, new)


def _conj(iterable, unary='', op='&&'):
    return f' {op} '.join(
        f'{unary}({s})' for s in iterable)
