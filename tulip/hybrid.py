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
#
"""Classes representing hybrid dynamical systems."""
import collections.abc as _abc
import itertools as _itr
import logging
import pprint as _pp
import textwrap as _tw
import typing as _ty
import warnings as _warn

import numpy as np
import polytope as pc

import tulip.graphics as _graphics


__all__ = [
    'LtiSysDyn',
    'PwaSysDyn',
    'SwitchedSysDyn']
_logger = logging.getLogger(__name__)


IJ = tuple[int, int]
maybe_array = (
    np.ndarray |
    None)
maybe_float = (
    float |
    None)
Hull = pc.Polytope
Polytope = (
    Hull |
    pc.Region)
TimeSemantics = _ty.Literal[
    'discrete',
    'sampled']


def _indent(
        s:
            str,
        n:
            int
        ) -> str:
    space = ' '
    prefix = n * space
    return _tw.indent(
        s,
        prefix=prefix)


class LtiSysDyn:
    r"""Represents discrete-time continuous-state dynamics.

    Specifically, dynamics of the form:

    ```
    s[t + 1] = A * s[t] + B * u[t] + E * d[t] + K
    ```

    subject to the constraints:

    ```
    u[t] \in Uset
    d[t] \in Wset
    s[t] \in domain
    ```

    where:
    - `u[t]` the control input
    - `d[t]` the disturbance input
    - `s[t]` the system state

    Attributes:

    - `A`, `B`, `E`, `K`, (matrices)
    - `Uset`, `Wset`, (each a `polytope.Polytope`)
    - `domain` (`polytope.Polytope` or `polytope.Region`)
    - `time_semantics`:
      - `'discrete'`: if the system is
        originally a discrete-time system, or
      - `'sampled'`: if the system is sampled from
        a continuous-time system)
    - timestep: A positive real number containing the
      timestep (for sampled system)

    as defined above.


    Note
    ====
    For state-dependent bounds on the input,

    ```
    [u[t]; s[t]] \in Uset
    ```

    can be used.


    Relevant
    ========
    `PwaSysDyn`,
    `SwitchedSysDyn`,
    `polytope.Polytope`
    """

    def __init__(
            self,
            A:
                maybe_array=None,
            B:
                maybe_array=None,
            E:
                maybe_array=None,
            K:
                maybe_array=None,
            Uset:
                Hull |
                None=None,
            Wset:
                Hull |
                None=None,
            domain:
                Polytope |
                None=None,
            time_semantics:
                TimeSemantics |
                None=None,
            timestep:
                float |
                None=None):
        if Uset is None:
            _warn.warn('Uset not given to `LtiSysDyn()`')
        elif not isinstance(Uset, pc.Polytope):
            raise TypeError('`Uset` has to be a Polytope')
        if domain is None:
            _warn.warn('Domain not given to `LtiSysDyn()`')
        type_error = (
            domain is not None and
            not isinstance(
                domain,
                pc.Polytope | pc.Region))
        if type_error:
            raise TypeError(
                '`domain` has to be '
                'a `Polytope` or `Region`')
        # check dimensions agree
        if A is not None:
            _assert_square_array(A)
            _, mA = A.shape
            self._check_domain(domain, mA)
        if B is not None:
            self._check_b_array(A, B)
            self._check_uset(A, B, Uset)
        if E is not None:
            self._check_e_array(A, E, Wset)
        if K is not None:
            self._check_k_array(A, K)
        self.A = A
        self.B = B
        if K is None and len(A) != 0:
            K = np.zeros([mA, 1])
        if K is not None:
            K = K.reshape(K.size, 1)
        self.K = K
        if E is None and len(A) != 0:
            self.E = np.zeros([mA, 1])
            self.Wset = pc.Polytope()
        else:
            self.E = E
            self.Wset = Wset
        self.Uset = Uset
        self.domain = domain
        # Check that timestep and semantics are valid.
        _check_time_data(time_semantics, timestep)
        self.time_semantics = time_semantics
        self.timestep = timestep

    @staticmethod
    def _check_domain(
            domain:
                Polytope |
                None,
            a_columns:
                int
            ) -> None:
        """Raise `ValueError` if unexpected."""
        if domain is None:
            return
        domain_ok = (
            domain.dim == a_columns)
        if domain_ok:
            return
        raise ValueError(
            '`domain.dim != A.shape[1]`')

    @staticmethod
    def _check_b_array(
            a_array:
                np.ndarray,
            b_array:
                np.ndarray
            ) -> None:
        """Raise `ValueError` if unexpected."""
        _assert_2d_array(b_array)
        a_n_rows, _ = a_array.shape
        b_n_rows, _ = b_array.shape
        if a_n_rows == b_n_rows:
            return
        raise ValueError(
            'The arrays `A` and `B` '
            'must have the same '
            'number of rows. Got: '
            f'{a_array.shape = } and '
            f'{b_array.shape = }')

    @staticmethod
    def _check_uset(
            a_array:
                np.ndarray,
            b_array:
                np.ndarray,
            u_set:
                Polytope |
                None
            ) -> None:
        """Raise `ValueError` if unexpected."""
        _assert_square_array(a_array)
        _, a_n_columns = a_array.shape
        _, b_n_columns = b_array.shape
        if u_set is None:
            return
        u_set_ok = (
            u_set.dim in
                (b_n_columns,
                 b_n_columns + a_n_columns))
        if u_set_ok:
            return
        raise ValueError(
            '`Uset.dim != B.shape[1]`'
            ' and `!= B.shape[1] + A.shape[1]`.'
            f'{u_set.dim = }, '
            f'{b_array.shape = }'
            f'{a_array.shape = }')

    @staticmethod
    def _check_e_array(
            a_array:
                np.ndarray,
            e_array:
                np.ndarray,
            wset:
                Polytope |
                None
            ) -> None:
        """Raise `ValueError` if unexpected."""
        _assert_2d_array(e_array)
        a_n_rows, _ = a_array.shape
        e_n_rows, e_n_columns = e_array.shape
        if a_n_rows != e_n_rows:
            raise ValueError(
                '`A` and `E` must have '
                'same number of rows. '
                'Got instead: '
                f'{a_array.shape = } and'
                f'{e_array.shape = }')
        wset_ok = (
            wset is None or
            wset.dim == e_n_columns)
        if wset_ok:
            return
        raise ValueError(
            '`Wset.dim != E.size[1]`')

    @staticmethod
    def _check_k_array(
            a:
                np.ndarray,
            k:
                np.ndarray
            ) -> None:
        """Raise `ValueError` if unexpected."""
        _assert_2d_array(k)
        a_n_rows, _ = a.shape
        k_n_rows, k_n_columns = k.shape
        if a_n_rows != k_n_rows:
            raise ValueError(
                '`A` and `K` must have '
                'same number of rows. '
                f'Got: A = {a.shape} '
                f'and K = {k.shape}')
        if k_n_columns == 1:
            return
        raise ValueError(
            '`K` must be a column vector. '
            'Got instead: '
            f'K.shape = {k.shape}')

    def __str__(self):
        indent_size = 3
        def format_(
                name:
                    str,
                value
                ) -> str:
            formatted_value = _indent(
                str(value), indent_size)
            return (
                f'{name} =\n'
                f'{formatted_value}')
        names = [
            'A', 'B', 'E', 'K',
            'Uset', 'Wset']
        values = [
            self.A, self.B, self.E, self.K,
            self.Uset, self.Wset]
        pairs = zip(names, values)
        return '\n'.join(_itr.starmap(
            format_, pairs))

    def plot(
            self,
            ax=None,
            color:
                maybe_array=None,
            show_domain:
                bool=True,
            res:
                IJ=(5, 5),
            **kwargs):
        if color is None:
            color = np.random.rand(3)
        x, res = pc.grid_region(
            self.domain,
            res=res)
        n = self.A.shape[0]
        DA = self.A - np.eye(n)
        v = DA.dot(x) + self.K
        if ax is None:
            ax, _ = _graphics.newax()
        if show_domain:
            self.domain.plot(ax, color)
        _graphics.quiver(
            x, v, ax,
            **kwargs)
        return ax


class PwaSysDyn:
    """Specifies a polytopic piecewise-affine system.

    Attributes:

    - `list_subsys`: list of `LtiSysDyn`

    - `domain`: domain over which piecewise-affine
      system is defined.

    - `time_semantics`: either
      - `'discrete'` (if system is originally
        a discrete-time system) or
      - `'sampled'` (if system is sampled from
        a continuous-time system)

    - `timestep`: A positive real number that contains
      the timestep (for sampled systems)

    For the system to be well-defined the domains of
    its subsystems should be mutually exclusive
    (modulo intersections with empty interior) and
    cover the domain.

    Relevant
    ========
    `LtiSysDyn`,
    `SwitchedSysDyn`,
    `polytope.Polytope`
    """

    def __init__(
            self,
            list_subsys:
                list[LtiSysDyn] |
                None=None,
            domain:
                Polytope |
                None=None,
            time_semantics:
                TimeSemantics |
                None=None,
            timestep:
                maybe_float=None,
            overwrite_time:
                bool=True):
        """Constructor.

        @param overwrite_time:
            If `True`, then overwrite any time data
            in the objects in `list_subsys` with
            the data in `time_semantics` and
            `timestep` variables.

            Otherwise, check that the time data
            of the objects in `list_subsys` are
            consistent with `time_semantics` and
            `timestep`.
        """
        if list_subsys is None:
            list_subsys = list()
        if domain is None:
            _warn.warn(
                'requires argument `domain`')
        type_error = (
            domain is not None and
            not isinstance(
                domain,
                pc.Polytope | pc.Region))
        if type_error:
            raise TypeError(
                '`domain` has to be '
                'a `Polytope` or `Region`')
        if list_subsys:
            if domain is None:
                raise ValueError(
                    'Argument `domain` must '
                    'not be `None` when argument '
                    '`list_subsys` is nonempty.')
            uncovered_dom = domain.copy()
            n = list_subsys[0].A.shape[1]
                # State-space dimension
            m = list_subsys[0].B.shape[1]
                # Input-space dimension
            p = list_subsys[0].E.shape[1]
                # Disturbance-space dimension
            for subsys in list_subsys:
                uncovered_dom = uncovered_dom.diff(
                    subsys.domain)
                dims_differ = (
                    n != subsys.A.shape[1] or
                    m != subsys.B.shape[1] or
                    p != subsys.E.shape[1])
                if dims_differ:
                    raise ValueError(
                        'state, input, disturbance '
                        'dimensions have to be the '
                        'same for all subsystems')
            if not pc.is_empty(uncovered_dom):
                raise ValueError(
                    'subdomains must cover the domain')
            for x in _itr.combinations(list_subsys, 2):
                if pc.is_fulldim(x[0].domain.intersect(x[1].domain)):
                    raise ValueError(
                        'subdomains have to be mutually '
                        'exclusive')
        self.list_subsys = list_subsys
        self.domain = domain
        # Input time semantics
        _check_time_data(time_semantics, timestep)
        if overwrite_time:
            _push_time_data(
                self.list_subsys,
                time_semantics,
                timestep)
        else:
            _check_time_consistency(
                list_subsys,
                time_semantics,
                timestep)
        self.timestep = timestep
        self.time_semantics = time_semantics

    def __str__(self):
        newlines = 2 * '\n'
        dashes = 30 * '-'
        spaces = 3 * ' '
        s = (
            'Piecewise-Affine System Dynamics\n'
            f'{dashes}{newlines}'
            f'{spaces}Domain:\n\n' +
            _indent(str(self.domain), n=6) + '\n')
        for i, sys in enumerate(self.list_subsys):
            sys_str = _indent(str(sys), n=6)
            s += (
                f'{spaces}Subsystem: {i}\n'
                f'{sys_str}')
        return s

    @classmethod
    def from_lti(
            cls,
            A:
                maybe_array=None,
            B:
                maybe_array=None,
            E:
                maybe_array=None,
            K:
                maybe_array=None,
            Uset:
                Hull=None,
            Wset:
                Hull=None,
            domain:
                Polytope=None
            ) -> 'PwaSysDyn':
        if A is None:
            A = list()
        if B is None:
            B = list()
        if E is None:
            E = list()
        if K is None:
            K = list()
        lti_sys = LtiSysDyn(
            A, B, E, K, Uset, Wset, domain)
        return cls([lti_sys], domain)

    def plot(
            self,
            ax=None,
            show_domain:
                bool=True,
            **kwargs):
        if ax is None:
            ax, _ = _graphics.newax()
        for subsystem in self.list_subsys:
            subsystem.plot(
                ax,
                color=np.random.rand(3),
                show_domain=show_domain,
                **kwargs)
        return ax


class SwitchedSysDyn:
    """Represent hybrid systems switching between dynamic modes.

    Represents a system with switching modes
    that depend on both discrete:

    - `n_env` environment variables (uncontrolled)
    - `n_sys` system variables (controlled)

    Attributes:

    - `disc_domain_size`: 2-`tuple` of numbers of modes
      - type: `(n_env, n_sys)`

    - `env_labels`: (optional) labels for
      discrete environment variables.
      - type: `list` of `len(n_env)`
      - default: `range(n_env)`

    - `disc_sys_labels`: (optional) labels for
      discrete system variables
      - type: `list` of `len(n_sys)`
      - default: `range(n_sys)`

    - `dynamics`: mapping mode 2-`tuple`s to
      active dynamics:

      ```
      (env_label, sys_label) -> PwaSysDyn
      ```

      - type: `dict`
      - default: If no `env_label` or `sys_label passed`,
        then default to `int` indices `(i, j)` `PwaSysDyn`.

    - `cts_ss`: continuous state-space over which
      the hybrid system is defined.
      - type: `polytope.Region`

    - `time_semantics`: either
      - `'discrete'` (if the system is originally
        a discrete-time system) or
      - `'sampled'` (if the system is sampled from
        a continuous-time system)

    - `timestep`: A positive real number that
      contains the timestep (for sampled systems)


    Note
    ====
    We assume that system and environment switching
    modes are independent of one another.
    (Use LTL statement to make it not so.)


    Relevant
    ========
    `LtiSysDyn`,
    `PwaSysDyn`,
    `polytope.Region`
    """

    def __init__(
            self,
            disc_domain_size:
                IJ=(1, 1),
            dynamics:
                dict[
                    tuple,
                    PwaSysDyn] |
                None=None,
            cts_ss=None,
            env_labels:
                list |
                None=None,
            disc_sys_labels:
                list |
                None=None,
            time_semantics:
                TimeSemantics |
                None=None,
            timestep:
                float |
                None=None,
            overwrite_time:
                bool=True):
        """Constructor.

        @param overwrite_time:
            If `True`, then overwrite any time data in
            the objects in `list_subsys` with the data in
            `time_semantics` and `timestep` variables.

            Otherwise, check that the time data of the
            objects in `list_subsys` are consistent with
            `time_semantics` and `timestep`.
        """
        if dynamics is None:
            dynamics = dict()
        # check that the continuous
        # domain is specified
        if cts_ss is None:
            _warn.warn(
                'requires continuous '
                'state-space `cts_ss`')
        else:
            if not isinstance(
                    cts_ss,
                    pc.Polytope |
                    pc.Region):
                raise TypeError(
                   '`cts_ss` must be '
                   'a `Polytope` or `Region`')
        self.disc_domain_size = disc_domain_size
        # If label numbers agree with
        # `disc_domain_size`, then use them.
        # Otherwise, ignore the labels.
        n_env, n_sys = disc_domain_size
        self._env_labels = self._check_labels(
            n_env, env_labels)
        self._disc_sys_labels = self._check_labels(
            n_sys, disc_sys_labels)
        # Check that each dynamics key is a valid mode,
        # i.e., a valid combination of
        # environment and system labels.
        modes = self.all_mode_combs
        undefined_modes = set(
            dynamics.keys()).difference(modes)
        if undefined_modes:
            raise ValueError(
                '`dynamics` keys are inconsistent'
                ' with discrete-mode labels.\n'
                f'Undefined modes:\n{undefined_modes}')
        missing_modes = set(modes).difference(
            dynamics.keys())
        if missing_modes:
            _warn.warn(
                f'Missing the modes:\n{missing_modes}'
                '\n Make sure you did not '
                'forget any modes,\n'
                'otherwise this is fine.')
        if not all(
                isinstance(sys, PwaSysDyn)
                for sys in dynamics.values()):
            raise TypeError(
                'For each mode, the dynamics '
                'must be `PwaSysDyn`.\n'
                'Got instead: '
                f'{dynamics.values()}')
        self.dynamics = dynamics
        self.cts_ss = cts_ss
        _check_time_data(time_semantics, timestep)
        if overwrite_time:
            _push_time_data(
                self.dynamics.values(),
                time_semantics,
                timestep)
        else:
            _check_time_consistency(
                list(dynamics.values()),
                time_semantics,
                timestep)
        self.timestep = timestep
        self.time_semantics = time_semantics

    def __str__(self):
        n_env, n_sys = self.disc_domain_size
        newlines = 2 * '\n'
        dashes = 30 * '-'
        spaces_4 = 4 * ' '
        spaces_6 = 6 * ' '
        s = (
            'Hybrid System Dynamics\n'
            f'{dashes}\n'
            'Modes:\n'
            f'{spaces_4}Environment ({n_env} modes):\n'
            + spaces_6 + _pp.pformat(
                self.env_labels, indent=3)
                + newlines
            + spaces_4 + f'System: ({n_sys} modes)\n'
            + spaces_6 + _pp.pformat(
                self.disc_sys_labels, indent=3)
                + newlines
            + 'Continuous State Space:\n\n'
            + _indent(str(self.cts_ss), 4) + '\n'
            'Dynamics:\n')
        for mode, pwa in self.dynamics.items():
            s += (
                f'{spaces_4}mode: {mode}\n' +
                f'{spaces_4}dynamics:\n'
                    + _indent(str(pwa), 8)
                    + newlines)
        return s

    def _check_labels(
            self,
            n:
                int,
            labels:
                list |
                None
            ) -> (
                list |
                None):
        ok = (
            labels is None or
            len(labels) == n)
        if ok:
            return labels
        raise ValueError(
            'number of environment labels '
            'is inconsistent with discrete '
            'domain size.\n'
            'Ignoring given environment labels.\n'
            'Defaulting to integer labels.')

    @property
    def all_mode_combs(self):
        """Return all possible combinations of modes."""
        modes = [
            (a, b)
            for a in self.env_labels
            for b in self.disc_sys_labels]
        _logger.debug(f'Available modes: {modes}')
        return modes

    @property
    def modes(self):
        if self.dynamics is None:
            _warn.warn(
                'No dynamics defined (`None`).')
            return None
        return self.dynamics.keys()

    @property
    def env_labels(self) -> list:
        if self._env_labels is not None:
            return self._env_labels
        return list(range(
            self.disc_domain_size[0]))

    @property
    def disc_sys_labels(self) -> list:
        if self._disc_sys_labels is not None:
            return self._disc_sys_labels
        return list(range(
            self.disc_domain_size[1]))

    @classmethod
    def from_pwa(
            cls,
            list_subsys:
                list |
                None=None,
            domain:
                Polytope=None
            ) -> 'SwitchedSysDyn':
        if list_subsys is None:
            list_subsys = list()
        pwa_sys = PwaSysDyn(
            list_subsys, domain)
        return cls(
            (1, 1),
            {(0, 0): pwa_sys},
            domain)

    @classmethod
    def from_lti(
            cls,
            A:
                maybe_array=None,
            B:
                maybe_array=None,
            E:
                maybe_array=None,
            K:
                maybe_array=None,
            Uset:
                Hull=None,
            Wset:
                Hull=None,
            domain:
                Polytope=None
            ) -> 'SwitchedSysDyn':
        if A is None:
            A = list()
        if B is None:
            B = list()
        if E is None:
            E = list()
        if K is None:
            K = list()
        pwa_sys = PwaSysDyn.from_lti(
            A, B, E, K,
            Uset, Wset, domain)
        return cls(
            (1, 1),
            {(0, 0): pwa_sys},
            domain)


def _push_time_data(
        systems:
            _abc.Iterable,
        time_semantics:
            TimeSemantics |
            None,
        timestep:
            float |
            None
        ) -> None:
    """Overwrite the time data in `systems`.

    Emits warnings if overwriting existing data.
    """
    for system in systems:
        overwriting_time_semantics = (
            system.time_semantics != time_semantics and
            system.time_semantics is not None)
        if overwriting_time_semantics:
            _warn.warn(
                'Overwriting existing '
                'time-semantics data.')
        overwriting_timestep_data = (
            system.timestep != timestep and
            system.timestep is not None)
        if overwriting_timestep_data:
            _warn.warn(
                'Overwriting existing '
                'timestep data.')
        system.time_semantics = time_semantics
        system.timestep = timestep
        # Overwrite LTI in system if
        # the system is piecewise-affine
        if isinstance(system, PwaSysDyn):
            _push_time_data(
                system.list_subsys,
                time_semantics,
                timestep)


def _check_time_data(
        semantics:
            str,
        timestep:
            int |
            float
        ) -> None:
    """Assert time semantics, timestep are correct.

    Raise `ValueError` if not.

    @param timestep:
        any positive number
    """
    if semantics not in ['sampled', 'discrete', None]:
        raise ValueError(
            'Time semantics must be discrete or '
            'sampled (sampled from continuous time system).')
    if semantics == 'discrete' and timestep is not None:
        raise ValueError(
            'Discrete semantics must not have a timestep')
    if timestep is not None:
        error_string = (
            'Timestep must be a positive real number, '
            'or unspecified.')
        if timestep <= 0:
            raise ValueError(error_string)
        if not isinstance(timestep, int | float):
            raise TypeError(error_string)


def _check_time_consistency(
        system_list:
            list[
                LtiSysDyn |
                PwaSysDyn],
        time_semantics:
            TimeSemantics,
        timestep:
            float |
            None
        ) -> None:
    """Assert homogeneous semantics in `system_list`.

    Checks that all the dynamical systems
    in `system_list` have the same
    time semantics and timestep.
    Raises `ValueError` if not.
    """
    # Check that time semantics
    # for all subsystems match
    for index in range(len(system_list) - 1):
        timesteps_differ = (
            system_list[index].timestep !=
            system_list[index + 1].timestep)
        if timesteps_differ:
            raise ValueError(
                'Not all timesteps in given '
                'systems are the same.')
        time_semantics_differ = (
            system_list[index].time_semantics !=
            system_list[index + 1].time_semantics)
        if time_semantics_differ:
            raise ValueError(
                'Not all time-semantics are the same.')
    # Check that time semantics for all subsystems
    # match specified system and timestep
    if system_list[0].timestep != timestep:
        raise ValueError(
            'Timestep of subsystems do not match '
            'specified timestep.')
    if system_list[0].time_semantics != time_semantics:
        raise ValueError(
            'Time semantics of subsystems do not match '
            'specified time semantics.')


def _assert_square_array(
        array:
            np.ndarray
        ) -> None:
    _assert_2d_array(array)
    n_rows, n_columns = array.shape
    if n_rows == n_columns:
        return
    raise ValueError(
        'Expected square array, '
        f'got instead {n_rows = } and '
        f'{n_columns = }')


def _assert_2d_array(
        array:
            np.ndarray
        ) -> None:
    if array.ndim == 2:
        return
    raise ValueError(
        'Expected 2-dimensional (2d) array, '
        'but got instead '
        f'{array.ndim}-dimensional array.')
