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
"""
Classes representing hybrid dynamical systems.
"""
from __future__ import absolute_import

import logging
logger = logging.getLogger(__name__)

from warnings import warn
import itertools
from pprint import pformat

import numpy as np
import polytope as pc

# inline imports:
#
# from tulip.graphics import newax, quiver

def _indent(s, n):
    s = s.split('\n')
    w = n*' '
    return w + ('\n'+w).join(s)

class LtiSysDyn(object):
    """Represent discrete-time continuous-state dynamics::

        s[t+1] = A*s[t] + B*u[t] + E*d[t] + K

    subject to the constraints::

        u[t] \in Uset
        d[t] \in Wset
        s[t] \in domain

    where:
        - u[t] the control input
        - d[t] the disturbance input
        - s[t] the system state

    A LtiSysDyn object contains the fields:

        - A, B, E, K, (matrices)
        - Uset, Wset, (each a C{polytope.Polytope})
        - domain (C{polytope.Polytope} or C{polytope.Region})
        - time_semantics: 'discrete' (if system is originally a discrete-time
          system) or 'sampled' (if system is sampled from a continuous-time
          system)
        - timestep: A positive real number containing the timestep (for sampled
          system)

    as defined above.

    Note
    ====
    For state-dependent bounds on the input,::
        [u[t]; s[t]] \in Uset
    can be used.

    See Also
    ========
    L{PwaSysDyn}, L{SwitchedSysDyn}, C{polytope.Polytope}
    """
    def __init__(self, A=None, B=None, E=None, K=None,
                 Uset=None,Wset=None, domain=None, time_semantics=None,
                 timestep=None):

        if Uset is None:
            warn('Uset not given to LtiSysDyn()')
        elif not isinstance(Uset, pc.Polytope):
            raise Exception('`Uset` has to be a Polytope')
        if domain is None:
            warn("Domain not given to LtiSysDyn()")
        if ((domain is not None) and
            (not (isinstance(domain, pc.Polytope) or
                isinstance(domain, pc.Region))
            )
        ):
            raise Exception('`domain` has to be a Polytope or Region')

        # check dimensions agree
        try:
            nA, mA = A.shape
        except:
            raise TypeError('A matrix must be 2d array')
        if nA != mA:
            raise ValueError('A must be square')
        if domain is not None:
            if domain.dim != mA:
                raise Exception('domain.dim != A.size[1]')

        if B is not None:
            try:
                nB, mB = B.shape
            except:
                raise TypeError('B matrix must be 2d array')
            if nA != nB:
                raise ValueError('A and B must have same number of rows')
            if Uset is not None:
                if (Uset.dim != mB) and (Uset.dim != mB + nA):
                    msg = 'Uset.dim != B.size[1]'
                    msg += ' and != B.size[1] + A.size[1]'
                    raise Exception(msg)

        if E is not None:
            try:
                nE, mE = E.shape
            except:
                raise TypeError('E matrix must be 2d array')
            if nA != nE:
                raise ValueError('A and E must have same number of rows')
            if Wset is not None:
                if Wset.dim != mE:
                    raise Exception('Wset.dim != E.size[1]')

        if K is not None:
            try:
                nK, mK = K.shape
            except:
                raise TypeError('K column vector must be 2d array')

            if nA != nK:
                raise ValueError('A and K must have same number of rows')
            if mK != 1:
                raise ValueError('K must be a column vector')

        self.A = A
        self.B = B

        if K is None:
            if len(A) != 0:
                self.K = np.zeros([mA, 1])
            else:
                self.K = K
        else:
            self.K = K.reshape(K.size,1)

        if E is None and (len(A) != 0):
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


    def __str__(self):
        n = 3
        output = 'A =\n' + _indent(str(self.A), n)
        output += '\nB =\n' + _indent(str(self.B), n)
        output += '\nE =\n' + _indent(str(self.E), n)
        output += '\nK =\n' + _indent(str(self.K), n)
        output += '\nUset =\n' + _indent(str(self.Uset), n)
        output += '\nWset =\n' + _indent(str(self.Wset), n)
        return output

    def plot(self, ax=None, color=np.random.rand(3), show_domain=True,
             res=(5, 5), **kwargs):
        try:
            from tulip.graphics import newax, quiver
        except:
            logger.error('failed to import graphics')
            warn('pyvectorized not found. No plotting.')
            return

        (x, res) = pc.grid_region(self.domain, res=res)
        n = self.A.shape[0]
        DA = self.A - np.eye(n)
        v = DA.dot(x) + self.K

        if ax is None:
            ax, fig = newax()

        if show_domain:
            self.domain.plot(ax, color)

        quiver(x, v, ax, **kwargs)

        return ax

class PwaSysDyn(object):
    """PwaSysDyn class for specifying a polytopic piecewise affine system.
    A PwaSysDyn object contains the fields:

      - C{list_subsys}: list of L{LtiSysDyn}

      - C{domain}: domain over which piecewise affine system is defined,
          type: polytope.Polytope or polytope.Region

      - C{time_semantics}: 'discrete' (if system is originally a discrete-time
       system) or 'sampled' (if system is sampled from a continuous-time
       system)

      - C{timestep}: A positive real number containing the timestep (for sampled
        systems)

    For the system to be well-defined the domains of its subsystems should be
    mutually exclusive (modulo intersections with empty interior) and cover the
    domain.

    See Also
    ========
    L{LtiSysDyn}, L{SwitchedSysDyn}, C{polytope.Polytope}
    """
    def __init__(self, list_subsys=[], domain=None, time_semantics=None,
                 timestep=None, overwrite_time=True):
        """
        @type overwrite_time: bool
        @param overwrite_time: If true, then overwrites any time data in the
                               objects in C{list_subsys} with the data in
                               C{time_semantics} and C{timestep} variables.
                               Otherwise checks that the time data of the
                               objects in C{list_subsys} are consistent with
                               C{time_semantics} and C{timestep}.
        """

        if domain is None:
            warn("Domain not given to PwaSysDyn()")

        if ((domain is not None) and
            (not (isinstance(domain, pc.Polytope) or
                isinstance(domain, pc.Region))
            )
        ):
            raise Exception("PwaSysDyn: `domain` has to be a Polytope or Region")

        if len(list_subsys) > 0:
            uncovered_dom = domain.copy()
            n = list_subsys[0].A.shape[1]  # State space dimension
            m = list_subsys[0].B.shape[1]  # Input space dimension
            p = list_subsys[0].E.shape[1]  # Disturbance space dimension
            for subsys in list_subsys:
                uncovered_dom = uncovered_dom.diff(subsys.domain)
                if (n!=subsys.A.shape[1] or m!=subsys.B.shape[1] or
                    p!=subsys.E.shape[1]):
                    raise Exception("PwaSysDyn: state, input, disturbance " +
                                    "dimensions have to be the same for all " +
                                     "subsystems")
            if not pc.is_empty(uncovered_dom):
                raise Exception("PwaSysDyn: subdomains must cover the domain")
            for x in itertools.combinations(list_subsys, 2):
                if pc.is_fulldim(x[0].domain.intersect(x[1].domain) ):
                    raise Exception("PwaSysDyn: subdomains have to be mutually"+
                        " exclusive")

        self.list_subsys = list_subsys
        self.domain = domain

        # Input time semantics
        _check_time_data(time_semantics, timestep)
        if overwrite_time:
            _push_time_data(self.list_subsys, time_semantics, timestep)
        else:
            _check_time_consistency(list_subsys, time_semantics, timestep)
        self.timestep = timestep
        self.time_semantics = time_semantics


    def __str__(self):
        s = 'Piecewise-Affine System Dynamics\n'
        s += 30 * '-' + 2*'\n'

        s += 3*' ' + 'Domain:\n\n'
        s += _indent(str(self.domain), n=6) + '\n'

        for i, sys in enumerate(self.list_subsys):
            s += 3*' ' + 'Subsystem: ' + str(i) +'\n'
            s += _indent(str(sys), n=6)
        return s

    @classmethod
    def from_lti(cls, A=[], B=[], E=[], K=[],
                 Uset=None, Wset=None,domain=None):
        lti_sys = LtiSysDyn(A,B,E,K,Uset,Wset,domain)
        return cls([lti_sys], domain)

    def plot(self, ax=None, show_domain=True, **kwargs):
        try:
            from tulip.graphics import newax
        except:
            logger.error('failed to import graphics')
            return

        if ax is None:
            ax, fig = newax()

        for subsystem in self.list_subsys:
            subsystem.plot(ax, color=np.random.rand(3),
                           show_domain=show_domain, **kwargs)
        return ax

class SwitchedSysDyn(object):
    """Represent hybrid systems switching between dynamic modes.

    A C{SwitchedSysDyn} represents a system with switching modes
    that depend on both discrete:

        - n_env environment variables (uncontrolled)
        - n_sys system variables (controlled)

    A C{SwitchedSysDyn} object contains the fields:

     - C{disc_domain_size}: 2-tuple of numbers of modes
       type: (n_env, n_sys)

     - C{env_labels}: (optional) labels for discrete environment variables
       type: list of len(n_env)
       default: range(n_env)

     - C{disc_sys_labels}: (optional) labels for discrete system variables
       type: list of len(n_sys)
       default: range(n_sys)

     - C{dynamics}: mapping mode 2-tuples to active dynamics::

         (env_label, sys_label) -> PwaSysDyn

       type: dict
       default: If no env_label or sys_label passed,
       then default to int indices (i,j) L{PwaSysDyn}.

     - C{cts_ss}: continuous state space over which hybrid system is defined.
       type: C{polytope.Region}

     - C{time_semantics}: 'discrete' (if system is originally a discrete-time
       system) or 'sampled' (if system is sampled from a continuous-time
       system)

     - C{timestep}: A positive real number containing the timestep (for sampled
       systems)


    Note
    ====
    We assume that system and environment switching modes are
    independent of one another.  (Use LTL statement to make it not so.)

    See Also
    ========
    L{LtiSysDyn}, L{PwaSysDyn}, C{polytope.Region}
    """
    def __init__(self, disc_domain_size=(1,1),
                 dynamics=None, cts_ss=None,
                 env_labels=None, disc_sys_labels=None, time_semantics=None,
                 timestep=None, overwrite_time=True):
        """
        @type overwrite_time: bool
        @param overwrite_time: If true, then overwrites any time data in the
                               objects in C{list_subsys} with the data in
                               C{time_semantics} and C{timestep} variables.
                               Otherwise checks that the time data of the
                               objects in C{list_subsys} are consistent with
                               C{time_semantics} and C{timestep}.
        """

        # check that the continuous domain is specified
        if cts_ss is None:
            warn('continuous state space not given to SwitchedSysDyn')
        else:
            if not isinstance(cts_ss, (pc.Polytope, pc.Region) ):
                raise Exception('SwitchedSysDyn: ' +
                   '`cts_ss` must be a Polytope or Region')

        self.disc_domain_size = disc_domain_size

        # If label numbers agree with disc_domain_size, then use them.
        # Otherwise, ignore the labels.
        n_env, n_sys = disc_domain_size

        self._env_labels = self._check_labels(n_env, env_labels)
        self._disc_sys_labels = self._check_labels(n_sys, disc_sys_labels)

        # Check each dynamics key is a valid mode,
        # i.e., a valid combination of env and sys labels.
        if dynamics is not None:
            modes = self.all_mode_combs

            undefined_modes = set(dynamics.keys()).difference(modes)

            if undefined_modes:
                msg = 'SwitchedSysDyn: `dynamics` keys inconsistent'
                msg += ' with discrete mode labels.\n'
                msg += 'Undefined modes:\n' + str(undefined_modes)
                raise ValueError(msg)

            missing_modes = set(modes).difference(dynamics.keys())

            if missing_modes:
                msg = 'Missing the modes:\n' + str(missing_modes)
                msg += '\n Make sure you did not forget any modes,\n'
                msg += 'otherwise this is fine.'
                warn(msg)

            if not all([isinstance(sys, PwaSysDyn)
                        for sys in dynamics.values()]):
                msg = 'For each mode dynamics must be PwaSysDyn.\n'
                msg += 'Got instead: ' +str(type(sys))
                raise Exception(msg)

        self.dynamics = dynamics
        self.cts_ss = cts_ss

        _check_time_data(time_semantics, timestep)
        if overwrite_time:
            _push_time_data(self.dynamics.values(), time_semantics, timestep)
        else:
            _check_time_consistency(dynamics.values(), time_semantics, timestep)
        self.timestep = timestep
        self.time_semantics = time_semantics

    def __str__(self):
        n_env, n_sys = self.disc_domain_size

        s = 'Hybrid System Dynamics\n'
        s += 30 * '-' + '\n'

        s += 'Modes:\n'
        s += 4*' ' + 'Environment (' + str(n_env) + ' modes):\n'
        s += 6*' ' + pformat(self.env_labels, indent=3) + 2*'\n'
        s += 4*' ' + 'System: (' + str(n_sys) + ' modes)\n'
        s += 6*' ' + pformat(self.disc_sys_labels, indent=3) + 2*'\n'

        s += 'Continuous State Space:\n\n'
        s += _indent(str(self.cts_ss), 4) + '\n'

        s += 'Dynamics:\n'
        for mode, pwa in self.dynamics.iteritems():
            s += 4*' ' + 'mode: ' + str(mode) + '\n'
            s += 4*' ' + 'dynamics:\n' + _indent(str(pwa), 8) +'\n\n'
        return s

    def _check_labels(self, n, labels):
        # don't complain for default
        if labels is None:
            return None

        # len exists ?
        try:
            # is len correct ?
            if len(labels) != n:
                msg = 'number of environment labels is inconsistent'
                msg += ' with discrete domain size.\n'
                msg += 'Ignoring given environment labels.\n'
                msg += 'Defaulting to integer labels.'
                warn(msg)

                return None
        except:
            warn('Environment labels of type: ' +
                 type(labels) + 'have no len()')
            return None
        return labels

    @property
    def all_mode_combs(self):
        """Return all possible combinations of modes.
        """
        modes = [(a,b) for a in self.env_labels
                           for b in self.disc_sys_labels]

        logger.debug('Available modes: ' + str(modes) )
        return modes

    @property
    def modes(self):
        if self.dynamics is None:
            warn('No dynamics defined (None).')
            return None
        return self.dynamics.keys()

    @property
    def env_labels(self):
        if self._env_labels is None:
            return range(self.disc_domain_size[0])
        else:
            return self._env_labels

    @property
    def disc_sys_labels(self):
        if self._disc_sys_labels is None:
            return range(self.disc_domain_size[1])
        else:
            return self._disc_sys_labels

    @classmethod
    def from_pwa(cls, list_subsys=[], domain=None):
        pwa_sys = PwaSysDyn(list_subsys,domain)
        return cls((1,1), {(0,0):pwa_sys}, domain)

    @classmethod
    def from_lti(cls, A=[], B=[], E=[], K=[],
                 Uset=None, Wset=None,domain=None):
        pwa_sys = PwaSysDyn.from_lti(A, B, E, K,
                                     Uset, Wset, domain)
        return cls((1,1), {(0,0):pwa_sys}, domain)


def _push_time_data(system_list, time_semantics, timestep):
    """Overwrite the time data in system list. Throws warnings if overwriting
    existing data."""

    for system in system_list:
        if (system.time_semantics != time_semantics) and (system.time_semantics
            is not None):
            warn('Overwriting existing time semantics data.')
        if (system.timestep != timestep) and (system.timestep is not None):
            warn('Overwriting existing timestep data.')
        system.time_semantics = time_semantics
        system.timestep = timestep

        # Overwrite LTI in system if system is a PWA
        if isinstance(system, PwaSysDyn):
            _push_time_data(system.list_subsys, time_semantics, timestep)



def _check_time_data(semantics, timestep):
    """Checks that time semantics and timestep are correctly specified. Raises
    ValueErrors if that's not the case.

    @type semantics: string
    @param timestep: any positive number
    @type timestep: int or float or long

    @rtype: None
    """

    if semantics not in ['sampled', 'discrete', None]:
        raise ValueError('Time semantics must be discrete or ' +
            'sampled (sampled from continuous time system).')

    if ((semantics == 'discrete') and (timestep is not None)):
        raise ValueError('Discrete semantics must not have a timestep')

    if timestep is not None:
        error_string = 'Timestep must be a positive real number or unspecified.'
        if timestep <= 0:
            raise ValueError(error_string)
        if not isinstance(timestep, (int, float, long)):
            raise TypeError(error_string)



def _check_time_consistency(system_list, time_semantics, timestep):
    """Checks that all the dynamical systems in system_list have the same time
    semantics and timestep. Raises ValueError if not the case.

    @type system_list: list of L{LtiSysDyn} or L{PwaSysDyn}
    @rtype: None
    """

    # Check that time semantics for all subsystems match
    for ind in range(len(system_list)-1):

        if system_list[ind].timestep != system_list[ind+1].timestep:
            raise ValueError('Not all timesteps in child systems are the same.')

        if system_list[ind].time_semantics != system_list[ind+1].time_semantics:
            raise ValueError('Not all time semantics are the same.')


    # Check that time semantics for all subsystems match specified system and
    # timestep
    if system_list[0].timestep != timestep:
        raise ValueError('Timestep of subsystems do not match specified ' +
                         'timestep.')

    if system_list[0].time_semantics != time_semantics:
        raise ValueError('Time semantics of subsystems do not match ' +
                         'specified time semantics.')
