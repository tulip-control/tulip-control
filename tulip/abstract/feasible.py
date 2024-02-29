# Copyright (c) 2011-2016 by California Institute of Technology
# Copyright (c) 2016 by The Regents of the University of Michigan
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
# 3. Neither the name of the copyright holder(s) nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
# COPYRIGHT HOLDERS OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
"""Reachability between polytopes.

Functions to check reachability between polytopes,
for linear discrete-time and time-invariant systems.

Primary functions:
- `solve_feasible`
- `createLM`
- `get_max_extreme`


Relevant
========
`find_controller`
"""
import collections.abc as _abc
import logging

import numpy as np
import polytope as pc

import tulip.hybrid as _hyb


__all__ = [
    'is_feasible',
    'solve_feasible']


_logger = logging.getLogger(__name__)
Polytope = (
    pc.Polytope |
    pc.Region)


def is_feasible(
        from_region:
            Polytope,
        to_region:
            Polytope,
        sys:
            _hyb.LtiSysDyn,
        N:
            int,
        closed_loop:
            bool=True,
        use_all_horizon:
            bool=False,
        trans_set:
            Polytope |
            None=None
        ) -> bool:
    """Whether `to_region` is reachable `from_region`.

    Returns `True` if `to_region` is
    reachable from all points in `from_region`.

    For details read function `solve_feasible`.
    """
    S0 = solve_feasible(
        from_region, to_region, sys, N,
        closed_loop, use_all_horizon,
        trans_set)
    return from_region <= S0


def solve_feasible(
        P1:
            Polytope,
        P2:
            Polytope,
        ssys:
            _hyb.LtiSysDyn,
        N:
            int=1,
        closed_loop:
            bool=True,
        use_all_horizon:
            bool=False,
        trans_set:
            Polytope |
            None=None,
        max_num_poly=5
        ) -> Polytope:
    r"""Compute from where `P2` is `N`-reachable.

    Computes a polytope `S0`, such that:

    ```tla
    S0 \subseteq trans_set
    ```

    from which polytope `P2` is reachable in
    - *exactly* `N` steps
        (when `use_all_horizon is False`),
        or
    - *within* `N` steps
        (when `use_all_horizon is True`).

    `N`-reachable means reachable in horizon `N`.
    The system dynamics are `ssys`.
    The closed-loop algorithm solves for
    one step at a time, which keeps the
    dimension of the polytopes down.

    Time semantics:

    - `use_all_horizon = False`:
      fixed sampling period of
      discrete-valued environment variables.
      Reachability in exactly `N` steps.

    - `use_all_horizon = True`:
      sampling period that varies and
      is chosen by the system, depending on
      how many steps are taken during
      each trajectory from `P1` to `P2`.

      Reachability in `1..N` steps,
      with an under-approximation of
      the attractor set.

      If the system dynamics do not allow
      staying at the same state,
      then disconnected polytopes can arise,
      possibly causing issues in
      the computation.

      Consider decreasing the sampling period
      used for discretizing the associated
      continuous-time dynamical system.

    @param N:
        horizon length
    @param closed_loop:
        If `True`, then take 1 step at a time.
        This keeps down polytope dimension and
        handles disturbances better.
    @param use_all_horizon:
        Used for closed loop algorithm.

        - If `True`, then check for
          reachability in `< N` steps,

        - otherwise, in exactly `N` steps.
    @param trans_set:
        If specified,
        then force transitions to be in this set.
        Otherwise, `P1` is used.
    @return:
        polytope of states from
        which `P2` is reachable
    """
    if closed_loop:
        if use_all_horizon:
            return _underapproximate_attractor(
                P1, P2, ssys, N,
                trans_set=trans_set)
        else:
            return _solve_closed_loop_fixed_horizon(
                P1, P2, ssys, N,
                trans_set=trans_set)
    else:
        if use_all_horizon:
            raise ValueError(
                'The argument '
                '`use_all_horizon = True` '
                'has no effect if '
                '`closed_loop = False`')
        return solve_open_loop(
            P1, P2, ssys, N,
            trans_set=trans_set,
            max_num_poly=max_num_poly)


def _solve_closed_loop_fixed_horizon(
        P1:
            Polytope,
        P2:
            Polytope,
        ssys:
            _hyb.LtiSysDyn,
        N:
            int,
        trans_set:
            Polytope |
            None=None
        ) -> Polytope:
    """Underapproximate set that can reach `P2`.

    Underapproximate states in
    polytope `P1` that can reach
    polytope `P2` in `N` steps,
    with `N > 0`.

    If intermediate polytopes are convex,
    then the result is exact and
    not an underapproximation.

    @param ssys:
        system dynamics
    @param N:
        horizon length (positive number)
    @param trans_set:
        If provided,
        then intermediate steps are allowed
        to be in `trans_set`.
        Otherwise, `P1` is used.
    """
    assert N > 0, N
    p1 = P1.copy()
        # initial set
    p2 = P2.copy()
        # terminal set
    if trans_set is None:
        pinit = p1
    else:
        pinit = trans_set
    # backwards in time
    for i in range(N, 0, -1):
        # first step from P1
        if i == 1:
            pinit = p1
        p2 = solve_open_loop(
            pinit, p2, ssys, 1, trans_set)
        p2 = pc.reduce(p2)
        if not pc.is_fulldim(p2):
            return pc.Polytope()
    return p2


def _solve_closed_loop_bounded_horizon(
        P1:
            Polytope,
        P2:
            Polytope,
        ssys:
            _hyb.LtiSysDyn,
        N:
            int,
        trans_set:
            Polytope |
            None=None
        ) -> Polytope:
    """Underapproximate set that can reach `P2`.

    Underapproximates the set of states in
    polytope `P1` that can reach
    polytope `P2` in at most `N` steps.

    For details, read the docstring of
    the function
    `_solve_closed_loop_fixed_horizon`.
    """
    _print_horizon_warning()
    p1 = P1.copy()
        # initial set
    p2 = P2.copy()
        # terminal set
    if trans_set is None:
        pinit = p1
    else:
        pinit = trans_set
    # backwards in time
    s = pc.Region()
    for i in range(N, 0, -1):
        # first step from P1
        if i == 1:
            pinit = p1
        p2 = solve_open_loop(
            pinit, p2, ssys, 1, trans_set)
        p2 = pc.reduce(p2)
        # running union
        s = s.union(
            p2,
            check_convex=True)
        s = pc.reduce(s)
        # empty target polytope ?
        if not pc.is_fulldim(p2):
            break
    if not pc.is_fulldim(s):
        return pc.Polytope()
    s = pc.reduce(s)
    return s


def _underapproximate_attractor(
        P1:
            Polytope,
        P2:
            Polytope,
        ssys:
            _hyb.LtiSysDyn,
        N:
            int,
        trans_set:
            Polytope |
            None=None
        ) -> Polytope:
    """Underapproximate `N`-step attractor of `P2`.

    Underapproximates the `N`-step attractor
    of polytope `P2`, with `N > 0`.

    For details, read the docstring of
    the function
    `_solve_closed_loop_fixed_horizon`.
    """
    assert N > 0, N
    _print_horizon_warning()
    p1 = P1.copy()
        # initial set
    p2 = P2.copy()
        # terminal set
    if trans_set is None:
        pinit = p1
    else:
        pinit = trans_set
    # backwards in time
    for i in range(N, 0, -1):
        # first step from P1
        if i == 1:
            pinit = p1
        r = solve_open_loop(
            pinit, p2, ssys, 1, trans_set)
        p2 = p2.union(
            r,
            check_convex=True)
        p2 = pc.reduce(p2)
        # empty target polytope ?
        if not pc.is_fulldim(p2):
            return pc.Polytope()
    return r


def _print_horizon_warning() -> None:
    print(
        'WARNING: different timing '
        'semantics and assumptions '
        'from the case of fixed horizon. '
        'Also, depending on dynamics, '
        'disconnected polytopes can arise, '
        'which may cause issues in '
        'the `polytope` package.')


def solve_open_loop(
        P1:
            Polytope,
        P2:
            Polytope,
        ssys:
            _hyb.LtiSysDyn,
        N:
            int,
        trans_set:
            Polytope |
            None=None,
        max_num_poly:
            int=5
        ) -> Polytope:
    r1 = P1.copy()
        # Initial set
    r2 = P2.copy()
        # Terminal set
    # use the max_num_poly largest
    # volumes for reachability
    r1 = volumes_for_reachability(
        r1, max_num_poly)
    r2 = volumes_for_reachability(
        r2, max_num_poly)
    if len(r1) > 0:
        start_polys = r1
    else:
        start_polys = [r1]
    if len(r2) > 0:
        target_polys = r2
    else:
        target_polys = [r2]
    # union of s0 over all
    # polytope combinations
    s0 = pc.Polytope()
    for p1 in start_polys:
        for p2 in target_polys:
            cur_s0 = poly_to_poly(
                p1, p2, ssys, N, trans_set)
            s0 = s0.union(
                cur_s0,
                check_convex=True)
    return s0


def poly_to_poly(
        p1:
            Polytope,
        p2:
            Polytope,
        ssys:
            _hyb.LtiSysDyn,
        N:
            int,
        trans_set:
            Polytope |
            None=None
        ) -> Polytope:
    """Compute `s0` for open-loop `N`-reachability.

    Computes `s0` for open-loop
    polytope-to-polytope `N`-reachability.
    """
    p1 = p1.copy()
    p2 = p2.copy()
    if trans_set is None:
        trans_set = p1
    # stack polytope constraints
    L, M = createLM(
        ssys, N, p1, trans_set, p2)
    s0 = pc.Polytope(L, M)
    s0 = pc.reduce(s0)
    # Project polytope s0 onto lower dim
    n = np.shape(ssys.A)[1]
    dims = range(1, n + 1)
    s0 = s0.project(dims)
    return pc.reduce(s0)


def volumes_for_reachability(
        part:
            list[Polytope],
        max_num_poly:
            int
        ) -> Polytope:
    if len(part) <= max_num_poly:
        return part
    vol_list = np.zeros(len(part))
    for i in range(len(part)):
        vol_list[i] = part[i].volume
    ind = np.argsort(- vol_list)
    temp = list()
    for i in ind[range(max_num_poly)]:
        temp.append(part[i])
    part = pc.Region(temp, [])
    return part


def createLM(
        ssys:
            _hyb.LtiSysDyn,
        N:
            int,
        list_P:
            Polytope |
            list[Polytope],
        Pk:
            pc.Polytope |
            None=None,
        PN:
            pc.Polytope |
            None=None,
        disturbance_ind:
            list[int] |
            None=None
        ) -> tuple[
            np.ndarray,
            np.ndarray]:
    r"""Compute the components of the polytope:

    ```
    L [x(0)' u(0)' ... u(N-1)']' <= M
    ```

    which stacks the following constraints:

    - `x(t + 1) = A * x(t) + B * u(t) + E * d(t)`
    - `\A k:  [u(k); x(k)] \in ssys.Uset`

    If `list_P` is a `Polytope`:

    - `x(0) \in list_P if list_P`
    - `x(k) \in Pk` for `k \in 1..(N - 1)`
    - `x(N) \in PN`

    If `list_P` is a `list` of polytopes:

    - `x(k) \in list_P[k]` for `k \in 0..N`

    The returned polytope describes the
    intersection of the polytopes
    for all possible

    @param ssys:
        system dynamics
    @param N:
        horizon length
    @param disturbance_ind:
        In which `k`s
        the disturbance should be
        taken into account.
        Default is `1..N`.
    """
    if not isinstance(list_P, _abc.Iterable):
        list_P = [list_P] + (N - 1) * [Pk] + [PN]
    if disturbance_ind is None:
        disturbance_ind = range(1, N + 1)
    A = ssys.A
    B = ssys.B
    E = ssys.E
    K = ssys.K
    D = ssys.Wset
    PU = ssys.Uset
    n = A.shape[1]
        # State space dimension
    m = B.shape[1]
        # Input space dimension
    p = E.shape[1]
        # Disturbance space dimension
    # non-zero disturbance matrix E ?
    if not np.all(E == 0):
        if not pc.is_fulldim(D):
            E = np.zeros(K.shape)
    list_len = np.array(
        [P.A.shape[0]
         for P in list_P])
    sumlen = np.sum(list_len)
    LUn = np.shape(PU.A)[0]
    Lk = np.zeros([sumlen, n + N * m])
    LU = np.zeros([LUn * N, n + N * m])
    Mk = np.zeros([sumlen, 1])
    MU = np.tile(
        PU.b.reshape(PU.b.size, 1),
        (N, 1))
    Gk = np.zeros([sumlen, p * N])
    GU = np.zeros([LUn * N, p * N])
    K_hat = np.tile(K, (N, 1))
    B_diag = B
    E_diag = E
    for i in range(N - 1):
        B_diag = _block_diag2(B_diag, B)
        E_diag = _block_diag2(E_diag, E)
    A_n = np.eye(n)
    A_k = np.zeros([n, n * N])
    sum_vert = 0
    for i in range(N + 1):
        Li = list_P[i]
        if not isinstance(Li, pc.Polytope):
            _logger.warning(
                f'createLM: Li of type: {type(Li)}')
        # FOR M
        idx = range(
            sum_vert,
            sum_vert + Li.A.shape[0])
        Mk[idx, :] = (
            Li.b.reshape(Li.b.size, 1) -
            Li.A.dot(A_k).dot(K_hat))
        # FOR G
        if i in disturbance_ind:
            idx = np.ix_(
                range(
                    sum_vert,
                    sum_vert + Li.A.shape[0]),
                range(Gk.shape[1]))
            Gk[idx] = Li.A.dot(A_k).dot(E_diag)
            if (PU.A.shape[1] == m + n) and (i < N):
                A_k_E_diag = A_k.dot(E_diag)
                d_mult = np.vstack([
                    np.zeros([m, p * N]),
                    A_k_E_diag
                    ])
                idx = np.ix_(
                    range(
                        LUn * i,
                        LUn * (i + 1)),
                    range(p * N))
                GU[idx] = PU.A.dot(d_mult)
        # FOR L
        AB_line = np.hstack([A_n, A_k.dot(B_diag)])
        idx = np.ix_(
            range(
                sum_vert,
                sum_vert + Li.A.shape[0]),
            range(
                0,
                Lk.shape[1]))
        Lk[idx] = Li.A.dot(AB_line)
        if i >= N:
            continue
        if PU.A.shape[1] == m:
            idx = np.ix_(
                range(
                    i * LUn,
                    (i + 1) * LUn),
                range(
                    n + m * i,
                    n + m * (i + 1)))
            LU[idx] = PU.A
        elif PU.A.shape[1] == m + n:
            uk_line = np.zeros([m, n + m * N])
            idx = np.ix_(
                range(m),
                range(
                    n + m * i,
                    n + m * (i + 1)))
            uk_line[idx] = np.eye(m)
            A_mult = np.vstack([
                uk_line,
                AB_line])
            b_mult = np.zeros([m + n, 1])
            b_mult[range(m, m + n), :] = A_k.dot(K_hat)
            idx = np.ix_(
                range(
                    i * LUn,
                    (i + 1) * LUn),
                range(n + m * N))
            LU[idx] = PU.A.dot(A_mult)
            MU[
                range(
                    i * LUn,
                    (i + 1) * LUn),
                :] -= PU.A.dot(b_mult)
        # Iterate
        sum_vert += Li.A.shape[0]
        A_n = A.dot(A_n)
        A_k = A.dot(A_k)
        idx = np.ix_(
            range(n),
            range(
                i * n,
                (i + 1) * n))
        A_k[idx] = np.eye(n)
    # Get disturbance sets
    if not np.all(Gk == 0):
        G = np.vstack([Gk, GU])
        D_hat = get_max_extreme(G, D, N)
    else:
        D_hat = np.zeros([sumlen + LUn * N, 1])
    # Put together matrices L, M
    L = np.vstack([Lk, LU])
    M = np.vstack([Mk, MU]) - D_hat
    _logger.debug(
        'Computed S0 polytope: '
        '`L x <= M`, where:\n'
        '\t`L = \n`'
        f'{L}\n\t M = \n{M}\n')
    return L, M


def get_max_extreme(
        G:
            np.ndarray,
        D:
            pc.Polytope,
        N:
            int
        ) -> np.ndarray:
    """Calculate the array `d_hat` such that:

    ```
    d_hat = max(G * DN_extreme)
    ```

    where `DN_extreme` are the vertices
    of the set `D^N`.

    This is used to describe the polytope:

    ```
    L * x <= M - G * d_hat
    ```

    Calculating `d_hat` is equivalent to
    taking the intersection of the polytopes:

    ```
    L * x <= M - G * d_i
    ```

    for every possible `d_i` in the set of
    extreme points to `D^N`.

    @param G:
        The matrix to maximize with respect to
    @param D:
        Polytope describing the disturbance set
    @param N:
        Horizon length
    @return:
        `d_hat` mentioned above,
        as an array that describes
        the maximum possible effect from
        the disturbance.
    """
    D_extreme = pc.extreme(D)
    nv = D_extreme.shape[0]
    dim = D_extreme.shape[1]
    DN_extreme = np.zeros([
        dim * N,
        nv**N])
    for i in range(nv**N):
        # Last N digits are indices we want
        ind = np.base_repr(
            i,
            base=nv,
            padding=N)
        for j in range(N):
            DN_extreme[
                range(
                    j * dim,
                    (j + 1) * dim),
                i] = D_extreme[
                    int(ind[-j - 1]),
                    :]
    d_hat = np.amax(
        np.dot(G, DN_extreme),
        axis=1)
    return d_hat.reshape(d_hat.size, 1)


def _block_diag2(
        A:
            np.ndarray,
        B:
            np.ndarray
        ) -> np.ndarray:
    """Similar to `block_diag()` in `scipy.linalg`.

    This function is similar to the function
    `scipy.linalg.block_diag`,
    but is restricted to 2 inputs.

    Old versions of the linear algebra
    package in SciPy (i.e., `scipy.linalg`)
    do not have a `block_diag()` function.

    Providing `_block_diag2()` here until
    most folks are using sufficiently
    up-to-date SciPy installations
    improves portability.
    """
    if len(A.shape) == 1:
        A = np.array([A])
            # Cast 1d array into matrix
    if len(B.shape) == 1:
        B = np.array([B])
    C = np.zeros(
        (A.shape[0] + B.shape[0],
         A.shape[1] + B.shape[1]))
    C[:A.shape[0], :A.shape[1]] = A.copy()
    C[A.shape[0]:, A.shape[1]:] = B.copy()
    return C
