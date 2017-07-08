"""Usage example for the function `abstract.get_input`.

To be run after `continuous.py`, in same session.
For example, within an `ipython` interactive session in
this directory:

run ../continuous.py
run -i test_get_input.py
"""
from __future__ import print_function

from tulip.abstract import get_input, find_discrete_state
from polytope import is_inside
import numpy as np

def integrate(sys_dyn, x0, u_seq):
    # is the continuous transition correct ?
    N = u_seq.shape[0]
    x = x0.reshape(x0.size, 1)

    A = sys_dyn.A
    B = sys_dyn.B

    if len(sys_dyn.K) == 0:
        K = np.zeros(x.shape)
    else:
        K = sys_dyn.K

    print('started continuous transition')
    m = u_seq[0, :].size
    for i in range(N):
        u = u_seq[i, :].reshape(m, 1)
        x = A.dot(x) + B.dot(u) + K

        print('Discrete time: k = ' +str(i) )
        print('\t u[' +str(i) +"]' = " +str(u.T) )
        print('\t x[' +str(i) +"]' = " +str(x.T) +'\n')

    print('completed continuous transition iteration')
    return x

x0 = np.array([0.5, 0.6])
start = find_discrete_state(x0, disc_dynamics.ppp)
end = 14

start_poly = disc_dynamics.ppp.regions[start]
end_poly = disc_dynamics.ppp.regions[end]

if not is_inside(start_poly, x0):
    raise Exception('x0 \\notin start_poly')

start_state = start
end_state = end

post = disc_dynamics.ts.states.post(start_state)
print(post)
if not end_state in post:
    raise Exception('end \\notin post(start)')

u_seq = get_input(x0, sys_dyn, disc_dynamics,
              start, end)
print('Computed input sequence: u = ')
print(u_seq)

x = integrate(sys_dyn, x0, u_seq)

# arrived at target ?
if not is_inside(end_poly, x):
    raise Exception('incorrect continuous transition')
else:
    print('arrived at target Region/Polytope')
