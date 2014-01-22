"""
Tests for tulip.hybrid module
"""
import numpy as np

from tulip import hybrid
from tulip import polytope as pc

subsystems = []

# subsystem 0
A = np.eye(2)
B = np.eye(2)

Uset = pc.box2poly([[0.0, 1.0], [0.0, 1.0]])
domain0 = pc.box2poly([[0.0, 2.0], [0.0, 2.0]])

subsystems += [hybrid.LtiSysDyn(A, B, Uset=Uset, domain=domain0)]

# subsystem 1
domain1 = pc.box2poly([[2.0, 4.0], [0.0, 2.0]])

subsystems += [hybrid.LtiSysDyn(A, B, Uset=Uset, domain=domain1)]

# PWA system
domain = domain0.union(domain1)
pwa = hybrid.PwaSysDyn(subsystems, domain)

# Switched system (mode dynamics the same, just testing code)
dom = (2, 2)
dyn = {
    ('a', 'c'):pwa,
    ('a', 'd'):pwa,
    ('b', 'c'):pwa,
    ('b', 'd'):pwa
}
env_labels = ['a', 'b']
sys_labels = ['c', 'd']

hyb = hybrid.HybridSysDyn(
    disc_domain_size=dom,
    dynamics=dyn,
    cts_ss=domain,
    env_labels=env_labels,
    disc_sys_labels=sys_labels
)

assert(hyb.disc_domain_size == dom)
assert(hyb.dynamics == dyn)
assert(hyb.env_labels == env_labels)
assert(hyb.disc_sys_labels == sys_labels)
assert(hyb.cts_ss == domain)
