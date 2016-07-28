"""
Tests for tulip.hybrid module
"""
import numpy as np

from nose.tools import raises
from tulip import hybrid
import polytope as pc

def switched_system_test():
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
    }
    env_labels = ['a', 'b']
    sys_labels = ['c', 'd']

    hyb = hybrid.SwitchedSysDyn(
        disc_domain_size=dom,
        dynamics=dyn,
        cts_ss=domain,
        env_labels=env_labels,
        disc_sys_labels=sys_labels
    )

    print(hyb)

    assert(hyb.disc_domain_size == dom)
    assert(hyb.dynamics == dyn)
    assert(hyb.env_labels == env_labels)
    assert(hyb.disc_sys_labels == sys_labels)
    assert(hyb.cts_ss == domain)


class time_semantics_test:
    """Test out time semantics for hybrid systems module."""
    def setUp(self):
        self.A1 = np.eye(2)
        self.A2 = np.array([[0, 1], [0, 0]])
        self.B1 = np.array([[0] ,[1]])
        self.B2 = np.array([[1], [0]])
        self.poly1 = pc.Polytope.from_box([[0, 1], [0, 1]])
        self.poly2 = pc.Polytope.from_box([[1, 2], [0, 1]])
        self.total_box = pc.Region(list_poly=[self.poly1, self.poly2])
        self.Uset = pc.Polytope.from_box([[0, 1]])

    def tearDown(self):
        self.A1 = None
        self.A2 = None
        self.B1 = None
        self.B2 = None
        self.poly1 = None
        self.poly2 = None
        self.total_box = None
        self.Uset = None

    def test_correct_lti_construction(self):
        LTI1 = hybrid.LtiSysDyn(A=self.A1, B=self.B1,
                                Uset=self.Uset, domain=self.poly1,
                                time_semantics='discrete', timestep=None)
        LTI2 = hybrid.LtiSysDyn(A=self.A2, B=self.B2,
                                Uset=self.Uset, domain=self.poly2,
                                time_semantics='sampled', timestep=.1)
        LTI3 = hybrid.LtiSysDyn(A=self.A2, B=self.B2,
                                Uset=self.Uset, domain=self.poly1)
        LTI4 = hybrid.LtiSysDyn(A=self.A1, B=self.B2,
                                Uset=self.Uset, domain=self.poly2)
        assert(LTI1.time_semantics == 'discrete')
        assert(LTI2.time_semantics == 'sampled')
        assert(LTI1.timestep is None)
        assert(LTI2.timestep == .1)
        assert(LTI3.time_semantics is None)
        assert(LTI3.timestep is None)
        assert(LTI4.time_semantics is None)
        assert(LTI4.timestep is None)

    def test_correct_pwa_construction(self):
        # Putting pwa together successfully, without time overwrite
        LTI1 = hybrid.LtiSysDyn(A=self.A1, B=self.B1,
                                Uset=self.Uset, domain=self.poly1,
                                time_semantics='sampled', timestep=.1)
        LTI2 = hybrid.LtiSysDyn(A=self.A2, B=self.B2,
                                Uset=self.Uset, domain=self.poly2,
                                time_semantics='sampled', timestep=.1)
        LTI3 = hybrid.LtiSysDyn(A=self.A2, B=self.B2,
                                Uset=self.Uset, domain=self.poly1)
        LTI4 = hybrid.LtiSysDyn(A=self.A1, B=self.B2,
                                Uset=self.Uset, domain=self.poly2)
        PWA1 = hybrid.PwaSysDyn(list_subsys=[LTI3, LTI4], domain=self.total_box,
                                overwrite_time=False)

        # Putting pwa together successfully, with time overwrite
        PWA2 = hybrid.PwaSysDyn(list_subsys=[LTI1, LTI2], domain=self.total_box,
                                time_semantics='sampled', timestep=.1,
                                overwrite_time=True)

    @raises(ValueError)
    def test_pwa_difftseman_among_subsys(self):
        """Different time semantics among LtiSysDyn subsystems of PwaSysDyn"""
        LTI1 = hybrid.LtiSysDyn(A=self.A1, B=self.B1,
                                Uset=self.Uset, domain=self.poly1,
                                time_semantics='sampled', timestep=.1)
        LTI2 = hybrid.LtiSysDyn(A=self.A2, B=self.B2,
                                Uset=self.Uset, domain=self.poly2,
                                time_semantics='discrete')
        PWA = hybrid.PwaSysDyn(list_subsys=[LTI1, LTI2], domain=self.total_box,
                               time_semantics='sampled', timestep=.1,
                               overwrite_time=False)

    @raises(ValueError)
    def test_pwa_difftstep_among_subsys(self):
        """Different timesteps among LtiSysDyn subsystems of PwaSysDyn"""
        LTI1 = hybrid.LtiSysDyn(A=self.A1, B=self.B1,
                                Uset=self.Uset, domain=self.poly1,
                                time_semantics='sampled', timestep=.1)
        LTI2 = hybrid.LtiSysDyn(A=self.A2, B=self.B2,
                                Uset=self.Uset, domain=self.poly2,
                                time_semantics='sampled', timestep=.2)
        PWA = hybrid.PwaSysDyn(list_subsys=[LTI1, LTI2], domain=self.total_box,
                               time_semantics='sampled', timestep=.1,
                               overwrite_time=False)

    @raises(ValueError)
    def test_pwa_difftseman_from_subsys(self):
        """LtiSysDyn subsystems time semantics do not match that of PwaSysDyn"""
        LTI1 = hybrid.LtiSysDyn(A=self.A1, B=self.B1,
                                Uset=self.Uset, domain=self.poly1,
                                time_semantics='sampled', timestep=.1)
        LTI2 = hybrid.LtiSysDyn(A=self.A2, B=self.B2,
                                Uset=self.Uset, domain=self.poly2,
                                time_semantics='sampled', timestep=.1)
        PWA = hybrid.PwaSysDyn(list_subsys=[LTI1, LTI2], domain=self.total_box,
                               time_semantics='discrete', overwrite_time=False)

    @raises(ValueError)
    def test_pwa_difftstep_from_subsys(self):
        """LtiSysDyn subsystems timesteps do not match that of PwaSysDyn"""
        LTI1 = hybrid.LtiSysDyn(A=self.A1, B=self.B1,
                                Uset=self.Uset, domain=self.poly1,
                                time_semantics='sampled', timestep=.1)
        LTI2 = hybrid.LtiSysDyn(A=self.A2, B=self.B2,
                                Uset=self.Uset, domain=self.poly2,
                                time_semantics='sampled', timestep=.1)
        PWA = hybrid.PwaSysDyn(list_subsys=[LTI1, LTI2], domain=self.total_box,
                               time_semantics='sampled', timestep=.2,
                               overwrite_time=False)

    @raises(ValueError)
    def test_pwa_invalid_semantics(self):
        LTI1 = hybrid.LtiSysDyn(A=self.A1, B=self.B1,
                                Uset=self.Uset, domain=self.poly1,
                                time_semantics='sampled', timestep=.1)
        LTI2 = hybrid.LtiSysDyn(A=self.A2, B=self.B2,
                                Uset=self.Uset, domain=self.poly2,
                                time_semantics='sampled', timestep=.1)
        PWA = hybrid.PwaSysDyn(list_subsys=[LTI1, LTI2], domain=self.total_box,
                               time_semantics='hello')

    @raises(ValueError)
    def test_disctime_errtstep(self):
        """Discrete time semantics yet given timestep"""
        LTI = hybrid.LtiSysDyn(A=self.A1, B=self.B1,
                               Uset=self.Uset, domain=self.poly1,
                               time_semantics='discrete', timestep=.1)


    @raises(ValueError)
    def test_lti_invalid_semantics(self):
        LTI = hybrid.LtiSysDyn(A=self.A1, B=self.B1,
                               Uset=self.Uset, domain=self.poly1,
                               time_semantics='hello', timestep=.1)

    @raises(ValueError)
    def test_nonpositive_timestep(self):
        LTI = hybrid.LtiSysDyn(A=self.A1, B=self.B1,
                               Uset=self.Uset, domain=self.poly1,
                               time_semantics='sampled', timestep=0)

    @raises(TypeError)
    def test_timestep_wrong_type(self):
        LTI = hybrid.LtiSysDyn(A=self.A1, B=self.B1,
                               Uset=self.Uset, domain=self.poly1,
                               time_semantics='sampled', timestep='.1')


class SwitchedSysDyn_test:
    def setUp(self):
        self.A1 = np.eye(2)
        self.A2 = np.array([[0, 1], [0, 0]])
        self.B1 = np.array([[0] ,[1]])
        self.B2 = np.array([[1], [0]])
        self.poly1 = pc.Polytope.from_box([[0, 1], [0, 1]])
        self.poly2 = pc.Polytope.from_box([[1, 2], [0, 1]])
        self.total_box = pc.Region(list_poly=[self.poly1, self.poly2])
        self.Uset = pc.Polytope.from_box([[0, 1]])
        self.env_labels = ('hi', 'hello')
        self.sys_labels = ('mode1',)
        self.disc_domain_size = (2, 1)
        self.LTI1 = hybrid.LtiSysDyn(A=self.A1, B=self.B1,
                                     Uset=self.Uset, domain=self.poly1,
                                     time_semantics='sampled', timestep=.1)
        self.LTI2 = hybrid.LtiSysDyn(A=self.A2, B=self.B2,
                                     Uset=self.Uset, domain=self.poly2,
                                     time_semantics='sampled', timestep=.1)
        self.LTI3 = hybrid.LtiSysDyn(A=self.A1, B=self.B1,
                                     Uset=self.Uset, domain=self.poly2,
                                     time_semantics='sampled', timestep=.1)
        self.LTI4 = hybrid.LtiSysDyn(A=self.A2, B=self.B2,
                                     Uset=self.Uset, domain=self.poly1,
                                     time_semantics='sampled', timestep=.1)
        self.PWA1 = hybrid.PwaSysDyn(list_subsys=[self.LTI1, self.LTI2],
                                     domain=self.total_box,
                                     time_semantics='sampled', timestep=.1)
        self.PWA2 = hybrid.PwaSysDyn(list_subsys=[self.LTI3, self.LTI4],
                                     domain=self.total_box,
                                     time_semantics='sampled', timestep=.1)
        self.dynamics1 = {(self.env_labels[0], self.sys_labels[0]): self.PWA1,
                          (self.env_labels[1], self.sys_labels[0]): self.PWA2}

    def tearDown(self):
        self.A1 = None
        self.A2 = None
        self.B1 = None
        self.B2 = None
        self.poly1 = None
        self.poly2 = None
        self.total_box = None
        self.Uset = None
        self.env_labels = None
        self.sys_labels = None
        self.disc_domain_size = None
        self.LTI1 = None
        self.LTI2 = None
        self.LTI3 = None
        self.LTI4 = None
        self.PWA1 = None
        self.PWA2 = None
        self.dynamics1 = None

    @raises(ValueError)
    def test_hybrid_difftstep_from_subsys(self):
        """LtiSysDyn subsystems timesteps do not match that of SwitchedSysDyn"""
        hybrid.SwitchedSysDyn(disc_domain_size=self.disc_domain_size,
                            dynamics=self.dynamics1, env_labels=self.env_labels,
                            disc_sys_labels=self.sys_labels,
                            time_semantics='hello', timestep=.1,
                            overwrite_time=True)

    @raises(ValueError)
    def test_hybrid_fail_check_time_consistency(self):
        # fail _check_time_consistency
        hybrid.SwitchedSysDyn(disc_domain_size=self.disc_domain_size,
                            dynamics=self.dynamics1, env_labels=self.env_labels,
                            disc_sys_labels=self.sys_labels,
                            time_semantics='sampled', timestep=.2,
                            overwrite_time=False)

    def test_correct_switched_construction(self):
        switched1 = hybrid.SwitchedSysDyn(disc_domain_size=self.disc_domain_size,
                                        dynamics=self.dynamics1,
                                        env_labels=self.env_labels,
                                        disc_sys_labels=self.sys_labels,
                                        time_semantics='sampled', timestep=.1,
                                        overwrite_time=True)
        switched2 = hybrid.SwitchedSysDyn(disc_domain_size=self.disc_domain_size,
                                        dynamics=self.dynamics1,
                                        env_labels=self.env_labels,
                                        disc_sys_labels=self.sys_labels,
                                        time_semantics='sampled', timestep=.1,
                                        overwrite_time=False)
        assert(switched1.time_semantics == 'sampled')
        assert(switched2.time_semantics == 'sampled')
        assert(switched1.timestep == .1)
        assert(switched2.timestep == .1)
