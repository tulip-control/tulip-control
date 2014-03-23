"""
Tests for tulip.hybrid module
"""
import numpy as np

from tulip import hybrid
from tulip import polytope as pc

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
    
    hyb = hybrid.HybridSysDyn(
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



def time_semantics_test():
    """Tests out time semantics for hybrid systems module."""

    # Specify four LTI systems:
    A1 = np.eye(2)
    A2 = np.array([[0, 1], [0, 0]])
    B1 = np.array([[0] ,[1]])
    B2 = np.array([[1], [0]])
    poly1 = pc.Polytope.from_box([[0, 1], [0, 1]])
    poly2 = pc.Polytope.from_box([[1, 2], [0, 1]])
    total_box = pc.Region(list_poly=[poly1, poly2])
    Uset = pc.Polytope.from_box([[0, 1], [0, 1]])

    # Test correct time semantics for LTIs, should produce no errors
    LTI1 = hybrid.LtiSysDyn(A=A1, B=B1, Uset=Uset, domain=poly1,
                            time_semantics='discrete', timestep=None)
    LTI2 = hybrid.LtiSysDyn(A=A2, B=B2, Uset=Uset, domain=poly2,
                            time_semantics='sampled', timestep=.1)
    LTI3 = hybrid.LtiSysDyn(A=A2, B=B2, Uset=Uset, domain=poly1)
    LTI4 = hybrid.LtiSysDyn(A=A1, B=B2, Uset=Uset, domain=poly2)
    assert(LTI1.time_semantics == 'discrete')
    assert(LTI2.time_semantics == 'sampled')
    assert(LTI1.timestep is None)
    assert(LTI2.timestep == .1)
    assert(LTI3.time_semantics is None)
    assert(LTI3.timestep is None)
    assert(LTI4.time_semantics is None)
    assert(LTI4.timestep is None)

    # Test incorrect time semantics, should produce errors

    # Discrete time semantics, existing timestep
    try:
        LTI = hybrid.LtiSysDyn(A=A1, B=B1, Uset=Uset, domain=poly1,
                               time_semantics='discrete', timestep=.1)
        raise AssertionError('time semantic test failed')
    except ValueError:
        pass

    # invalid semantics
    try:
        LTI = hybrid.LtiSysDyn(A=A1, B=B1, Uset=Uset, domain=poly1,
                               time_semantics='hello', timestep=.1)
        raise AssertionError
    except ValueError:
        pass

    # nonpositive timestep
    try:
        LTI = hybrid.LtiSysDyn(A=A1, B=B1, Uset=Uset, domain=poly1,
                               time_semantics='sampled', timestep=0)
        raise AssertionError
    except ValueError:
        pass

    # timestep is wrong type
    try:
        LTI = hybrid.LtiSysDyn(A=A1, B=B1, Uset=Uset, domain=poly1,
                               time_semantics='sampled', timestep='.1')
        raise AssertionError
    except TypeError:
        pass


    # Putting pwa together successfully, without time overwrite
    LTI1 = hybrid.LtiSysDyn(A=A1, B=B1, Uset=Uset, domain=poly1,
                            time_semantics='sampled', timestep=.1)
    LTI2 = hybrid.LtiSysDyn(A=A2, B=B2, Uset=Uset, domain=poly2,
                            time_semantics='sampled', timestep=.1)
    PWA1 = hybrid.PwaSysDyn(list_subsys=[LTI3, LTI4], domain=total_box,
                            overwrite_time=False)

    # Putting pwa together successfully, with time overwrite
    PWA2 = hybrid.PwaSysDyn(list_subsys=[LTI1, LTI2], domain=total_box,
                            time_semantics='sampled', timestep=.1,
                            overwrite_time=True)


    # Putting pwa togther unsuccessfully

    # different time semantics among subsystems
    try:
        LTI1 = hybrid.LtiSysDyn(A=A1, B=B1, Uset=Uset, domain=poly1,
                            time_semantics='sampled', timestep=.1)
        LTI2 = hybrid.LtiSysDyn(A=A2, B=B2, Uset=Uset, domain=poly2,
                            time_semantics='discrete')
        PWA = hybrid.PwaSysDyn(list_subsys=[LTI1, LTI2], domain=total_box,
                               time_semantics='sampled', timestep=.1,
                               overwrite_time=False)
        raise AssertionError
    except ValueError:
        pass

    # different timesteps among subsystems
    try:
        LTI1 = hybrid.LtiSysDyn(A=A1, B=B1, Uset=Uset, domain=poly1,
                            time_semantics='sampled', timestep=.1)
        LTI2 = hybrid.LtiSysDyn(A=A2, B=B2, Uset=Uset, domain=poly2,
                            time_semantics='sampled', timestep=.2)
        PWA = hybrid.PwaSysDyn(list_subsys=[LTI1, LTI2], domain=total_box,
                               time_semantics='sampled', timestep=.1,
                               overwrite_time=False)
        raise AssertionError
    except ValueError:
        pass

    # time semantics don't match what is specified for pwa system
    try:
        LTI1 = hybrid.LtiSysDyn(A=A1, B=B1, Uset=Uset, domain=poly1,
                                time_semantics='sampled', timestep=.1)
        LTI2 = hybrid.LtiSysDyn(A=A2, B=B2, Uset=Uset, domain=poly2,
                                time_semantics='sampled', timestep=.1)
        PWA = hybrid.PwaSysDyn(list_subsys=[LTI1, LTI2], domain=total_box,
                               time_semantics='discrete', overwrite_time=False)
        raise AssertionError
    except ValueError:
        pass

    # timesteps don't match what is specified for pwa system
    try:
        LTI1 = hybrid.LtiSysDyn(A=A1, B=B1, Uset=Uset, domain=poly1,
                                time_semantics='sampled', timestep=.1)
        LTI2 = hybrid.LtiSysDyn(A=A2, B=B2, Uset=Uset, domain=poly2,
                                time_semantics='sampled', timestep=.1) 
        PWA = hybrid.PwaSysDyn(list_subsys=[LTI1, LTI2], domain=total_box,
                               time_semantics='sampled', timestep=.2,
                               overwrite_time=False)
        raise AssertionError
    except ValueError:
        pass


    # PWA fails _check_time_data
    try:
        PWA = hybrid.PwaSysDyn(list_subsys=[LTI1, LTI2], domain=total_box,
                               time_semantics='hello')
        raise AssertionError 
    except ValueError:
        pass


    # Correct switched system constructions
    env_labels = ('hi', 'hello')
    sys_labels = ('mode1',)
    disc_domain_size = (2, 1)
    LTI1 = hybrid.LtiSysDyn(A=A1, B=B1, Uset=Uset, domain=poly1,
        time_semantics='sampled', timestep=.1)
    LTI2 = hybrid.LtiSysDyn(A=A2, B=B2, Uset=Uset, domain=poly2,
        time_semantics='sampled', timestep=.1)
    LTI3 = hybrid.LtiSysDyn(A=A1, B=B1, Uset=Uset, domain=poly2,
        time_semantics='sampled', timestep=.1)
    LTI4 = hybrid.LtiSysDyn(A=A2, B=B2, Uset=Uset, domain=poly1,
        time_semantics='sampled', timestep=.1)
    PWA1 = hybrid.PwaSysDyn(list_subsys=[LTI1, LTI2], domain=total_box,
        time_semantics='sampled', timestep=.1)
    PWA2 = hybrid.PwaSysDyn(list_subsys=[LTI3, LTI4], domain=total_box,
        time_semantics='sampled', timestep=.1)
    dynamics1 = { (env_labels[0], sys_labels[0]) : PWA1,
                  (env_labels[1], sys_labels[0]) : PWA2 }
    dynamics2 = { (env_labels[0], sys_labels[0]) : PWA2,
                  (env_labels[1], sys_labels[0]) : PWA1 }
    switched1 = hybrid.HybridSysDyn(disc_domain_size=disc_domain_size, 
        dynamics=dynamics1, env_labels=env_labels, disc_sys_labels=sys_labels,
        time_semantics='sampled', timestep=.1, overwrite_time=True)
    switched2 = hybrid.HybridSysDyn(disc_domain_size=disc_domain_size, 
        dynamics=dynamics1, env_labels=env_labels, disc_sys_labels=sys_labels,
        time_semantics='sampled', timestep=.1, overwrite_time=False)
    assert(switched1.time_semantics == 'sampled')
    assert(switched2.time_semantics == 'sampled')
    assert(switched1.timestep == .1)
    assert(switched2.timestep == .1)
    

    # Hybrid system can fail _check_time_data
    try:
        switched1 = hybrid.HybridSysDyn(disc_domain_size=disc_domain_size, 
            dynamics=dynamics1, env_labels=env_labels, 
            disc_sys_labels=sys_labels, time_semantics='hello', 
            timestep=.1, overwrite_time=True)
        raise AssertionError
    except ValueError:
        pass

    # Hybrid system fails _check_time_consistency
    try:
        switched1 = hybrid.HybridSysDyn(disc_domain_size=disc_domain_size, 
            dynamics=dynamics1, env_labels=env_labels, 
            disc_sys_labels=sys_labels, time_semantics='sampled', 
            timestep=.2, overwrite_time=False)
        raise AssertionError
    except ValueError:
        pass
