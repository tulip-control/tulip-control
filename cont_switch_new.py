#!/usr/bin/env python
#
# Copyright (c) 2011 - 2014 by California Institute of Technology
# and 2014-2015 The Regents of the University of Michigan
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
# WARNING: This example may not yet be working.  Please check again in
#          the upcoming release.
#
"""
This is code that takes a continuous state space with different modes 
and discretizes it. 
"""

import sys
import numpy as np
from scipy import sparse as sp
import time
import logging
import multiprocessing as mp
import math
logger = logging.getLogger(__name__)

from tulip import spec, hybrid,synth
from tulip.synth import _spec_plus_sys, strategy2mealy
from tulip.interfaces import jtlv, gr1c
from polytope import box2poly
from tulip.abstract import prop2part, discretize, add_grid 
from tulip.abstract import part2convex, find_discrete_state
from tulip.abstract import find_equilibria, get_input
from tulip.abstract import discretization as ds
from tulip.abstract.discretization import create_prog_map
from tulip.abstract import prop2partition as p2p
from tulip import transys as trs
import polytope as pc
from pprint import pformat

from tulip.transys.labeled_graphs import LabeledDiGraph, str2singleton
from tulip.abstract.plot import plot_partition


sys.setrecursionlimit(10000)
"""
Added:
-create_prog_map,get_postarea,get_postarea_transitions, 
discretize_modeonlyswitched to discretization.py
-class AugmentedFiniteTransitionSystem to transys.py
-find_equilibria to prop2partition.py
"""

# Kaushik - ignore for now. Relevant part of this code has been copied at the bottom. This
# is the original code - so I didn't delete it.
"""
def simulation(disc_dynamics,ctrl):
    Tc = [1.3]
    Th = [-1.]
    P = [-1.3]

    s0_part = find_discrete_state([Tc[0],Th[0],P[0]],disc_dynamics.ppp)
    s0_loc = disc_dynamics.ppp2ts[s0_part]

    mach = synth.determinize_machine_init(ctrl, {'loc':s0_loc})
    sim_hor = 130

    (s1, dum) = mach.reaction('Sinit', {'level': 1})
    (s1, dum) = mach.reaction(s1, {'level': 1})
    for sim_time in range(sim_hor):
        for i in range(3):
            if np.array([Tc[sim_time*N],Th[sim_time*N],P[sim_time*N]]) in cont_sys.list_subsys[i].domain:
                sysnow=i
        u = get_input(
                np.array([Tc[sim_time*N],Th[sim_time*N],P[sim_time*N]]),
                cont_sys.list_subsys[sysnow],
                disc_dynamics,
                s0_part,
                disc_dynamics.ppp2ts.index(dum['loc']),
                mid_weight=100.0,
                test_result=True)

    for ind in range(N):
        x = np.dot(
                cont_sys.list_subsys[sysnow].A, [Tc[-1],Th[-1],P[-1]]
                ) + np.dot(cont_sys.list_subsys[sysnow].B,u[ind]) + cont_sys.list_subsys[sysnow].K.flatten()
        Tc.append(x[0])
        Th.append(x[1])
        P.append(x[2])

    s0_part = find_discrete_state([Tc[-1],Th[-1],P[-1]],disc_dynamics.ppp)
    s0_loc = disc_dynamics.ppp2ts[s0_part]
    print s0_loc, dum['loc']
    if pc.is_inside(disc_dynamics.ppp[disc_dynamics.ppp2ts.index(dum['loc'])],[Tc[-1],Th[-1],P[-1]]):
        s0_part = disc_dynamics.ppp2ts.index(dum['loc'])
    if sim_time <= 10:
        (s1, dum) = mach.reaction(s1, {'level': 1})
    elif sim_time <= 50:
        (s1, dum) = mach.reaction(s1, {'level': 0})
    else:
        (s1, dum) = mach.reaction(s1, {'level': 2})
"""

abs_tol=1e-7

A_off=np.array([[0.9998,0.],[0.,1.]])
A_heat=np.array([[0.9998,0.0002],[0.,1.]])
A_cool=np.array([[0.9998,-0.0002],[0.,1.]])
A_on=np.array([[0.9998,0.0002],[0.,1.]])

K_off=np.array([[0.0032],[0.]])
K_heat=np.array([[0.],[0.01]])
K_cool=np.array([[0.],[-0.01]])
K_on=np.array([[0.],[0.]])

B_zero= np.array([[0., 0.], [ 0., 0.]])

cont_state_space = box2poly([[15., 24.],[15., 24.]])
cont_props = {}
cont_props['LOW'] = box2poly([[17., 19.], [20., 22.]])
cont_props['HIGH'] = box2poly([[21., 22.], [20., 22.]])
cont_props['OUTSIDE'] = box2poly([[24.,25.],[24.,25.]])
orig_props=set(cont_props)
out=[]
out.append(box2poly([[24.,25.],[24.,25.]]))


sdyn_off = hybrid.LtiSysDyn(A_off,  B_zero, None, K_off, None, None, cont_state_space)
sdyn_heat = hybrid.LtiSysDyn(A_heat,  B_zero, None, K_heat, None, None, cont_state_space)
sdyn_cool = hybrid.LtiSysDyn(A_cool,B_zero, None, K_cool, None, None, cont_state_space)
sdyn_on = hybrid.LtiSysDyn(A_on,  B_zero, None, K_on, None, None, cont_state_space)

pwa_off = hybrid.PwaSysDyn(list_subsys=[sdyn_off],domain=cont_state_space)#,time_semantics='sampled',timestep=0.1)
pwa_heat = hybrid.PwaSysDyn(list_subsys=[sdyn_heat],domain=cont_state_space)
pwa_cool= hybrid.PwaSysDyn(list_subsys=[sdyn_cool],domain=cont_state_space)
pwa_on = hybrid.PwaSysDyn(list_subsys=[sdyn_on],domain=cont_state_space)

#print pwa_off

sys_dyn={}
sys_dyn['regular','off']=pwa_off
sys_dyn['regular','heat']=pwa_heat
sys_dyn['regular','cool']=pwa_cool
sys_dyn['regular','on']=pwa_on

ssd=hybrid.SwitchedSysDyn(disc_domain_size=(1,4),
                 dynamics=sys_dyn, cts_ss=cont_state_space,
                 env_labels=['regular'], disc_sys_labels=['off', 'heat','cool','on'], time_semantics='sampled',
                 timestep=0.1, overwrite_time=True)
print ssd;
owner='env'
#is_convex=True
#grid_size=-1.
#eps=0.3
#cont_dyn={}
#trans={}
#p2p.find_equilibria(ssd=ssd,cont_props=cont_props,eps=eps)
#cont_part = p2p.prop2part(ssd.cts_ss, cont_props)

#if is_convex:
#   cont_part, new2old = part2convex(cont_part)
#   print "Convexify DONE!!"
#  

#if grid_size==-1.:
#   ref_grid=cont_part
#else:
#   ref_grid=p2p.add_grid(ppp=cont_part, grid_size=grid_size)
#   
#    
#prog_map=create_prog_map(ssd.modes,ref_grid)




#print prog_map

#abstMOS=ds.discretize_modeonlyswitched(ssd=ssd,cont_props=cont_props, owner=owner, grid_size=0.5,
#                                visualize=False,eps=0.3, is_convex=True,
#                                N=1,abs_tol=1e-7)
#print abstMOS


import pickle as pl
abstMOS=pl.load( open( "abstMOS.p", "rb" ) )

#env_vars ={}
#env_vars['level'] = (0,1)
#sys_vars ={'trackfl':(0,2)}

#env_init = set()
#env_prog = set()
#env_safe = set() 

#sys_init = set()
#sys_safe = {'!OUTSIDE',
#    'LOW -> (trackfl=0)',
#    'HIGH -> (trackfl=1)',
#    '(trackfl=0) -> X ((trackfl=0) || !(level=0))',
#    '(trackfl=1) -> X ((trackfl=1) || !(level=1))'
#}

#sys_prog = {
#    'LOW || !(level=0)',
#    'HIGH || !(level=1)'
#    }



##env_vars = set() #{'off','on','heat','cool'}
##env_init = set()                # empty set
##env_prog = set()
##env_safe = set()
#cnt=0;
##safe = ''

##env_safe = set()
##prog=set()

#""" Creates a set of environment progress assumptions based on 
#!<>[](~eq_pnti & modei) == []<>(eq_pnti or !modei). 

#Is that assumption right?
#Is eq_pnt still used?
#Is it required? - My guess - not yet.
#""" 
## for x in abstMOS.ts.sys_actions:
##     sp='(eqpnt_'
##     sp+=x
##     sp+=' && sys_actions = !"'
##     sp+=x
##     sp+='")'
##     # sp+=x
##     # for y in abstMOS.ts.progress_map[x]:
##     #     for i,z in enumerate(y):
##     #         sp+=z
##     #         if i!=len(y)-1:
##     #             sp+=' || '
##     # sp+=') && '
##     # sp+=x[1]
##     # sp+=')'
##     prog|={sp}
##env_prog=prog

#""" Assumption/prog - !<>[](~eq_pnti & modei) == []<>(eq_pnti or !modei)
## Are there stable & unstable eq pnts? - In this example - yes. Should there be? - Don't know.
## The synth function should append the above assumption automatically on detecting AOFTS.

#"""
## System variables and requirements
##sys_vars=set()

##sys_init = set()            
##sys_prog = {'LOW'}
##sys_safe = {'!OUTSIDE', 'LOW -> X LOW'}
#specs = spec.GRSpec(env_vars, sys_vars, env_init, sys_init,
#                    env_safe, sys_safe, env_prog, sys_prog)



##specs = _spec_plus_sys(
##        specs, env=abstMOS.ts, sys=None,
##        ignore_env_init=False,
##        ignore_sys_init=False,
##        bool_states=False,
##        bool_actions=False)

##strategy = gr1c.synthesize(specs)




#gr_fts = synth.synthesize('gr1c', specs, env=abstMOS.ts,rm_deadends=False)
##print (gr_fts)

#if not gr_fts.save('gr_fts.eps'):
#    print(gr_fts)


gr_fts=pl.load( open( "gr_fts.p", "rb" ) )
disc_dynamics=abstMOS
ctrl=gr_fts
Tc = [16.0]
Th = [16.0]

s0_part = find_discrete_state([Tc[0],Th[0]],disc_dynamics.ppp)
s0_loc = disc_dynamics.ppp2ts[s0_part]
##mach = synth.determinize_machine_init(ctrl,{'sys_actions':'on'}) # - to be used if we want a certain mode only
ctrl=synth.determinize_machine_init(ctrl)
sim_hor = 15000
N=1 # N = number of steps between each sampled transition

(s1, dum) = ctrl.reaction('Sinit', {'level': 1, 'eloc': s0_loc})
#inputs= {'eloc': s0_loc, 'level': 0}
#import itertools
#for k in itertools.permutations(inputs.iteritems(),2):
#	print k
#(s1, dum)= ctrl.reaction('Sinit' , {'eqpnt_cool': 0, 'eqpnt_off': 1, 'eqpnt_on': 1, 'env_actions': 'regular', 'HIGH': 0, 'OUTSIDE': 0, 'LOW': 0, 'eqpnt_heat': 0, 'sys_actions': 'heat', 'eloc': s0_loc})
##(s1, dum) = mach.reaction(s1, {'on': 1}) # - possible different way to give input? 
s0_locprev=0
for sim_time in range(sim_hor):
    sysnow=('regular',dum['sys_actions'])

    for ind in range(N):
        x = np.dot(
                ssd.dynamics[sysnow].list_subsys[0].A, [Tc[-1],Th[-1]]
                ) + ssd.dynamics[sysnow].list_subsys[0].K.flatten()
        Tc.append(x[0])
        Th.append(x[1])

    s0_part = find_discrete_state([Tc[-1],Th[-1]],disc_dynamics.ppp)
    s0_loc = disc_dynamics.ppp2ts[s0_part]
    if s0_loc != s0_locprev:
    	print s1, s0_loc, dum['sys_actions']
    s0_locprev=s0_loc
    (s1, dum) = ctrl.reaction(s1, {'level': 1, 'eloc': s0_loc})

###import pickle as pl
#### pl.dump(specs,open("jtlv_specs.p","wb"))
#### self_trans=pl.load(open("trans_self.p","rb"))
