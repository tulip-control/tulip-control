#!/usr/bin/env python
#
# WARNING: This example may not yet be working.  Please check again in
#          the upcoming release.
#
"""
This is code that takes a continuous state space with different modes 
and discretizes it. 
"""
#
# Note: This code is commented to allow components to be extracted into
# the tutorial that is part of the users manual.  Comments containing
# strings of the form @label@ are used for this purpose.

# @import_section@


import sys
import numpy as np
from scipy import sparse as sp
import time
import logging
import multiprocessing as mp
logger = logging.getLogger(__name__)

from tulip import spec, hybrid,synth
from polytope import box2poly
from tulip.abstract import prop2part, discretize, add_grid, post_area, get_transitions, part2convex
from tulip import transys as trs
from tulip.abstract import discretization as ds
import polytope as pc
# @import_section_end@
from tulip.transys.labeled_graphs import LabeledDiGraph, str2singleton

from tulip.abstract.plot import plot_partition


abs_tol=1e-7

def normalize(IA,K):
    if IA.size > 0:
        IAnorm = np.sqrt(np.sum(IA*IA,1)).flatten()     
        pos = np.nonzero(IAnorm > 1e-10)[0]
        IA = IA[pos, :]
        K = K[pos]
        IAnorm = IAnorm[pos]           
        mult = 1/IAnorm
        for i in xrange(IA.shape[0]):
            IA[i,:] = IA[i,:]*mult[i]
        K = K.flatten()*mult
    return IA,K

def check_self_trans(cont_dyn,mode,cont_ss,cont_props,outside_props,pm_props):
    A=cont_dyn[mode].A
    K=cont_dyn[mode].K.T[0]
    #cont_ss.b=np.array([cont_ss.b]).T
    I=np.eye(len(A),dtype=float)
    rank_IA=np.linalg.matrix_rank(I-A)
    concat=np.hstack((I-A,K.reshape(len(A),1)))
    rank_concat=np.linalg.matrix_rank(concat)
    soln=pc.Polytope()
    props_sym='eqpnt_'+str(mode[1])
    pm_props|={props_sym}

    if (rank_IA==rank_concat):
        if (rank_IA==len(A)):
            equil=np.dot(np.linalg.inv(I-A),K)
            print "Equilibrium Points: "+str(mode)
            print equil
            print "---------------------------------"
            if (equil[0]>=(-cont_ss.b[2]) and equil[0]<=cont_ss.b[0] and equil[1]>=(-cont_ss.b[3]) and equil[1]<=cont_ss.b[1]):
                delta=equil/100
                soln=box2poly([[equil[0]-delta[0], equil[0]+delta[0]],[equil[1]-delta[1], equil[1]+delta[1]]]) 
            else:
                soln=box2poly([[24.,25.],[24.,25.]])
                outside_props|={props_sym}
        elif (rank_IA<len(A)):
            #eps=abs(min(np.amin(K),np.amin(I-A)))
            #eps=0.0005
            eps=0.2
            if eps==0:
                eps=abs(min(np.amin(-K),np.amin(A-I)))
            IAn,Kn = normalize(I-A,K)
            soln=pc.Polytope(np.vstack((IAn,-IAn)), np.hstack((Kn+eps,-Kn+eps)))

            print "First soln: "+str(mode)
            print soln
            print "---------------------------------"
            relevantsoln=pc.intersect(soln,cont_ss,abs_tol)
            if pc.is_empty(relevantsoln):
                print "Intersect "+str(mode)+" is empty"
            else:
                print "Intersect "+str(mode)+" is not empty - good job!!"
            print relevantsoln
            print "---------------------------------"

            if(pc.is_empty(relevantsoln) & ~pc.is_empty(soln)):
                soln=box2poly([[24.,25.],[24.,25.]])
                outside_props|={props_sym}
            else:
                soln=relevantsoln
    
    else:
        #Assuming trajectories go to infinity
        soln=box2poly([[24.,25.],[24.,25.]])
        outside_props|={props_sym}
        print str(mode)+" trajectories go to infinity! No solution"

    print "Normalized soln: "+str(mode)
    print soln
    print "---------------------------------"
    cont_props[props_sym]=soln
    #put soln as a cont_part and see what happens
    #intersect with the domain and then reduce?
    
    #A matrix is critically stable. then eqpnt is neither stable nor unstable. and "A" becomes on a rotational matrix and you get oscillations, mostly rotations along an ellipsoid


def post_trans(ppp, sys_dyn, N=1, abs_tol=1e-7):

    list_post_area={}
    list_extp_d=pc.extreme(sys_dyn.Wset)
    transitions = np.zeros([len(ppp.regions),(len(ppp.regions)+1)], dtype = int)
    if list_extp_d==None:
        for i in range(0,len(ppp.regions)):
            list_post_area[i]=[]
            p_current=ppp.regions[i]
            for m in range(len(ppp.regions[i].list_poly)):
                extp=pc.extreme(p_current.list_poly[m])
                j=1
                post_extp_N=extp
                while j <=N:
                     post_extp_N=np.dot(post_extp_N,sys_dyn.A.T)+sys_dyn.K.T
                     j+=1
                post_area_hull=pc.qhull(post_extp_N)
                list_post_area[i].append(post_area_hull)

                for k in range(0,len(ppp.regions)):
                    inters_region=pc.intersect(list_post_area[i][m],ppp.regions[k])
                    if (pc.is_empty(inters_region)== False and i!=k):
                        trans=1
                    else:
                        trans=0
                    transitions[i,k]=trans

                inters=pc.mldivide(post_area_hull,ppp.domain)
                if pc.is_empty(inters)== False:
                    transend=1
                else:
                    transend=0
                transitions[i,len(ppp.regions)]=transend

    return transitions

def mppt(q,i,ppp, sys_dyn, N=1, abs_tol=1e-7):
    global logger
    logger = mp.log_to_stderr()
    global transitions
    name = mp.current_process().name
    print('PPP Region: ' + str(i) + ', on: ' + str(name))

    list_post_area=[]
    p_current=ppp.regions[i]
    for m in range(len(ppp.regions[i].list_poly)):
        extp=pc.extreme(p_current.list_poly[m])
        j=1
        post_extp_N=extp
        while j <=N:
             post_extp_N=np.dot(post_extp_N,sys_dyn.A.T)+sys_dyn.K.T
             j+=1
        post_area_hull=pc.qhull(post_extp_N)
        list_post_area.append(post_area_hull)

        for k in range(0,len(ppp.regions)):
            inters_region=pc.intersect(ppp.regions[k],post_area_hull)
            if (pc.is_empty(inters_region)== False and i!=k):
                trans=1
            else:
                trans=0
            transitions[i,k]=trans

        inters=pc.mldivide(post_area_hull,ppp.domain)
        if pc.is_empty(inters)== False:
            transend=1
        else:
            transend=0
        transitions[i,len(ppp.regions)]=transend

    q.put((i, transitions))
    print('Worker: ' + str(name) + ' finished.')

def pti_loop(ppp,sys_dyn,N=1,abs_tol=1e-7):
    global logger
    logger.info('Extreme parallel discretize_switched started')
    global transitions
    list_extp_d=pc.extreme(sys_dyn.Wset)
    transitions = np.zeros([len(ppp.regions),(len(ppp.regions)+1)], dtype = int)
    mode_args=dict()
    que=mp.Queue()

    if list_extp_d==None:
        for i in range(0,len(ppp.regions)):
            mode_args[i]=(que,i,ppp,sys_dyn,N,abs_tol)

        jobs=[mp.Process(target=mppt, args=args)
            for args in mode_args.itervalues()]

        for job in jobs:
            job.start()

        transitions = dict()
        for job in jobs:
            mode, trans = que.get()
            transitions[mode] = trans
        
        for job in jobs:
            job.join()
    
    return transitions


def all_trans(ppp,post_area):

    transitions = np.zeros([(len(ppp.regions)+1),len(ppp.regions)], dtype = int)
    list_intersect_region=[]
    for i in range(0,len(ppp.regions)):
        for j in range(0,len(ppp.regions)):
            trans=0
            transend=0
            for m in range(len(ppp.regions[i].list_poly)):
                inters_region=pc.intersect(ppp.regions[j],post_area[i][m])
                if pc.is_empty(inters_region)== False:
                    trans=1
                if j==0:
                    inters=pc.mldivide(post_area[i][m],ppp.domain)   
                    if pc.is_empty(inters)== False:
                        transend=1
            transitions[i,j]=trans
            if j==0:
                transitions[len(ppp.regions),i]=transend                
    
    return transitions

def multiproc_posttrans(q,mode,ref_grid,cont_dyn,N=1,abs_tol=1e-7):
    global logger
    logger = mp.log_to_stderr()
    
    name = mp.current_process().name
    print('Discretization mode: ' + str(mode) + ', on: ' + str(name))
    
    trans = pti_loop(ref_grid, cont_dyn, N,abs_tol)
    
    q.put((mode, trans))
    print('Worker: ' + str(name) + ' finished.')

def multiproc_ptloop(ref_grid,cont_dyn,N=1,abs_tol=1e-7):
    global logger
    logger.info('parallel discretize started')
    q=mp.Queue()
    mode_args = dict()
    for mode in ssd.modes:
        mode_args[mode]=(q,mode,ref_grid,cont_dyn)

    jobs=[mp.Process(target=multiproc_posttrans, args=args)
            for args in mode_args.itervalues()]
    for job in jobs:
        job.start()

    transitions = dict()
    for job in jobs:
        mode, trans = q.get()
        transitions[mode] = trans
    
    for job in jobs:
        job.join()
    
    return transitions

A_off=np.array([[0.9998,0.],[0.,1.]])
A_heat=np.array([[0.9998,0.0002],[0.,1.]])
A_cool=np.array([[0.9998,0.0002],[0.,1.]])
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
# cont_props['eqpnt_off'] = box2poly([[17.,25.],[21.,25.]])
# cont_props['outside'] = box2poly([[24.,25.],[24.,25.]])
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


sys_dyn={}
sys_dyn['regular','off']=pwa_off
sys_dyn['regular','heat']=pwa_heat
sys_dyn['regular','cool']=pwa_cool
sys_dyn['regular','on']=pwa_on
ssd=hybrid.SwitchedSysDyn(disc_domain_size=(1,4),
                 dynamics=sys_dyn, cts_ss=cont_state_space,
                 env_labels=['regular'], disc_sys_labels=['off', 'heat','cool','on'], time_semantics='sampled',
                 timestep=0.1, overwrite_time=True)
cont_dyn={}
outside_props=set()
pm_props=set()
rel_soln={}
for mode in ssd.modes:
    cont_dyn[mode] = ssd.dynamics[mode].list_subsys[0]
    check_self_trans(cont_dyn,mode,cont_state_space,cont_props,outside_props,pm_props)


visualize = True
p2p=time.time()
cont_part = prop2part(cont_state_space, cont_props)
print "Prop2Part Time: "+str(time.time()-p2p)
print "Prop2Part DONE!!"
plot_partition(cont_part, show=visualize)
conv=time.time()
cont_part, new2old = part2convex(cont_part)
print "Convexify Time: "+str(time.time()-conv)
print "Convexify DONE!!"
plot_partition(cont_part, show=visualize)

ref_grid=cont_part 
#ref_grid=add_grid(ppp=cont_part, grid_size=4.)
#plot_partition(ref_grid, show=visualize)

afts = trs.OpenFTS()
test_afts = trs.AOFTS()
cont_dyn={}
postar={}
trans={}
test_trans={}
xx={}
cnt=0
actions_per_mode= {
                (e,s):{'env_actions':str(e), 'sys_actions':str(s)}
                for e,s in ssd.modes
                }


xz={}
# mptrans={}
# mpstart=time.time()
# for mode in ssd.modes:
#     cont_dyn[mode]=ssd.dynamics[mode].list_subsys[0]
#     mptrans=multiproc_ptloop(ref_grid,cont_dyn,1)
# mpdur=time.time()-mpstart
# print "MP Discretization Time: %.6f"%mpdur

# print "No. of CPUs: %.1f"%mp.cpu_count()

# emptrans={}
# empstart=time.time()
# for mode in ssd.modes:
#     cont_dyn[mode]=ssd.dynamics[mode].list_subsys[0]
#     emptrans[mode]=pti_loop(ref_grid,cont_dyn[mode],1)
# empdur=time.time()-empstart
# print "Extreme MP Discretization Time: %.6f"%empdur


ddst=time.time()

for mode in ssd.modes:
    cont_dyn[mode] = ssd.dynamics[mode].list_subsys[0]
    postar[mode] = post_area(ref_grid,cont_dyn[mode],1) 
    trans[mode] = get_transitions(ref_grid,postar[mode])
    test_trans[mode]=post_trans(ref_grid,cont_dyn[mode],1)
    xx[mode]=test_trans[mode]-trans[mode]
    xz[mode]=sp.lil_matrix(xx[mode])
dddur=time.time()-ddst
 

# for mode in ssd.modes:
#     # r,c=trans[mode].shape
#     # trans[mode]=np.vstack((trans[mode],np.zeros((1,c))))
#     # trans[mode][c-1][c-1]=1
#     # adj=sp.lil_matrix(trans[mode])
#     # if cnt==0:
#     #     afts_states = range(adj.shape[0]-1)
#     #     afts_states = trs.prepend_with(afts_states, 's')
#     #     afts.states.add_from(set(afts_states))
#     #     afts.states.add('sOut')
#     #     afts_states.append('sOut')
#     #     afts.atomic_propositions.add_from(set(ref_grid.prop_regions))
#     #     afts.states.initial.add('s0')
#     #     ppp2ts = afts_states ##Used for labelling the states when no. of states=no. of regions
#     #     for (i, state) in enumerate(ppp2ts):
#     #         if i==c-1:
#     #             props=set(['OUTSIDE'])
#     #             #props=set()
#     #         else:
#     #             props =  ref_grid[i].props
#     #         afts.states[state]['ap'] = props
#     # afts.env_actions.add_from([str(e) for e,s in ssd.modes])
#     # afts.sys_actions.add_from([str(s) for e,s in ssd.modes])
#     # afts.transitions.add_adj(adj=adj,adj2states=afts_states,**actions_per_mode[mode])

#     tr,tc=test_trans[mode].shape
#     test_trans[mode]=np.vstack((test_trans[mode],np.zeros((1,tc))))
#     test_trans[mode][tc-1][tc-1]=1 

#     test_adj=sp.lil_matrix(test_trans[mode])
#     if cnt==0:
#         test_afts_states = range(test_adj.shape[0]-1)
#         test_afts_states = trs.prepend_with(test_afts_states, 's')
#         test_afts.states.add_from(set(test_afts_states))
#         test_afts.states.add('sOut')
#         test_afts_states.append('sOut')
#         test_afts.atomic_propositions.add_from(set(orig_props))
#         test_afts.progress_map.add_from(pm_props)
#         test_afts.states.initial.add('s0')
#         test_ppp2ts = test_afts_states ##Used for labelling the states when no. of states=no. of regions
#         for (i, state) in enumerate(test_ppp2ts):
#             props=set()
#             pmp=set()
#             if i==tc-1:
#                 props=set(['OUTSIDE'])
#                 pmp|=outside_props
#                 #props=set()
#             else:
#                 for p in ref_grid[i].props:
#                     if p in orig_props:
#                         props|={p}
#                     else: 
#                         pmp|={p}
#             test_afts.states[state]['ap'] = props
#             test_afts.states[state]['pm'] = pmp
#         cnt=2

#     test_afts.env_actions.add_from([str(e) for e,s in ssd.modes])
#     test_afts.sys_actions.add_from([str(s) for e,s in ssd.modes])
#     test_afts.transitions.add_adj(adj=test_adj,adj2states=test_afts_states,**actions_per_mode[mode])



# print "POSTTRANS-Trans ------------------ "
# print xz
# # ref_grid.regions.append(pc.Region(out))
# # # abst=ds.AbstractModeSwitched(ppp=ref_grid, ts=afts,ppp2ts= afts_states,modes=ssd.modes)
# # test_abst=ds.AbstractModeSwitched(ppp=ref_grid,ts=test_afts,ppp2ts= test_afts_states,modes=ssd.modes)
# # ds.plot_mode_partitions(test_abst, show_ts=True, only_adjacent=False)

# #if True:
#     #ds.plot_mode_partitions(abst, show_ts=True, only_adjacent=False)
#     #ds.plot_mode_partitions(test_abst, show_ts=True, only_adjacent=False)


# #print(ref_grid)
# #print(afts)


# env_vars = set()  
# env_init = set()                # empty set
# env_prog = set()
# #env_prog = {'eqpnt_off || !(sys_actions = off)','eqpnt_on || !(sys_actions = on)','eqpnt_heat || !(sys_actions = heat)','eqpnt_cool || !(sys_actions = cool)'}  
# #env_prog = {'eqpnt_off || !off','eqpnt_on || !on','eqpnt_heat || !heat','eqpnt_cool || !cool'}  
# env_safe = set()                # empty set
# ''' Assumption/prog - !<>[](~eq_pnti & modei) == []<>(eq_pnti or !modei or out)
# # stable & unstable eq pnts. 
# # Define A(O)FTS, and as a starting point, include labels to show environment progress. 
# # Synth func appends the above assumption automatically on detecting AOFTS. env_to_spec
# # new attribute to define eq_pnti and modei. Include a new dict for 'eq' similarly to 'ap' for states. 
# # 'sys_actions' in ctrl.outputs, if True ->gr1c else jtlv

# '''
# # System variables and requirements
# sys_vars = set() 
# sys_init = set()            
# sys_prog = {'LOW','HIGH'}               # []<>home
# sys_safe = {'!OUTSIDE'}

# specs = spec.GRSpec(env_vars, sys_vars, env_init, sys_init,
#                     env_safe, sys_safe, env_prog, sys_prog)

# jt_ctrl_untrim = synth.synthesize('jtlv', specs, env=test_afts, trim_aut=False)
# jt_ctrl_trim = synth.synthesize('jtlv', specs, env=test_afts, trim_aut=True)
# # ctrl_start=time.time()
# gr_ctrl_untrim = synth.synthesize('gr1c', specs, env=test_afts, trim_aut=False)
# gr_ctrl_trim = synth.synthesize('gr1c', specs, env=test_afts, trim_aut=True)
# # ctrldur=time.time()-ctrl_start
# print "Discretization of Discretization Time: %.6f"%dddur
# #print "Synthesist of Controller Time: %.6f"%ctrldur

# import pickle as pl
# # pl.dump(specs,open("stripedv1_specs.p","wb"))
# # pl.dump(test_afts,open("stripedv2_afts.p","wb"))
# # # yy={}
# # # yz={}
# # # self_trans=pl.load(open("trans_self.p","rb"))
# # # for mode in ssd.modes:
# # #     yy[mode]=trans[mode]-self_trans[mode]
# # #     yz[mode]=sp.lil_matrix(yy[mode])

# # # print "CURRENT-PREV  -------------------------"
# # # print yz
# # pl.dump(trans,open("trans_self.p","wb"))

# # if not jt_ctrl_untrim.save('eqpnt_3_jtlv_untrimmed.png'):
# #     print(ctrl)
# # if not jt_ctrl_trim.save('eqpnt_3_jtlv_trimmed.png'):
# #     print(ctrl)
# # if not gr_ctrl_untrim.save('eqpnt_4_gr1c_untrimmed.png'):
# #     print(ctrl)
# # if not gr_ctrl_trim.save('eqpnt_4_gr1c_trimmed.png'):
# #     print(gr_ctrl_trim)
# #eqpnt_1 indicates first iteration of eqpnt with K+2eps and K-eps and with K+eps and K-eps, with eps=min(K,A)
# #when eps=min(K,A)/10, controller is unable to be synthesized. 
# # eqpnt_3 is with the env assumptions. For gr1c to work, it must be !sys_actions=off, while for jtlv it must be !off