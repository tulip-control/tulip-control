import tulip
from tulip import transys, spec, synth
from tulip.abstract.horizon import RHTLPProb
import logging
logging.basicConfig(filename='example.log',level=logging.DEBUG)

sys = transys.OpenFTS()

################
#Trans-Grid-gen#
################

def box_tans(width, height, var):
    transitions = {}
    for x in range(width):
        for y in range(height):
            transitions[var+str(y*width+x)] = set([var+str(y*width+z) for z in
                                               range(max(0, x-1),
                                                     min(width, x+2))])
            transitions[var+str(y*width+x)].update([var+str(z*width+x)for z in
                                                    range(max(0, y-1),
                                                          min(height, y+2))])
    return transitions

def road(length, var):
    trans = box_tans(2, length, var)

    env_vars = set(['O'+var+str(x) for x in xrange(0,length*2)])

    #Not adjacent obstacles
    env_safe = ['&&'.join(['!(O'+var+str(x)+'&&O'+var+str(x+1)+')'
                           for x in range(0,length*2,2)])]
    #Not diagonal obstacles
    env_safe.append('&&'.join(['!(O'+var+str(x)+'&&O'+var+str(x+3)+')'
                               for x in range(0,(length-1)*2,2)]))
    env_safe.append('&&'.join(['!(O'+var+str(x+1)+'&&O'+var+str(x+2)+')'
                               for x in range(0,(length-1)*2,2)]))
    #Obstacles does not move
    for y in xrange(0,length):
        env_safe.append('&&'.join(['(('+var+str(y*2)+'&&O'+var+str(x*2)+')->X(O'+var+str(x*2)+'))'
                                   for x in range(max(0,y-2),min(length,y+2))]))
        env_safe.append('&&'.join(['(('+var+str(y*2)+'&&O'+var+str(x*2+1)+')->X(O'+var+str(x*2+1)+'))'
                                   for x in range(max(0,y-2),min(length,y+2))]))
        env_safe.append('&&'.join(['(('+var+str(y*2+1)+'&&O'+var+str(x*2)+')->X(O'+var+str(x*2)+'))'
                                   for x in range(max(0,y-2),min(length,y+2))]))
        env_safe.append('&&'.join(['(('+var+str(y*2+1)+'&&O'+var+str(x*2+1)+')->X(O'+var+str(x*2+1)+'))'
                                   for x in range(max(0,y-2),min(length,y+2))]))

    sys_safe = '&&'.join(['!('+var+str(x)+'&&O'+var+str(x)+')' for x in xrange(0,length*2)])

    return (trans, env_vars, env_safe, sys_safe)


########
#ROAD 1#
########
r1_len = 3
trans1, e_vars1, e_safe1, s_safe1 = road(r1_len, 'R')
sys.states.add_from(trans1.keys())
for key, value in trans1.items():
    sys.transitions.add_comb({key}, value)

########
#ROAD 2#
########
r2_len = 2
trans2, e_vars2, e_safe2, s_safe2  = road(r2_len, 'Q')
sys.states.add_from(trans2.keys())
for key, value in trans2.items():
    sys.transitions.add_comb({key}, value)

###############
#Connect Roads#
###############
sys.transitions.add_comb({"R"+str(r1_len*2-1)}, {"Q0"})
sys.transitions.add_comb({"R"+str(r1_len*2-3)}, {"Q1"})
sys.transitions.add_comb({"Q0"}, {"R"+str(r1_len*2-1)})
sys.transitions.add_comb({"Q1"}, {"R"+str(r1_len*2-3)})

##########
#Env spec#
##########
#env_vars = e_vars2
env_vars = e_vars1.union(e_vars2)
env_init = ''
env_prog = ''

##Restrictions on environment positions in connection
#env_safe = '&&'.join(e_safe2)
env_safe = "("+'&&'.join(e_safe1)+")&&("+'&&'.join(e_safe2)+")"
env_safe = env_safe+"&&"+"!(OR"+str(r1_len*2-1)+"&&"+"OR"+str(r1_len*2-3)+")"
env_safe = env_safe+"&&"+"!(OR"+str(r1_len*2-1)+"&&"+"OQ1)"
env_safe = env_safe+"&&"+"!(OR"+str(r1_len*2-3)+"&&"+"OQ0)"

#Restrictions on environment changes in connection
env_safe = env_safe + '&&((R'+str(r1_len*2-1)+'&&OQ0)->X(OQ0))'
env_safe = env_safe + '&&((R'+str(r1_len*2-1)+'&&OQ1)->X(OQ1))'
env_safe = env_safe + '&&((R'+str(r1_len*2-2)+'&&OQ0)->X(OQ0))'
env_safe = env_safe + '&&((R'+str(r1_len*2-2)+'&&OQ1)->X(OQ1))'
env_safe = env_safe + '&&((R'+str(r1_len*2-3)+'&&OQ0)->X(OQ0))'
env_safe = env_safe + '&&((R'+str(r1_len*2-3)+'&&OQ1)->X(OQ1))'
env_safe = env_safe + '&&((R'+str(r1_len*2-4)+'&&OQ0)->X(OQ0))'
env_safe = env_safe + '&&((R'+str(r1_len*2-4)+'&&OQ1)->X(OQ1))'

##########
#Sys Spec#
##########

sys_vars = {}
#sys_init = 'Q0||Q1'
sys_init = 'R0||R1'
goal_ap = 'WQ'+str(r2_len-1)
sys_prog = goal_ap

#Add goal AP
sys.atomic_propositions.add_from({goal_ap})
sys.states.add('Q'+str(r2_len*2-1), ap={goal_ap})
sys.states.add('Q'+str(r2_len*2-2), ap={goal_ap})

#sys_safe = s_safe2
sys_safe = "("+s_safe1+")&&("+s_safe2+")"

specs = spec.GRSpec(env_vars, sys_vars, env_init, sys_init,
                    env_safe, sys_safe, env_prog, sys_prog)

#############
#RHTLP Parts#
#############

#w_part = ['W0','W1','W2','W3','W4']
#w_part = ['WQ'+str(x) for x in xrange(0,r2_len)]
w_part = ['WR'+str(x) for x in xrange(0,r1_len)]
w_part.extend(['WQ'+str(x) for x in xrange(0,r2_len)])

#w_mapping = {'W0':'W1','W1':'W2','W2':'W3',...,'WN':'WN'}
#w_mapping = {}
w_mapping = {'WR'+str(x):
                 'WR'+str(min(r1_len-1, x+1)) for x in xrange(0,r1_len)}
w_mapping.update({'WQ'+str(x):
                      'WQ'+str(min(r2_len-1, x+1)) for x in xrange(0,r2_len)})
w_mapping.update({"WR"+str(r1_len-1):"WQ0"})

#w_plan_set = {'W4':set(['W4']),'W3':set(['W3','W4']),...,'W1':set(['W1','W2'])}
#w_plan_set = {}
w_plan_set = {'WR'+str(x1):
                  set(['WR'+str(x2) for x2 in range(x1, min(r1_len, x1+2))])
              for x1 in xrange(0, r1_len)}
w_plan_set.update({'WQ'+str(x1):
                  set(['WQ'+str(x2) for x2 in range(x1, min(r2_len, x1+2))])
              for x1 in xrange(0, r2_len)})
w_plan_set.update({"WR"+str(r1_len-1):{"WR"+str(r1_len-1),"WQ0"}})

w_plan_set.update({"WR"+str(r1_len-1):
                       {"WR"+str(r1_len-1), "WR"+str(r1_len-2), "WQ0"}})

#partition_mapping = {'W0': ['R0','R1'],, 'W1': ['R2','R3'],...'W4': ['R8','R9']}
#partition_mapping = {}
partition_mapping = {'WR'+str(x1):
                         ['R'+str(x2) for x2 in range(x1*2, x1*2+2)]
                     for x1 in xrange(0,r1_len)}
partition_mapping.update({'WQ'+str(x1):
                              ['Q'+str(x2) for x2 in range(x1*2, x1*2+2)]
                          for x1 in xrange(0,r2_len)})

phi1 = '&&'.join(['(!(R'+str(x)+'&&OR'+str(x)+'))'
                  for x in xrange(0,r1_len*2)])
phi2 = '&&'.join(['(!(Q'+str(x)+'&&OQ'+str(x)+'))'
                  for x in xrange(0,r2_len*2)])

#phi = phi2
phi = phi1+"&&"+phi2

#############
#Create Prob#
#############
#end_state=False and repl_prog=False causes next FALSE, which crashes jtlv
rhprob = RHTLPProb(disc_dyn=sys, specs=specs,\
                   parts=w_part, mappings=w_mapping, plan_sets=w_plan_set,\
                   partition_mapping=partition_mapping, phi=phi,
                   repl_prog=False, add_end_state=False)

#phi = rhprob.generate_phi()

specs = rhprob.reduce_spec()

aut = rhprob.synthesize()#extra_phi=phi)