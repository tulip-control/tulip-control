from tulip import transys, spec
from tulip.abstract.horizon import RHTLPProb
# We label the states using the following picture
#
#          +----+    +----+----+
#          | X1 |    | X4 | X6 |
#     +----+----+----+----+----+----+
#     | X0 |    | X3 |         | X8 |
#     +----+----+----+----+----+----+
#          | X2 |    | X5 | X7 |
#          +----+    +----+----+

sys = transys.OpenFTS()

# Define the states of the system
sys.states.add_from(['X0', 'X1', 'X2', 'X3', 'X4', 'X5','X6','X7','X8'])

# Define the allowable transitions
sys.transitions.add_comb({'X0'}, {'X1', 'X2'})
sys.transitions.add_comb({'X1','X2'}, {'X3'})
sys.transitions.add_comb({'X3'}, {'X4', 'X5'})
sys.transitions.add_comb({'X4'}, {'X6'})
sys.transitions.add_comb({'X5'}, {'X7'})
sys.transitions.add_comb({'X6','X7','X8'}, {'X8'})

#Create partition mapping instead of creating APs
partition_mapping = {}
partition_mapping['W0'] = ['X8']
partition_mapping['W1'] = ['X4','X5','X6','X7']
partition_mapping['W2'] = ['X3']
partition_mapping['W3'] = ['X1','X2']
partition_mapping['W4'] = ['X0']

#Xor obs1 and obs2. Not allowed to change obs1 or obs2 when observe is set
env_vars = {'obs'}
env_init = ''
env_prog = ''
env_safe = '((obs&&observe)->X(obs))&&(((!obs)&&observe)->X(!obs))'

sys_vars = {'observe'}
sys_init = 'W4'
sys_prog = 'W0'

#Create specification
# Not in X6 if obs1, not in X7 if obs2.
# Observe is true after X1
sys_safe = '!(X6&&obs)&&!(X7&&!obs)&&((X1||observe)<->X(observe))'

specs = spec.GRSpec(env_vars, sys_vars, env_init, sys_init,
                    env_safe, sys_safe, env_prog, sys_prog)
#
# Define the horizons
w_part = ['W0','W1','W2','W3','W4']
w_mapping = {'W0':'W0','W1':'W0','W2':'W0','W3':'W1','W4':'W2'}
w_plan_set = {'W4':set(['W4','W3','W2']),\
              'W3':set(['W3','W2','W1']),\
              'W2':set(['W2','W1','W0']),\
              'W1':set(['W1','W0']),\
              'W0':set(['W0'])}

#Invariant
#phi = '!X2&&((W2||W1)->observe)&&((X4||X6)->!obs1)&&((X5||X7)->!obs2)'
#since -> or <--> not implemented:
phi = '!X2&&(((W2||W1)&&observe)||!(W2||W1))&&(((X4||X6)&&!obs)||!(X4||X6))&&(((X5||X7)&&obs)||!(X5||X7))'

#Try un-commenting this line and generate phi
#phi = ''

rhprob = RHTLPProb(disc_dyn =sys, specs = specs,\
                   parts= w_part, mappings=w_mapping, plan_sets=w_plan_set,\
                   phi = phi, repl_prog = True, add_end_state = True,\
                   partition_mapping = partition_mapping)

#Generate invariant if needed
phi = rhprob.generate_phi()

#Synthesize controllers
aut = rhprob.synthesize(extra_phi=phi)