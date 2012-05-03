#!/usr/bin/env python
"""
The example is an extension of robot_discrete_simple.py by including continuous dynamics. 

Nok Wongpiromsarn (nok@cds.caltech.edu)
September 2, 2010

minor refactoring by SCL <slivingston@caltech.edu>
1 May 2011.

Small modifications by Yuchen Lin.
12 Aug 2011
"""

#@importvar@
import sys, os
from numpy import array

from tulip import *
import tulip.polytope as pc


# Specify where the smv file, spc file and aut file will go
testfile = 'robot_simple'
path = os.path.abspath(os.path.dirname(sys.argv[0]))
smvfile = os.path.join(path, 'specs', testfile+'.smv')
spcfile = os.path.join(path, 'specs', testfile+'.spc')
autfile = os.path.join(path, 'specs', testfile+'.aut')

# Environment variables
env_vars = {'park' : 'boolean'}

# Discrete system variable
# Introduce a boolean variable X0reach to handle the spec [](park -> <>X0)
# X0reach starts with TRUE. 
# [](next(X0reach) = X0 | (X0reach & !park))
sys_disc_vars = {'X0reach' : 'boolean'}
#@importvar_end@

# Continuous state space
#@contdyn@
cont_state_space = pc.Polytope(array([[1., 0.],[-1., 0.], [0., 1.], [0., -1.]]),
                               array([[3.],[0.],[2.],[0.]]))

# Continuous proposition
cont_props = {}
for i in xrange(0, 3):
    for j in xrange(0, 2):
        prop_sym = 'X' + str(3*j + i)
        cont_props[prop_sym] = pc.Polytope(array([[1., 0.],[-1., 0.], [0., 1.], [0., -1.]]),
                                           array([[float(i+1)],[float(-i)],[float(j+1)],[float(-j)]]))

# Continuous dynamics: \dot{x} = u_x, \dot{y} = u_y
A = array([[1.1052, 0.],[ 0., 1.1052]])
B = array([[1.1052, 0.],[ 0., 1.1052]])
U = pc.Polytope(array([[1., 0.],[-1., 0.], [0., 1.], [0., -1.]]), array([[1.],[1.],[1.],[1.]]))
sys_dyn = discretize.CtsSysDyn(A,B,[],[],U,[])
#@contdyn_end@

#@discretize@
# Compute the proposition preserving partition of the continuous state space
cont_partition = prop2part.prop2part2(cont_state_space, cont_props)


# Discretize the continuous state space
disc_dynamics = discretize.discretize(cont_partition, sys_dyn, verbose=2)
#@discretize_end@

# Optional: plot the discretized state space
# from tulip.polytope.plot import plot_partition
# plot_partition(disc_dynamics, plot_transitions=True)

# Spec
assumption = 'X0reach & []<>(!park)'
guarantee = '[]<>X5 & []<>(X0reach)'
guarantee += ' & [](next(X0reach) = (X0 | (X0reach & !park)))'

#@gencheckcomp@
# Generate input to JTLV
prob = jtlvint.generateJTLVInput(env_vars, sys_disc_vars, [assumption, guarantee],
                                 {}, disc_dynamics, smvfile, spcfile, verbose=2)

# Check realizability
realizability = jtlvint.checkRealizability(smv_file=smvfile, spc_file=spcfile,
                                           aut_file=autfile, verbose=3)

# Compute an automaton
jtlvint.computeStrategy(smv_file=smvfile, spc_file=spcfile, aut_file=autfile,
                        priority_kind=3, verbose=3)
aut = automaton.Automaton(autfile, [], 3)

# Remove dead-end states from automaton.
aut.trimDeadStates()
#@gencheckcomp_end@


if not aut.writeDotFile("rdsimple_example.dot"):
    print "Error occurred while generating DOT file."
else:
    try:
        call("dot rdsimple_example.dot -Tpng -o rdsimple_example.png".split())
    except:
        print "Failed to create image from DOT file. To do so, try\n\ndot rdsimple_example.dot -Tpng -o rdsimple_example.png\n"





# Simulate.
#@sim@
num_it = 30
env_states = [{'X0reach': True}]
for i in range(1, num_it):
    if (i%3 == 0):
        env_states.append({'park':True})
    else:
        env_states.append({'park':False})

graph_vis = raw_input("Do you want to open in Gephi? (y/n)") == 'y'
destfile = 'rsimple_example.gexf'
label_vars = ['park', 'cellID', 'X0reach']
delay = 2
vis_depth = 3
aut_states = grsim.grsim([aut], aut_trans_dict={}, env_states=env_states,
                         num_it=num_it, deterministic_env=False,
                         graph_vis=graph_vis, destfile=destfile,
                         label_vars=label_vars, delay=delay,
                         vis_depth=vis_depth)

# Save discrete dynamics.
f = open('rsimple_example_disc_dynamics.txt', 'w')
f.write(str(disc_dynamics.list_prop_symbol) + '\n')
for i in xrange(0, len(disc_dynamics.list_region)):
    f.write(str(disc_dynamics.list_region[i].list_prop))
    f.write('\n')
f.close()
#@sim_end@
