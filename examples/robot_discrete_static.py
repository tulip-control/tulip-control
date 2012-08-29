#!/usr/bin/env python
"""
Non-adversarial version of robot_discrete_simple

Nok Wongpiromsarn (nok@cds.caltech.edu)
August 3, 2010

minor refactoring by SCL <slivingston@caltech.edu>
1 May 2011.

Small modifications by Yuchen Lin.
12 Aug 2011

Static by Nick Spooner <ns507@cam.ac.uk>
27 June 2012

Updated to new solver interface by Nick Spooner <ns507@cam.ac.uk>
29 Aug 2012
"""

#@import_section@
import sys, os
from subprocess import call

from tulip import *
from tulip import solver
import tulip.polytope as pc
#@import_section_end@


# Specify where the smv file, spc file and aut file will go
#@filename_section@
testfile = 'robot_discrete_static'
path = os.path.abspath(os.path.dirname(sys.argv[0]))
smvfile = os.path.join(path, 'specs', testfile+'.smv')
pmlfile = os.path.join(path, 'specs', testfile+'.pml')
spcfile = os.path.join(path, 'specs', testfile+'.spc')
autfile = os.path.join(path, 'specs', testfile+'.aut')
#@filename_section_end@

# Specify the environment variables
#@envvar_section@
# No environment
env_vars = {}
#@envvar_section_end@

# Specify the discrete system variable
# Introduce a boolean variable X0reach to handle the spec [](park -> <>X0)
# X0reach starts with TRUE. 
# [](next(X0reach) = X0 | (X0reach & !park))
#@sysdiscvar_section@
# This is now unnecessary?
#sys_disc_vars = {'X0reach' : 'boolean'}
sys_disc_vars = {}
#@sysdiscvar_section_end@

# Specify the transition system representing the continuous dynamics
#@ts_section@
disc_dynamics = prop2part.PropPreservingPartition(list_region=[], list_prop_symbol=[])

# These following propositions specify in which cell the robot is,
# i.e., Xi means that the robot is in cell Ci
disc_dynamics.list_prop_symbol = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5'] 
disc_dynamics.num_prop = len(disc_dynamics.list_prop_symbol)

# Regions. Note that the first argument of Region(poly, prop) should
# be a list of polytopes. But since we are not dealing with the actual
# controller, we will just fill it with a string (think of it as a
# name of the region).  The second argument of Region(poly, prop) is a
# list that specifies which propositions in cont_props above is
# satisfied. As specified below, regioni satisfies proposition Xi.
region0 = pc.Region('C0', [1, 0, 0, 0, 0, 0])
region1 = pc.Region('C1', [0, 1, 0, 0, 0, 0])
region2 = pc.Region('C2', [0, 0, 1, 0, 0, 0])
region3 = pc.Region('C3', [0, 0, 0, 1, 0, 0])
region4 = pc.Region('C4', [0, 0, 0, 0, 1, 0])
region5 = pc.Region('C5', [0, 0, 0, 0, 0, 1])
disc_dynamics.list_region = [region0, region1, region2, region3, region4, region5]
disc_dynamics.num_regions = len(disc_dynamics.list_region)

# The transition relation between regions. disc_dynamics.trans[i][j] =
# 1 if starting from region j, the robot can move to region i while
# only staying in the union of region i and region j.
disc_dynamics.trans =   [[1, 1, 0, 1, 0, 0], \
                         [1, 1, 1, 0, 1, 0], \
                         [0, 1, 1, 0, 0, 1], \
                         [1, 0, 0, 1, 1, 0], \
                         [0, 1, 0, 1, 1, 1], \
                         [0, 0, 1, 0, 1, 1]]
#@ts_section_end@

#@specification@
assumption = ''
# reach the goal infinitely often
guarantee = '[]<>(X5)'
#@specification_end@

solverin = solver.generateSolverInput(sys_disc_vars, [assumption, guarantee],
                                 {}, disc_dynamics, "rdstatic_example.mdl",
                                 {"X0" : True}, "SPIN")
# Solve and save result to AUT file 
solverin.solve("rdstatic_example.aut")
# Load automaton from file
aut = solverin.automaton()
# Convert propositions back to original form (i.e. X0, X1, ...)
solver.restore_propositions(aut, disc_dynamics)
# Remove the module name from each automaton state
aut.stripNames()

# Remove dead-end states from automaton.
aut.trimDeadStates()
#@compaut_end@


# Visualize automaton with DOT file

# This example uses environment vs. system turn distinction.  To
# disable it, just use (the default),
if not aut.writeDotFile(fname="rdstatic_example.dot", hideZeros=False):
# if not aut.writeDotFile("rdstatic_example.dot",
#                         distinguishTurns={"env": prob.getEnvVars().keys(),
#                                           "sys": prob.getSysVars().keys()},
#                         turnOrder=("env", "sys")):
# if not aut.writeDotFileEdged(fname="rdstatic_example.dot",
#                              env_vars = prob.getEnvVars().keys(),
#                              sys_vars = prob.getSysVars().keys(),
#                              hideZeros=True):
    print "Error occurred while generating DOT file."
else:
    try:
        call("dot rdstatic_example.dot -Tpng -o rdstatic_example.png".split())
    except:
        print "Failed to create image from DOT file. To do so, try\n\ndot rdstatic_example.dot -Tpng -o rdstatic_example.png\n"


# Simulate.
#@sim@
num_it = 30
graph_vis = raw_input("Do you want to open in Gephi? (y/n)") == 'y'
destfile = 'rdstatic_example.gexf'
label_vars = ['cellID']
env_states = []
delay = 2
vis_depth = 3
aut_states = grsim.grsim([aut], aut_trans_dict={}, env_states=env_states,
                         num_it=num_it, deterministic_env=False,
                         graph_vis=graph_vis, destfile=destfile,
                         label_vars=label_vars, delay=delay,
                         vis_depth=vis_depth)
#@sim_end@
