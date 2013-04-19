#!/usr/bin/env python
""" A script to use TuLiP from commandline

Input: .yaml file that containts a problem description
Output: .xml file that contains the problem description and synthesized automatton

Command line usage: python tulip_synth.py your_yaml_file.yaml

Necmiye Ozay (necmiye@cds.caltech.edu)
April 19, 2013
"""

#@importvardyn@
import sys, os
from numpy import array

from tulip import *
from tulip import parsespec
from tulip import conxml

testfile = 'temp'
path = os.path.abspath(os.path.dirname(sys.argv[0]))
smvfile = os.path.join(testfile+'.smv')
spcfile = os.path.join(testfile+'.spc')
autfile = os.path.join(testfile+'.aut')

inputfile = sys.argv[1]

(sys_dyn, initial_partition, N, assumption, guarantee, env_vars, sys_disc_vars) = conxml.readYAMLfile(inputfile)

disc_dynamics = discretize.discretize(initial_partition, sys_dyn, closed_loop=True, \
                    N=N, min_cell_volume=0.1, verbose=0)
                    
prob = jtlvint.generateJTLVInput(env_vars, sys_disc_vars, [assumption, guarantee],
                    {}, disc_dynamics, smvfile, spcfile, verbose=0)
                    
jtlvint.computeStrategy(smv_file=smvfile, spc_file=spcfile, aut_file=autfile,
                    priority_kind=3, verbose=0)
                    
aut = automaton.Automaton(autfile, [], 0)

# Remove dead-end states from automaton
aut.trimDeadStates()

conxml.writeXMLfile("rsimple_example.xml", prob, [assumption, guarantee], sys_dyn, aut, pretty=True)

os.remove(smvfile)
os.remove(spcfile)
os.remove(autfile)



