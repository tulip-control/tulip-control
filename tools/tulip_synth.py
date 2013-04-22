#!/usr/bin/env python
""" A script to use TuLiP (with jtlv) from command line
(other solvers not supported yet)

Input: .yaml file that contains a problem description
Output: .xml file that contains the problem description and synthesized automaton

Command line usage: python tulip_synth.py your_yaml_file.yaml [output_file.xml]

Necmiye Ozay (necmiye@cds.caltech.edu)
April 19, 2013. 
Minor edits Apr 21, 2013.
"""

import sys, os
from numpy import array

from tulip import *
from tulip import conxml

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print "Usage: %s your_yaml_file.yaml [output_file.xml]" % sys.argv[0]
        exit(1)
        
    inputfile = sys.argv[1]

    (sys_dyn, initial_partition, N, assumption, guarantee, env_vars, sys_disc_vars) = \
                        conxml.readYAMLfile(inputfile)

    disc_dynamics = discretize.discretize(initial_partition, sys_dyn, closed_loop=True, \
                        N=N, min_cell_volume=0.1, verbose=0)
                        
    ### jtlv specific part (should be within an if-then statement when solver is optional) ###
    testfile = 'temp'
    path = os.path.abspath(os.path.dirname(sys.argv[0]))
    smvfile = os.path.join(testfile+'.smv')
    spcfile = os.path.join(testfile+'.spc')
    autfile = os.path.join(testfile+'.aut')
                    
    prob = jtlvint.generateJTLVInput(env_vars, sys_disc_vars, [assumption, guarantee],
                        {}, disc_dynamics, smvfile, spcfile, verbose=0)
                    
    jtlvint.computeStrategy(smv_file=smvfile, spc_file=spcfile, aut_file=autfile,
                        priority_kind=3, verbose=0)
    
                    
    aut = automaton.Automaton(autfile, [], 0)
    
    # Remove temporary files
    os.remove(smvfile)
    os.remove(spcfile)
    os.remove(autfile)
    
    ### end of jtlv specific part ###

    # Remove dead-end states from automaton
    aut.trimDeadStates()
    
    # If the output filename is not specified use the input filename for output
    if len(sys.argv) < 3:
        outputfile = inputfile[:-5]+'.xml'
    else:
        outputfile = sys.argv[2]

    conxml.writeXMLfile(outputfile, prob, [assumption, guarantee], sys_dyn, aut, pretty=True)



