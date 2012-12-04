#!/usr/bin/env python
"""
Example of parallel simulation: Executing a communication protocol to transmit integer data

Clemens Wiltsche (wclemens@cds.caltech.edu)
18 Jul 2012
"""

#@import_section@
import sys, os

from tulip import jtlvint, automaton
#@import_section_end@


########
# Sender
########

# Specify where the smv file, spc file and aut file will go
#@filename_section@
sendfile = 'send'
path = os.path.abspath(os.path.dirname(sys.argv[0]))
smvfile = os.path.join(path, 'specs', sendfile+'.smv')
spcfile = os.path.join(path, 'specs', sendfile+'.spc')
autfile = os.path.join(path, 'specs', sendfile+'.aut')
#@filename_section_end@

#@specification_section@
data = 3 # the maximum value that could be transmitted

env_vars = {'ack' : 'boolean'}
sys_disc_vars = {'req' : range(0,data)}
# empty dynamics - not needed for this example
disc_dynamics = None

#GR1 specification
assumption = '[](req=0 -> <>(!ack))\n'
assumption += ' & [](ack & next(!ack) -> req=0)\n'
assumption += ' & [](!ack & next(ack) -> req!=0)\n'

guarantee = '[](ack -> next(req=0))\n' # VI
guarantee += ' & [](!ack -> next(req=1 | req=2))\n'
guarantee += ' & [](req!=-1 & next(req=0) -> ack)\n'
guarantee += ' & [](req=-1 & next(req!=0) -> !ack)\n'
for d in xrange(1,data):
    guarantee += ' & [](req=%i & next(req!=%i) -> next(req=0))\n' % (d, d) #V
for d in xrange(1,data):
    guarantee += ' & [](req=%i & !ack -> next(req=%i))\n' % (d, d) # IV
#@specification_section_end@

# Generate JTLV input
#@geninput@
prob = jtlvint.generateJTLVInput(env_vars, sys_disc_vars, [assumption, guarantee], \
                                   {}, disc_dynamics, smvfile, spcfile, file_exist_option='r', verbose=2)
#@eninout_end@

# Construct automaton
#@compaut@
jtlvint.computeStrategy(smv_file=smvfile, spc_file=spcfile, aut_file=autfile, \
                                    priority_kind=3, init_option=1, heap_size='-Xmx2g', file_exist_option='r', verbose=3)
sender_aut = automaton.Automaton(autfile, [], 3)
#@compaut_end@


########
# Receiver
########

#@filename_section@
recvfile = 'recv'
path = os.path.abspath(os.path.dirname(sys.argv[0]))
smvfile = os.path.join(path, 'specs', recvfile+'.smv')
spcfile = os.path.join(path, 'specs', recvfile+'.spc')
autfile = os.path.join(path, 'specs', recvfile+'.aut')
#@filename_section_end@

#@specification_section@
env_vars = {'req' : range(0,data)}
sys_disc_vars = {'ack' : 'boolean', 'sink' : range(0,data)}
# Empty dynamics
disc_dynamics = None

# GR1 specification
assumption = '[](req!=0 & next(req=0) -> ack)\n' #X
assumption += ' & [](req=0 & next(req!=0) -> !ack)\n' #IX

guarantee = '[](req=0 -> next(!ack))\n' # XII
guarantee += ' & [](req!=0 -> next(ack))\n' # XI
guarantee += ' & [](!ack & next(ack) -> req!=0)\n' # XIV
guarantee += ' & [](ack & next(!ack) -> req=0)\n' # XIII
guarantee += ' & [](req!=0 -> <>('
for d in xrange(1,data):
    if d != 1:
        guarantee += ' | '
    guarantee += '(sink=%i & req=%i)' % (d, d)
guarantee += '))\n' #XV
#@specification_section_end@

# Generate JTLV input
#@geninput@
prob = jtlvint.generateJTLVInput(env_vars, sys_disc_vars, [assumption, guarantee], \
                                   {}, disc_dynamics, smvfile, spcfile, file_exist_option='r', verbose=2)
#@geninput_end@

# Construct automaton
#@compaut@
jtlvint.computeStrategy(smv_file=smvfile, spc_file=spcfile, aut_file=autfile, \
                                    priority_kind=3, init_option=1, heap_size='-Xmx2g', file_exist_option='r', verbose=3)

receiver_aut = automaton.Automaton(autfile, [], 3)
#@compaut_end@


#####################
# Parallel simulation
#####################

#@sim@
from tulip.parallelsim import Strategy

# Initial values of variables
V = {'ack': 0, 'req': 0, 'sink': 0}

# Strategies
sender_strat = Strategy(sender_aut, V, ['ack'], ['req'], 'sender')#, runtime=10)
receiver_strat = Strategy(receiver_aut, V, ['req'], ['ack', 'sink'], 'receiver')#, runtime=10)

# Start threads

print "\n#######################"
print "# Starting Simulation #"
print "#######################\n"

sender_strat.start()
receiver_strat.start()
try:
    while True:
        pass
except KeyboardInterrupt:
    sender_strat.runtime = 1
    receiver_strat.runtime = 1
#@sim_end@
