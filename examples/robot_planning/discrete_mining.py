'''
Testing mining assumption algorithm on the discrete.py example.
System transition described in

'''

from tulip import transys, spec, synth
from tulip.interfaces import slugs, mining
import os
start_time = os.times()[0]

def fixed_mutex(varnames):
    """Create mutual exclusion formulae from iterable of variables.

    E.g., given a set of variable names {"a", "b", "c"}, return a set
    of formulae {"a -> ! (c || b)", "c -> ! (b)"}.
    """
    mutex = set()
    numVars = len(varnames)
    varnames = list(varnames)
    for i in range(numVars):
        count = 0
        mut_str = varnames[i] + ' -> !('
        for j in range(numVars):
            if (varnames[j] != varnames[i]):
                target = varnames[j]
                mut_str += target
                count += 1
                if (count < numVars-1):
                    mut_str += ' || '
        mut_str += ')'
        mutex |= {mut_str}
    return mutex

#
# Environment variables and specification
#
# The environment can issue a park signal that the robot just respond
# to by moving to the lower left corner of the grid.  We assume that
# the park signal is turned off infinitely often.
#
# @environ_section@
env_vars = {'park'}
env_init = set()                # empty set
env_prog = {'!park'}
env_safe = set()                # empty set
# @environ_section_end@

#
# System specification
#
# The system specification is that the robot should repeatedly revisit
# the upper right corner of the grid while at the same time responding
# to the park signal by visiting the lower left corner.  The LTL
# specification is given by
#
#     []<> home && [](park -> <>lot)
#
# Since this specification is not in GR(1) form, we introduce the
# variable X0reach that is initialized to True and the specification
# [](park -> <>lot) becomes
#
#     [](X (X0reach) <-> lot || (X0reach && !park))
#

# @specs_setup_section@
# Augment the system description to make it GR(1)
sys_vars = {'X0reach', 'home', 'loc1', 'loc2', 'loc5', 'loc3', 'loc4', 'lot'}          # infer the rest from TS
sys_init = {'X0reach', 'home'}
sys_prog = {'home'}             # []<>home
sys_safe = {'(X (X0reach) <-> lot) || (X0reach && !park)'}
sys_safe |= {'(X (park) <-> ! lot)'}
# transition system TODO: automcatically generate this perhaps from FTS object
sys_safe |= {
    'home -> X (loc1 || loc3)',
    'loc1 -> X (home || loc2 || loc4)',
    'loc2 -> X (loc1 || lot || loc5)',
    'loc3 -> X (home || loc4)',
    'loc4 -> X (loc3 || loc1 || loc5 || lot)',
    'loc5 -> X (loc2 || loc4)',
    'lot -> X (loc2 || loc4)'
}
sys_safe |= fixed_mutex({'home', 'loc1', 'loc2', 'loc3', 'loc4', 'loc5'})
sys_safe |= {'lot -> !(home || loc1 || loc2 || loc3 || loc4)', 'loc5 <-> lot'}

sys_prog |= {'X0reach'}
# @specs_setup_section_end@

# @specs_create_section@
# Create the specification
specs = spec.GRSpec(env_vars, sys_vars, env_init, sys_init,
                    env_safe, sys_safe, env_prog, sys_prog)
# @specs_create_section_end@

# Generate input and output signals using the var_list from counter-strategy machine
input_signal = list(env_vars)
output_signal = list(sys_vars)
print input_signal
print output_signal

# Using input and output signals, generate candidates in three categories
candidates = mining._generate_candidate_list(input_signal, output_signal)
print candidates

# Define the trace
trace = ''

# Mine assumptions required by this
mining.mine_assumption(specs, candidates, trace)

end_time = os.times()[0]
msg = 'Total time: ' +\
      str(end_time - start_time) + '[sec]'
print(msg)
