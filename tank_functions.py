import numpy as np
import re

from tulip import abstract
import tulip.polytope as pc

def get_transitions(part, ssys, N=10, closed_loop=True, trans_length=1, abs_tol=1e-7):
    
    # Initialize matrix for pairs to check
    IJ = part.adj.copy()
    if trans_length > 1:
        k = 1
        while k < trans_length:
            IJ = np.dot(IJ, part.adj)
            k += 1
        IJ = (IJ > 0).astype(int)
        
    # Initialize output
    transitions = np.zeros([part.num_regions,part.num_regions], dtype = int)

    # Do the abstraction
    while np.sum(IJ) > 0:
        ind = np.nonzero(IJ)
        i = ind[1][0]
        j = ind[0][0]
        IJ[j,i] = 0
        
        si = part.list_region[i]
        sj = part.list_region[j]
        
        # Use original cell as trans_set
        S0 = abstract.discretize.solveFeasable(si,sj,ssys,N, closed_loop=closed_loop,
                trans_set=part.orig_list_region[part.orig[i]])
        
        diff = pc.mldivide(si, S0)
        vol2 = pc.volume(diff)
                    
        if vol2 < abs_tol:
            transitions[j,i] = 1        
        else:
            transitions[j,i] = 0
            
    return transitions
    
def merge_partitions(part1, part2):

    if (part1.num_prop != part2.num_prop) or \
        (len(part1.orig_list_region) != len(part2.orig_list_region)):
        raise Exception("merge: partitions don't have the same origin")
    
    if (part1.domain != part2.domain) or (part1.list_prop_symbol != part2.list_prop_symbol):
        raise Exception("merge: partitions don't have the same origin")

    new_list = []
    orig = []
    parent_1 = []
    parent_2 = []
    for i in range(part1.num_regions):
        for j in range(part2.num_regions):
            isect = pc.intersect(part1.list_region[i],
                                 part2.list_region[j])
            rc, xc = pc.cheby_ball(isect)
            if rc > 1e-5:
                if len(isect) == 0:
                    isect = pc.Region([isect], [])
                isect.list_prop = part1.list_region[i].list_prop
                new_list.append(isect)
                parent_1.append(i)
                parent_2.append(j)
                if part1.orig[i] != part2.orig[j]:
                    raise Exception("not same orig, partitions don't have the \
                                      same origin.")
                orig.append(part1.orig[i])
    
    adj = np.zeros([len(new_list), len(new_list)], dtype=int)
    for i in range(len(new_list)):
        for j in range(i+1, len(new_list)):
            if (part1.adj[parent_1[i], parent_1[j]] == 1) or \
                    (part2.adj[parent_2[i], parent_2[j]] == 1) or \
                    (parent_1[i] == parent_1[j]) or \
                    (parent_2[i] == parent_2[j]):
                if pc.is_adjacent(new_list[i], new_list[j]):
                    adj[i,j] = 1
                    adj[j,i] = 1
        adj[i,i] = 1
            
    return abstract.PropPreservingPartition(domain=part1.domain,
                    num_prop=part1.num_prop, list_region=new_list, num_regions=len(new_list),
                    adj=adj, trans=None, list_prop_symbol=part1.list_prop_symbol, 
                    orig_list_region=part1.orig_list_region, orig=np.array(orig))

def create_files(ppp, trans1, trans2, control_var, val1, val2, env_vars,
                 sys_disc_vars, spec, smvfile, spcfile):
    
    prob = rhtlp.SynthesisProb()
    prob.createProbFromDiscDynamics(disc_dynamics=ppp, env_vars=env_vars,
                                    sys_disc_vars=sys_disc_vars, spec=spec)
                                
    env_vars = prob.getEnvVars()
    sys_vars = prob.getSysVars()
    
    # Write smv file
    f = open(smvfile, 'w')
    f.write('MODULE main \n')
    f.write('\tVAR\n')
    f.write('\t\te : env();\n')
    f.write('\t\ts : sys();\n\n')
    
    # Environment
    f.write('MODULE env \n')
    f.write('\tVAR\n')
    for var, val in env_vars.iteritems():
        f.write('\t\t' + var + ' : ' + val + ';\n')
    
    # System
    f.write('\nMODULE sys \n')
    f.write('\tVAR\n')
    for var, val in sys_vars.iteritems():
        f.write('\t\t' + var + ' : ' + val + ';\n')
    f.close()
    
    spec = prob.getSpec()
    disc_cont_var = prob.getDiscretizedContVar()
    
    assumption = spec[0]
    guarantee = spec[1]
    
    assumption = re.sub(r'\b'+'True'+r'\b', 'TRUE', assumption)
    guarantee = re.sub(r'\b'+'True'+r'\b', 'TRUE', guarantee)
    assumption = re.sub(r'\b'+'False'+r'\b', 'FALSE', assumption)
    guarantee = re.sub(r'\b'+'False'+r'\b', 'FALSE', guarantee)
    
    # Replace any environment variable var in spec with e.var and replace any 
    # system variable var with s.var
    for var in env_vars.keys():
        assumption = re.sub(r'\b'+var+r'\b', 'e.'+var, assumption)
        guarantee = re.sub(r'\b'+var+r'\b', 'e.'+var, guarantee)
    for var in sys_vars.keys():
        assumption = re.sub(r'\b'+var+r'\b', 's.'+var, assumption)
        guarantee = re.sub(r'\b'+var+r'\b', 's.'+var, guarantee)
    
    # Add assumption on the possible initial state of the system and all the possible 
    # values of the environment
    env_values_formula = ''
    for var, reg in env_vars.iteritems():
        all_values = re.findall('[-+]?\d+', reg)
        if (len(all_values) > 0):
            if (len(env_values_formula) > 0):
                env_values_formula += ' & '
            current_env_values_formula = ''
            for val in all_values:
                if (len(current_env_values_formula) > 0):
                    current_env_values_formula += ' | '
                current_env_values_formula += 'e.' + var + '=' + val
            env_values_formula += '(' + current_env_values_formula + ')'
    
    sys_values_formula = ''
    for var, reg in sys_vars.iteritems():
        all_values = re.findall('[-+]?\d+', reg)
        if (len(all_values) > 0):
            if (len(sys_values_formula) > 0):
                sys_values_formula += ' & '
            current_sys_values_formula = ''
            for val in all_values:
                if (len(current_sys_values_formula) > 0):
                    current_sys_values_formula += ' | '
                current_sys_values_formula += 's.' + var + '=' + val
            sys_values_formula += '(' + current_sys_values_formula + ')'
    
    addAnd = False
    if (len(assumption) > 0 and not(assumption.isspace())):
        assumption = assumption 
        addAnd = True
    if (len(env_values_formula) > 0):
        if (addAnd):
            assumption = assumption + ' &\n'
        assumption = assumption + '-- all initial environment states\n'
        assumption = assumption + '\t(' + env_values_formula + ')'
        assumption = assumption + ' &\n-- possible values of environment variables\n'
        assumption = assumption + '\t[](next(' + env_values_formula + '))'
    
    f = open(spcfile, 'w')
    
    # Assumption
    f.write('LTLSPEC\n')
    f.write(assumption)
    f.write('\n;\n')
    
    # Guarantee
    f.write('\nLTLSPEC\n')
    addAnd = False
    if (len(guarantee) > 0 and not(guarantee.isspace())):
        f.write(guarantee)
        addAnd = True
    if (len(sys_values_formula) > 0):
        if (addAnd):
            f.write(' &\n')
        f.write('-- all initial system states\n')
        f.write('\t(' + sys_values_formula + ')')
        addAnd = True        
    
    # Transitions for continuous dynamics
    for from_region in xrange(0,ppp.num_regions):
        to_regions = [j for j in range(0,ppp.num_regions) if \
                          trans1[j][from_region]]
        if (addAnd):
            f.write(' &\n')
        if (from_region == 0):
            f.write('-- transition relations for continuous dynamics\n')
        f.write('\t[](((e.' + control_var + ' = ' + str(val1) + ') & (s.' + disc_cont_var + ' = ' +
                    str(from_region) + ')) -> next(')
        if (len(to_regions) == 0):
            f.write('FALSE')
        for i, to_region in enumerate(to_regions):
            if (i > 0):
                f.write(' | ')
            f.write('(s.' + disc_cont_var + ' = ' + str(to_region) + ')')
        f.write('))')
        addAnd = True
        
    for from_region in xrange(0,ppp.num_regions):
        to_regions = [j for j in range(0,ppp.num_regions) if \
                          trans2[j][from_region]]
        if (addAnd):
            f.write(' &\n')
        f.write('\t[](((e.' + control_var + ' = ' + str(val2) + ') & (s.' + disc_cont_var + ' = ' +
                    str(from_region) + ')) -> next(')
        if (len(to_regions) == 0):
            f.write('FALSE')
        for i, to_region in enumerate(to_regions):
            if (i > 0):
                f.write(' | ')
            f.write('(s.' + disc_cont_var + ' = ' + str(to_region) + ')')
        f.write('))')
        addAnd = True

    
    # Transitions
    # For discrete transitions
    if (len(sys_values_formula) > 0):
        if (addAnd):
            f.write(' &\n')
        f.write('-- possible values of discrete system variables\n');
        f.write('\t[](next(' + sys_values_formula + '))')
    
    f.write('\n;')
    f.close()
