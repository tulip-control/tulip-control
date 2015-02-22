"""
mining.py
Assumption mining algorithm

Given an LTL formula
phi = phi_e -> phi_s
Mine additional envrionment assumptions required for a synthesis
"""
from tulip import synth, spec, transys
from tulip.abstract import horizon
from tulip.interfaces import slugs, aut_nusmv, buchi_smv
import re, sys, os

NUSMV_PATH = '/usr/local/bin/nusmv'
LTL2BA_PATH = 'ltl2ba/'
FINAL_RESULT = [] # A set of mined assumptions for return
ERROR_SPECS = [] # A list of specs mined that had error in model checking

# THE MAIN FUNCTION
def mine_assumption(grinput, candidates, trace):
    '''Returns the mined assumption required to make the system realizable'''
    print '==========================================='
    print 'Current Specification : \n' + grinput.__str__()
    print '===========================================\n'

    if (synth.is_realizable('jtlv', grinput)):
        print '============================================================='
        print 'The specification is realizable, and mined assumptions are : \n'
        for i in FINAL_RESULT:
            print i
        print '============================================================='
        print 'Final specification: '
        print grinput
        print '============================================================='
        print 'Saving the controller strategy..'
        ctrl = synth.synthesize('jtlv', grinput)
        if not ctrl.save('controller.png'):
            print(ctrl)
        return

    specs = grinput
    counter_aut = synth.synthesize('slugs', specs)
    print 'Counter-strategy machine generated'
    print 'Now converting aut file to smv...'
    aut_nusmv.aut_to_smv(counter_aut, 'counter_result.smv', specs) # convert
    print 'Conversion to smv complete.'
    print '==========================================='
    print 'Starting mining...'
    print '==========================================='
    [mined_assumption, category, candidates_updated] = \
                        mine(counter_aut, 'counter_result.smv', candidates, trace, specs)
    if (mined_assumption == False):
        print "Error: Candidates from the given template are not enough."
        print '       Could not make the specification realizable.'
        print 'So far, the mined assumptions are: \n'
        for i in FINAL_RESULT:
            print i
        exit(1)
    if category == 'prog':
        FINAL_RESULT.append('[]<>(' + mined_assumption + ')')
        specs.env_prog.append(mined_assumption)
    elif (category == 'resp') or (category == 'safe'):
        FINAL_RESULT.append('[](' + mined_assumption + ')')
        specs.env_safety.append(mined_assumption)
    #Update the given specification
    specs = spec.GRSpec(specs.env_vars, specs.sys_vars, specs.env_init, specs.sys_init,
                        specs.env_safety, specs.sys_safety, specs.env_prog, specs.sys_prog)

    mine_assumption(specs, candidates_updated, trace)

def mine(counter_aut, counter_machine, candidates, trace, specs):
    '''Set of tests to see if the candidate assumption should be added to
    the current assumptions, counterstrategy returned in the form of Moore Machine'''
    for idx in range(len(candidates)):
        while candidates[idx]:
            result = candidates[idx].pop()
            aut_nusmv.aut_to_smv(counter_aut, counter_machine, specs)
            #print '******CANDIDATES REMAINING******'
            #for i in candidates:
            #    print i
            #print '********************************'
            if idx == 0:
                category = 'safe'
                print '=================================================='
                print 'Testing SAFETY properties from the template...'
                print '--------------------------------------------------'
                if (trace_check(trace, result)):
                    #the argument given as string
                    if (model_check(counter_machine, result, category)): 
                        if is_satisfiable(specs, result, category):
                            print '+=============================================+'
                            print '|   Mined Assumption: '
                            print '|   [] (' + result + ')'
                            print '+=============================================+\n\n'
                            return [result, category, candidates]
            elif idx == 1:
                category = 'prog'
                print '================================================'
                print 'Testing PROGRESS properties from the template...'
                print '------------------------------------------------'
                if (trace_check(trace, result)):
                    #the argument given as string
                    if (model_check(counter_machine, result, category)): 
                        if is_satisfiable(specs, result, category):
                            print '+=============================================+'
                            print '|   Mined Assumption:'
                            print '|   []<> (' + result + ')'
                            print '+=============================================+\n\n'
                            return [result, category, candidates]
            elif idx == 2:
                category = 'resp'
                print '=============================================='
                print 'Testing RESPONSE properties from the template...'
                print '----------------------------------------------'
                if (trace_check(trace, result)):
                    #the argument given as string
                    if (model_check(counter_machine, result, category)): 
                        if is_satisfiable(specs, result, category):
                            print '+=============================================+'
                            print '|   Mined Assumption:'
                            print '|   [] (' + result + ')'
                            print '+=============================================+\n\n'
                            return [result, category, candidates]
    return [False, None, candidates]


# TODO: receding horizon mining approach
def rh_mine(rhprob, trace):
    '''Applying mining assumption algo on receding horizon problem backwards
    @param rhprob: RHTLPProb object'''
    for i in range(len(rhprob.parts)-1):
        mined = mine_assumption(sys, rhprob.shortRHTLProbs[rhprob.parts[i]].local_spec, \
                        rhprob.shortRHTLProbs[rhprob.parts[i]].local_spec.env_vars,
                        rhprob.shortRHTLProbs[rhprob.parts[i]].local_spec.sys_vars, trace)
        mined = synth._conj(mined)
        rhprob.shortRHTLProbs[rhprob.parts[i+1]].local_spec.sys_safe |= set(mined)
    raise NotImplementedError


def model_check(smv_file, specs, category):
    '''Model Checking with specification via calling NuSMV
    @param smv_file: input smv file used to model check for the specification given
    @param specs: specification used for the model checking (given in String)
    @param category: specifies what type of input spec is (safety, progress, or response)'''
    new_file = _add_gr_to_nusmv(smv_file, specs, category)
    print 'Model Checking in progress...'
    print '------------------------------------------------'
    print 'Calling NuSMV with following arguments: '
    print '  NuSMV path: ' + NUSMV_PATH
    if category == 'prog':
        print '  Candidate Specification: G F ' + specs
    elif (category == 'safe') or (category == 'resp'):
        print '  Candidate Specification: G ' + specs
    start = os.times()[0]
    os.system(NUSMV_PATH + ' ' + new_file + ' > model_check_results.txt')
    f = open('model_check_results.txt', 'r')
    for line in f:
        if (line.find('-- specification') >= 0):
            if (line.find('true') >= 0):
                end = os.times()[0]
                print 'Model Checking returned True'
                #print 'Run Time: ' + str(end - start) + '[sec]'
                print '------------------------------------------------\n'
                f.close()
                return True
            elif (line.find('false') >= 0):
                end = os.times()[0]
                print 'Model Checking returned False'
                #print 'Run Time: ' + str(end - start) + '[sec]'
                print '------------------------------------------------\n\n'
                f.close()
                return False
    print 'Model Checking Failed. Check dependencies.\n'
    ERROR_SPECS.append(specs)
    f.close()


def trace_check(trace, specs): #TODO: think about how traces are going to look like
    '''Trace checking with specification'''
    if trace == '': #if there is go given trace
        print 'No trace given: proceeding to model checking...'
        return True
    else:
        g = open('trace.smv', 'w')
        g.write('MODULE main\nVAR\n')

        os.system(NUSMV_PATH + ' trace.smv' +  + ' > trace_check_results.txt')
        f = open('results.txt', 'r')
        for line in f:
            if (line.find('-- specification') >= 0):
                if (line.find('true') >= 0):
                    return True
                elif (line.find('false') >= 0):
                    return False


# TODO: NuSMV satisfiability
def is_satisfiable(specs, assumption, category):
    '''Test if the given assumption is trivial for the original specification.
    @param specs: original specification (target)
    @param assumption: specification to be added, and thus to be tested
    @param category: category of the assumption to added, (progress, safety, or response)
    @return TRUE if satisfiable, FALSE is not satisfiable'''
    new_spec = []
    for i in specs.env_safety:
        if '!' in i:
            j = i.replace('!', '! ')
        if '->' in i: # for response properties
            j = '(' + i + ')'
        new_spec.append('[] ' + j)
    for i in specs.env_prog:
        if '!' in i:
            j = i.replace('!', '! ')
        new_spec.append('[] <> ' + j)
    new_spec = ' && '.join(new_spec)
    print "Satisfiability Testing on Environment Assumption :"
    print str(new_spec)
    # Create a transition system (buchi automaton) corresponding to this env spec
    os.system(LTL2BA_PATH + ' -f "' + new_spec + '" > buchi_file.txt')

    syntax = ['!', '->', 'X', '||']
    # split the components so that it can be used to put s. for nusmv
    components = assumption.split() 
    new_components = []
    for i in components:
        if i in syntax: # these don't mean anything
            new_components.append(i)
        if '(' in i: #if there is parenthesis, stip it down
            i = i.strip('(')
            i = i.strip(')')
            new_components.append(i)
        elif i not in syntax:
            new_components.append(i)
    new = ' '.join(new_components)

    # Write the assumption to the file
    buchi_smv.buchi_to_smv('buchi_smv.smv', 'buchi_file.txt', 
                            specs.env_init, specs.env_safety, specs.env_prog, assumption)
    f = open('buchi_smv.smv', 'a')
    f.write('LTLSPEC ')
    if category == 'safe' or category == 'resp':
        print "Testing : G (" + str(new) + ")"
        f.write('G (' + new + ') & G !(state = block)\n')
    elif category == 'prog':
        print "Testing : G F (" + str(new) + ")" 
        f.write('G F (' + new + ') & G !(state = block)\n')
    else:
        print 'Category not defined: specify what type of the candidate assumption is'
        return
    f.close()

    # Run NuSMV on the transition created and test with respect to the mined assumption
    os.system(NUSMV_PATH + ' buchi_smv.smv > satTest.txt')

    g = open('satTest.txt', 'r')
    for line in g:
        if (line.find('-- specification') >= 0):
            if (line.find('true') >= 0):
                print 'Satisfiability Checking returned True'
                print '------------------------------------------------\n'
                g.close()
                return True
            elif (line.find('false') >= 0):
                print 'Satisfiability Checking returned False'
                print '------------------------------------------------\n\n'
                g.close()
                return False
    print 'Satisfiability Checking Failed. Check dependencies.\n'


#TODO: return the contradictory mined assumption that goes against the original spec
def return_contradictory(specs, assumption, category):
    '''returns some part of the original spec that contradicts the mined assumption
    @param specs: GRSpec object
    @param assumption: string representation of the mined assumption'''
    if category == 'safe':
        _assumption = re.sub('[()]', '', assumption)
        if '||' in _assumption:
            for i in _assumption.split(' || '):
                if i[0] == '!':
                    exit(1)

        if _assumption[0] == '!':
            possible.append()

    raise NotImplementedError


def _generate_candidate_list(input_sig, output_sig):
    '''generate GR specification template using the input and output signals
    @param input_sig : environment specification variables
    @param output_sig : system specification variables
    @return list of candidate assumptions'''
    env_progress = []
    env_response = []
    env_safety = []
    if len(input_sig) == 1:
        env_progress += [str(input_sig[0]), '!' + str(input_sig[0])]
        env_safety += [str(input_sig[0]), '!' + str(input_sig[0])]
        for k in (input_sig + output_sig):
            env_response += [str(k) + ' -> X (' + str(input_sig[0]) + ')', 
                             '!' + str(k) + ' -> X (' + str(input_sig[0]) + ')',
                             str(k) + ' -> X (!' + str(input_sig[0]) + ')', 
                             '!' + str(k) + ' -> X (!' + str(input_sig[0]) + ')']
    else:
        for i in input_sig[:-1]:
            env_progress += [str(i), '!' + str(i)]
            env_safety += [str(i), '!' + str(i)]
            for j in input_sig[input_sig.index(i)+1:]:
                env_safety += [str(i) + ' || ' + str(j), '!' + str(i) + ' || ' + str(j),
                               str(i) + ' || !' + str(j), '!' + str(i) + ' || !' + str(j)]
            for k in (input_sig + output_sig):
                env_response += [str(k) + ' -> X (' + str(i) + ')', 
                                 '!' + str(k) + ' -> X (' + str(i) + ')',
                                 str(k) + ' -> X (!' + str(i) + ')', 
                                 '!' + str(k) + ' -> X (!' + str(i) + ')']

    print '======================================='
    print 'Candidates Generated from GR Template: '
    print '---------------------------------------'
    print '\nSafety   :'
    for i in env_safety:
        print '  [](' + i + ')'
    print 'Progress :'
    for i in env_progress:
        print '  []<>(' + i + ')'
    print '\nResponse :'
    for i in env_response:
        print '  [](' + i + ')'
    print '======================================='
    return [env_safety, env_progress, env_response]


def _add_gr_to_nusmv(smv_file, specs, category):
    '''Converts the gr specification into appropriate form and insert it in the smv file
    for model checking, specs given in string, and assumed to be following the templates'''
    new_file = smv_file
    syntax = ['!', '->', 'X', '||']
    # split the components so that it can be used to put 's.' for nusmv
    components = specs.split() 
    new_components = []
    for i in components:
        _i = re.sub('[()]', '', i)
        if _i in syntax: # these symbols don't mean anything
            new_components.append(_i)
        else:
            if '!' in _i:
                new_components.append('!s.' + _i[1:])
            else:
                new_components.append('s.' + _i)

    new = ' '.join(new_components)
    f = open(new_file, 'a')
    f.write('LTLSPEC ')
    if category == 'safe':
        _new = new.replace(' || ', ' | ')
        f.write('! G ' + _new + '\n')
    elif category == 'resp':
        f.write('! G ' + new + '\n')
    elif category == 'prog':
        f.write('! G F ' + new + '\n')
    else:
        print 'Category not defined: specify what type of the candidate assumption is'
        return
    f.close()
    return new_file


def _check_numeric(s):
    '''checks if the given string is composed only of numerical values'''
    try:
        int(s)
    except ValueError:
        return False
    return True
