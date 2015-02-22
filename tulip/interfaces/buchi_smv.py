"""
buchi_smv.py
Reponsible for converting buchi automaton description file into an equivalent
smv file for satisfiability checking
"""
import re

def buchi_to_smv(smv_file, buchi_file, env_init, env_safety, env_prog, assumption):
    '''converts buchi automaton into equivalent nusmv file
    @param env_safety: list of environment safety spec of the problem
    @param env_prog : list of environment progress spec of the problem'''
    f = open(buchi_file, 'r')
    g = open(smv_file, 'w')

    nodes = [] # list of node labels of the automaton
    labels = set()
    transitions = [] # list of lists (conditions)
    targets = [] # list of list of target nodes

    for line in f:
        # First add the nodes of the automaton
        if (line.find(':\n') >= 0):
            nodes.append(line[:-2].strip())
            transitions.append([])
            targets.append([])
        # If we find the line with conditions listed
        elif (line.find(':: ') >= 0):
            tmp = line[4:-1]
            trans = tmp.split(' -> goto ')[0]
            if '||' in trans: # the label is disjunction of two vairables
                _trans_list = []
                for i in trans.split(' || '):
                    _trans_list.append(re.sub('[()]', '', i))
                for j in _trans_list:
                    if '!' == j[0]:
                        labels.add(j[1:])
                    else:
                        labels.add(j)
                _trans = re.sub('[()]', '', trans)
                _trans = _trans.replace('||', '|')
                transitions[len(nodes)-1].append('(' + _trans + ')')
            elif '&&' in trans: # the transition is conjunction of two variables
                _trans_list = []
                for i in trans.split(' && '):
                    _trans_list.append(re.sub('[()]', '', i))
                for j in _trans_list:
                    if '!' == j[0]:
                        labels.add(j[1:])
                    else:
                        labels.add(j)
                _trans = re.sub('[()]', '', trans)
                _trans = _trans.replace('&&', '&')
                transitions[len(nodes)-1].append(_trans)
            else:
                if '!' in trans:
                    _trans = re.sub('[()]', '', trans)
                    labels.add(_trans[1:])
                    transitions[len(nodes)-1].append(_trans)
                elif trans == '(1)':
                    transitions[len(nodes)-1].append('TRUE')
                else:
                    _trans = re.sub('[()]', '', trans)
                    labels.add(_trans)
                    transitions[len(nodes)-1].append(_trans)
        	targetState = tmp.split(' -> goto ')[1]
            targets[len(nodes)-1].append(targetState)
    
    # Add the additinal transitions to guaratee the exhaustive condition
    # for nusmv conditions
    for i in xrange(len(nodes)):
        neg_spec = '!('
        tmp = ''
        for s in transitions[i]:
        	tmp = '(' + str(s) + ')'
        	neg_spec += tmp
        	neg_spec += ' | '
        neg_spec = neg_spec[:-3] 
        neg_spec += ')'
        transitions[i].append(neg_spec)
        targets[i].append('block')
    
    # Then check if the given assumptions have any unknown variables. 
    notIncluded = set()
    if '->' in assumption:
        _t = ''
        tmp = assumption.split(' -> X ')
        for t in tmp:
            _t = re.sub('[()]', '', t)
            if '!' in _t:
                _t = _t[1:]
                if _t not in labels:
                    notIncluded.add(_t)
            else:
                if _t not in labels:
                    notIncluded.add(_t)

    # now write to the smv file
    g.write('MODULE main\n  VAR\n')
    g.write('    state : {' + ', '.join(nodes) + ', block};\n')
    for i in labels:
        g.write('    ' + i + ' : boolean;\n')
    for i in notIncluded:
        g.write('    ' + i + ' : boolean;\n')
    g.write('  ASSIGN\n')
    g.write('    init(state) := ' + nodes[0] + ';\n')
    
    #TODO: for deciding the initial condition for nusmv file, 
    # also need to consider the added assumptions as well.
    for i in labels:
        if (i in env_init + env_safety + env_prog):
            boolval = 'TRUE'
        else:
            boolval = 'FALSE'
        g.write('    init(' + i +') := ' + boolval + ';\n')

    g.write('    next(state) := \n      case\n')
    for i in range(len(nodes)):
        for j in range(len(transitions[i])):
            g.write('        state = ' + nodes[i] + ' & ' + transitions[i][j] +\
                    ' : ' + targets[i][j] + ';\n')
    g.write('        state = block & TRUE : block;\n')
    g.write('      esac;\n')
    f.close()
    g.close()

