'''
aut_nusmv.py
Contains relevant functions for converting the .aut file of strategy or counter-strategy
into corresponding equivalent implementation in NuSMV for model checking.
'''
import re, os

# This process assumes that transLabels[0], which is the first transition conditions,
# contains all the keys that should be used to describe the transition variables.
def aut_to_smv(aut_file, smv_file, specs):
    '''@param aut_file: input .aut file describing finite transition machine (Moore Machine)
       @param smv_file: output .smv file used for NuSMV model checking
       @param specs: GR specification of the original problem
       @param sys: Finite Transition System (FTS) object
       Converts aut file into an equivalent smv file for model checking'''
    input_file = open(aut_file, 'r')
    output = open(smv_file, 'w')
    stateLabels = [] # list of string state labels
    transLabels = [] # list of dict of values of varivables
    targetLabels = [] # list of list of string target states
    start_time = os.times()[0]
    for line in input_file:
        if (line.find('State ') >= 0):
            stateID = re.search('State (\d+)', line)
            stateLabels.append(stateID.group(1))
            transLabel = dict(re.findall('(\w+):(\w+)', line))
            for key, val in transLabel.iteritems():
                if (transLabel[key] == '1'):
                    transLabel[key] = 'TRUE'
                elif (transLabel[key] == '0'):
                    transLabel[key] = 'FALSE'
            transLabels.append(transLabel)

        if (line.find('successors') >= 0):
            if (line.find('no successors') >= 0):
                lst = []
                lst.append(stateID.group(1))
                targetLabels.append(lst)
            else:
                targetStates = re.findall(' (\d+)', line)
                targetLabels.append(targetStates)

    # WRITING INTO THE OUTPUT FILE
    output.write('MODULE main\nVAR\n')
    output.write('  s : state;\n')
    output.write('  proc : process move(s);\n\n')
    output.write('MODULE state\nVAR\n')
    output.write('  value : {')
    for i in range(len(stateLabels)-1):
        output.write(stateLabels[i] + ', ')
    output.write(stateLabels[-1] + '};\n')
    for item in transLabels[0].keys():
        if _check_numeric(item):
            output.write('  loc' + item + ' : boolean;\n')
        else:
            output.write('  ' + item + ' : boolean;\n')
    output.write('INIT\n')

    print 'Finding initial conditions...'

    # Set up the initial conditions
    # dict form. find the initial conditions
    initConditions = _find_initConditions(transLabels[0], specs) 
    # find the initial state that satisfies all the initial conditions given
    initState = _find_initState(stateLabels, transLabels, initConditions)
    # find the initial state that satisfies all the initial conditions given
    #initState = _better_find_initState(aut_file, transLabels[0], initConditions)

    #diagnostic part... delete after completion
    if initState == None: #initState is not found
        print 'Check the following dependencies: '
        print '   Initial Conditions : '
        for key, val in initConditions.iteritems():
            print '    ' + key + ' : ' + val
        exit(1)

    output.write('  value = ' + initState + '\n')
    for key, val in initConditions.iteritems():
        if _check_numeric(key):
            output.write('  & loc' + key + ' = ' + val + '\n')
        else:
            output.write('  & ' + key + ' = ' + val + '\n')

    output.write('\nMODULE move(s)\nASSIGN\n  next(s.value) := \n    case\n')
    # put in value transitions
    for i in range(len(stateLabels)):
        output.write('      s.value = ' + stateLabels[i] + ' : ')
        if len(targetLabels[i]) == 1:
            output.write(targetLabels[i][0] + ';\n')
        else: # if there is more than one successors
            output.write('{')
            for idx in range(len(targetLabels[i])-1):
                output.write(targetLabels[i][idx] + ', ')
            output.write(targetLabels[i][-1])
            output.write('};\n')
    output.write('      TRUE : s.value;\n    esac;\n')
    # now the conditions must be listed out
    for item in transLabels[0].keys():
        if _check_numeric(item):
            output.write('  next(s.loc' + item + ') := \n    case\n')
        else:
            output.write('  next(s.' + item + ') := \n    case\n')
        for i in range(len(stateLabels)):
            for j in range(len(targetLabels[i])):
                idx = stateLabels.index(targetLabels[i][j])
                #copy the current transLabel to compare it with the target's transLabel
                tmp = transLabels[i] 
                for diffKeys, diffVals in transLabels[idx].iteritems():
                    tmp[diffKeys] = diffVals
                try:
                    output.write('      s.value = ' + stateLabels[i] +
                    ' & next(s.value) = ' + targetLabels[i][j] + ' : ' + tmp[item] + ';\n')
                except KeyError:
                    if _check_numeric(item):
                        output.write('      s.value = ' + stateLabels[i] +
                        ' & next(s.value) = ' + targetLabels[i][j] + ' : s.loc' + item + ';\n')
                    else:
                        output.write('      s.value = ' + stateLabels[i] +
                        ' & next(s.value) = ' + targetLabels[i][j] + ' : s.' + item + ';\n')

        if _check_numeric(item):
            output.write('      TRUE : s.loc' + item + ';\n    esac;\n')
        else:
            output.write('      TRUE : s.' + item + ';\n    esac;\n')

    output.write('FAIRNESS\n  running\n\n')
    print 'File creation complete.'
    output.close()
    input_file.close()


def _find_initConditions(transLabel, specs):
    '''finds the initial condition for the transition values of the counterstrategy machine
    should satisfy returns dictionary
    ASSUMES that all the spec variables are in the form of sets, each element separated by comma'''
    tmp = []
    result = dict()
    for i in transLabel.keys():
        if _check_numeric(i):
            tmp.append('loc' + i)
        else:
            tmp.append(i)
    for i in tmp:
        if (i in specs.sys_vars.keys()) or (i in specs.env_vars.keys()):
            if (i in specs.sys_init) or (i in specs.sys_safety) or \
               (i in specs.env_init) or (i in specs.env_safety):
                result[i] = 'TRUE'
                # Check for added assumptions that may contribute to the initial condition
            else:
                result[i] = 'FALSE'
    #Added assumptions should be taken into account
    for j in specs.env_safety:
        if '||' in j:
            _j = j.replace(' ', '').split('||')
            for k in _j:
                if k[0] == '!':
                    result[k[1:]] = result[k[1:]] + ' | FALSE'
                else:
                    result[k] = result[k] + ' | TRUE'
    return result


def _find_initState(stateLabels, transLabels, initial_conditions):
    '''Find the initial state of the counter-strategy machine based on the initial conditions given'''
    for i in range(len(transLabels)):
        flag = 0
        for key, val in initial_conditions.iteritems():
            try:
                if transLabels[i][key] == val:
                    flag += 1
                else:
                    if '|' in val:
                        if transLabels[i][key] in val.replace(' ','').split('|'):
                            flag += 1
            except KeyError:
                pass
        if flag == len(transLabels[0]):
            if cmp(transLabels[i], initial_conditions) == 0:
                return stateLabels[i]
    print 'Initial state not found on the counter-strategy machine.'


def _better_find_initState(aut_file, transLabel, initial_conditions):
    '''find the initial state of the counter-strategy machine based on the initial conditions given
    @update: this methods goes through the file itself rather than going over saved list of labels'''
    f = open(aut_file, 'r')
    line = f.readlines[2]
    f.close()
    key_order = list(re.findall('(\w+):', line))
    tmp_line = ''
    tmp = dict()
    for key, val in initial_conditions.iteritems():
        if val == 'TRUE':
            tmp[key] = '1'
        elif val == 'FALSE':
            tmp[key] = '0'

    for k in key_order:
        for key in tmp.keys():
            if k == key:
                tmp_line += k
                tmp_line += ':'
                tmp_line += tmp[k]
                tmp_line += ', '

    tmp_line = tmp_line[:len(tmp_line)-2]
    g = open(aut_file, 'r')
    for line in g:
        if line.find(tmp_line) >= 0:
            stateID = re.search('State (\d+)', line)
            g.close()
            return stateID.group(1)
    print 'Initial state not found on the counter-strategy machine'
    raise NotImplementedError


def _check_numeric(s):
    '''checks if the given string is composed only of numerical values'''
    try:
        int(s)
    except ValueError:
        return False
    return True
