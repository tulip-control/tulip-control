"""
Export the controller synthesized by TuLiP

- only support Deterministic Mealy machine, with 1 init state
- only support controller synthesized by gr1c, NOT jtlv!!
"""
import numpy


def mealyexport(ctrl, filename="mealydata"):

    def input2num(sublabel_dict, inputs):
        result = 0
        tmp = 1
        for inputname, inputset in inputs.items():
            inputset = list(inputset)
            valuation = sublabel_dict[inputname]
            idx = inputset.index(valuation)
            result = result + idx * tmp
            tmp = tmp * len(inputset)

        return result

    fh = open('include/'+filename+".h", 'w')
    fc = open('sources/'+filename+".c", 'w')

    fh.write('''\
#ifndef __MEALY_H__
#define __MEALY_H__\n''')
    fh.write('#include <ecos.h>\n')
    fc.write('#include "'+filename+'.h"\n')

    inputs = ctrl.inputs

    if inputs is None:
        raise NotImplementedError

    fh.write('extern idxint nInputVariable;\n')
    fh.write('extern idxint nInputValue[];\n')

    numinputs = 1

    index = 0
    fc.write('idxint nInputVariable = ' + str(len(inputs.items())) + ';\n')
    fc.write('idxint nInputValue[] = {')
    for inputname, inputset in inputs.items():
        fc.write(str(len(inputset))+",")
        numinputs = numinputs * len(inputset)
        index = index + 1

    fc.write('};\n')

    fh.write("#define NUM_INPUT "+str(numinputs)+"\n")

    initialstate = ctrl.states.initial.pop()

    numstates = len(ctrl.states())

    fh.write("#define NUM_STATE "+str(numstates)+"\n")

    fh.write("extern int initState;\n")
    fh.write("extern int transition[NUM_STATE][NUM_INPUT];\n")
    fh.write("extern int output[NUM_STATE][NUM_INPUT];\n")

    statelist = []
    # create a list of states
    for state in ctrl.states():
        statelist.append(state)

    fc.write("int initState = "+str(statelist.index(initialstate))+";\n")

    # create the transition and output table:
    transition = numpy.zeros((numstates, numinputs), dtype='intc')
    output = numpy.zeros((numstates, numinputs), dtype='intc')
    for state in ctrl.states():
        index = statelist.index(state)
        trans = ctrl.transitions.find([state])
        for (from_state_, to_state, sublabel_dict) in trans:
            inputinedx = input2num(sublabel_dict, inputs)
            transition[index][inputinedx] = statelist.index(to_state)
            output[index][inputinedx] = sublabel_dict['loc']

    fc.write("int transition[NUM_STATE][NUM_INPUT]={\n")
    for i in range(0, numstates):
        fc.write("{")
        for j in range(0, numinputs):
            fc.write(str(transition[i][j])+",")
        fc.write("},\n")
    fc.write("};\n\n")

    fc.write("int output[NUM_STATE][NUM_INPUT]={\n")
    for i in range(0, numstates):
        fc.write("{")
        for j in range(0, numinputs):
            fc.write(str(output[i][j])+",")
        fc.write("},\n")
    fc.write("};\n\n")

    fh.write('int value2index(idxint inputValue[]);\n')

    fc.write('int value2index(idxint inputValue[])\n{\n')
    fc.write('	idxint result = 0;\n')
    fc.write('	int tmp = 1;\n')
    fc.write('	idxint valuation;\n')
    fc.write('	int i;\n')
    fc.write('	for(i=0;i<nInputVariable;i++)\n')
    fc.write('	{\n')
    fc.write('		valuation = inputValue[i]%nInputValue[i];\n')
    fc.write('		result += valuation * tmp;\n')
    fc.write('		tmp *= nInputValue[i];\n')
    fc.write('	}\n')
    fc.write('	return result;\n')
    fc.write('}\n')

    fh.write("#endif")
    fc.close()
    fh.close()
