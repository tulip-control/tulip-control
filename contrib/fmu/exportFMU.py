"""
Export the TuLiP output as an FMU
"""
import uuid
import numpy
import os
from ctrlexport import mealyexport
from pppexport import pppexport
from poly2str import matrix2str, polytope2str


def exportXML(ctrl, pwa, uid, filename="modelDescription"):
    """Generate the modelDescription.xml for the FMU

    @param ctrl: tulip.transys.machines.MealyMachine
    @param pwa: tulip.abstract,discretization.AbstractPwa
    @param uid: the uid the FMU (uuid.UUID)
    @param filename: a string of the export filename
    @rtype: None
    """
    f = open(filename+".xml", 'w')

    f.write('<fmiModelDescription\n')
    f.write('    description="Discrete Time Controller"\n')
    f.write('    fmiVersion="1.5"\n')
    f.write('    guid="{' + str(uid)+'}"\n')
    f.write('    modelName="TuLiPFMU">\n\n')

    f.write('<CoSimulation modelIdentifier="TuLiPFMU" \
            canHandleVariableCommunicationStepSize="true" \
            canHandleEvents="true" \
            canProvideMaxStepSize="true"/>\n')

    f.write('<ModelVariables>\n')
    # number of state
    n = pwa.pwa.B.shape[0]
    # number of control
    m = pwa.pwa.B.shape[1]
    # output real variable: the control output
    for i in range(0, m):
        f.write('<ScalarVariable name="u'+str(i)+'" \
                valueReference="'+str(i)+'" \
                description="output" \
                causality="output">')
        f.write('<Real/>\n')
        f.write('</ScalarVariable>\n')

    # input real variable: the current state of the system
    for i in range(0, n):
        f.write('<ScalarVariable name="y'+str(i)+'" \
                valueReference="'+str(i+m)+'" \
                description="input" \
                causality="input">')
        f.write('<Real/>\n')
        f.write('</ScalarVariable>\n')

    # input discrete variable
    i = 0
    for inputname, inputset in ctrl.inputs.items():
        f.write('<ScalarVariable name="'+inputname+'" \
                valueReference="'+str(i+m+n)+'" \
                description="input" \
                causality="input">')
        f.write('<Integer/>')
        f.write('</ScalarVariable>')
        i = i+1

    f.write('</ModelVariables>\n')
    f.write('</fmiModelDescription>\n')
    f.close()


def exportFMUheader(uid, tick, filename="TuLiPFMU"):
    """Export the TuLiPFMU.h file

    @param uid: the uid of the FMU (uuid.UUID)
    @param tick: the time for one MPC step
    @param filename: a string of the export filename
    @rtype: None
    """
    f = open("include/" + filename + ".h", 'w')
    f.write('#ifndef __TULIPFMU_H__\n')
    f.write('#define __TULIPFMI_H__\n')
    f.write('#define MODEL_IDENTIFIER TuLiPFMU\n')
    f.write('#define MODEL_GUID "{' + str(uid) + '}"\n')
    f.write('#define EPSILON 1e-9\n')
    f.write('#define TICK_PERIOD ' + str(tick) + '\n')
    f.write('#endif\n')
    f.close()


def exportSysData(ctrl, pwa, initState, initRegion, filename="data"):
    """Export the data.c file

    @param ctrl: tulip.transys.machines.MealyMachine
    @param pwa: tulip.abstract,discretization.AbstractPwa
    @param initState: the initial state of the continuous system (numpy.array)
    @param initRegion: the initial region which the initial state belongs to
    @param filename: a string of the export filename
    @rtype: None
    """
    f = open("sources/" + filename + ".c", 'w')
    A = pwa.pwa.A
    B = pwa.pwa.B
    n = B.shape[0]
    # assuming full state observation, hence m = n
    m = B.shape[0]
    p = B.shape[1]
    f.write('#include "data.h"\n')
    f.write('idxint n = ' + str(n) + ';\n')
    f.write('idxint m = ' + str(m) + ';\n')
    f.write('idxint p = ' + str(p) + ';\n')
    f.write('pfloat A[] = {' + matrix2str(A) + '};\n')
    f.write('pfloat B[] = {' + matrix2str(B) + '};\n')
    f.write('idxint totalSteps = ' + str(pwa.disc_params['N']) + ';\n')
    f.write('Polytope *input_bound;\n')

    f.write(polytope2str(pwa.pwa.Uset, 'pu'))
    f.write('\nvoid init_input_bound(void)\n{\n')
    f.write('   input_bound=create_poly(puk,pul,puA,pub,pucenter);\n')
    f.write('}\n\n')

    f.write('void free_input_bound(void)\n{\n')
    f.write('		free(input_bound);\n')
    f.write('}\n')

    f.write('pfloat x0[] = {' + matrix2str(initState) + '};\n')
    f.write('idxint dRegion0 = ' + str(initRegion) + ';\n')
    f.close()


def exportFMU(ctrl, pwa, initState, initRegion, tick=1):
    """Generate the fmu

    @param ctrl: tulip.transys.machines.MealyMachine
    @param pwa: tulip.abstract,discretization.AbstractPwa
    @param initState: the initial state of the continuous system (numpy.array)
    @param initRegion: the initial region which the initial state belongs to
    @param tick: the time for one MPC step
    @rtype: None
    """
    uid = uuid.uuid1()
    exportXML(ctrl, pwa, uid)
    exportFMUheader(uid, tick)
    exportSysData(ctrl, pwa, initState, initRegion)
    pppexport(pwa.ppp)
    mealyexport(ctrl)
    os.system("make fmu")
