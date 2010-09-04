#!/usr/bin/env python

""" 
-------------------------------------------------------------------
discretizeM.py --- Interface to MATLAB implementation of discretize
-------------------------------------------------------------------

Created by Ufuk Topcu, 8/30/10
Modified by Nok Wongpiromsarn, 9/3/10

:Version: 0.1.0
"""
import os, time, subprocess
import pickle 
import pdb
from numpy import *
from scipy import *
from polytope_computations import *
from copy import deepcopy
from polyhedron import Vrep, Hrep
from prop2part import PropPreservingPartition
from errorprint import printWarning, printError, printInfo

matfile_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), \
                               'matlab', 'tmpmat')
to_matfile = os.path.join(matfile_dir, 'dataToMatlab.mat')
from_matfile = os.path.join(matfile_dir, 'dataFromMatlab.mat')
donefile = os.path.join(matfile_dir, 'done.txt')

class CtsSysDyn:
    """
    CtsSysDyn class for specifying the continuous dynamics:

        s[t+1] = A*s[t] + B*u[t] + E*w[t]
        u[t] \in Uset - polytope object
        d[t] \in Wset - polytope object

    A CtsSysDyn object contains the fields A, B, E, Uset and Wset as defined above.
    
    **Constructor**:
    
    **CtsSysDyn** ([ `A` = [][, `B` = [][, `E` = [][, `Uset` = [][, `Wset` = []]]]]])
    """

    def __init__(self, A=[], B=[], E=[], Uset=[], Wset=[]):
        self.A = A
        self.B = B
        self.E = E
        self.Uset = Uset
        self.Wset = Wset

def discretizeM(part, ssys, N = 10, auto=True, minCellVolume = 0.1, \
                    maxNumIterations = 5, useClosedLoopAlg = True, \
                    useAllHorizonLength = True, useLargeSset = True, \
                    timeout = -1, maxNumPoly = 5, verbose = 0):
    """
    Discretize the continuous state space using MATLAB implementation.
    
    Input:
    
    - `part`: a PropPreservingPartition object
    - `ssys`: a CtsSysDyn object
    - `N`: horizon length
    - `auto`: a boolean that indicates whether to automatically run the MATLAB  
      implementation of discretize.
    - `minCellVolume`: the minimum volume of cells in the resulting partition
    - `maxNumIterations`: the maximum number of iterations
    - `useClosedLoopAlg`: a boolean that indicates whether to use the closed loop algorithm.
      For the difference between the closed loop and the open loop algorithm, 
      see Borrelli, F. Constrained Optimal Control of Linear and Hybrid Systems, 
      volume 290 of Lecture Notes in Control and Information Sciences. Springer. 2003.
    - `useAllHorizonLength`: a boolean that indicates whether all the horizon length up
      to probStruct.N can be used. This option is relevant only when the closed 
      loop algorithm is used.
    - `useLargeSset`: a boolean that indicates whether when solving the reachability
      problem between subcells of the original partition, the cell of the
      original partition should be used for the safe set.
    - `timeout`: timeout (in seconds) for polytope union operation. 
      If negative, the timeout won't be used. Note that using timeout requires MATLAB
      parallel computing toolbox.
    - `maxNumPoly`: the maximum number of polytopes in a region used in computing reachability.
    - `verbose`: level of verbosity of
    """
    if (os.path.isfile(globals()["to_matfile"])):
        os.remove(globals()["to_matfile"])
    if (os.path.isfile(globals()["from_matfile"])):
        os.remove(globals()["from_matfile"])
    if (os.path.isfile(globals()["donefile"])):
        os.remove(globals()["donefile"])
    
    starttime = time.time()
    discretizeToMatlab(part, ssys, N, minCellVolume, \
                           maxNumIterations, useClosedLoopAlg, \
                           useAllHorizonLength, useLargeSset, \
                           timeout, maxNumPoly, verbose)

    if (auto):
        try:
            mpath = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'matlab')
            mcommand = "addpath('" + mpath + "');"
            mcommand += "try, runDiscretizeMatlab; catch, disp(lasterr); quit; end;"
            mcommand += "quit;"
            cmd = subprocess.call( \
                ["matlab", "-nojvm", "-nosplash", "-r", mcommand])
            auto = True
        except:
            printError("Cannot run matlab. Please make sure that MATLAB is in your PATH.")
            auto = False

        if (not os.path.isfile(globals()["donefile"]) or \
                os.path.getmtime(globals()["donefile"]) <= \
                os.path.getmtime(globals()["to_matfile"])):
            printError("Discretization failed!")
            auto = False

    if (not auto):
        printInfo("\nPlease run 'runDiscretizeMatlab' in the 'matlab' folder.\n")
        print("Waiting for MATLAB output...")

        while (not os.path.isfile(globals()["donefile"]) or \
                   os.path.getmtime(globals()["donefile"]) <= \
                   os.path.getmtime(globals()["to_matfile"])):
            if (verbose > 0):
                print("Waiting for MATLAB output...")
            time.sleep(10)

    dyn = discretizeFromMatlab(part)
    return dyn

	 
def discretizeToMatlab(part, ssys, N = 10, minCellVolume = 0.1, \
                           maxNumIterations = 5, useClosedLoopAlg = True, \
                           useAllHorizonLength = True, useLargeSset = True, \
                           timeout = -1, maxNumPoly = 5, verbose = 0):
    """
    Generate an input file for MATLAB implementation of discretize.
    
    Input:
    
    - `part`: a PropPreservingPartition object
    - `ssys`: a CtsSysDyn object
    - `N`: horizon length
    - `minCellVolume`: the minimum volume of cells in the resulting partition
    - `maxNumIterations`: the maximum number of iterations
    - `useClosedLoopAlg`: a boolean that indicates whether to use the closed loop algorithm.
      For the difference between the closed loop and the open loop algorithm, 
      see Borrelli, F. Constrained Optimal Control of Linear and Hybrid Systems, 
      volume 290 of Lecture Notes in Control and Information Sciences. Springer. 2003.
    - `useAllHorizonLength`: a boolean that indicates whether all the horizon length up
      to probStruct.N can be used. This option is relevant only when the closed 
      loop algorithm is used.
    - `useLargeSset`: a boolean that indicates whether when solving the reachability
      problem between subcells of the original partition, the cell of the
      original partition should be used for the safe set.
    - `timeout`: timeout (in seconds) for polytope union operation. 
      If negative, the timeout won't be used. Note that using timeout requires MATLAB
      parallel computing toolbox.
    - `maxNumPoly`: the maximum number of polytopes in a region used in computing reachability.
    - `verbose`: level of verbosity of
    """

    import scipy.io as sio
    data = {}
    adj = deepcopy(part.adj)
    for i in xrange(0, len(adj)):
        adj[i][i] = 1
    data['adj'] = adj
    data['minCellVolume'] = minCellVolume
    data['maxNumIterations'] = maxNumIterations
    data['useClosedLoopAlg'] = useClosedLoopAlg
    data['useAllHorizonLength'] = useAllHorizonLength
    data['useLargeSset'] = useLargeSset
    data['timeout'] = timeout
    data['maxNumPoly'] = maxNumPoly
    data['verbose'] = verbose
    data['A'] = ssys.A
    data['B'] = ssys.B
    data['E'] = ssys.E
    data['Uset'] = ssys.Uset
    data['Wset'] = ssys.Wset
    data['N'] = N
    numpolyvec = []
    den = []
    for i1 in range(0,len(part.list_region)):
        numpolyvec.append(len(part.list_region[i1].list_poly))
        for i2 in range(0,len(part.list_region[i1].list_poly)):
            pp = part.list_region[i1].list_poly[i2]
            data['Reg'+str(i1+1)+'Poly'+str(i2+1)+'Ab'] = concatenate((pp.A,pp.b),1)
    data['numpolyvec'] = numpolyvec
    matfile = globals()["to_matfile"]
    if (not os.path.exists(os.path.abspath(os.path.dirname(matfile)))):
        os.mkdir(os.path.abspath(os.path.dirname(matfile)))
    sio.savemat(matfile, data)
    print('MATLAB input saved to ' + matfile)
	
	
def discretizeFromMatlab(origPart):
    """
    Load the data from MATLAB discretize implementation.

    Input:

    - origPart: a PropPreservingPartition object
    """
    import scipy.io as sio
    matfile = globals()["from_matfile"]
    if (os.path.getmtime(matfile) <= os.path.getmtime(globals()["to_matfile"])):
        printWarning("The MATLAB output file is older than the MATLAB input file.")
        cont = raw_input('Continue [c]?: ')
        if (cont.lower() != 'c'):
            return False

    print('Loading MATLAB output from ' + matfile)
    data = sio.loadmat(matfile)
    trans = data['trans']
    a1 = data['numNewCells']
    numNewCells = zeros((a1.shape[0],1))
    numNewCells[0:,0] = a1[:,0]
    newCellVol = data['newCellVol']
    num_cells = data['num_cells'][0][0]
    a2 = data['numpoly']
    numpoly = zeros(a2.shape)
    numpoly[0:,0:] = a2[0:,0:]
	
    regs = []
    for i1 in range(0,num_cells):
        for i2 in range(0,numNewCells[i1]):
            polys = []
            props = []
            for i3 in range(0,int(numpoly[i1,i2])):
                Ab = data['Cell'+str(i1)+'Reg'+str(i2)+'Poly'+str(i3)+'Ab']
                A = deepcopy(Ab[:,0:-1])
                b = zeros((A.shape[0],1))
                b[0:,0] = deepcopy(Ab[:,-1])
                polys.append(Polytope(A,b))

            props = origPart.list_region[i1].list_prop
            regs.append(Region(polys,props))	
				
    domain = deepcopy(origPart.domain)
    num_prop = deepcopy(origPart.num_prop)
    num_regions = len(regs)
    list_prop_symbol = deepcopy(origPart.list_prop_symbol)
    newPartition = PropPreservingPartition(domain, num_prop, regs, num_regions, [], \
                                               trans, list_prop_symbol)
    return newPartition
	
	
	
		
		
	
	
	
