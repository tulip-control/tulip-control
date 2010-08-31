
from numpy import *
from scipy import *
import pickle 
import pdb
from polytope_computations import *
from copy import deepcopy
from polyhedron import Vrep, Hrep





class CtsSysDyn:
	def __init__(self, A=[], B=[], E=[], Uset=[], Wset=[]):
		'''
		s[t+1] = A*x[t] + B*u[t] + E*w[t]
		u[t] \in Uset - polytope object
		d[t] \in Wset - polytope object
		'''
		self.A = A
		self.B = B
		self.E = E
		self.Uset = Uset
		self.Wset = Wset
		 

def discretizeToMatlab(part, adj, ssys, minCellVolume = 0.1, \
			maxNumIterations = 100, useClosedLoopAlg = True, \
			useAllHorizonLength = True, useLargeSset = True):
	import scipy.io as sio
	data = {}
	data['adj'] = adj;
	data['minCellVolume'] = minCellVolume
	data['maxNumIterations'] = maxNumIterations
	data['useClosedLoopAlg'] = useClosedLoopAlg
	data['useAllHorizonLength'] = useAllHorizonLength
	data['useLargeSset'] = useLargeSset
	data['A'] = ssys.A
	data['B'] = ssys.B
	data['E'] = ssys.E
	data['Uset'] = ssys.Uset
	data['Wset'] = ssys.Wset
	numpolyvec = []
	den = []
	for i1 in range(0,len(part.list_region)):
		numpolyvec.append(len(part.list_region[i1].list_poly))
		for i2 in range(0,len(part.list_region[i1].list_poly)):
			pp = part.list_region[i1].list_poly[i2]
			data['Reg'+str(i1+1)+'Poly'+str(i2+1)+'Ab'] = concatenate((pp.A,pp.b),1)
	data['numpolyvec'] = numpolyvec
	sio.savemat('dataToMatlab.mat',data)
	
	
def discretizeFromMatlab(origPart):
	import scipy.io as sio
	data = sio.loadmat('dataFromMatlab.mat')
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
				props.append(origPart.list_region[i1].list_prop)
			
			regs.append(Region(polys,props))	
				
	domain = deepcopy(origPart.domain)
	num_prop = deepcopy(origPart.num_prop)
	num_regions = len(regs)
	list_prop_symbol = deepcopy(origPart.list_prop_symbol)
	newPartition = PropPreservingPartition(domain, num_prop, regs, num_regions, [], trans, list_prop_symbol)
	
	return newPartition, trans, numNewCells, newCellVol
	
	
	
		
		
	
	
	
