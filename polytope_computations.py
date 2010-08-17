
""" polytope_computations.py --- A computational geometry module for polytope computations

Functions: 
	- __boundingBoxRegion__, __findBoundingBox__
	- __isAdjacentPoly2__, __isAdjacentPolyOuter__, isAdjacentRegions
	- polySimplify
	- isNonEmptyInterior
	- Intersect, regionIntersectPoly
	- Diff, regionDiffPoly
	- ChebyBallRad
	- PolyUnionPoly, RegionUnionPoly, Union
	- volumeRegion, __volumePoly__
	- Projection
	
Classes:
	- Region
	- Polytope

Created by N. Ozay, 8/15/10 (necmiye@cds.caltech.edu)
Modified by U. Topcu, 8/16/10
"""

from numpy import *
from scipy import *
from cvxopt import blas, lapack, solvers
from cvxopt import matrix
import cvxopt
import pdb
from copy import deepcopy
from polyhedron import Vrep, Hrep



solvers.options['show_progress'] = False



def projectionV(polyIn,dimToElim): 
	flag = 1
	polyOut = []#Hrep([],[])
	try:
		ext = deepcopy(polyIn.generators)
		dimToKeep = [i1 for i1 in range(0,polyIn.A.shape[1]) if (i1 in dimToElim)==False]
		ext = ext[:,dimToKeep]
		polyOut = Vrep(ext)
		polyOut = Hrep(polyOut.A.copy(),polyOut.b.copy())
	except:
		flag = 0
	return flag, polyOut


def num_bin(N, places=8):
	def bit_at_p(N, p):
		''' find the bit at place p for number n '''
		two_p = 1 << p   # 2 ^ p, using bitshift, will have exactly one
				# bit set, at place p
		x = N & two_p    # binary composition, will be one where *both* numbers
				# have a 1 at that bit.  this can only happen 
				# at position p.  will yield  two_p if  N has a 1 at 
				# bit p
		return int(x > 0)
	bits = []
	for x in xrange(places):
		bits.append(bit_at_p(N, x))
	return bits
	
def __boundingBoxRegion__(reg1):
	l_bounds = inf*ones((reg1.list_poly[0].A.shape[1],1))
	u_bounds = -inf*ones((reg1.list_poly[0].A.shape[1],1))
	for i1 in range(0,len(reg1.list_poly)):
		aa = __findBoundingBox__(reg1.list_poly[i1])
		l_bounds = minimum(l_bounds,aa[0])
		u_bounds = maximum(u_bounds,aa[1])
	return l_bounds, u_bounds
	

def __findBoundingBox__(poly):
	"""Returns the bounding box of the input polytope"""
	A_arr = poly.A.copy()
	b_arr = poly.b.copy()
	neq, nx = A_arr.shape
	A = matrix(A_arr)
	b = matrix(b_arr)
	#In = matrix(eye(nx))
	l_bounds = zeros((nx,1))
	u_bounds = zeros((nx,1))
	for i in range(nx):
		c = zeros((nx,1))
		c[i] = 1
		c = matrix(c)
		sol=solvers.lp(c,A,b)
		l_bounds[i] = sol['primal objective']
		c = zeros((nx,1))
		c[i] = -1
		c = matrix(c)
		sol=solvers.lp(c,A,b)
		u_bounds[i] = -sol['primal objective']
	return l_bounds, u_bounds

		
		
def __isAdjacentPoly2__(poly1,poly2,M1n=None,M2n=None, tol=10.0e-6):
	"""Checks adjacency of two polytopes"""
	
	'''Warning: Chebyshev ball radius is at least tol/2 for adjacents, if you want to change
	the tol here, make sure you have a compatible tol in isNonEmptyInterior function'''
	
	A1_arr = poly1.A.copy()
	A2_arr = poly2.A.copy()
	b1_arr = poly1.b.copy()
	b2_arr = poly2.b.copy()
	# Find the candidate facets parameters
	neq1 = M1n.shape[1]
	neq2 = M2n.shape[1]
	dummy = dot(M1n.T,M2n)
	#print 'minimum is',dummy.min()
	cand = (dummy==dummy.min())
	for i in range(neq1):
		for j in range(neq2):
			if cand[i,j]: break
		if cand[i,j]: break

	b1_arr[i] += tol
	b2_arr[j] += tol	
	dummy = Polytope(concatenate((A1_arr,A2_arr)),concatenate((b1_arr,b2_arr)))
	if isNonEmptyInterior(dummy):
		return 1
	else:
		return 0

		
def isAdjacentRegion(reg1,reg2):
	"""Checks if two regions share a face"""
	for i in reg1.list_poly:
		for j in reg2.list_poly:
			adj = __isAdjacentPolyOuter__(i,j)
			if adj==1:
				return 1
	return 0
	
	
def __isAdjacentPolyOuter__(poly1,poly2):
	"""Eliminates a non-adjacent pair by checking hyperplane normals"""
	#start_time = time()
	M1 = concatenate((poly1.A,poly1.b),1).T
	#M1 = vstack([poly1.A.T,poly1.b]) #stack A and b
	M1row = 1/sqrt(sum(M1**2,0))
	M1n = dot(M1,diag(M1row)) #normalize the magnitude
	M2 = concatenate((poly2.A,poly2.b),1).T
	#M2 = vstack([poly2.A.T,poly2.b])
	M2row = 1/sqrt(sum(M2**2,0))
	M2n = dot(M2,diag(M2row))
	#time_elapsed = time()-start_time
	#print 'normal test', time_elapsed
	#print dot(M1n.T,M2n)
	if any(dot(M1n.T,M2n)<-0.99):
		#print 'here'
		return __isAdjacentPoly2__(poly1,poly2,M1n,M2n)
	else:
		return 0

	
def polySimplify(poly,nonEmptyBounded=1):
	"""
	Removes redundant inequalities in the hyperplane representation of the polytope with the algorithm described at http://www.ifor.math.ethz.ch/~fukuda/polyfaq/node24.html by solving one LP for each facet
	Warning:
	- nonEmptyBounded == 0 case is not tested much.
	"""
	 
	A_arr = poly.A.copy()
	b_arr = poly.b.copy()
	neq, nx = A_arr.shape
	# first eliminate the linearly dependent rows corresponding to the same hyperplane
	M1 = concatenate((poly.A,poly.b),1).T
	M1row = 1/sqrt(sum(M1**2,0))
	M1n = dot(M1,diag(M1row)) 
	M1n = M1n.T
	keep_row = []
	for i in range(neq):
		keep_i = 1
		for j in range(i+1,neq):
			if dot(M1n[i].T,M1n[j])>0.999999:
				keep_i = 0
		if keep_i:
			keep_row.append(i)
	
	A_arr = A_arr[keep_row]
	b_arr = b_arr[keep_row]
	neq, nx = A_arr.shape
	
	if nonEmptyBounded:
		if neq<=nx+1:
			return Polytope(A_arr,b_arr)
	
	# Now eliminate hyperplanes outside the bounding box
	if neq>3*nx:
		lb, ub = __findBoundingBox__(poly)
		#print lb.shape,ub.shape
		#cand = -(dot((A_arr>0)*A_arr,ub-lb)-(b_arr-dot(A_arr,lb).T).T<-1e-4)
		cand = -(dot((A_arr>0)*A_arr,ub-lb)-(b_arr-dot(A_arr,lb))<-1e-4)
		A_arr = A_arr[cand.squeeze()]
		b_arr = b_arr[cand.squeeze()]
	
	neq, nx = A_arr.shape
	if nonEmptyBounded:
		if neq<=nx+1:
			return Polytope(A_arr,b_arr)
	
	# Finally eliminate the rest of the redundancies
	del keep_row[:] #empty list
	for i in range(neq):
		A = matrix(A_arr)
		s = -matrix(A_arr[i])
		t = b_arr[i]
		b = matrix(b_arr)
		b[i] += 1
		sol=solvers.lp(s,A,b)
		if -sol['primal objective']>t:
			keep_row.append(i)
	polyOut = Polytope(A_arr[keep_row],b_arr[keep_row])
	return polyOut
			

def isNonEmptyInterior(poly1,tol=10.0e-7):
	"""Checks the radius of the Chebyshev ball to decide polytope degenaracy"""
	A = poly1.A.copy()
	nx = A.shape[1]
	A = matrix(c_[A,-sqrt(sum(A*A,1))])
	b = matrix(poly1.b)
	c = matrix( r_[zeros((nx,1)),[[1]]])
	sol=solvers.lp(c,A,b)
	if sol['status']=='primal infeasible':
		return False
	elif -sol['x'][nx]<=tol:
		#print 'radius',-sol['x'][nx]
		return False
	else:
		#print 'radius',-sol['x'][nx]
		return True

def Intersect(reg1,reg2):
	'''Three separate cases seem more efficient because of the inefficiency of out union function'''
	if len(reg2.list_poly)==1:
		RegOut = regionIntersectPoly(reg1,reg2.list_poly[0])
	elif len(reg1.list_poly)==1:
		RegOut = regionIntersectPoly(reg2,reg1.list_poly[0])
	else:
		RegOut = Region([],[])
		for aux in reg2.list_poly:
			aux1 = regionIntersectPoly(reg1,aux)
			RegOut = Union(RegOut,aux1)
	return RegOut
    	
def Diff(reg1,reg2):
	RegOut = regionDiffPoly(reg1,reg2.list_poly[0])
	for aux in reg2.list_poly[1:len(reg2.list_poly)]:
		aux1 = regionDiffPoly(reg1,aux)
		RegOut = Intersect(RegOut,aux1)
	return RegOut
    
		

def regionIntersectPoly(region1, poly1):
	"""Performs set intersection"""
	result_region = Region()
	for j in range(len(region1.list_poly)): #j polytope counter
		dummy = (Polytope(concatenate((region1.list_poly[j].A, poly1.A)),
			concatenate((region1.list_poly[j].b,poly1.b))))
		if isNonEmptyInterior(dummy): #non-empty interior
			dummy = polySimplify(dummy)
			result_region.list_poly.append(dummy)
	return result_region
	
def regionDiffPoly(region1, poly1):
	"""Performs set difference"""
	result_region = Region()
	A_poly = poly1.A.copy()
	b_poly = poly1.b.copy()
	for j in range(len(region1.list_poly)):
		dummy = (Polytope(concatenate((region1.list_poly[j].A,A_poly)),concatenate((region1.list_poly[j].b, b_poly))))
		if isNonEmptyInterior(dummy):
			num_halfspace = A_poly.shape[0]
			for k in range(pow(2,num_halfspace)-1): #loop to keep polytopes ofthe region disjoint
				signs = num_bin(k,places=num_halfspace)
				A_now = A_poly.copy()
				b_now = b_poly.copy()
				for l in range(len(signs)):
					if signs[l]==0:
						A_now[l] = -A_now[l]
						b_now[l] = -b_now[l]
				dummy = (Polytope(concatenate((region1.list_poly[j].A,A_now)),concatenate((region1.list_poly[j].b, b_now))))
				if isNonEmptyInterior(dummy):
					dummy = polySimplify(dummy)
					result_region.list_poly.append(dummy)
		else:	
			result_region.list_poly.append(deepcopy(region1.list_poly[j]))
	return result_region


'''def regionDiffPoly(region1, poly1):
	"""Performs set difference"""
	result_region = Region()
	for j in range(len(region1.list_poly)):
		A_poly = poly1.A.copy()
		b_poly = poly1.b.copy()
		num_halfspace = A_poly.shape[0]
		for k in range(pow(2,num_halfspace)-1): #loop to keep polytopes of the region disjoint
			signs = num_bin(k,places=num_halfspace)
			A_now = A_poly.copy()
			b_now = b_poly.copy()
			for l in range(len(signs)): 
				if signs[l]==0:
					A_now[l] = -A_now[l]
					b_now[l] = -b_now[l]
			dummy = (Polytope(concatenate((region1.list_poly[j].A, A_now)),concatenate((region1.list_poly[j].b, 				b_now))))
			if isNonEmptyInterior(dummy):
				dummy = polySimplify(dummy)
				result_region.list_poly.append(dummy)
	return result_region'''
	
def ChebyBallRad(poly1):
	radius = 0
	A = poly1.A
	nx = A.shape[1]
	H = cvxopt.matrix(c_[A,-sqrt(sum(A*A,1))])
	K = cvxopt.matrix(poly1.b)
	C = cvxopt.matrix( r_[zeros((nx,1)),[[1]]])
	solvers.options['show_progress'] = False
	sol=solvers.lp(C,H,K)
	if (sol['status'] !='primal infeasible'):
		radius = -sol['primal objective']
		xsol = double(sol['x'])
	return radius, xsol
	
	
def PolyUnionPoly(poly1,poly2):
	RegOut = Region([poly1,poly2],[])
	return RegOut
	
	
def RegionUnionPoly(reg1,poly1):
	aux = reg1.list_poly
	aux.append(poly1)
	regOut = Region(aux,[])
	return regOut
	
	
def Union(reg1,reg2):
	aux = reg1.list_poly
	aux.extend(reg2.list_poly)
	regOut = Region(aux,[])
	return regOut

def volumeRegion(reg1):
	vol = 0
	for i1 in range(0,len(reg1.list_poly)):
		vol = vol + __volumePoly__(reg1.list_poly[i1])
	return vol
    
    
def __volumePoly__(poly1):
	n = poly1.A.shape[1]
	if n == 1:
		N = 50
	elif n == 2:
		N = 500
	elif n ==3:
		N = 3000
	else:
		N = 10000
	
	l_b, u_b = __findBoundingBox__(poly1)
	x = tile(l_b,(1,N)) + rand(n,N)*tile(u_b-l_b,(1,N))
	aux = dot(poly1.A,x)-tile(poly1.b,(1,N))
	aux = nonzero(all(((aux < 0)==True),0))[0].shape[0]
	vol = prod(u_b-l_b)*aux/N
	return vol

	
def saveListRegMat(Xlist,fn):
	import scipy.io as sio
	data = {}
	data['k'] = len(Xlist)
	for k in range(0,len(Xlist)):
		X = Xlist[k]
		data['Nk'+str(k)] = [len(X.list_poly)]
		for i1 in range(0,len(X.list_poly)):
			data['Ak'+str(k)+'i'+str(i1)] = deepcopy(X.list_poly[i1].A)
			data['bk'+str(k)+'i'+str(i1)] = deepcopy(X.list_poly[i1].b)
	sio.savemat(fn,data)

def saveMat(A,b):
	import scipy.io as sio
	data = {}
	data['A'] = A
	data['b'] = b
	sio.savemat('test.mat',data)
	
def projection(polyIn,dimToElim):
	
	'''OrigPoly1 = Hrep(polyIn.A,polyIn.b)
	ddd = Vrep(OrigPoly1.generators)
	ReachFrom1 = Hrep(ddd.A,ddd.b)
	auxx = zeros((ReachFrom1.A.shape[0],1))
	auxx[0:,0] = ReachFrom1.b
	polyIn = Polytope(ReachFrom1.A,auxx)'''
	polyIn = polySimplify(polyIn)
	Aaux = polyIn.A
	baux = polyIn.b
	ind = []
	for i1 in dimToElim:  
		#pdb.set_trace()
		#print i1, Aaux.shape
		a = Aaux[:,i1]
		nc = Aaux.shape[0]
		positive = [i2 for i2 in range(0,nc) if a[i2] > 1e-9]
		negative = [i2 for i2 in range(0,nc) if a[i2] < -1e-9]
		null = [i2 for i2 in range(0,nc) if (i2 in positive)==False if (i2 in negative)==False]
					
		nr = len(null) + len(positive)*len(negative)
		nc = Aaux.shape[0]
		C = zeros((nr,nc))
	   
		row = 0
		for j in positive:
			for k in negative:       
				C[row,j] = -a[k]
				C[row,k] = a[j]
				row += 1    
			   
		for j in null:
			C[row,j] = 1
			row += 1
		#pdb.set_trace()
		Aaux = dot(C,Aaux)
		baux = dot(C,baux)
		ind.append(i1)
		indK = [i4 for i4 in range(0,Aaux.shape[1]) if (i4 in ind)==False]
		#pdb.set_trace()
		auxP = Polytope(Aaux[:,indK],baux)
		auxP = polySimplify(auxP)
		#pdb.set_trace()
		Aaux = zeros((auxP.A.shape[0],Aaux.shape[1]))
		Aaux[:,indK] = deepcopy(auxP.A)
		baux = deepcopy(auxP.b)
	
	# We probably have an extra call to simplify here. To be sorted later....
	indK = [i5 for i5 in range(0,Aaux.shape[1]) if (i5 in dimToElim)==False]
	Aaux = Aaux[:,indK]	
	auxP = Polytope(Aaux,baux)
	#pdb.set_trace()
	auxP = polySimplify(auxP)
	#pdb.set_trace()
	return auxP


class Region:
	"""Region class with following fields

	-list_poly: proposition preserving regions	
	-list of propositions: binary vector that show which prop holds in which region
	"""
	
	def __init__(self, list_poly=[], list_prop=[]):
		self.list_poly = list_poly[:]
		self.list_prop = list_prop[:]

		
class Polytope:
	"""Polytope class with following fields
	
	-A: a numpy array for the hyperplane normals in hyperplane representation of a polytope
	-b:  a numpy array for the hyperplane offsets in hyperplane representation of a polytope
	"""
	def __init__(self,A,b):
		self.A = A.copy()
		self.b = b.copy()