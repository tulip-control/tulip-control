
""" polytope_computations.py --- A computational geometry module for polytope computations

Necmiye Ozay (necmiye@cds.caltech.edu)
August 15, 2010
"""

from numpy import *
from cvxopt import blas, lapack, solvers
from cvxopt import matrix

solvers.options['show_progress'] = False


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
	M1 = vstack([poly1.A.T,poly1.b]) #stack A and b
	M1row = 1/sqrt(sum(M1**2,0))
	M1n = dot(M1,diag(M1row)) #normalize the magnitude
	M2 = vstack([poly2.A.T,poly2.b])
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
	
	Warning:Should be used for bounded polytopes with nonempty interior
	"""
	 
	A_arr = poly.A.copy()
	b_arr = poly.b.copy()
	neq, nx = A_arr.shape
	
	# first eliminate the linearly dependent rows corresponding to the same hyperplane
	M1 = vstack([poly.A.T,poly.b]) #stack A and b
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
		cand = -(dot((A_arr>0)*A_arr,ub-lb)-(b_arr-dot(A_arr,lb).T).T<-1e-4)
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
	return result_region


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