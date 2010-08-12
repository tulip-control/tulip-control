
""" prop2part.py --- Proposition preserving partition module

Necmiye Ozay (necmiye@cds.caltech.edu)
August 7, 2010
"""

from numpy import *
import numpy
from polyhedron import Vrep, Hrep
from time import time
import copy
from cvxopt import blas, lapack, solvers
from cvxopt import matrix

solvers.options['show_progress'] = False


def matrixRank(matIn, tol=10.0e-12):
	"""
	Finds the rank of the matrix (by thresholding singular values)
	"""
	u, d, v = linalg.svd(matIn)
	return sum(d>=tol)


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
	

def __isAdjacentPolyOld__(poly1,poly2,M1n=None,M2n=None):
	"""Checks adjacency of two polytopes"""
	#start_time = time()
	dummy = Hrep(concatenate((poly1.A,poly2.A)),concatenate((poly1.b,poly2.b)))
	if dummy.generators.shape[0]>0:		
		#time_elapsed = time()-start_time
		#print 'feasibility', time_elapsed
		return 1
	else:
		return 0
		
		
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
	dummy = Hrep(concatenate((A1_arr,A2_arr)),concatenate((b1_arr,b2_arr)))
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


def __polySimplifyOld(poly): #check if there is a better way of eliminating redundancy
	'''Warning:Should be used for polytopes with nonempty interior'''
	dummy = Vrep(poly.generators)
	poly = Hrep(dummy.A,dummy.b)
	return poly
	
	
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
			return Hrep(A_arr,b_arr)
	
	# Now eliminate hyperplanes outside the bounding box
	if neq>3*nx:
		lb, ub = __findBoundingBox__(poly)
		cand = -(dot((A_arr>0)*A_arr,ub-lb)-(b_arr-dot(A_arr,lb).T).T<-1e-4)
		A_arr = A_arr[cand.squeeze()]
		b_arr = b_arr[cand.squeeze()]
	
	neq, nx = A_arr.shape
	if nonEmptyBounded:
		if neq<=nx+1:
			return Hrep(A_arr,b_arr)
	
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
	polyOut = Hrep(A_arr[keep_row],b_arr[keep_row])
	return polyOut
	

def __isNonEmptyInteriorOld(poly1):
	#Function to check if the polytope is degenerate
	num_var = poly1.A.shape[1]
	dummy = poly1.generators.copy()
	num_gen = dummy.shape[0]
	if num_gen<=num_var:
		return False
	elif num_var<=2: #number of generators>2 in 2d is sufficient for non-emptiness
		return True
	else: #check rank
		#print 'HERE'
		moveOrigin = dummy[0]
		points = dummy[1:]
		points = points-tile(moveOrigin,(num_gen-1,1))
		if matrixRank(points)<num_var:
			return False
		else:
			return True
			

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
		dummy = (Hrep(concatenate((region1.list_poly[j].A, poly1.A)),
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
			dummy = (Hrep(concatenate((region1.list_poly[j].A, A_now)),concatenate((region1.list_poly[j].b, 				b_now))))
			if isNonEmptyInterior(dummy):
				dummy = polySimplify(dummy)
				result_region.list_poly.append(dummy)
	return result_region


def prop2part2(state_space, cont_props):
	"""Main function that takes a domain (state_space) and a list of propositions (cont_props), and
	returns a proposition preserving partition of the state space"""
	num_props = len(cont_props)
	list_regions = []
	first_poly = [] #Initial Region's list_poly atribute 
	first_poly.append(state_space)
	list_regions.append(Region(list_poly=first_poly))
	mypartition = PropPreservingPartition(domain=copy.deepcopy(state_space), num_prop=num_props, list_region=list_regions)
	for prop_count in range(num_props):
		num_reg = mypartition.num_regions
		prop_holds_reg = []
		prop_holds_poly = []
		prop_not_holds_poly = []
		for i in range(num_reg): #i region counter
			region_now = mypartition.list_region[i].list_poly[:];#if [:] is omitted, acts like pointer
			#loop for prop holds
			prop_holds_reg.append(0)
			prop_holds_poly[:] = []
			list_prop_now = mypartition.list_region[i].list_prop[:]
			for j in range(len(region_now)): #j polytope counter
				prop_holds_poly.append(0)
				dummy = (Hrep(concatenate((region_now[j].A, cont_props[prop_count].A)),
					concatenate((region_now[j].b,cont_props[prop_count].b))))
				if isNonEmptyInterior(dummy):
					#dummy = polySimplify(dummy)
					mypartition.list_region[i].list_poly[j] = polySimplify(dummy)
					prop_holds_reg[-1] = 1
					prop_holds_poly[-1] = 1
			count = 0
			for hold_count in range(len(prop_holds_poly)):
				if prop_holds_poly[hold_count]==0:
					mypartition.list_region[i].list_poly.pop(hold_count-count)
					count+=1
			if len(mypartition.list_region[i].list_poly)>0:
				mypartition.list_region[i].list_prop.append(1)
			#loop for prop does not hold
			mypartition.list_region.append(Region(list_poly=[],list_prop=list_prop_now))
			for j in range(len(region_now)):
				valid_props = cont_props[prop_count] #eliminateRedundantProps(region_now[j],cont_props[prop_count])
				A_prop = valid_props.A.copy()
				b_prop = valid_props.b.copy()
				num_halfspace = A_prop.shape[0]
				for k in range(pow(2,num_halfspace)-1):
					signs = num_bin(k,places=num_halfspace)
					A_now = A_prop.copy()
					b_now = b_prop.copy()
					for l in range(len(signs)): 
						if signs[l]==0:
							A_now[l] = -A_now[l]
							b_now[l] = -b_now[l]
					dummy = (Hrep(concatenate((region_now[j].A, A_now)),concatenate((region_now[j].b, 							b_now))))
					if isNonEmptyInterior(dummy):
						#dummy = polySimplify(dummy)
						mypartition.list_region[-1].list_poly.append(polySimplify(dummy))
			if len(mypartition.list_region[-1].list_poly)>0:
				mypartition.list_region[-1].list_prop.append(0)
			else:
				mypartition.list_region.pop()
		count = 0
		for hold_count in range(len(prop_holds_reg)):
			if prop_holds_reg[hold_count]==0:
				mypartition.list_region.pop(hold_count-count)
				count+=1
		num_reg = len(mypartition.list_region)
		mypartition.num_regions = num_reg
	adj = numpy.zeros((num_reg,num_reg),int8)
	for i in range(num_reg):
		for j in range(i+1,num_reg):
			adj[i,j] = isAdjacentRegion(mypartition.list_region[i],mypartition.list_region[j])
	adj =  adj+adj.T
	mypartition.adj = adj.copy()
	return mypartition


class Region:
	"""Region class with following fields

	-list_poly: proposition preserving regions	
	-list of propositions: binary vector that show which prop holds in which region
	"""
	
	def __init__(self, list_poly=[], list_prop=[]):
		self.list_poly = list_poly[:]
		self.list_prop = list_prop[:]


class PropPreservingPartition:
	"""Partition class with following fields
	
	-domain: the domain we want to partition, type: polytope
	-num_prop: number of propositions
	-list_region: proposition preserving regions, type: list of Region
	-num_regions: length of the above list
	-adj: a matrix showing which regions are adjacent
	-trans: a matrix showing which region is reachable from which region
	-list_prop_symbol: list of symbols of propositions
	"""
	
	def __init__(self, domain=None, num_prop=0, list_region=[], num_regions=0, adj=0, trans=0, list_prop_symbol=None):
		self.domain = domain
		self.num_prop = num_prop
		self.list_region = list_region[:]
		self.num_regions = len(list_region)
		self.adj = adj
		self.trans = trans
		self.list_prop_symbol = list_prop_symbol
		
#class Polytope:
#	def __init__(self):
#		self.A = A
#		self.b = b


#def test():
if __name__ == "__main__":
	start_time = time()
	domain = array([[0., 2.],[0., 2.]])

	domain_poly_A = array(vstack([eye(2),-eye(2)]))
	domain_poly_b = array(r_[domain[:,1],-domain[:,0]])
	state_space = Hrep(domain_poly_A, domain_poly_b)

	cont_props = []
	
	A0 = array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.], [0.,-1.]])
	b0 = array([1., 0., 1., 0., -0.])
	#A0 = array([[1., 0.], [-1., 0.]])
	#b0 = array([1., 0.])
	cont_props.append(Hrep(A0, b0))
	#p1 = polySimplify(Hrep(A0, b0))
	
	A1 = array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
	b1 = array([1., 0., 2., -1.])
	#A1 = array([[-1., 0.]])
	#b1 = array([-1.5])
	cont_props.append(Hrep(A1, b1))
	
	A1 = array([[-1., 0.]])
	b1 = array([-1.7])
	cont_props.append(Hrep(A1, b1))
	
	A1 = array([[-1., 0.]])
	b1 = array([-1.9])
	cont_props.append(Hrep(A1, b1))
	
	A1 = array([[-1., 0.]])
	b1 = array([-0.5])
	cont_props.append(Hrep(A1, b1))
	
	A1 = array([[-1., 0.]])
	b1 = array([-1.8])
	cont_props.append(Hrep(A1, b1))

	A1 = array([[1., 1.]])
	b1 = array([2])
	cont_props.append(Hrep(A1, b1))
	
	A2 = array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
	b2 = array([2., -1., 1., 0.])
	cont_props.append(Hrep(A2, b2))

	A3 = array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
	b3 = array([2., -1., 2., -1.])
	cont_props.append(Hrep(A3, b3))
	
	mypartition = prop2part2(state_space, cont_props)
	
	#print len(mypartition.list_region)
	A4 = array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
	b4 = array([0.5, 0., 0.5, 0.])
	poly1 = Hrep(A4,b4)

	r1 = regionDiffPoly(mypartition.list_region[3],poly1)
	
	verbose = 1
	if verbose:
		for j in range(len(r1.list_poly)):
			x = r1.list_poly[j].generators
			print x
	
	xx = mypartition.adj
	print xx
	time_elapsed = time()-start_time
	
	if verbose:
		print 'time', time_elapsed
		#print sys.getrefcount(numpy.dtype('float64'))
		print '\nAdjacency matrix:\n', mypartition.adj
	
		for i in range(mypartition.num_regions):
			print i+1,"th region has the following generators"
			for j in range(len(mypartition.list_region[i].list_poly)):
				print 'polytope',j,'in region',i+1
				print mypartition.list_region[i].list_poly[j].generators
			print 'region',i+1,'proposition list is',mypartition.list_region[i].list_prop,'\n'
			
	visualization = 1
	if visualization:
		nx = domain_poly_A.shape[1]
		if nx!=2:
			print "Can only plot 2D partitions!"
		elif 0:
			import pylab
			colors = ['bo','go','ro','mo','co','ko','yo']
			num_pnt = 1000
			points = 2*random.random((2,num_pnt))
			pylab.figure(1, facecolor='w')
			for num in range(num_pnt):
				for i in range(mypartition.num_regions):
					for j in range(len(mypartition.list_region[i].list_poly)):
						p = mypartition.list_region[i].list_poly[j]
						if all(dot(p.A,points[:,num]) <= p.b):
							pylab.plot(points[0,num],points[1,num],colors[(i%7)])
			#for i in range(100):
			#	pylab.text(x_coord[i], y_coord[i], i)
			pylab.title('My Proposition Preserving Partition')
		else:
			import pylab
			colors = ['b','g','r','m','c','w','y', '0.8','0.3']
			no_c = len(colors)
			pylab.figure(1, facecolor='w')
			for i in range(mypartition.num_regions):
				for j in range(len(mypartition.list_region[i].list_poly)):
					xx = mypartition.list_region[i].list_poly[j].generators
					pylab.fill(xx[:,0],xx[:,1],colors[(i%no_c)],edgecolor='none')
					xx = vstack((xx[(1,0),:],xx[2:,:]))
					pylab.fill(xx[:,0],xx[:,1],colors[(i%no_c)],edgecolor='none')
					cntr = sum(xx,0)/xx.shape[0]
					pylab.text(cntr[0],cntr[1],str(i+1))
			pylab.title('My Proposition Preserving Partition')
			
			
def test2():
#if __name__ == "__main__":
	count =1
	for i in range(1):
		test()
		count +=1
	print count-1, 'repeated tests'
	
