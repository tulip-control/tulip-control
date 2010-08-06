from numpy import *
from polyhedron import Vrep, Hrep
from time import time
#import copy
from cvxopt import blas, lapack, solvers
from cvxopt import matrix

solvers.options['show_progress'] = True


def matrixRank(matIn, tol=10.0e-12):
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


def isAdjacentPoly(poly1,poly2):
	#start_time = time()
	dummy = Hrep(concatenate((poly1.A,poly2.A)),concatenate((poly1.b,poly2.b)))
	if dummy.generators.shape[0]>0:		
		#time_elapsed = time()-start_time
		#print 'feasibility', time_elapsed
		return 1
	else:
		return 0

		
def isAdjacentRegion(reg1,reg2):
	for i in reg1.list_poly:
		for j in reg2.list_poly:
			adj = isAdjacentPoly2(i,j)
			if adj==1:
				return 1
	return 0
	
def isAdjacentPoly2(poly1,poly2):
	#start_time = time()
	M1 = vstack([poly1.A.T,poly1.b])
	M1row = 1/sqrt(sum(M1**2,0))
	M1n = dot(M1,diag(M1row))
	M2 = vstack([poly2.A.T,poly2.b])
	M2row = 1/sqrt(sum(M2**2,0))
	M2n = dot(M2,diag(M2row))
	#time_elapsed = time()-start_time
	#print 'normal test', time_elapsed
	#print dot(M1n.T,M2n)
	if any(dot(M1n.T,M2n)<-0.999999999):
		#print 'here'
		return isAdjacentPoly(poly1,poly2)
	else:
		return 0


def polySimplify(poly): #check if there is a better way of eliminating redundancy
	'''Warning:Should be used for polytopes with nonempty interior'''
	dummy = Vrep(poly.generators)
	poly = Hrep(dummy.A,dummy.b)
	return poly


def isNonEmptyInterior(poly1):
	"""Function to check if the polytope is degenerate"""
	num_var = poly1.A.shape[1]
	dummy = poly1.generators[:]
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

def isNonEmptyInterior2(poly1,tol=10.0e-12):
	A = matrix(poly1.A)
	b = matrix(poly1.b)
	c = matrix(zeros((A.shape[1],1)))
	sol=solvers.lp(c,A,b)
	if sol['status']=='primal infeasible':
		return False
	elif any(array(A*sol['x']-b)>=-tol):
		return False
	else:
		return True
		

def regionIntersectPoly(region1, poly1):
	"""Function to perform set intersection"""
	result_region = Region()
	for j in range(len(region1.list_poly)): #j polytope counter
		dummy = (Hrep(concatenate((region1.list_poly[j].A, poly1.A)),
			concatenate((region1.list_poly[j].b,poly1.b))))
		if isNonEmptyInterior(dummy): #non-empty interior
			dummy = polySimplify(dummy)
			result_region.list_poly.append(dummy)
	return result_region


def regionDiffPoly(region1, poly1):
	"""Function to perform set difference"""
	result_region = Region()
	for j in range(len(region1.list_poly)):
		A_poly = poly1.A
		b_poly = poly1.b
		num_halfspace = A_poly.shape[0]
		for k in range(pow(2,num_halfspace)-1): #loop to keep polytopes of the region disjoint
			signs = num_bin(k,places=num_halfspace)
			A_now = A_poly*1
			b_now = b_poly*1
			for l in range(len(signs)): 
				if signs[l]==0:
					A_now[l] = -A_now[l]
					b_now[l] = -b_now[l]
			dummy = (Hrep(concatenate((region1.list_poly[j].A, A_now)),concatenate((region1.list_poly[j].b, 				b_now))))
			if isNonEmptyInterior(dummy):
				dummy = polySimplify(dummy)
				result_region.list_poly.append(dummy)
	return result_region
							

def eliminateRedundantProps(poly,prop):
	return prop  # Remember to implement this!


def prop2part2(state_space, cont_props):
	num_props = len(cont_props)
	list_regions = []
	first_poly = [] #Initial Region's list_poly atribute 
	first_poly.append(state_space)
	list_regions.append(Region(list_poly=first_poly))
	partition = PropPreservingPartition(domain=state_space, num_prop=num_props, list_region=list_regions)
	for prop_count in range(num_props):
		num_reg = partition.num_regions
		prop_holds_reg = []
		prop_holds_poly = []
		prop_not_holds_poly = []
		for i in range(num_reg): #i region counter
			region_now = partition.list_region[i].list_poly[:];#if [:] is omitted, acts like pointer
			#loop for prop holds
			prop_holds_reg.append(0)
			prop_holds_poly[:] = []
			list_prop_now = partition.list_region[i].list_prop[:]
			for j in range(len(region_now)): #j polytope counter
				prop_holds_poly.append(0)
				dummy = (Hrep(concatenate((region_now[j].A, cont_props[prop_count].A)),
					concatenate((region_now[j].b,cont_props[prop_count].b))))
				if isNonEmptyInterior(dummy):
					dummy = polySimplify(dummy)
					partition.list_region[i].list_poly[j] = dummy
					prop_holds_reg[-1] = 1
					prop_holds_poly[-1] = 1
			count = 0
			for hold_count in range(len(prop_holds_poly)):
				if prop_holds_poly[hold_count]==0:
					partition.list_region[i].list_poly.pop(hold_count-count)
					count+=1
			if len(partition.list_region[i].list_poly)>0:
				partition.list_region[i].list_prop.append(1)
			#loop for prop does not hold
			partition.list_region.append(Region(list_poly=[],list_prop=list_prop_now))
			for j in range(len(region_now)):
				valid_props = eliminateRedundantProps(region_now[j],cont_props[prop_count])
				A_prop = valid_props.A
				b_prop = valid_props.b
				num_halfspace = A_prop.shape[0]
				for k in range(pow(2,num_halfspace)-1):
					signs = num_bin(k,places=num_halfspace)
					A_now = A_prop*1
					b_now = b_prop*1
					for l in range(len(signs)): 
						if signs[l]==0:
							A_now[l] = -A_now[l]
							b_now[l] = -b_now[l]
					dummy = (Hrep(concatenate((region_now[j].A, A_now)),concatenate((region_now[j].b, 							b_now))))
					if isNonEmptyInterior(dummy):
						dummy = polySimplify(dummy)
						partition.list_region[-1].list_poly.append(dummy)
			if len(partition.list_region[-1].list_poly)>0:
				partition.list_region[-1].list_prop.append(0)
			else:
				partition.list_region.pop()
		count = 0
		for hold_count in range(len(prop_holds_reg)):
			if prop_holds_reg[hold_count]==0:
				partition.list_region.pop(hold_count-count)
				count+=1
		num_reg = len(partition.list_region)
		partition.num_regions = num_reg
	"""
	reg_list_now = copy.deepcopy(partition.list_region)
	adj = zeros((num_reg,num_reg))
	for i in range(num_reg):
		for j in range(i+1,num_reg):
			adj[i,j] = isAdjacentRegion(reg_list_now[i],reg_list_now[j])
	partition.adj=adj+adj.T
	"""
	adj = zeros((num_reg,num_reg))
	for i in range(num_reg):
		for j in range(i+1,num_reg):
			adj[i,j] = isAdjacentRegion(partition.list_region[i],partition.list_region[j])
	partition.adj=adj+adj.T
	return partition


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
		-adjacency list: list of each regions neighbors
		-list_prop_symbol: list of symbols of propositions
	"""
	
	def __init__(self, domain=None, num_prop=0, list_region=[], num_regions=0, adj=0, list_prop_symbol=None):
		self.domain = domain
		self.num_prop = num_prop
		self.list_region = list_region[:]
		self.num_regions = len(list_region)
		self.adj = adj
		self.list_prop_symbol = list_prop_symbol


if __name__ == "__main__":
	start_time = time()
	domain = array([[0., 2.],[0., 2.]])

	domain_poly_A = mat(vstack([eye(2),-eye(2)]))
	domain_poly_b = mat(r_[domain[:,1],-domain[:,0]])
	state_space = Hrep(domain_poly_A, domain_poly_b)

	cont_props = []
	
	#A0 = [[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]
	#b0 = [1., 0., 1., 0.]
	A0 = [[1., 0.], [-1., 0.]]
	b0 = [1., 0.]
	cont_props.append(Hrep(A0, b0))

	#A1 = [[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]
	#b1 = [1., 0., 2., -1.]
	A1 = [[-1., 0.]]
	b1 = [-1.5]
	cont_props.append(Hrep(A1, b1))
	
	A1 = [[-1., 0.]]
	b1 = [-1.7]
	cont_props.append(Hrep(A1, b1))
	
	A1 = [[-1., 0.]]
	b1 = [-1.9]
	cont_props.append(Hrep(A1, b1))
	
	A1 = [[-1., 0.]]
	b1 = [-0.5]
	cont_props.append(Hrep(A1, b1))
	
	A1 = [[-1., 0.]]
	b1 = [-1.8]
	cont_props.append(Hrep(A1, b1))

	A2 = [[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]
	b2 = [2., -1., 1., 0.]
	cont_props.append(Hrep(A2, b2))

	A3 = [[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]
	b3 = [2., -1., 2., -1.]
	cont_props.append(Hrep(A3, b3))
	
	partition = prop2part2(state_space, cont_props)
	
	print len(partition.list_region)
	A4 = [[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]
	b4 = [0.5, 0., 0.5, 0.]
	#poly1 = Hrep(A4,b4)

	#r1 = regionDiffPoly(partition.list_region[3],poly1)
	#for j in range(len(r1.list_poly)):
	#	x = r1.list_poly[j].generators
	#	print'\n generators of polytope',j, 'of resulting region'
	#	for row in x:
	#		print row
	
	#xx = partition.adj
	#print xx
	time_elapsed = time()-start_time
	print 'time', time_elapsed
	print partition.num_prop
	
	#del partition
	#del cont_props
	#print '\nAdjacency matrix:\n', partition.adj
	
	#for i in range(partition.num_regions):
	#	print i+1,"th region has the following generators"
	#	for j in range(len(partition.list_region[i].list_poly)):
	#		print 'polytope',j,'in region',i+1
	#		print partition.list_region[i].list_poly[j].generators
	#	print 'region',i+1,'proposition list is',partition.list_region[i].list_prop,'\n'
	"""
	reg_list_now = copy.deepcopy(partition.list_region)
	adj = zeros((num_reg,num_reg))
	for i in range(num_reg):
		for j in range(i+1,num_reg):
			adj[i,j] = isAdjacentRegion(reg_list_now[i],reg_list_now[j])
	partition.adj=adj+adj.T"""
