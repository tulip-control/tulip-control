#
# Copyright (c) 2011 by California Institute of Technology
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 
# 3. Neither the name of the California Institute of Technology nor
#    the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CALTECH
# OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
#
#
#  Reference: 
#
#  C. Bradford Barber, David P. Dobkin, Hannu Huhdanpa,
#  The quickhull algorithm for convex hulls,
#  ACM Transactions on Mathematical Software,
#  Volume 22 Issue 4, Dec. 1996,

"""
Implementation of the Quickhull algorithm for computing convex hulls.

Created by P. Nilsson, 8/2/11 
"""

import numpy as np

numpy_ver = np.version.version.split('.')
numpy_ver = float(numpy_ver[0] + str('.') + numpy_ver[1])

class Facet: 
    """A class describing a facet (n-1 dimensional face) of an n dimensional polyhedron with the 
    following fields:
    N.B. Polyhedron is assumed to contain the origin (inside and outside are defined accordingly)
    
    - `outside`: a list of points outside the facet
    - `vertices`: the vertices of the facet in a n*n matrix where each row denotes a vertex
    - `neighbors`: a list of other facets with which the facet shares n-1 vertices
    - `normal`: a normalized vector perpendicular to the facet, pointing "out"
    - `distance`: the normal distance of the facet from origo
    """
    
    
    def __init__(self,points):
        
        self.outside = []
        self.vertices = points
        self.neighbors = []
        self.normal = None
        self.distance = None
        
        sh = np.shape(points)
        A0 = np.hstack([points,np.ones([sh[0],1])])
        b0 = np.zeros([sh[0],1])
        b = np.vstack([np.zeros([sh[0],1]),1])
        c = np.zeros(sh[1]+1)
        c[-1] = -1.
        A = np.vstack([A0,c])
        sol = np.linalg.solve(A,b)

        xx = sol[0:sh[1]]
        mult = np.sqrt(np.sum(xx**2))
        n = xx/mult
        d = sol[sh[1]]/mult
        
        # Test to check that n is >outer< normal
        if np.sum(n.flatten()*points[0]) < 0:
            n = -n
        self.normal = n
        self.distance = -d
    
    def get_furthest(self):
        """Returns the point in outside the furthest away from the facet"""
        N = len(self.outside)
        if N == 1:
            ret = self.outside[0]
            del self.outside[0]
            return ret
        else:
            p0 = self.outside[0]
            inddel = 0
            for i in range(1,N):
                if p0.distance < self.outside[i].distance:
                    p0 = self.outside[i]
                    inddel = i
            del self.outside[inddel]
            return p0
            
class Outside_point:
    """Class containing the coordinates of a point and the distance to the
    facet to which the point is assigned"""
    def __init__(self, coordinates, distance):
        self.distance = distance
        self.coordinates = coordinates

def distance(p,fac1):
    """Calculate the distance from a facet to a point"""
    n = fac1.normal
    d = fac1.distance
    return np.sum(n.flatten()*p.flatten()) - d
    
def is_neighbor(fac1, fac2, abs_tol=1e-7):
    """Determine if two facets share d-1 vertices"""
    vert1 = fac1.vertices
    vert2 = fac2.vertices
    sh = np.shape(vert1)
    dim = sh[1]
    
    same = 0
    for ii in range(dim):
        v1 = vert1[ii,:]
        for jj in range(dim):
            v2 = vert2[jj,:]           
            if np.all(np.abs(v1-v2) < abs_tol):
                same += 1
                break
    return (same == dim - 1)                

def quickhull(POINTS, abs_tol=1e-7):
    """Compute the convex hull of a set of points
    
    Input:
    - `points`: a n*d np array where each row denotes a point
    
    Output:
    - `A,b,vertices`: `A` and `b describing the convex hull polytope as
                      A x <= b (H-representation). `vertices is a list of all 
                      the points in the convex hull (V-representation).
    """
    
    POINTS = POINTS.astype('float')
    sh = np.shape(POINTS)
    dim = sh[1]
    npt = sh[0]
    
    if npt <= dim:
        # Convex hull is empty
        return np.array([]),np.array([]),None
    
    # Check if convex hull is fully dimensional
    u, s, v = np.linalg.svd(np.transpose(POINTS - POINTS[0,:]))
    rank = np.sum(s > 1e-15)
    
    if rank < dim:
        print("Warning: " +
            "convex hull is not fully dimensional, returning empty polytope")
        return np.array([]),np.array([]),None
    
    # Choose starting simplex by choosing maximum points in random directions
    rank = 0
    while rank < dim:
        ind = []
        d = 0
        while d < dim + 1:
            rand = np.random.rand(dim) - 0.5
            test = np.dot(POINTS,rand)
            index = np.argsort(test)
            i = 0
            b = index[i] in ind
            while b:
                i += 1
                b = index[i] in ind
            ind.append(index[i])
            d += 1
        startsimplex = POINTS[ind,:]  
        u, s, v = np.linalg.svd(np.transpose(startsimplex - startsimplex[0,:]))
        rank = np.sum(s > 1e-10)
    
    unassigned_points = POINTS[np.setdiff1d(range(npt),ind),:]
                
    # Center starting simplex around origo by translation
    xc = np.zeros(dim)
    for ii in range(dim+1):
        xc += startsimplex[ii,:]/(dim+1)
    
    startsimplex = startsimplex - xc
    unassigned_points = unassigned_points - xc
    
    Forg = []
    F = []
    R = []
    
    for i in range(dim+1):
        ind = np.setdiff1d(np.arange(dim+1),[i])
        fac = Facet(startsimplex[ind,:]) 
        Forg.append(fac)     

    if npt == dim+1:  
        # If only d+1 facets, we already have convex hull
        num = len(Forg)
        A = np.zeros([num,dim])
        b = np.zeros([num,1])
        vert = np.zeros([num*dim,dim])
        for ii in range(num):
            vert[ np.ix_(range(ii*dim,(ii+1)*dim)), : ] = Forg[ii].vertices + xc
            A[ii,:] = Forg[ii].normal.flatten()
            b[ii] = Forg[ii].distance
        if numpy_ver > 1.59:
            vert = np.unique(vert.view([('',vert.dtype)]*vert.shape[1])).view(vert.dtype).reshape(-1,vert.shape[1])
        else:
            vert = np.unique1d(vert.view([('',vert.dtype)]*vert.shape[1])).view(vert.dtype).reshape(-1,vert.shape[1])
        b = b.flatten() + np.dot(A,xc.flatten())  
        return A,b.flatten(),vert
        
    for ii in range(len(Forg)):               
        # In the starting simplex, all facets are neighbors
        for jj in range(ii+1,len(Forg)):
            fac1 = Forg[ii]
            fac2 = Forg[jj]
            ind = np.setdiff1d(np.arange(dim+1),[ii,jj])
            fac1.neighbors.append(fac2)
            fac2.neighbors.append(fac1)       
             
    for fac1 in Forg:
        # Assign outside points to facets        
        npt = np.shape(unassigned_points)[0]
        keep_list = np.ones(npt, dtype=int)
                
        for ii in range(npt):
            if npt == 1:
                pu = unassigned_points
            else:
                pu = unassigned_points[ii,:]
            
            d = distance(pu,fac1)
            if d > abs_tol:
                op = Outside_point(pu.flatten(), d)
                fac1.outside.append(op)
                keep_list[ii] = 0
        
        if len(fac1.outside) > 0:
            F.append(fac1)
        ind = np.nonzero(keep_list)[0]
    
        if len(ind) > 0:
            unassigned_points = unassigned_points[ind,:]
        else:
            unassigned_points = None
            break
        
    # We now have a collection F of facets with outer points!
    # Selecting the point furthest away from a facet
    while len(F) > 0:
        
        facet = F[0]
        p = facet.get_furthest().coordinates
                    
        V = []      # Initialize visible set
                    # Want to add all facets that are visible from p
        Ncoll = []  # Set of unvisited neighbors
        visited = []
                
        V.append(facet)             # facet itself is visible by definition
        visited.append(facet)       # facet is visited
        for N in facet.neighbors:   # add all neighbors for visit
            Ncoll.append(N)
        
        while len(Ncoll) > 0:       # Visit all neighbours
            N = Ncoll[0]
            visited.append(N)
            
            if distance(p,N) > abs_tol:
                V.append(N)
                for neighbor in N.neighbors:
                    if (neighbor not in visited) & (neighbor not in Ncoll):
                        Ncoll.append(neighbor)
            del Ncoll[0]
        
        # Should now have all visible facets in V     
                
        NV = []                     
        unassigned_points = None
        
        for fac1 in V:
            # Move points from facets in V to the set unassigned_points
            N = len(fac1.outside)
            for ii in range(N):
                if unassigned_points == None:
                    unassigned_points = np.array([fac1.outside[ii].coordinates])
                else:
                    unassigned_points = np.vstack([unassigned_points,fac1.outside[ii].coordinates])
            
        for fac1 in V:                
            # Figure out the boundary of V, and create new facets            
            for fac2 in fac1.neighbors:
                if not (fac2 in V):                  
                    # fac1 is on the boundary!
                    # Create new facet from intersection between fac1 and fac2
                    # and p
                    vert1 = fac1.vertices
                    vert2 = fac2.vertices
                    for ii in range(dim):
                        p1 = vert1[ii,:]
                        test = np.sum(vert2 == p1,1)
                        if not np.any(test==dim):
                            ind = np.setdiff1d(np.arange(dim),np.array([ii]))
                            points = vert1[ind]
                            break 
                    points = np.vstack([p,points])
                                        
                    # Vertex points are in points
                    R = Facet(points)
                    
                    R.neighbors.append(fac2)
                    fac2.neighbors.append(R)
                    
                    NV.append(R)

        # Establish other neighbor relations in NV
        for ii in range(len(NV)):
            for jj in range(ii+1,len(NV)):
                if is_neighbor(NV[ii],NV[jj],abs_tol=abs_tol):
                    NV[ii].neighbors.append(NV[jj])
                    NV[jj].neighbors.append(NV[ii])

        # Assign unassigned points to facets in NV,
        # and add facets to F or Forg
        for fac1 in NV:
            if unassigned_points == None:
                Forg.append(fac1)
                continue
            npt = np.shape(unassigned_points)[0]
            keep_list = np.ones(npt, dtype=int)
            
            for ii in range(npt):
                if npt == 1:
                    pu = unassigned_points
                else:
                    pu = unassigned_points[ii,:]
                d = distance(pu,fac1) 
                if d > abs_tol:
                    op = Outside_point(pu.flatten(), d)
                    fac1.outside.append(op)
                    keep_list[ii] = 0
            
            if len(fac1.outside) > 0:
                F.append(fac1)
                Forg.append(fac1)
            else:
                Forg.append(fac1)
            ind = np.nonzero(keep_list)
            if len(ind[0]) > 0:
                unassigned_points = unassigned_points[ind[0],:]
            else:
                unassigned_points = None
        
        # Delete facets in V, and neighbor references
        for fac1 in V:
            for fac2 in fac1.neighbors:
                fac2.neighbors.remove(fac1)
            if fac1 in F:
                F.remove(fac1)
            Forg.remove(fac1)       
            fac1.neighbors = []
        V = []
    
    num = len(Forg)   
    A = np.zeros([num,dim])
    b = np.zeros([num,1])
    vert = np.zeros([num*dim,dim])
    for ii in range(num):
        vert[ np.ix_(range(ii*dim,(ii+1)*dim)), : ] = Forg[ii].vertices + xc
        A[ii,:] = Forg[ii].normal.flatten()
        b[ii] = Forg[ii].distance
    if numpy_ver > 1.59:
        vert = np.unique(vert.view([('',vert.dtype)]*vert.shape[1])).view(vert.dtype).reshape(-1,vert.shape[1])
    else:
        vert = np.unique1d(vert.view([('',vert.dtype)]*vert.shape[1])).view(vert.dtype).reshape(-1,vert.shape[1])
    b = b.flatten() + np.dot(A,xc.flatten())

    return A,b.flatten(),vert