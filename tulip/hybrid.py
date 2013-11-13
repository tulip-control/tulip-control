# Copyright (c) 2011, 2012, 2013 by California Institute of Technology
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

""" 
Classes:
    - LtiSysDyn
    - PwaSysDyn
    - HybridSysDyn
    
NO, 2 Jul 2013.
"""
import numpy as np
import itertools
import polytope as pc

class LtiSysDyn:
    """LtiSysDyn class for specifying the continuous dynamics:

        s[t+1] = A*s[t] + B*u[t] + E*d[t] + K
        u[t] \in Uset - polytope object
        d[t] \in Wset - polytope object
        s[t] \in domain -polytope object

    A LtiSysDyn object contains the fields A, B, E, K, Uset, Wset and domain
    as defined above.
    
    Note: For state-dependent bounds on the input, [u[t];s[t]] \in Uset can
    be used.
    
    **Constructor**:
    
	**LtiSysDyn** ([ `A` = [][, `B` = [][, `E` = [][, `K` = [][, `Uset` = [][,
	`Wset` = [][, `domain`[]]]]]]]])
    """
    def __init__(self, A=[], B=[], E=[], K=[], Uset=None,Wset=None,domain=None):
        
        if Uset == None:
            print "Warning: Uset not given in LtiSysDyn()"
        
        if (Uset != None) & (not isinstance(Uset, pc.Polytope)):
            raise Exception("LtiSysDyn: `Uset` has to be a Polytope")
           
        if domain == None:
            print "Warning: domain is not given in LtiSysDyn()"
        
        if (domain != None) & (not isinstance(domain, pc.Polytope)):
            raise Exception("LtiSysDyn: `domain` has to be a Polytope")

        self.A = A
        self.B = B
        
        if len(K) == 0:
            if len(A) != 0:
                self.K = np.zeros([A.shape[1], 1])
            else:
                self.K = K
        else:
            self.K = K.reshape(K.size,1)

        if (len(E) == 0) & (len(A) != 0):
            self.E = np.zeros([A.shape[1], 1])
            self.Wset = pc.Polytope()
        else:
            self.E = E
            self.Wset = Wset
        
        self.Uset = Uset
        self.domain = domain

    def __str__(self):
        output = "A =\n"+str(self.A)
        output += "\nB =\n"+str(self.B)
        output += "\nE =\n"+str(self.E)
        output += "\nK =\n"+str(self.K)
        output += "\nUset =\n"+str(self.Uset)
        output += "\nWset =\n"+str(self.Wset)
        return output
        
class PwaSysDyn:
    """PwaSysDyn class for specifying a piecewise affine system.
    A PwaSysDyn object contains the fields:
    
    - `list_subsys`: list of LtiSysDyn
    
	- `domain`: domain over which piecewise affine system is defined, type:
	  polytope
    
	For the system to be well-defined the domains of its subsystems should be
	mutually exclusive (modulo intersections with empty interior) and cover the
	domain.
    """
    def __init__(self, list_subsys=[], domain=None):
        
        if domain == None:
            print "Warning: domain not given in PwaSysDyn()"
        
        if (domain != None) and \
               (not (isinstance(domain, pc.Polytope) or isinstance(domain, pc.Region))):
            raise Exception("PwaSysDyn: `domain` has to be a Polytope or Region")

        if len(list_subsys) > 0:
            uncovered_dom = domain.copy()
            n = list_subsys[0].A.shape[1]  # State space dimension
            m = list_subsys[0].B.shape[1]  # Input space dimension
            p = list_subsys[0].E.shape[1]  # Disturbance space dimension
            for subsys in list_subsys:
                uncovered_dom = pc.mldivide(uncovered_dom, subsys.domain)
                if (n!=subsys.A.shape[1] or m!=subsys.B.shape[1] or 
				    p!=subsys.E.shape[1]):
                    raise Exception("PwaSysDyn: state, input, disturbance " + 
					                "dimensions have to be the same for all " +
									"subsystems")
            if not pc.is_empty(uncovered_dom):
                raise Exception("PwaSysDyn: subdomains must cover the domain")
            for x in itertools.combinations(list_subsys, 2):
                if pc.is_fulldim(pc.intersect(x[0].domain,x[1].domain)):
                    raise Exception("PwaSysDyn: subdomains have to be mutually"+
					                " exclusive")
        
        self.list_subsys = list_subsys
        self.domain = domain
        
    @classmethod
    def from_lti(cls, A=[], B=[], E=[], K=[], Uset=None, Wset=None,domain=None):
        lti_sys = LtiSysDyn(A,B,E,K,Uset,Wset,domain)
        return cls([lti_sys], domain)

class HybridSysDyn:
	"""HybridSysDyn class for specifying hybrid systems with discrete and
	continuous variables.
    
    A HybridSysDyn object contains the fields:
    
	- `disc_domain_size`: A 2-tuple of integers showing the number of discrete
	  environment (uncontrolled) and system (controlled) variables respectively
	  (i.e., switching modes) 
    
	- `env_labels`: A list of length disc_domain_size[0], optional field for
	  definining labels for discrete environment variables
    
	- `disc_sys_labels`: A list of length disc_domain_size[1], optional field
	  for definining labels for discrete system variables

    - `dynamics`: a dictionary mapping (env_label, sys_label) -> PwaSysDyn. If
	  we are not using env_label and sys_label, then it makes indices (i,j) to
	  PwaSysDyn.
    
	- `cts_ss`: continuous state space over which hybrid system is defined,
	  type: Region
    
	- `time_semantics`: TBD. Current default semantics are discrete-time. State
	  s[t] and discrete environment env[t] are observed and continuous input
	  u[t] and discrete system variable m[t] are determined based on env[t] and
	  s[t] (synchronously at time t).
   
    Note: We assume that system and environment switching modes are independent
	of one another. (Use LTL statement to make it not so.)
    """
	def __init__(self, disc_domain_size=(1,1),dynamics=None,cts_ss=None,**args):

		# check that the continuous domain is specified
		if cts_ss is None:
			print "Warning: continuous state space not given in HybridSysDyn()"
		if (cts_ss is not None) and \
               (not (isinstance(cts_ss, pc.Polytope) or isinstance(cts_ss, pc.Region))):
			raise Exception("HybridSysDyn: `cts_ss` has to be a Polytope or Region")

		# Get the labels and, if there are the right number of them, use them.
		# Otherwise, ignore the labels.
		self.env_labels = args.get('env_labels', range(disc_domain_size[0]))
		self.disc_sys_labels = args.get('disc_sys_labels', 
		                                range(disc_domain_size[1]))
		use_env_labels = True
		use_sys_labels = True
		if (self.env_labels is not None) and \
				(len(self.env_labels) != disc_domain_size[0]):
			use_env_labels = False
			print "Warning: number of environment labels is inconsistent with"+\
			      "discrete domain size. Ignoring the environment labels."
		if self.env_labels is None:
			use_env_labels = False
		if (self.disc_sys_labels is not None) and \
				(len(self.disc_sys_labels) != disc_domain_size[1]):
			print "Warning: number of discrete system labels is inconsistent" +\
			      "with discrete domain size. Ignoring the system labels."
			use_sys_labels = False
		if self.disc_sys_labels is None:
			use_sys_labels = False

		# Check that the keys of the dynamics hash line up
		if dynamics is not None:
			if use_env_labels:
				env_list = self.env_labels
			else:
				env_list = range(disc_domain_size[0])
			if use_sys_labels:
				sys_list = self.disc_sys_labels
			else:
				sys_list = range(disc_domain_size[1])
			check_keys = [(a,b) for a in env_list for b in sys_list]
			if set(dynamics.keys()) != set(check_keys):
				raise Exception("HybridSysDyn: keys in `dynamics` are " + 
					            "inconsistent with discrete mode labels")
		self.dynamics = dynamics
		self.cts_ss = cts_ss
        
	@classmethod
	def from_pwa(cls, list_subsys=[], domain=None):
		pwa_sys = PwaSysDyn(list_subsys,domain)
		return cls((1,1), {(0,0):pwa_sys}, domain)

	@classmethod
	def from_lti(cls, A=[], B=[], E=[], K=[], Uset=None, Wset=None,domain=None):
		pwa_sys = PwaSysDyn.from_lti(A,B,E,K,Uset,Wset,domain)
		return cls((1,1), {(0,0):pwa_sys}, domain)
