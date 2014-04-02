# Copyright (c) 2014 by California Institute of Technology
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

"""
Contains XML import and export functions so that we can export (and not
recompute) Tulip data structures.
"""

import numpy
import scipy.sparse
import xml.etree.ElementTree as ET

import polytope
from tulip import transys
from tulip import hybrid
from tulip import abstract


# Global names used in tags
N_PROPS = 'props'
N_PROP = 'prop'
N_LOWERBOUND = 'lb'
N_UPPERBOUND = 'ub'
N_FULLDIM = 'fulldim'
N_MINREP = 'minrep'
N_VOLUME = 'volume'
N_DIMENSION = 'dim'
N_BBOX = 'bbox'
N_CHEBR = 'chebR'
N_CHEBXC = 'chebXc'
N_POLYLIST = 'list_poly'
N_REGIONLIST = 'list_reg'
N_KEY = 'key'
N_VALUE = 'value'
N_KEYVALUEPAIR = 'pair'
N_DOMAIN = 'domain'
N_PROPREG = 'proplist'
N_ADJ = 'adj'
N_ITEM = 'item'


# data types used in xml file (attributes)
T_STRING = 'str'
T_INT = 'int'
T_FLOAT = 'float'
T_BOOL = 'bool'
T_POLYTOPE = 'polytope'
T_MATRIX = 'ndarray'
T_REGION = 'region'
T_DICT = 'dict'
T_PPP = 'PPP'
T_ADJ = 'adjmatrix'
T_TUPLE = 'tuple'
T_LIST = 'list'
T_LTISYS = 'LtiSysDyn'
T_PWASYS = 'PwaSysDyn'
T_HYBRIDSYS = 'SwitchedSysDyn'


def _make_pretty(tree, indent=1):
	"""Modifies the tail attribute of an XML tree so that the resulting file
	isn't just one messy line of text.

	@type tree: L{xml.etree.ElementTree.Element}
	@type indent: L{int}
	@rtype: None
	"""

	tab_string = '\n' + '\t'*indent

	# If a tree has children, put tabs in front of the first child
	if tree.getchildren():
		if tree.text is not None:
			tree.text = tab_string + tree.text
		else:
			tree.text = tab_string

	# Number of children in the tree
	N = len(tree.getchildren())

	# Put tabs after each child in a list (to indent the next one), 
	# one less tab after the final child
	for index, child in enumerate(tree):
		_make_pretty(child, indent=indent+1)
		if index < N - 1:
			child.tail = tab_string
		else:
			child.tail = '\n' + '\t'*(indent-1)



def exportXML(data, filename, tag=None):
	"""Exports a Tulip data structure into an XML file.

	@param data: The Tulip data structure to export into an xml lfile
	@type data: L{Polytope}, L{Region}, L{PropPreservingPartition},
		L{SwitchedSysDyn}, L{PwaSysDyn}, L{LtiSysDyn}
	@param filename: The name of the XML file to export to.
	@type filename: L{string}
	@param tag: (Optional) What we want the first tag of the XML file to read.
	@type tag: L{string}

	@rtype: L{None}
	"""

	if tag is None:
		tag = "object"

	tree = _export_xml(data, None, tag)
	_make_pretty(tree)
	pretty_string = ET.tostring(tree)

	xmlfile = open(filename, 'w')
	xmlfile.write(pretty_string)
	xmlfile.close()



def importXML(filename):
	"""Takes a Tulip XML file and returns a Tulip data structure.

	@param filename: XML file containing exported data to import
	@type filename: L{string}

	@return: the data structure exported into the file.
	@rtype: L{Polytope}, L{Region}, L{PropPreservingPartition},
		L{SwitchedSysDyn}, L{PwaSysDyn}, L{LtiSysDyn}
	"""

	# Load the file into an xml tree
	xmltree = ET.parse(filename)

	# send the root to the general tree parser
	return _import_xml(xmltree.getroot())



def _import_xml(node):
	"""Takes a XML tree and returns the Tulip data structure the tree was made
	from.

	@type tree: L{xml.ElementTree.etree.Element}

	@rtype: stuff
	"""

	# Get the type of data this is
	nodetype = node.attrib['type']

	# Call the right import function

	# Python types
	if nodetype == T_STRING:
		return node.text
	elif nodetype == T_INT:
		return int(node.text)
	elif nodetype == T_FLOAT:
		return float(node.text)
	elif nodetype == T_BOOL:
		return bool(node.text)
	elif nodetype == T_MATRIX:
		return eval('numpy.array(' + node.text + ')')
	elif nodetype == T_TUPLE:
		return _import_tuple(node)
	elif nodetype == T_DICT:
		return _import_dictionary(node)
	elif nodetype == T_LIST:
		return _import_list(node)
	elif nodetype == T_ADJ:
		return _import_adj(node)

	# Tulip data structures
	elif nodetype == T_POLYTOPE:
		return _import_polytope(node)
	elif nodetype == T_REGION:
		return _import_region(node)
	elif nodetype == T_PPP:
		return _import_ppp(node)
	elif nodetype == T_LTISYS:
		return _import_ltisys(node)
	elif nodetype == T_PWASYS:
		return _import_pwasys(node)
	elif nodetype == T_HYBRIDSYS:
		return _import_hybridsys(node)
	else:
		raise TypeError('Type ' + nodetype + ' is not supported.')


def _import_adj(node):

	# Get number of rows and columns
	N = _import_xml(node.findall('num_states')[0])

	# Make matrix
	sparse_matrix = scipy.sparse.lil_matrix((N,N))

	# Get entries and fill them in with ones
	entries = _import_xml(node.findall('index_list')[0])
	for entry in entries:
		sparse_matrix[entry[0],entry[1]] = 1

	return sparse_matrix



def _import_ppp(node):
	
	# Domain
	domain_node = node.findall('domain')
	if domain_node:
		domain = _import_xml(domain_node[0])
	else:
		domain = None

	# Regions
	regions_node = node.findall('list_reg')
	if regions_node:
		list_regions = _import_xml(regions_node[0])
	else:
		list_regions = []

	# adj
	adj_node = node.findall('adj')
	if adj_node:
		adjmatrix = _import_xml(adj_node[0])
	else:
		adjmatrix = None

	# prop_regions
	prop_regions_node = node.findall('proplist')
	if prop_regions_node:
		prop_regions = _import_xml(prop_regions_node[0])
	else:
		prop_regions = None

	return abstract.prop2partition.PropPreservingPartition(domain=domain,
		regions=list_regions, adj=adjmatrix, prop_regions=prop_regions)



def _import_tuple(node):

	all_stuff = []
	for child in node:
		all_stuff.append(_import_xml(child))
	return tuple(all_stuff)


def _import_list(node):
	all_stuff = []
	for child in node:
		all_stuff.append(_import_xml(child))
	return all_stuff



def _import_dictionary(node):
	dictionary = {}
	for keyvaluepair in node:
		key = _import_xml(keyvaluepair.findall(N_KEY)[0])
		value = _import_xml(keyvaluepair.findall(N_VALUE)[0])
		dictionary[key] = value
	return dictionary


def _import_region(node):
	
	# Get the polytope list and import the polytopes
	polytope_list = node.findall(N_POLYLIST)[0]

	# Import the polytopes
	list_poly = _import_xml(polytope_list)

	return polytope.Region(list_poly=list_poly)


def _import_polytope(node):

	# Get the A matrix
	A = _import_xml(node.findall('A')[0])

	# Get the b matrix
	b = _import_xml(node.findall('b')[0])

	return polytope.Polytope(A=A, b=b)



def _import_ltisys(node):
	A = _import_xml(node.findall('A')[0])
	B = _import_xml(node.findall('B')[0])
	E = _import_xml(node.findall('E')[0])
	K = _import_xml(node.findall('K')[0])
	Uset = node.findall('Uset')
	Wset = node.findall('Wset')
	domain = node.findall('domain')

	if not Uset:
		Uset = None
	else:
		Uset = _import_xml(Uset[0])
	if not Wset:
		Wset = None
	else:
		Wset = _import_xml(Wset[0])
	if not domain:
		domain = None
	else:
		domain = _import_xml(domain[0])

	return hybrid.LtiSysDyn(A=A, B=B, E=E, K=K, Uset=Uset, Wset=Wset,
		domain=domain)



def _import_pwasys(node):

	domain = node.findall('domain')
	if domain:
		domain = _import_xml(domain[0])
	else:
		domain = None

	# Get list of ltisys
	ltilist = node.findall('ltilist')[0]
	list_subsys = _import_xml(ltilist)
	
	return hybrid.PwaSysDyn(list_subsys=list_subsys, domain=domain)



def _import_hybridsys(node):

	# Get parts, import the non-optional parts
	disc_domain_size = _import_xml(node.findall('disc_domain_size')[0])
	sys_labels = node.findall('sys_labels')
	env_labels = node.findall('env_labels')
	cts_ss = _import_xml(node.findall('cts_ss')[0])
	dynamics = _import_xml(node.findall('dynamics')[0])

	if sys_labels:
		sys_labels = _import_xml(sys_labels[0])
	else:
		sys_labels = None
	if env_labels:
		env_labels = _import_xml(env_labels[0])
	else:
		env_lables = None

	return hybrid.SwitchedSysDyn(disc_domain_size=disc_domain_size,
		dynamics=dynamics, cts_ss=cts_ss, env_labels=env_labels,
		disc_sys_labels=sys_labels)



def _export_xml(data, parent=None, tag=None):
	"""Exports Tulip data structures to XML structures for later import. This
	function is called both internal

	@param data: the data structure to be exported into an XML tree.
	@type data: L{numpy.ndarray} or L{Polytope} or L{Region} or 
	    L{FiniteTransitionSystem} or L{PropPreservingPartition} or
		L{AbstractSysDyn} or L{dict}
	@param parent: 
	@type parent: L{None} or L{xml.etree.ElementTree.Element} or
		L{xml.etree.ElementTree.SubElement}
	@type tag: L{None} or L{string}
	
	@return: None (if parent is None), or an xml tree
	@rtype: L{None} or L{xml.etree.ElementTree.Element} or
		L{xml.etree.ElementTree.SubElement}return: The data structure
	"""

	# Tulip types (parent might not exist)
	if isinstance(data, polytope.Polytope):
		if parent is None:
			return _export_polytope(data, parent, tag)
		else:
			_export_polytope(data, parent, tag)
	elif isinstance(data, polytope.Region):
		if parent is None:
			return _export_region(data, parent, tag)
		else:
			_export_region(data, parent, tag)

	elif isinstance(data, abstract.prop2partition.PropPreservingPartition):
		if parent is None:
			return _export_ppp(data, parent, tag)
		else:
			_export_ppp(data, parent, tag)

	elif isinstance(data, hybrid.LtiSysDyn):
		if parent is None:
			return _export_ltisys(data, parent, tag)
		else:
			_export_ltisys(data, parent, tag)
	
	elif isinstance(data, hybrid.PwaSysDyn):
		if parent is None:
			return _export_pwasys(data, parent, tag)
		else:
			_export_pwasys(data, parent, tag)
	
	elif isinstance(data, hybrid.SwitchedSysDyn):
		if parent is None:
			return _export_hybridsys(data, parent, tag)
		else:
			_export_hybridsys(data, parent, tag)


	# parent will always not be none
	elif (isinstance(data, int) or isinstance(data, numpy.int32)):
		if tag is None:
			tag = "integer"
		new_node = ET.SubElement(parent, tag, type=T_INT)
		new_node.text = str(data)

	elif isinstance(data, str):
		if tag is None:
			tag = "string"
		new_node = ET.SubElement(parent, tag, type=T_STRING)
		new_node.text = data

	elif (isinstance(data, bool) or isinstance(data, numpy.bool_)):
		if tag is None:
			tag = "bool"
		new_node = ET.SubElement(parent, tag, type=T_BOOL)
		new_node.text = str(data)

	elif isinstance(data, float):
		if tag is None:
			tag = "float"
		new_node = ET.SubElement(parent, tag, type=T_FLOAT)
		new_node.text = str(data)

	elif isinstance(data, dict):
		_export_dict(data, parent, tag)

	elif isinstance(data, numpy.ndarray):
		if tag is None:
			tag = "numpyarray"
		new_node = ET.SubElement(parent, tag, type=T_MATRIX)
		new_node.text = str(data.tolist())
	
	elif isinstance(data, tuple):
		_export_tuple(data, parent, tag)

	elif isinstance(data, list):
		_export_list(data, parent, tag)

	elif isinstance(data, scipy.sparse.lil.lil_matrix):
		_export_adj(data, parent, tag)

	elif isinstance(data, set):
		_export_set(data, parent, tag)

	# Type not found
	else:
		raise TypeError('Type ' + str(type(data)) + ' is not supported.')



def _export_ppp(ppp, parent, tag):
	"""
	@return: None (if parent is None), or an xml tree
	@rtype: L{None} or L{xml.etree.ElementTree.Element} or
		L{xml.etree.ElementTree.SubElement}
	"""
	if tag is None:
		tag = "PropPreservingPartition"

	if parent is None:
		tree = ET.Element(tag, type=T_PPP)
	else:
		tree = ET.SubElement(parent, tag, type=T_PPP)

	# Domain (polytope)
	_export_polytope(ppp.domain, tree, tag=N_DOMAIN)

	# regions (list of regions)
	_export_xml(ppp.regions, tree, N_REGIONLIST)

	# adj (adjacency matrix)
	_export_adj(ppp.adj, tree, N_ADJ)

	# prop regions (dictionary mapping strings to regions)
	_export_xml(ppp.prop_regions, tree, N_PROPREG)

	if parent is None:
		return tree
	

def _export_ltisys(ltisys, parent, tag=None):
	"""
	@return: None (if parent is None), or an xml tree
	@rtype: L{None} or L{xml.etree.ElementTree.Element} or
		L{xml.etree.ElementTree.SubElement}
	"""
	if tag is None:
		tag = "LtiSysDyn"
	if parent is None:
		tree = ET.Element(tag, type=T_LTISYS)
	else:
		tree = ET.SubElement(parent, tag, type=T_LTISYS)

	# State space matrices
	_export_xml(ltisys.A, tree, 'A')
	_export_xml(ltisys.B, tree, 'B')
	_export_xml(ltisys.E, tree, 'E')
	_export_xml(ltisys.K, tree, 'K')

	# Domain matrices
	if ltisys.Uset is not None:
		_export_polytope(ltisys.Uset, tree, 'Uset')
	if ltisys.Wset is not None:
		_export_polytope(ltisys.Wset, tree, 'Wset')
	if ltisys.domain is not None:
		_export_polytope(ltisys.domain, tree, 'domain')

	if parent is None:
		return tree



def _export_pwasys(pwasys, parent, tag=None):
	"""
	@return: None (if parent is None), or an xml tree
	@rtype: L{None} or L{xml.etree.ElementTree.Element} or
		L{xml.etree.ElementTree.SubElement}
	"""
	if tag is None:
		tag = "PwaSysDyn"
	if parent is None:
		tree = ET.Element(tag, type=T_PWASYS)
	else:
		tree = ET.SubElement(parent, tag, type=T_PWASYS)

	# Export domain
	if pwasys.domain is not None:
		_export_polytope(pwasys.domain, tree, 'domain')
	
	# Export lti list
	_export_list(pwasys.list_subsys, tree, 'ltilist')

	if parent is None:
		return tree


def _export_hybridsys(hybridsys, parent, tag=None):
	"""
	@return: None (if parent is None), or an xml tree
	@rtype: L{None} or L{xml.etree.ElementTree.Element} or
		L{xml.etree.ElementTree.SubElement}
	"""
	if tag is None:
		tag = "SwitchedSysDyn"
	if parent is None:
		tree = ET.Element(tag, type=T_HYBRIDSYS)
	else:
		tree = ET.SubElement(parent, tag, type=T_HYBRIDSYS)

	# cts_ss
	_export_xml(hybridsys.cts_ss, tree, tag="cts_ss")

	# disc_domain_size
	_export_xml(hybridsys.disc_domain_size, tree, tag="disc_domain_size")

	# disc_sys_labels
	_export_xml(hybridsys.disc_sys_labels, tree, tag="sys_labels")

	# env_labels
	_export_xml(hybridsys.env_labels, tree, tag="env_labels")

	# Dynamics
	_export_dict(hybridsys.dynamics, tree, tag="dynamics")

	if parent is None:
		return tree



def _export_polytope(poly, parent, tag=None):
	"""Builds an XML tree from a polytope
	
	@param poly: Polytope to export
	@type poly: Polytope

	@return: None (if parent is None), or an xml tree
	@rtype: L{None} or L{xml.etree.ElementTree.Element} or
		L{xml.etree.ElementTree.SubElement}
	"""

	if tag is None:
		tag = "polytope"

	if parent is None:
		tree = ET.Element(tag, type=T_POLYTOPE)
	else:
		tree = ET.SubElement(parent, tag, type=T_POLYTOPE)

	# A and b matrices
	_export_xml(poly.A, tree, 'A')
	_export_xml(poly.b, tree, 'b')


	# Optional parts

	# minimal representation (bool)
	if poly.minrep is not None:
		_export_xml(poly.minrep, tree, N_MINREP)

	# bounding box
	if poly.bbox is not None:
		bbox_node = ET.SubElement(tree, N_BBOX)
		_export_xml(poly.bbox[0], bbox_node, N_LOWERBOUND)
		_export_xml(poly.bbox[1], bbox_node, N_UPPERBOUND)

	# chebyshev center (ndarray)
	if poly.chebXc is not None:
		_export_xml(poly.chebXc, tree, N_CHEBXC)

	# chebyshev radius (float)
	if poly.chebR is not None:
		_export_xml(poly.chebR, tree, N_CHEBR)

	# dimension (integer)
	if poly.dim:
		_export_xml(poly.dim, tree, N_DIMENSION)

	# full dimension (bool)
	if poly.fulldim is not None:
		_export_xml(poly.fulldim, tree, N_FULLDIM)

	# volume (float)
	if poly.volume is not None:
		_export_xml(poly.volume, tree, N_VOLUME)

	# Return if there is no parent
	if parent is None:
		return tree



def _export_region(reg, parent, tag=None):
	"""
	@return: None (if parent is None), or an xml tree
	@rtype: L{None} or L{xml.etree.ElementTree.Element} or
		L{xml.etree.ElementTree.SubElement}
	"""
	if tag is None:
		tag = "region"

	if parent is None:
		tree = ET.Element(tag, type=T_REGION)
	else:
		tree = ET.SubElement(parent, tag, type=T_REGION)
	
	# Attach list of polytopes
	_export_list(reg.list_poly, tree, N_POLYLIST)


	# Attach optional parts of region:

	# Bounding box, two numpy arrays
	if reg.bbox is not None:
		bbox_node = ET.SubElement(tree, N_BBOX)
		_export_xml(reg.bbox[0], bbox_node, N_LOWERBOUND)
		_export_xml(reg.bbox[1], bbox_node, N_UPPERBOUND)

	# Dimension (integer)
	if reg.dim:
		_export_xml(reg.dim, tree, N_DIMENSION)

	# Fulldim (bool)
	if reg.fulldim is not None:
		_export_xml(reg.fulldim, tree, N_FULLDIM)

	# Volume (float)
	if reg.volume is not None:
		_export_xml(reg.volume, tree, N_VOLUME)

	# Chebyshev radius (float)
	if reg.chebR is not None:
		_export_xml(reg.chebR, tree, N_CHEBR)

	# Chebyshev center (array)
	if reg.chebXc is not None:
		_export_xml(reg.chebXc, tree, N_CHEBXC)

	# Propositions that hold in region (set of strings)
	if reg.props:
		_export_xml(reg.props, tree, N_PROPS)
		
	if parent is None:
		return tree



def _export_fts(fts, parent, tag):
	"""Converts a FiniteTransitionSystem or an OpenFiniteTransitionSystem into
	an xml tree.

	@type fts: L{FiniteTransitionSystem} or L{OpenFiniteTransitionSystem}
	@type parent:
	@type tag: L{None} or L{string}

	@return: None (if parent is None), or an xml tree
	@rtype: L{None} or L{xml.etree.ElementTree.Element} or
		L{xml.etree.ElementTree.SubElement}
	"""

	if tag is None:
		if type(fts) == transys.transys.OpenFiniteTransitionSystem:
			tag = "OFTS"
		elif type(fts) == transys.transys.FiniteTransitionSystem:
			tag = "FTS"
	if parent is None:
		if type(fts) == transys.transys.OpenFiniteTransitionSystem:
			tree = ET.Element(tag, type=T_OFTS)
		elif type(fts) == transys.transys.FiniteTransitionSystem:
			tree = ET.SubElement(parent, tag, type=T_FTS)
	
	# Export states

	# make and export transition matrix

	# 

	if parent is None:
		return tree
	


def _export_adj(matrix, parent, tag=None):
	"""Converts an adjacency matrix (scipy.sparse.lil.lil_matrix) into an xml
	tree.

	@param matrix: Sparce adjacency matrix.
	@type matrix: L{scipy.sparse.lil.lil_matrix}
	@type parent: L{None} or L{xml.etree.ElementTree.Element} or 
		L{xml.etree.ElementTree.SubElement}
	@type tag: L{string}

	@return: None (if parent is None), or an xml tree
	@rtype: L{None} or L{xml.etree.ElementTree.Element} or
		L{xml.etree.ElementTree.SubElement}
	"""

	if tag is None:
		tag = "adj"
	if parent is None:
		tree = ET.Element(tag, type=T_ADJ)
	else:
		tree = ET.SubElement(parent, tag, type=T_ADJ)

	# number of states, because the matrix must be square
	(M,N) = matrix.shape
	_export_xml(N, tree, "num_states")

	# list of nonzero indices
	(row_indices, col_indices) = matrix.nonzero()
	indices = []
	for i, row_ind in enumerate(row_indices):
		col_ind = col_indices[i]
		indices.append((row_ind, col_ind))
	_export_list(indices, tree, "index_list")

	if parent is None:
		return tree


def _export_dict(dictionary, parent, tag=None):
	"""Converts a dictionary into an XML tree. The key and value can be any
	supported type because the function calls _export_xml()

	@type dictionary: L{dict}
	@type parent: L{None} or L{xml.etree.ElementTree.Element} or
		L{xml.etree.ElementTree.SubElement}
	@type tag: L{string}

	@return: None, if parent is None. An XML tree if parent is a node
	@rtype: L{xml.etree.ElementTree.Element} or L{None}
	"""

	if tag is None:
		tag = "dict"
	if parent is None:
		tree = ET.Element(tag, type=T_DICT)
	else:
		tree = ET.SubElement(parent, tag, type=T_DICT)

	# Make key value pairs
	for key, value in dictionary.iteritems():
		pair_node = ET.SubElement(tree, N_KEYVALUEPAIR)
		_export_xml(key, parent=pair_node, tag=N_KEY)
		_export_xml(value, parent=pair_node, tag=N_VALUE)

	if parent is None:
		return tree


def _export_tuple(tup, parent, tag=None):

	if tag is None:
		tag = "tuple"
	if parent is None:
		tree = ET.Element(tag, type=T_TUPLE)
	else:
		tree = ET.SubElement(parent, tag, type=T_TUPLE)

	for item in tup:
		_export_xml(item, parent=tree, tag=N_ITEM)

	if parent is None:
		return tree



def _export_list(lst, parent, tag=None):

	if tag is None:
		tag = "list"
	if parent is None:
		tree = ET.Element(tag, type=T_LIST)
	else:
		tree = ET.SubElement(parent, tag, type=T_LIST)

	for item in lst:
		_export_xml(item, parent=tree, tag=N_ITEM)

	if parent is None:
		return tree


def _export_set(s, parent, tag=None):

	if tag is None:
		tag = "set"
	if parent is None:
		tree = ET.Element(tag, type="set")
	else:
		tree = ET.SubElement(parent, tag, type="set")

	for item in s:
		_export_xml(item, parent=tree, tag=N_ITEM)

	if parent is None:
		return tree
