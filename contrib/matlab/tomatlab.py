"""
Saves LtiSysDyn, PwaSysDyn, and HybridSysDyn into a .mat file.
"""

from tulip import hybrid, polytope, abstract
import scipy.io


def export(system, filename):
	"""Saves a Tulip dynamical system as a .mat file that can be imported into
	the MPT toolbox.

	@param system: L{LtiSysDyn}, L{PwaSysDyn}, or L{HybridSysDyn} to be saved in
		the .mat file.
	@param filename: String containing name of the .mat file to be created.
	@rtype: None
	"""

	if isinstance(system, hybrid.LtiSysDyn):
		output = lti_export(system)
		output['type'] = 'LtiSysDyn'
	elif isinstance(system, hybrid.PwaSysDyn):
		output = pwa_export(system)
		output['type'] = 'PwaSysDyn'
	elif isinstance(system, hybrid.HybridSysDyn):
		output = hybrid_export(system)
		output['type'] = 'HybridSysDyn'
	elif isinstance(system, polytope.Region):
		output = reg_export(system)
		output['type'] = 'Region'
	elif (isinstance(system, abstract.discretization.AbstractSwitched) or \
		isinstance(system, abstract.discretization.AbstractPwa)):
		output = export_locations(system)
		output['type'] = 'Abstraction'
	else:
		raise TypeError(str(type(system)) + ' is not supported.')
	scipy.io.savemat(filename, {'TulipObject': output}, oned_as='column')


def lti_export(ltisys):
	"""Saves a LtiSysDyn as a Matlab struct."""

	output = {}
	output['A'] = ltisys.A
	output['B'] = ltisys.B
	output['E'] = ltisys.E
	output['K'] = ltisys.K
	output['domain'] = poly_export(ltisys.domain)
	output['Uset'] = poly_export(ltisys.Uset)
	output['Wset'] = poly_export(ltisys.Wset)

	return output


def pwa_export(pwasys):

	output = {}
	output['domain'] = poly_export(pwasys.domain)
	ltisystems = []
	for subsystem in pwasys.list_subsys:
		ltisystems.append(lti_export(subsystem))
	output['subsystems'] = ltisystems

	return output


def hybrid_export(hybridsys):

	output = {}
	output['disc_domain_size'] = list(hybridsys.disc_domain_size)
	output['env_labels'] = list(hybridsys.env_labels)
	output['sys_labels'] = list(hybridsys.disc_sys_labels)
	output['cts_ss'] = poly_export(hybridsys.cts_ss)

	dynamics = []
	for label, system in hybridsys.dynamics.iteritems():
		system_dict = {}
		system_dict['label'] = label
		system_dict['pwasys'] = pwa_export(system)
		dynamics.append(system_dict)
	
	output['dynamics'] = dynamics

	return output


def poly_export(poly):
	"""Saves parts of a polytope as a dictionary for export to MATLAB.
	
	@param poly: L{Polytope} that will be exported.
	@return output: dictionary containing fields of poly
	"""

	output = {}
	if poly is not None:
		output['A'] = poly.A
		output['b'] = poly.b
	return output


def reg_export(reg):
	"""Saves a region as a dictionary for export to MATLAB.

	@type reg: L{Region}
	@return output: a dictionary containing a list of polytopes.
	"""
	output = {}
	poly_list = []
	for poly in reg.list_poly:
		poly_list.append(poly_export(poly))
	output['list_poly'] = poly_list
	return output


def export_locations(abstraction):
	"""Exports an abstraction to a .mat file
	
	@type abstraction: L{AbstractPwa} or L{AbstractSwitched}
	@rtype output: dictionary"""

	output = {}

	location_list = []
	for index, region in enumerate(abstraction.ppp.regions):
		location_dict = {}
		reg_dict = reg_export(region)
		location_dict['region'] = reg_dict
		location_dict['index'] = index
		location_list.append(location_dict)	

	output['abstraction'] = location_list

	return output
