"""
Only supports closed-loop and non-conservative simulation.
"""

from tulip import hybrid, polytope, abstract
import scipy.io
import tosimulink
import numpy


def export(filename, mealy_machine, system_dynamics=None, abstraction=None,
	control_horizon=None, R=None, r=None, Q=None, mid_weight=None):

	"""Creates two matlab files. One is a script that generates a Simulink model
	that contains a Stateflow Chart of the Mealy Machine, a block that contains
	a get_input function, a block that maps continuous state to discrete state,
	and a block that times simulation of get_input.

	@param system: L{LtiSysDyn}, L{PwaSysDyn}, or L{HybridSysDyn} to be saved in
		the .mat file.
	@param filename: String containing name of the .mat file to be created.
	@rtype: None
	"""

	# Check whether we're discrete or continuous
	if ((system_dynamics is None) and (abstraction is None)):
		is_continuous = False
	elif ((system_dynamics is not None) and (abstraction is not None)):
		is_continuous = True
	else:
		raise StandardError('Cannot tell whether system is continuous or ' + 
			'discrete. Please specify dynamics and abstraciton or neither.')


	# Skip all but last part if the system is discrete.
	if is_continuous:

		# Export system dynamics, get state and input dimension
		if isinstance(system_dynamics, hybrid.LtiSysDyn):
			dynamics_output = lti_export(system_dynamics)
			dynamics_output['type'] = 'LtiSysDyn'
			state_dimension = numpy.shape(system_dynamics.A)[0];
			input_dimension = numpy.shape(system_dynamics.B)[0];
		elif isinstance(system_dynamics, hybrid.PwaSysDyn):
			dynamics_output = pwa_export(system_dynamics)
			dynamics_output['type'] = 'PwaSysDyn'
			state_dimension = numpy.shape(system_dynamics.list_subsys[0].A)[0];
			input_dimension = numpy.shape(system_dynamics.list_subsys[0].B)[0];
		elif isinstance(system_dynamics, hybrid.HybridSysDyn):
			dynamics_output = hybrid_export(system_dynamics)
			dynamics_output['type'] = 'HybridSysDyn'
		else:
			raise TypeError(str(type(system)) + 
				' is not a supported type of system dynamics.')

		# Set default values for matrix weights and other optional parameters
		if control_horizon is None:
			raise ValueError('Control horizon must be given for continuous ' +
				'systems.')
		if R is None:
			R = numpy.zeros([state_dimension, state_dimension])
		if r is None:
			r = numpy.zeros([1, state_dimension])
		if Q is None:
			Q = numpy.eye(input_dimension)
		if mid_weight is None:
			mid_weight = 3


		# Miscellaneous simulation parameters that aren't in the dynamics
		sim_params = {}
		sim_params['horizon'] = control_horizon
		sim_params['state_weight'] = R
		sim_params['input_weight'] = Q
		sim_params['linear_weight'] = r
		sim_params['mid_weight'] = mid_weight
		
		# Final data dictionary
		output = {}
		output['system_dynamics'] = dynamics_output
		output['abstraction'] = export_locations(abstraction) #get abstraction
		output['sim_params'] = sim_params
		scipy.io.savemat(filename, output, oned_as='column')


	# Export the simulink model
	tosimulink.export(mealy_machine, is_continuous, control_horizon, R, r, Q, 
		mid_weight)




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
