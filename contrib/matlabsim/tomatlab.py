"""Export hybrid controller to Matlab.

Only supports closed-loop and non-conservative simulation.
"""
import numpy
import polytope
import scipy.io
from tulip import abstract
from tulip import hybrid


def export(
        filename, mealy_machine, system_dynamics=None, abstraction=None,
        disc_params=None, R=None, r=None, Q=None, mid_weight=None):
    """Creates two matlab files. One is a script that generates a Simulink model
    that contains a Stateflow Chart of the Mealy Machine, a block that contains
    a get_input function, a block that maps continuous state to discrete state,
    and a block that times simulation of get_input.

    @param filename: string ending in '.mat'
    @param system: L{LtiSysDyn}, L{PwaSysDyn}, or L{HybridSysDyn} to be saved in
        the .mat file.
    @param filename: String containing name of the .mat file to be created.
    @rtype: None
    """
    # Check whether we're discrete or continuous
    if ((system_dynamics is None) and (abstraction is None) and
        (disc_params is None)):
        is_continuous = False
    elif ((system_dynamics is not None) and (abstraction is not None) and
          (disc_params is not None)):
        is_continuous = True
    else:
        raise StandardError('Cannot tell whether system is continuous or ' +
            'discrete. Please specify dynamics and abstraciton and ' +
            'discretization parameters or none at all.')
    output = dict()
    output['is_continuous'] = is_continuous
    # Only export dynamics and abstraction and control weights if the system is
    # continuous
    if is_continuous:
        # Export system dynamics, get state and input dimension
        if isinstance(system_dynamics, hybrid.LtiSysDyn):
            dynamics_output = lti_export(system_dynamics)
            dynamics_output['type'] = 'LtiSysDyn'
            state_dimension = numpy.shape(system_dynamics.A)[1]
            input_dimension = numpy.shape(system_dynamics.B)[1]
        elif isinstance(system_dynamics, hybrid.PwaSysDyn):
            dynamics_output = pwa_export(system_dynamics)
            dynamics_output['type'] = 'PwaSysDyn'
            state_dimension = numpy.shape(system_dynamics.list_subsys[0].A)[1]
            input_dimension = numpy.shape(system_dynamics.list_subsys[0].B)[1]
        elif isinstance(system_dynamics, hybrid.SwitchedSysDyn):
            dynamics_output = switched_export(system_dynamics)
            dynamics_output['type'] = 'SwitchedSysDyn'
            # getting state and input dimension by looking at size of the A and
            # B matrices of one of the PWA subsystems
            pwa_systems = list(system_dynamics.dynamics.values())
            pwa_system = pwa_systems[0]
            state_dimension = numpy.shape(pwa_system.list_subsys[0].A)[1]
            input_dimension = numpy.shape(pwa_system.list_subsys[0].B)[1]
        else:
            raise TypeError(str(type(system_dynamics)) +
                ' is not a supported type of system dynamics.')
        output['system_dynamics'] = dynamics_output
        # Control weights. Set default values if needed.
        if R is None:
            R = numpy.zeros([state_dimension, state_dimension])
        if r is None:
            r = numpy.zeros([1, state_dimension])
        if Q is None:
            Q = numpy.eye(input_dimension)
        if mid_weight is None:
            mid_weight = 3
        control_weights = dict(
            state_weight=R,
            input_weight=Q,
            linear_weight=r,
            mid_weight=mid_weight)
        output['control_weights'] = control_weights
        # Simulation parameters; insert default discretization values if needed
        sim_params = dict()
        try:
            sim_params['horizon'] = disc_params['N']
        except KeyError:
            sim_params['horizon'] = 10
        try:
            sim_params['use_all_horizon'] = disc_params['use_all_horizon']
        except KeyError:
            sim_params['use_all_horizon'] = False
        try:
            sim_params['closed_loop'] = disc_params['closed_loop']
        except KeyError:
            sim_params['closed_loop'] = True
        if 'conservative' in disc_params:
            if disc_params['conservative'] is True:
                raise ValueError('MATLAB interface does not suport ' +
                                 'conservative simulation')
        output['simulation_parameters'] = sim_params
        # Abstraction
        output['abstraction'] = export_locations(abstraction)
    # Export the simulink model
    output['TS'] = export_mealy(mealy_machine, is_continuous)
    # Save file
    scipy.io.savemat(filename, output, oned_as='column')


def lti_export(ltisys):
    """Saves a LtiSysDyn as a Matlab struct."""
    output = dict(
        A=ltisys.A,
        B=ltisys.B,
        E=ltisys.E,
        K=ltisys.K,
        domain=poly_export(ltisys.domain),
        Uset=poly_export(ltisys.Uset),
        Wset=poly_export(ltisys.Wset))
    return output


def pwa_export(pwasys):
    """Return piecewise-affine system as Matlab struct."""
    output = dict()
    output['domain'] = poly_export(pwasys.domain)
    ltisystems = list()
    for subsystem in pwasys.list_subsys:
        ltisystems.append(lti_export(subsystem))
    output['subsystems'] = ltisystems
    return output


def switched_export(switchedsys):
    """Return switched system as Matlab struct."""
    output = dict()
    output['disc_domain_size'] = list(switchedsys.disc_domain_size)
    output['cts_ss'] = poly_export(switchedsys.cts_ss)
    dynamics = list()
    for label, system in switchedsys.dynamics.items():
        system_dict = dict()
        system_dict['env_act'] = label[0]
        system_dict['sys_act'] = label[1]
        system_dict['pwasys'] = pwa_export(system)
        dynamics.append(system_dict)
    output['dynamics'] = dynamics
    return output


def poly_export(poly):
    """Saves parts of a polytope as a dictionary for export to MATLAB.

    @param poly: L{Polytope} that will be exported.
    @return output: dictionary containing fields of poly
    """
    output = dict()
    if poly is not None:
        output['A'] = poly.A
        output['b'] = poly.b
    return output


def reg_export(reg):
    """Saves a region as a dictionary for export to MATLAB.

    @type reg: L{Region}
    @return output: a dictionary containing a list of polytopes.
    """
    output = dict()
    poly_list = list()
    for poly in reg.list_poly:
        poly_list.append(poly_export(poly))
    output['list_poly'] = poly_list
    return output


def export_locations(abstraction):
    """Exports an abstraction to a .mat file

    @type abstraction: L{AbstractPwa} or L{AbstractSwitched}
    @rtype output: dictionary"""
    output = dict()
    location_list = list()
    for index, region in enumerate(abstraction.ppp.regions):
        location_dict = dict()
        reg_dict = reg_export(region)
        location_dict['region'] = reg_dict
        location_dict['index'] = index
        location_list.append(location_dict)
    output['abstraction'] = location_list
    return output


def export_mealy_io(variables, values):
    """Return declarations of variable types.

    @rtype: list of dict
    """
    var_list = list()
    for ind, variable in enumerate(variables):
        var_dict = dict()
        var_dict['name'] = variable
        var_dict['values'] = list(values[ind])
        var_list.append(var_dict)
    return var_list


def export_mealy(mealy_machine, is_continuous):
    """Exports a Mealy Machine to data that can be put into a .mat file. Turns
    the Mealy Machine states into a list of dictionaries. Turns the Mealy
    Machine transitions into a list of transition matrices.

    Some of the exported information is a bit redundant, but makes writing the
    MATLAB code easier.

    @rtype: dict
    """
    SINIT = 'Sinit'
    output = dict()
    # map from Mealy nodes to value of variable "loc"
    node_to_loc = dict()
    for _, v, label in mealy_machine.edges(data=True):
        node_to_loc[v] = label['loc']
    # Export states as a list of dictionaries
    state_list = list()
    for u in mealy_machine.nodes():
        if u == SINIT:
            print('Skipping state "{s}".'.format(s=SINIT))
            continue
        state_dict = dict(name=u)
        # For a continuous system, export the 'loc' variable
        if is_continuous:
            state_dict.update(loc=node_to_loc[u])
        state_list.append(state_dict)
    output['states'] = state_list
    # Get list of environment and system variables
    env_vars = mealy_machine.inputs.keys()
    env_values = list(mealy_machine.inputs.values())
    sys_vars = mealy_machine.outputs.keys()
    sys_values = list(mealy_machine.outputs.values())
    output['inputs'] = export_mealy_io(env_vars, env_values)
    output['outputs'] = export_mealy_io(sys_vars, sys_values)
    # Transitions will be exported as a 2D list of dictionaries. The only
    # purpose of this block here is to separate the inputs from the outputs to
    # make the MATLAB code easier to write
    transitions = list()
    for transition_tuple in mealy_machine.transitions.find():
        # Ignore transitions from Sinit
        if transition_tuple[0] == 'Sinit':
            continue
        transition_vals = transition_tuple[2]
        transition_inputs = {var: str(transition_vals[var])
                             for var in env_vars}
        transition_outputs = {var: str(transition_vals[var])
                              for var in sys_vars}
        transition_dict = dict()
        transition_dict['start_state'] = transition_tuple[0]
        transition_dict['end_state'] = transition_tuple[1]
        transition_dict['inputs'] = transition_inputs
        transition_dict['outputs'] = transition_outputs
        transitions.append(transition_dict)
    output['transitions'] = transitions
    # Initial states are the states that have transitions from Sinit. Initial
    # transitions (for the purposes of execution in Stateflow), are the
    # transitions coming from the states that transition from Sinit.
    Sinit_transitions = mealy_machine.transitions.find(from_states=['Sinit'])
    initial_states = dict()
    for _, u, label in Sinit_transitions:
        initial_states[u] = label['loc']
    initial_transitions = mealy_machine.transitions.find(
        from_states=initial_states)
    initial_trans = list()
    for init_transition in initial_transitions:
        transition_vals = init_transition[2]
        trans_dict = dict()
        trans_dict['state'] = init_transition[1]
        trans_dict['inputs'] = {var: str(transition_vals[var])
                                for var in env_vars}
        trans_dict['outputs'] = {var: str(transition_vals[var])
                                 for var in sys_vars}
        orig_state = init_transition[0]
        orig_loc = initial_states[orig_state]
        trans_dict['start_loc'] =  orig_loc
        initial_trans.append(trans_dict)
    output['init_trans'] = initial_trans

    return output
