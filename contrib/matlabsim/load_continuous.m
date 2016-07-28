function [regions, MPTsys, control_weights, simulation_parameters, systype] =...
    load_continuous(matfile, timestep)

% Load .mat file
TulipObject = load(matfile);

% Get the system
MPTsys = createMPTsys(TulipObject.system_dynamics, timestep);
systype = TulipObject.system_dynamics.type;

% List of regions in each abstraction and polytopes in each abstraction
regions = createAbstraction(TulipObject.abstraction.abstraction);


% Control weights in receding horizon control
control_weights.state_weight = double(TulipObject.control_weights.state_weight);
control_weights.input_weight = double(TulipObject.control_weights.input_weight);
control_weights.linear_weight = ...
    double(TulipObject.control_weights.linear_weight);
control_weights.mid_weight = double(TulipObject.control_weights.mid_weight);


% Simulation parameters
simulation_parameters = TulipObject.simulation_parameters;
simulation_parameters.horizon = double(simulation_parameters.horizon);



%------------------------------------------------------------------------------%
% Nested functions called above
%------------------------------------------------------------------------------%


% Reads the abstraction exported from Python and returns a cell array of
% Polyhedra structs. Each struct contains the location number of a region
% in the abstraction and a Polyhedron representing the region itself.
function region_list = createAbstraction(abstraction)

    num_regions = length(abstraction);
    region_list = cell(1, num_regions);

    for ind = 1:num_regions
        region_index = abstraction{ind}.index;
        polytope_list = abstraction{ind}.region.list_poly;

        % Because target sets must be in Polyhedron class in MPT, combine all
        % polyhedra within a region into one polyhedron.
        if length(polytope_list) > 1
            polytopes_in_region = cell(1, length(polytope_list));
            for jnd = 1:length(polytope_list)
                current_polytope_struct = polytope_list{jnd};
                polytopes_in_region{jnd} = Polyhedron('A', ...
                    current_polytope_struct.A, 'b', current_polytope_struct.b);
            end
            polytopes_in_region = PolyUnion([polytopes_in_region{:}]');
            polytopes_in_region.merge;
            polytope = polytopes_in_region.Set;
        else
            polytope = Polyhedron('A', polytope_list{1}.A, 'b', ...
                polytope_list{1}.b);
        end

        region_list{ind}.index = double(region_index);
        region_list{ind}.region = polytope;
    end

end



% Takes a struct exported from Python and imports a
%
% Notes:
%   - Domains of LTI systems are in input-state space.
%   - Forcing timestep to be an argument until time-semantics are
%     implemented in Tulip
function MPTsys = createMPTsys(system, timestep)

    if strcmp(system.type, 'LtiSysDyn')
        [MPTsys, xu] = createLTIsys(system, timestep);
        MPTsys = set_lti_bounds(MPTsys, xu, system);
    elseif strcmp(system.type, 'PwaSysDyn')
        MPTsys = createPWAsys(system, timestep);
    elseif strcmp(system.type, 'SwitchedSysDyn')
        num_modes = length(system.dynamics);
        MPTsys(num_modes).system = [];
        for ind = 1:num_modes
            MPTsys(ind).system = createPWAsys(system.dynamics{ind}.pwasys, ...
                timestep);
            MPTsys(ind).env_act = system.dynamics{ind}.env_act;
            MPTsys(ind).sys_act = system.dynamics{ind}.sys_act;
        end
    else
        error(['System type "' num2str(system.type) '" not recognized.']);
    end


end


function MPTsys = createPWAsys(system, timestep)
    % Import each LTI system and add it to a list
    num_lti = length(system.subsystems);
    lti_list = cell(1, num_lti);
    xu_list = cell(1, num_lti);
    for ind = 1:num_lti
        [lti_list{ind}, xu_list{ind}] = createLTIsys(system.subsystems{ind}, ...
            timestep);
        lti_list{ind} = set_lti_bounds(lti_list{ind}, xu_list{ind}, ...
            system.subsystems{ind});
    end

    % Create PWASystem from list of LTI systems.
    MPTsys = PWASystem([lti_list{:}]');

    % Set domain of system
    domain = Polyhedron('A', system.domain.A, 'b', system.domain.b);
    MPTsys.x.min = min(domain.V);
    MPTsys.x.max = max(domain.V);

    % Set input domain of system (need it iterate through all
    % subsystems
    MPTsys.u.min = lti_list{1}.u.min';
    MPTsys.u.max = lti_list{1}.u.max';
    for ind = 1:num_lti
        ltisys = lti_list{ind};
        MPTsys.u.min = min([MPTsys.u.min'; ltisys.u.min']);
        MPTsys.u.max = max([MPTsys.u.max'; ltisys.u.max']);
    end
end


% Nested function for importing LTIs.
function [LtiSys, state_input_region] = createLTIsys(lti_struct, timestep)

    % Create LTI System object
    LtiSys = LTISystem('A', double(lti_struct.A), ...
        'B', double(lti_struct.B), ...
        'f', double(lti_struct.K), ...
        'Ts', double(timestep));

    % Create the polytope constraining the domain of the state. How it's
    % done depends on whether it uses state and input constraints
    % separately or state-input constraints.
    input_dim = size(lti_struct.B, 2);
    if size(lti_struct.Uset.A,2) == input_dim
        polyA = blkdiag(lti_struct.domain.A, lti_struct.Uset.A);
        polyB = [lti_struct.domain.b; lti_struct.Uset.b];
        state_input_region = Polyhedron('A', polyA, 'b', polyB);
    else
        rows_A1 = size(lti_struct.domain.A,1);
        UA = lti_struct.Uset.A;
        UA = [UA(:,input_dim+1:end), UA(:,1:input_dim)];
        polyA = [lti_struct.domain.A, zeros(rows_A1,input_dim); ...
                 UA];
        polyB = [lti_struct.domain.b; lti_struct.Uset.b];
        state_input_region = Polyhedron('A', polyA, 'b', polyB);
    end

    % Set the domain
    LtiSys.setDomain('xu', state_input_region);
end


% Function that sets the state and input bounds of lti systems
function ltisys = set_lti_bounds(ltisys, state_input_region, lti_struct)

    % Set the domain of the system for big-M relaxation
    poly_state = Polyhedron('A', lti_struct.domain.A, 'b', lti_struct.domain.b);
    ltisys.x.min = min(poly_state.V);
    ltisys.x.max = max(poly_state.V);

    % Domain of inputs
    state_dim = size(lti_struct.A, 2);
    ltisys.u.min = min(state_input_region.V(:,state_dim+1:end));
    ltisys.u.max = max(state_input_region.V(:,state_dim+1:end));
end


end
