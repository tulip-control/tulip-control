function [regions, MPTsys, control_weights, horizon] = load_continuous(...
    matfile, timestep)

% Load .mat file
TulipObject = load(matfile);

% Get the system
MPTsys = createMPTsys(TulipObject.system_dynamics, timestep);

% Pull abstraction from .mat file
regions = createAbstraction(TulipObject.abstraction.abstraction);
regions = [regions{:}];

% Get control weights and parameters
horizon = double(TulipObject.sim_params.horizon);
control_weights.state_weight = double(TulipObject.sim_params.state_weight);
control_weights.input_weight = double(TulipObject.sim_params.input_weight);
control_weights.linear_weight = double(TulipObject.sim_params.linear_weight);
control_weights.mid_weight = double(TulipObject.sim_params.mid_weight);


%------------------------------------------------------------------------------%
% Nested functions called above
%------------------------------------------------------------------------------%


function region_list = createAbstraction(abstraction)
% Returns a list of Polyhedra.

num_regions = length(abstraction);
region_list = cell(1, num_regions);

for ind = 1:num_regions
    region_index = abstraction{ind}.index;
    polytope_list = abstraction{ind}.region.list_poly;
    
    if length(polytope_list) > 1
        little_poly_list = cell(1, length(polytope_list));
        for jnd = 1:length(polytope_list)
            little_poly = polytope_list{jnd};
            little_poly_list{jnd} = Polyhedron('A', little_poly.A, 'b', ...
                little_poly.b);
        end
        little_poly_list = PolyUnion([little_poly_list{:}]');
        little_poly_list.merge;
        polytope = little_poly_list.Set;
    else
        polytope = polytope_list{1};
    end
    %num_polytopes = length(polytope_list);
    %polyhedron_list = cell(1, num_polytopes);
    
    %for jnd = 1:num_polytopes
    %    polytope = polytope_list{jnd};
    %    polyhedron_list{jnd} = Polyhedron('A', polytope.A, 'b', polytope.b);
    %end
    %polyhedron_list = PolyUnion([polyhedron_list{:}]');
    
    region_list{ind}.index = region_index;
    region_list{ind}.region = Polyhedron('A', polytope.A, 'b', polytope.b);
end

end




function MPTsys = createMPTsys(system, timestep)
% Takes a struct exported from Python and imports a 
%
% Notes:
%   - Domains of LTI systems are in input-state space. 
%   - Using timestep = .1 second until time semantics are implemented for
%     Lti and Pwa systems in Tulip.


if strcmp(system.type, 'LtiSysDyn')
    MPTsys = createLTIsys(system, timestep);
    
    % set the domain for the Ltisystem
    domain = Polyhedron('A', system.domain.A, 'b', system.domain.b);
    domain_vertices = domain.V;
    MPTsys.x.min = min(domain_vertices);
    MPTsys.x.max = max(domain_vertices);
    Uset = Polyhedron('A', system.Uset.A, 'b', system.Uset.b);
    Uset_vertices = Uset.V;
    MPTsys.u.min = min(Uset_vertices);
    MPTsys.u.max = max(Uset_vertices);

elseif strcmp(system.type, 'PwaSysDyn')

    % Import each Lti and add it to a list
    n_lti = length(system.subsystems);
    lti_cell = cell(1, n_lti);
    lti_list = system.subsystems;
    for ind = 1:n_lti
        lti_cell{ind} = createLTIsys(lti_list{ind}, timestep);
    end

    % Create PWASystem from list of ltis.
    MPTsys = PWASystem([lti_cell{:}]');
    
    % Set the domain
    domain = Polyhedron('A', system.domain.A, 'b', system.domain.b);
    domain_vertices = domain.V;
    MPTsys.x.min = min(domain_vertices);
    MPTsys.x.max = max(domain_vertices);
    Uset = Polyhedron('A', system.Uset.A, 'b', system.Uset.b);
    Uset_vertices = Uset.V;
    MPTsys.u.min = min(Uset_vertices);
    MPTsys.u.max = max(Uset_vertices);
end


    % Nested function for importing LTIs. 
    function LtiSys_ret = createLTIsys(LtiSys, timestep)
        % Create LTI System object
        LtiSys_ret = LTISystem('A', LtiSys.A, 'B', LtiSys.B, 'f', ...
            LtiSys.K, 'Ts', timestep);

        % Number of polytopic constraints on state and input
        poly_n_state = size(LtiSys.domain.A, 1);
        poly_n_inputs = size(LtiSys.Uset.A, 1);

        % Dimension of state and inputs
        dim_state = size(LtiSys.A, 2);
        dim_inputs = size(LtiSys.B, 2);

        % Create the polytope constraining the domain of the state
        polyA = [LtiSys.domain.A, zeros(poly_n_state, dim_inputs); ...
                 zeros(poly_n_inputs, dim_state), LtiSys.Uset.A];
        polyB = [LtiSys.domain.b; LtiSys.Uset.b];
        state_input_region = Polyhedron('A', polyA, 'b', polyB);

        % Set the domain
        LtiSys_ret.setDomain('xu', state_input_region);
    end
end



end