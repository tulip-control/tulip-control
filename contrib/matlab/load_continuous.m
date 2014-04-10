function [regions, MPTsys, control_weights, simulation_parameters] = ...
    load_continuous(matfile, timestep)

% Load .mat file
TulipObject = load(matfile);


% Get the system
MPTsys = createMPTsys(TulipObject.system_dynamics, timestep);


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
            polytope = Polyhedron('A', polytope_list{1}.A, 'b', polytope_list{1}.b);
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
        MPTsys = createLTIsys(system, timestep);
        
        % Set the domain of the system
        domain = Polyhedron('A', system.domain.A, 'b', system.domain.b);
        MPTsys.x.min = min(domain.V);
        MPTsys.x.max = max(domain.V);
        Uset = Polyhedron('A', system.Uset.A, 'b', system.Uset.b);
        MPTsys.u.min = min(Uset.V);
        MPTsys.u.max = max(Uset.V);
        
    elseif strcmp(system.type, 'PwaSysDyn')

        % Import each LTI system and add it to a list
        num_lti = length(system.subsystems);
        lti_list = cell(1, num_lti);
        for ind = 1:num_lti
            lti_list{ind} = createLTIsys(system.subsystems{ind}, timestep);
        end

        % Create PWASystem from list of LTI systems.
        MPTsys = PWASystem([lti_list{:}]');
        
        % Set domain of system
        domain = Polyhedron('A', system.domain.A, 'b', system.domain.b);
        MPTsys.x.min = min(domain.V);
        MPTsys.x.max = max(domain.V);
        
        % Set input domain of system (need it iterate through all
        % subsystems
        for ind = 1:num_lti
            UsetA = system.subsystems{ind}.Uset.A;
            Usetb = system.subsystems{ind}.Uset.b;
            Uset = Polyhedron('A', UsetA, 'b', Usetb);
            MPTsys.u.min = min([MPTsys.u.min'; Uset.V]);
            MPTsys.u.max = max([MPTsys.u.max'; Uset.V]);
        end
        
    end


end



% Nested function for importing LTIs. 
function LtiSys = createLTIsys(lti_struct, timestep)

    % Create LTI System object
    LtiSys = LTISystem('A', lti_struct.A, 'B', lti_struct.B, 'f', ...
        lti_struct.K, 'Ts', timestep);

    % Create the polytope constraining the domain of the state
    polyA = blkdiag(lti_struct.domain.A, lti_struct.Uset.A);
    polyB = [lti_struct.domain.b; lti_struct.Uset.b];
    state_input_region = Polyhedron('A', polyA, 'b', polyB);

    % Set the domain
    LtiSys.setDomain('xu', state_input_region);
end



end