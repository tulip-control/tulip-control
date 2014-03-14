function MPTsys = createMPTsys(matfile, timestep)
% Loads the contents of a .mat file created by tompt.export() into a MPT
% model
%
% Notes:
%   - Domains of LTI systems are in input-state space. 
%   - Using timestep = .1 second until time semantics are implemented for
%     Lti and Pwa systems in Tulip.


% Load the file containing a variable called "TulipSystem"
TulipSystem = load(matfile);
TulipSystem = TulipSystem.TulipSystem;


if strcmp(TulipSystem.type, 'LtiSysDyn')
    MPTsys = createLTIsys(TulipSystem);
    
elseif strcmp(TulipSystem.type, 'PwaSysDyn')
    
    % Import each Lti and add it to a list
    n_lti = length(TulipSystem.subsystems);
    lti_cell = cell(1, n_lti);
    lti_list = TulipSystem.subsystems;
    for ind = 1:n_lti
        lti_cell{ind} = createLTIsys(lti_list{ind});
    end
    
    % Create PWASystem from list of ltis.
    MPTsys = PWASystem([lti_cell{:}]');
end


    % Nested function for  import LTIs. 
    function LtiSys_ret = createLTIsys(LtiSys)
        % Create LTI System object
        LtiSys_ret = LTISystem('A', LtiSys.A, 'B', LtiSys.B, 'f', LtiSys.K, ...
            'Ts', timestep);

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