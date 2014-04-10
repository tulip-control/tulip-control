% File that generates a Simulink model containing a receding horizon input
% and a Stateflow chart representing the Tulip automaton.

% TODO: Support use_all_horizon


load(matfile, 'is_continuous', 'TS');


% Load abstraction, mpt objects for continuous systems. Open blank model.
%-------------------------------------------------------------------------------
if is_continuous
    [regions, MPTsys, control_weights, simulation_parameters, systype] = ...
        load_continuous(matfile, timestep);
end
if strcmp(systype, 'SwitchedSysDyn')
    is_switched = true;
else
    is_switched = false;
end

bdclose(modelname); 
open_system(new_system(modelname))

if ~is_continuous
    set_param(modelname, 'Solver', 'VariableStepDiscrete');
    set_param(modelname, 'MaxStep', num2str(timestep));
end


% Create Tulip Controller
%-------------------------------------------------------------------------------
tulip_controller = add_block('sflib/Chart', [modelname '/TulipController']);



% Get handle on the chart
root = sfroot;
simulink_model = root.find('-isa', 'Simulink.BlockDiagram', '-and', 'Name', ...
    modelname);
mealy_machine = simulink_model.find('-isa', 'Stateflow.Chart', '-and', ...
    'Name', 'TulipController');
mealy_machine.StateMachineType = 'Mealy';


% Set chart time semantics
mealy_machine.ChartUpdate = 'DISCRETE';
if is_continuous
    machine_timestep = timestep*simulation_parameters.horizon;
    mealy_machine.SampleTime = num2str(machine_timestep);
else
    mealy_machine.SampleTime = num2str(timestep);
end


% Add states
num_states = length(TS.states);
state_handles = cell(1, num_states);
for ind = 1:num_states
    state_handles{ind} = Stateflow.State(mealy_machine);
    state_handles{ind}.position = [ 100, 100+150*(ind-1), 50, 50];
    state_handles{ind}.Name = ['s', num2str(ind-1)];
end

% Add inputs and outputs
num_inputs = size(TS.inputs, 1);
num_outputs = size(TS.outputs, 1);
input_handles = cell(1, num_inputs);
output_handles = cell(1, num_outputs);
for ind = 1:num_inputs
    input_handles{ind} = Stateflow.Data(mealy_machine);
    input_handles{ind}.Name = strtrim(TS.inputs(ind,:));
    input_handles{ind}.Scope = 'Input';
    

end
for ind = 1:num_outputs
    output_handles{ind} = Stateflow.Data(mealy_machine);
    output_handles{ind}.Name = strtrim(TS.outputs(ind,:));
    output_handles{ind}.Scope = 'Output';
    
    if (is_continuous && strcmp(output_handles{ind}.Name,'loc'))
        output_handles{ind}.Port = 1;
    end
    
    % TODO: Add handling for env_action and sys_action
end


% Add current location to list of inputs if system is continuous
if is_continuous
    current_loc = Stateflow.Data(mealy_machine);
    current_loc.Name = 'current_loc';
    current_loc.Scope = 'Input';
    current_loc.Port = 1;
end


% Add transitions
num_transitions = length(TS.transitions);
transition_handles = cell(1, num_transitions);
for ind = 1:num_transitions
    start_state_index = double(TS.transitions{ind}.start_state) + 1;
    end_state_index = double(TS.transitions{ind}.end_state) + 1;
    transition_handles{ind} = Stateflow.Transition(mealy_machine);
    transition_handles{ind}.Source = state_handles{start_state_index};
    transition_handles{ind}.Destination = state_handles{end_state_index};
    
    % Label strings on transition
    label_string = '[';
    for jnd = 1:num_inputs
        input_name = input_handles{jnd}.Name;
        input_value = eval(['TS.transitions{ind}.inputs.' input_name]);
        label_string = [label_string, '(', input_name '==' input_value ')', ...
            '&&'];
    end
    label_string = [label_string(1:end-2), ']{'];
    for jnd = 1:num_outputs
        output_name = output_handles{jnd}.Name;
        output_value = eval(['TS.transitions{ind}.outputs.' output_name]);
        label_string = [label_string output_name '=' output_value ';'];
    end
    label_string = [label_string '}'];
    transition_handles{ind}.LabelString = label_string;
end

% Add initial transitions
num_init_transitions = length(TS.init_trans);
init_handles = cell(1, num_init_transitions);
for ind = 1:num_init_transitions
    init_state_index = double(TS.init_trans{ind}.state) + 1;
    init_handles{ind} = Stateflow.Transition(mealy_machine);
    init_handles{ind}.Destination = state_handles{init_state_index};
    init_handles{ind}.DestinationOClock = 9;
    init_handles{ind}.SourceEndPoint = ...
        [state_handles{init_state_index}.Position(1) - 30, ...
         state_handles{init_state_index}.Position(2) + 25];
    init_handles{ind}.MidPoint = ...
        [state_handles{init_state_index}.Position(1) - 15, ...
         state_handles{init_state_index}.Position(2) + 25];
     
    % Label string on initial transitions
    label_string = '[';
    for jnd = 1:num_inputs
        input_name = input_handles{jnd}.Name;
        input_value = eval(['TS.init_trans{ind}.inputs.' input_name]);
        label_string = [label_string, '(', input_name '==' input_value ')', ...
            '&&'];
    end
    
    % Add current location to inputs if system is continuous
    if is_continuous
        current_loc = num2str(double(TS.states{init_state_index}.loc));
        label_string = [label_string '(current_loc==' current_loc ')]{'];
    else
        label_string = [label_string(1:end-2) ']{'];
    end
    
    % Initial outputs
    for jnd = 1:num_outputs
        output_name = output_handles{jnd}.Name;
        output_value = eval(['TS.init_trans{ind}.outputs.' output_name]);
        label_string = [label_string output_name '=' output_value ';'];
    end
    label_string = [label_string '}'];
    
    init_handles{ind}.LabelString = label_string;
end



% RHC blocks for continuous systems
%-------------------------------------------------------------------------------
if is_continuous
    
    % Move this to a good position
    set_param(tulip_controller, 'Position', '[495 27 600 153]');
    
    % Continuous state to discrete state
    c2d_block = add_block('built-in/MATLABFcn', [modelname '/Abstraction']);
    set_param(c2d_block, 'MATLABFcn', 'cont_to_disc');
    set_param(c2d_block, 'Position', '[330 34 420 86]');
        
    % RHC Subsystem Container
    rhc_subsys = add_block('built-in/Subsystem', [modelname '/RHC']);
    set_param(rhc_subsys, 'Orientation', 'left');
    rhc_output = add_block('built-in/Outport', [modelname '/RHC/u']);
    rhc_cont_input = add_block('built-in/Inport', ...
                               [modelname '/RHC/Continuous Input']);
    rhc_loc_input = add_block('built-in/Inport', ...
                              [modelname '/RHC/Location']);
    if is_switched
        rhc_env_input = add_block('built-in/Inport', ...
                                  [modelname '/RHC/Env Action']);
        rhc_sys_input = add_block('built-in/Inport', ...
                                  [modelname '/RHC/Sys Action']);
    end
    set_param(rhc_output, 'Position', '[820 50 840 70]');
    set_param(rhc_cont_input, 'Position', '[40 50 60 70]');
    set_param(rhc_loc_input, 'Position', '[40 110 60 130]');
    set_param(rhc_subsys, 'Position', '[295 364 420 466]');
                           
    % Horizon block (in RHC Subsystem)
    if simulation_parameters.closed_loop
        horizon_block = add_block('sflib/Chart', ...
                                  [modelname '/RHC/Control Horizon']);
        horizon_chart = simulink_model.find('-isa', 'Stateflow.Chart', ...
            '-and', 'Name', 'Control Horizon');
        
        % Output variable
        horizon_output = Stateflow.Data(horizon_chart);
        horizon_output.Name = 'horizon';
        horizon_output.Scope = 'Output';
        
        % Make states and transitions
        N = simulation_parameters.horizon;
        horizon_states = cell(1,N);
        for ind = 1:N
            horizon_states{ind} = Stateflow.State(horizon_chart);
            horizon_states{ind}.Name = ['s' num2str(ind)];
            horizon_states{ind}.position = [ 100, 100+150*(ind-1), 50, 50];
        end
        horizon_transitions = cell(1,N+1);
        for ind = 1:N
            horizon_transitions{ind} = Stateflow.Transition(horizon_chart);
            horizon_transitions{ind}.Source = horizon_states{ind};
            horizon_transitions{ind}.Destination = horizon_states{mod(ind,N)+1};
            horizon_transitions{ind}.LabelString = ...
                ['{horizon=' num2str(N-mod(ind,N)) '}'];
        end
        horizon_init = Stateflow.Transition(horizon_chart);
        horizon_init.Destination = horizon_states{1};
        horizon_init.DestinationOClock = 9;
        horizon_init.LabelString = ['{horizon=' num2str(N) '}'];
 
        % Sample Time
        horizon_chart.ChartUpdate = 'DISCRETE';
        horizon_chart.SampleTime = num2str(timestep);
    else
        horizon_block = add_block('built-in/Constant', ...
                                  [modelname '/Control Horizon']);
        set_param(horizon_block, 'Value', ...
                  num2str(simulation_parameters.horizon));
    end
    set_param(horizon_block, 'Position', '[40 309 100 361]');
    
    if ~is_switched
        % RHC block (in RHC Subsystem)
        rhc_block = add_block('built-in/MATLABFcn', ...
                              [modelname '/RHC/RHC Input']);
        set_param(rhc_block, 'MATLABFcn', 'get_input');
        set_param(rhc_block, 'SampleTime', num2str(timestep));
        input_dim = length(MPTsys.u.max);
        set_param(rhc_block, 'OutputDimensions', num2str(input_dim));
        set_param(rhc_block, 'Position', '[460 27 545 93]');

        % Mux for RHC block (in RHC Subsystem)
        rhc_mux = add_block('built-in/Mux', [modelname '/RHC/RHC Mux']);
        set_param(rhc_mux, 'DisplayOption', 'Bar');
        set_param(rhc_mux, 'Inputs', '3');
        set_param(rhc_mux, 'Position', '[370 19 375 101]');
    end
    
    % The Plant Subsystem
    plant = add_block('built-in/Subsystem', [modelname '/Plant']);
    control_input = add_block('built-in/Inport', [modelname '/Plant/u']);
    cont_state = add_block('built-in/Outport', [modelname '/Plant/contstate']);
    set_param(plant, 'Position', '[120, 197, 240, 263]');
    
    % Draw Transitions
    add_line(modelname, 'Abstraction/1','TulipController/1','autorouting','on');
    add_line(modelname, 'RHC/1', 'Plant/1', 'autorouting', 'on');
    add_line(modelname, 'Plant/1', 'Abstraction/1', 'autorouting', 'on');
    add_line(modelname, 'Plant/1', 'RHC/1', 'autorouting', 'on');
    add_line(modelname, 'TulipController/1', 'RHC/2', 'autorouting', 'on');
    if ~is_switched
        add_line([modelname '/RHC'], 'RHC Mux/1', 'RHC Input/1', ...
            'autorouting', 'on');
        add_line([modelname '/RHC'], 'Control Horizon/1', 'RHC Mux/3', ...
            'autorouting', 'on');
        add_line([modelname '/RHC'], 'Continuous Input/1', 'RHC Mux/1', ...
            'autorouting', 'on');
        add_line([modelname '/RHC'], 'Location/1', 'RHC Mux/2', 'autorouting',...
            'on');
        add_line([modelname '/RHC'], 'RHC Input/1', 'u/1', 'autorouting', 'on');
    end
end