clear all; close all; clc;


%-------------------------------------------------------------------------------
% Edit input arguments here
%-------------------------------------------------------------------------------
matfile = 'tulip_output.mat';
modelname = 'Tulip_Model';
timestep = 1;



%-------------------------------------------------------------------------------
% % Do not edit below this line
%-------------------------------------------------------------------------------
bdclose(modelname); 
open_system(new_system(modelname))


% Load apparatus for continuous systems if the system is continuous
load(matfile, 'is_continuous', 'TS');
if is_continuous
    [regions, MPTsys, control_weights, horizon] = ...
        load_continuous(matfile, timestep);
end


% CREATE TULIP CONTROLLER
add_block('sflib/Chart', [modelname '/TulipController']);


% Get handle on the chart
root = sfroot;
simulink_model = root.find('-isa', 'Simulink.BlockDiagram', '-and', 'Name', ...
    modelname);
mealy_machine = simulink_model.find('-isa', 'Stateflow.Chart', '-and', ...
    'Name', 'TulipController');
mealy_machine.StateMachineType = 'Mealy';


% Set chart time semantics
if is_continuous
    
else
    mealy_machine.ChartUpdate = 'DISCRETE';
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
    label_string = '[';
    for jnd = 1:num_inputs
        input_name = input_handles{jnd}.Name;
        input_value = eval(['TS.transitions{ind}.inputs.' input_name]);
        label_string = [label_string, '(', input_name '==' input_value ')', ...
            '&&'];
    end
    label_string = [label_string(1:end-2) ']'];
    init_handles{ind}.LabelString = label_string;
end


% CREATE RHC BLOCKS
if is_continuous
    
end