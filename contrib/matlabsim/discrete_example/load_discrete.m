% Define parameters
matfile = 'robot_discrete.mat';
timestep = 1;
modelname = 'Robot_Discrete';

% Load the model
load_tulip;

% Set position
set_param([modelname '/TulipController'], 'Position', '[180 34 250 126]');

% Add Park signal generator
park_signal = add_block('built-in/Subsystem', [modelname '/Park Signal']);
random_number = add_block('built-in/UniformRandomNumber', ...
    [modelname '/Park Signal/Random Number']);
rounder = add_block('built-in/Rounding', [modelname '/Park Signal/Round']);
park_out = add_block('built-in/Outport', [modelname '/Park Signal/park']);
set_param(random_number, 'Position', '[60 32 95 68]');
set_param(random_number, 'Minimum', '0');
set_param(random_number, 'Maximum', '1');
set_param(random_number, 'SampleTime', num2str(timestep));
set_param(rounder, 'Position', '[180 32 215 68]');
set_param(rounder, 'Operator', 'round');
set_param(park_out, 'Position', '[305 42 335 58]');
set_param(park_signal, 'Position', '[25 65 90 105]');
add_line(modelname, 'Park Signal/1', 'TulipController/1');
add_line([modelname '/Park Signal'], 'Random Number/1', 'Round/1');
add_line([modelname '/Park Signal'], 'Round/1', 'park/1');

% Add scopes
scope_home = add_block('built-in/Scope', [modelname '/Home']);
scope_lot = add_block('built-in/Scope', [modelname '/Lot']);
scope_X0reach = add_block('built-in/Scope', [modelname '/X0reach']);
scope_park = add_block('built-in/Scope', [modelname '/Park Output']);
scope_loc = add_block('built-in/Scope', [modelname '/Loc']);
set_param(scope_home, 'Position', '[335 20 365 50]');
set_param(scope_lot, 'Position', '[335 75 365 105]');
set_param(scope_X0reach, 'Position', '[335 135 365 165]');
set_param(scope_park, 'Position', '[135 109 165 141]');
set_param(scope_loc, 'Position', '[335 195 365 225]');

% Connect blocks
add_line(modelname, 'Park Signal/1', 'Park Output/1', 'autorouting', 'on');
add_line(modelname, 'TulipController/1', 'Home/1', 'autorouting', 'on');
add_line(modelname, 'TulipController/2', 'Lot/1', 'autorouting', 'on');
add_line(modelname, 'TulipController/3', 'X0reach/1', 'autorouting', 'on');
add_line(modelname, 'TulipController/4', 'Loc/1', 'autorouting', 'on');