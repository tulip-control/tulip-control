% Define parameters for the model
matfile = 'robot_continuous.mat';
modelname = 'Robot_Continuous';
timestep = .1;


% Load the model
load_tulip;


% Add LTI system into the plant
sysA = zeros(2,2);
sysB = eye(2);
sysC = eye(2);
sysD = zeros(2,2);
sysc = ss(sysA, sysB, sysC, sysD);
x_init = [1.5; .5];

lti_block = add_block('cstblocks/LTI System', [modelname '/Plant/robotdyn']);
set_param(lti_block, 'sys', 'sysc');
set_param(lti_block, 'IC', 'x_init');

set_param([modelname '/Plant/u'], 'Position', '[15 45 35 65]');
set_param([modelname '/Plant/contstate'], 'Position', '[295, 45, 315, 65]');
set_param(lti_block, 'Position', '[120 37 215 73]');

add_line([modelname '/Plant'], 'u/1', 'robotdyn/1', 'autorouting', 'on');
add_line([modelname '/Plant'], 'robotdyn/1', 'contstate/1', 'autorouting','on');


% Add park signal generator
park_signal = add_block('built-in/Subsystem', [modelname '/Park Signal']);
random_number = add_block('built-in/UniformRandomNumber', ...
    [modelname '/Park Signal/Random Number']);
rounder = add_block('built-in/Rounding', [modelname '/Park Signal/Round']);
park_out = add_block('built-in/Outport', [modelname '/Park Signal/park']);
set_param(random_number, 'Position', '[60 32 95 68]');
set_param(rounder, 'Position', '[180 32 215 68]');
set_param(park_out, 'Position', '[305 42 335 58]');
set_param(park_signal, 'Position', '[360 109 390 141]');
set_param(random_number, 'Maximum', '1');
set_param(random_number, 'Minimum', '0');
set_param(random_number, 'SampleTime', num2str(machine_timestep));
set_param(rounder, 'Operator', 'round');
add_line(modelname, 'Park Signal/1', 'TulipController/2');
add_line([modelname '/Park Signal'], 'Random Number/1', 'Round/1');
add_line([modelname '/Park Signal'], 'Round/1', 'park/1');


% Add scopes to outputs of discrete outputs, continuous state, park signal,
% and continuous input
scope_home = add_block('built-in/Scope', [modelname '/Home']);
scope_lot = add_block('built-in/Scope', [modelname '/Lot']);
scope_X0reach = add_block('built-in/Scope', [modelname '/X0reach']);
scope_state = add_block('built-in/Scope', [modelname '/Continuous State']);
scope_park = add_block('built-in/Scope', [modelname '/Park Output']);
scope_loc = add_block('built-in/Scope', [modelname '/Loc']);
scope_u = add_block('built-in/Scope', [modelname '/u']);
set_param(scope_u, 'Orientation', 'left');
set_param(scope_home, 'Position', '[695 105 725 135]');
set_param(scope_lot, 'Position', '[695 175 725 205]');
set_param(scope_X0reach, 'Position', '[695 240 725 270]');
set_param(scope_park, 'Position', '[445 155 475 185]');
set_param(scope_loc, 'Position', '[695 29 725 61]');
set_param(scope_state, 'Position', '[330 260 360 290]');
set_param(scope_u, 'Position', '[40 270 70 300]');
add_line(modelname, 'TulipController/1', 'Loc/1', 'autorouting', 'on');
add_line(modelname, 'TulipController/2', 'Home/1', 'autorouting', 'on');
add_line(modelname, 'TulipController/3', 'Lot/1', 'autorouting', 'on');
add_line(modelname, 'TulipController/4', 'X0reach/1', 'autorouting', 'on');
add_line(modelname, 'Plant/1', 'Continuous State/1', 'autorouting', 'on');
add_line(modelname, 'Park Signal/1', 'Park Output/1', 'autorouting', 'on');
add_line(modelname, 'RHC/1', 'u/1', 'autorouting', 'on');