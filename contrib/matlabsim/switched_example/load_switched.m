clear; close all; clc;

matfile = 'fuel_tank';
timestep = 1;
modelname = 'Fuel_Tank';

% Load Tulip Framework
load_tulip;

% Original discrete system
fuel_consumption = 1;
input_lb = 0;
input_ub = 3;
refill_rate = 3;
Ad = eye(2);
Bd = [-1; 1];
K1d = [0; -fuel_consumption];
K2d = [refill_rate; -fuel_consumption];

% Convert to continuous time
linearization_timestep = 1;
A = zeros(2,2);
B = Bd;
x0 = zeros(2,1);
x0dot1 = K1d;
x0dot2 = K2d;

% Initial state
x_init = [7.5; 7.8];

% Fill in (ideal) plant of Simulink Model
set_param(cont_state, 'Position', '[665 100 685 120]');
set_param(control_input, 'Position', '[30 120 50 140]');

plant_env_mode = add_block('built-in/Inport', [modelname '/Plant/mode']);
set_param(plant_env_mode, 'Position', '[30 180 50 200]');

integrator = add_block('built-in/Integrator', [modelname '/Plant/Integrator']);
set_param(integrator, 'Position', '[575 95 605 125]');
set_param(integrator, 'InitialCondition', 'x_init');

mode_if = add_block('built-in/If', [modelname '/Plant/If']);
normal_mode_key = num2str(input_value_map('normal'));
set_param(mode_if, 'Position', '[90 171 190 209]');
set_param(mode_if, 'IfExpression', ['u1 == ' normal_mode_key]);

mode_merge = add_block('built-in/Merge', [modelname '/Plant/Merge']);
set_param(mode_merge, 'Position', '[405 150 445 190]');

normalmode_subsys = add_block('built-in/Subsystem', ...
    [modelname '/Plant/Normal Mode']);
set_param(normalmode_subsys, 'Position', '[275 200 335 230]');
normalmode_const = add_block('built-in/Constant', ...
    [modelname '/Plant/Normal Mode/drift']);
set_param(normalmode_const, 'Position', '[25 85 55 115]');
set_param(normalmode_const, 'Value', 'x0dot1');
normalmode_out = add_block('built-in/Outport', ...
    [modelname '/Plant/Normal Mode/x0dot']);
set_param(normalmode_out, 'Position', '[160 90 180 110]');
normalmode_act = add_block('built-in/ActionPort', ...
    [modelname '/Plant/Normal Mode/If Action']);
set_param(normalmode_act, 'Position', '[80 15 140 45]');
add_line([modelname '/Plant/Normal Mode'], 'drift/1', 'x0dot/1');

refuelmode_subsys = add_block('built-in/Subsystem', ...
    [modelname '/Plant/Refuel Mode']);
set_param(refuelmode_subsys, 'Position', '[210 245 270 275]');
refuelmode_const = add_block('built-in/Constant', ...
    [modelname '/Plant/Refuel Mode/drift']);
set_param(refuelmode_const, 'Position', '[25 85 55 115]')
set_param(refuelmode_const, 'Value', 'x0dot2');
refuelmode_out = add_block('built-in/Outport', ...
    [modelname '/Plant/Refuel Mode/x0dot']);
set_param(refuelmode_out, 'Position', '[160 90 180 110]');
refuelmode_act = add_block('built-in/ActionPort', ...
    [modelname '/Plant/Refuel Mode/If Action']);
set_param(refuelmode_act, 'Position', '[80 15 140 45]');
add_line([modelname '/Plant/Refuel Mode'], 'drift/1', 'x0dot/1');

B_matrix = add_block('built-in/Gain', [modelname '/Plant/B']);
set_param(B_matrix, 'Position', '[80 115 110 145]');
set_param(B_matrix, 'Multiplication', 'Matrix(K*u)');
set_param(B_matrix, 'Gain', 'B');

A_matrix = add_block('built-in/Gain', [modelname '/Plant/A']);
set_param(A_matrix, 'Position', '[170 35 200 65]');
set_param(A_matrix, 'Multiplication', 'Matrix(K*u)');
set_param(A_matrix, 'Gain', 'A');

const = add_block('built-in/Constant', [modelname '/Plant/drift']);
set_param(const, 'Position', '[125 75 155 105]');
set_param(const, 'Value', 'A*x0');

add_stuff = add_block('built-in/Sum', [modelname '/Plant/Add']);
set_param(add_stuff, 'Position', '[485 30 540 190]');
set_param(add_stuff, 'Inputs', '++++');

add_line([modelname '/Plant'], 'Integrator/1', 'contstate/1');
add_line([modelname '/Plant'], 'Add/1', 'Integrator/1');

add_line([modelname '/Plant'], 'mode/1', 'If/1');
add_line([modelname '/Plant'], 'u/1', 'B/1');
add_line([modelname '/Plant'], 'Integrator/1', 'A/1', 'autorouting', 'on');
add_line([modelname '/Plant'], 'A/1', 'Add/1');
add_line([modelname '/Plant'], 'drift/1', 'Add/2');
add_line([modelname '/Plant'], 'B/1', 'Add/3');
add_line([modelname '/Plant'], 'Merge/1', 'Add/4', 'autorouting', 'on');
add_line([modelname '/Plant'], 'If/1', 'Normal Mode/Ifaction', ...
    'autorouting', 'on');
add_line([modelname '/Plant'], 'If/2', 'Refuel Mode/Ifaction', ...
    'autorouting', 'on');
add_line([modelname '/Plant'], 'Normal Mode/1', 'Merge/1', 'autorouting', 'on');
add_line([modelname '/Plant'], 'Refuel Mode/1', 'Merge/2', 'autorouting', 'on');


% Make environment signal
refueling = add_block('built-in/Subsystem', [modelname '/Refuel Signal']);
set_param(refueling, 'Position', '[330 125 420 185]');

refuel_signal = add_block('built-in/Outport', [modelname '/Refuel Signal/out']);
set_param(refuel_signal, 'Position', '[605 35 625 55]');

add_line(modelname, 'Refuel Signal/1', 'TulipController/2', ...
    'autorouting', 'on');
add_line(modelname, 'Refuel Signal/1', 'Plant/2', 'autorouting', 'on');
add_line(modelname, 'Refuel Signal/1', 'RHC/3', 'autorouting', 'on');

time_block = add_block('built-in/Clock', [modelname '/Refuel Signal/Time']);
set_param(time_block, 'Position', '[30 35 50 55]');

iftime = add_block('built-in/If', [modelname '/Refuel Signal/If']);
set_param(iftime, 'Position', '[110 17 220 68]');
set_param(iftime, 'IfExpression', 'u1 >= 14');

zero_subsys = add_block('built-in/Subsystem', ...
    [modelname '/Refuel Signal/Normal Mode']);
set_param(zero_subsys, 'Position', '[245 120 330 160]');
zero_act = add_block('built-in/ActionPort', ...
    [modelname '/Refuel Signal/Normal Mode/If Port']);
set_param(zero_act, 'Position', '[75 15 145 50]');
zero_const = add_block('built-in/Constant', ...
    [modelname '/Refuel Signal/Normal Mode/Zero']);
set_param(zero_const, 'Position', '[25 87 85 123]');
set_param(zero_const, 'Value', num2str(input_value_map('normal')));
zero_out = add_block('built-in/Outport', ...
    [modelname '/Refuel Signal/Normal Mode/Out']);
set_param(zero_out, 'Position', '[165 95 185 115]');
add_line([modelname '/Refuel Signal/Normal Mode'], 'Zero/1', 'Out/1');

one_subsys = add_block('built-in/Subsystem', ...
    [modelname '/Refuel Signal/Refuel Mode']);
set_param(one_subsys, 'Position', '[345 55 425 95]');
one_act = add_block('built-in/ActionPort', ...
    [modelname '/Refuel Signal/Refuel Mode/If Port']);
set_param(one_act, 'Position', '[75 15 145 50]');
one_const = add_block('built-in/Constant', ...
    [modelname '/Refuel Signal/Refuel Mode/One']);
set_param(one_const, 'Position', '[25 87 85 123]');
set_param(one_const, 'Value', num2str(input_value_map('refuel')));
one_out = add_block('built-in/Outport', ...
    [modelname '/Refuel Signal/Refuel Mode/Out']);
set_param(one_out, 'Position', '[165 95 185 115]');
add_line([modelname '/Refuel Signal/Refuel Mode'], 'One/1', 'Out/1');

mergetime = add_block('built-in/Merge', [modelname '/Refuel Signal/Merge']);
set_param(mergetime, 'Position', '[505 17 555 68]');

add_line([modelname '/Refuel Signal'], 'If/1', 'Refuel Mode/Ifaction', ...
    'autorouting', 'on');
add_line([modelname '/Refuel Signal'], 'If/2', 'Normal Mode/Ifaction', ...
    'autorouting', 'on');
add_line([modelname '/Refuel Signal'], 'Refuel Mode/1', 'Merge/1', ...
    'autorouting', 'on');
add_line([modelname '/Refuel Signal'], 'Normal Mode/1', 'Merge/2', ...
    'autorouting', 'on');
add_line([modelname '/Refuel Signal'], 'Time/1', 'If/1');
add_line([modelname '/Refuel Signal'], 'Merge/1', 'out/1');

% Model outputs
no_refuel = add_block('built-in/Scope', [modelname '/no_refuel']);
set_param(no_refuel, 'Position', '[705 20 725 40]');
vol_diff = add_block('built-in/Scope', [modelname '/vol_diff']);
set_param(vol_diff, 'Position', '[705 60 725 80]');
critical = add_block('built-in/Scope', [modelname '/critical']);
set_param(critical, 'Position', '[705 100 725 120]');
init_bool = add_block('built-in/Scope', [modelname '/initial']);
set_param(init_bool, 'Position', '[705 140 725 160]');

location = add_block('built-in/Scope', [modelname '/loc']);
set_param(location, 'Position', '[705 180 725 200]');
env_out = add_block('built-in/Scope', [modelname '/env_action']);
set_param(env_out, 'Position', '[495 200 515 220]');
u_out = add_block('built-in/Scope', [modelname '/u']);
set_param(u_out, 'Position', '[235 375 255 395]');
x_out = add_block('built-in/Scope', [modelname '/x']);
set_param(x_out, 'Position', '[345 255 365 275]');

add_line(modelname, 'TulipController/3', 'no_refuel/1', 'autorouting', 'on');
add_line(modelname, 'TulipController/4', 'vol_diff/1', 'autorouting', 'on');
add_line(modelname, 'TulipController/5', 'critical/1', 'autorouting', 'on');
add_line(modelname, 'TulipController/6', 'initial/1', 'autorouting', 'on');
add_line(modelname, 'TulipController/1', 'loc/1', 'autorouting', 'on');
add_line(modelname, 'Refuel Signal/1', 'env_action/1', 'autorouting', 'on');
add_line(modelname, 'Plant/1', 'x/1', 'autorouting', 'on');
add_line(modelname, 'RHC/1', 'u/1', 'autorouting', 'on');