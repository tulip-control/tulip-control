% Computes the receding horizon controller given a continuous state, an
% index of a target region, and a horizon.


function u = get_input(input)

horizon = input(end);
end_loc = input(end-1);
continuous_state = input(1:end-2);

% Get variables from base workspace
regions = evalin('base', 'regions');
control_weights = evalin('base', 'control_weights');
MPTsys = evalin('base', 'MPTsys');


% Get Chebyshev center of end region
end_region = regions{end_loc+1}.region;
end_cheby = end_region.chebyCenter;
end_cheby = double(end_cheby.x);
offset = control_weights.mid_weight*norm(continuous_state - end_cheby);

% Add weights to system
MPTsys.x.penalty = QuadFunction(control_weights.state_weight, ...
    control_weights.linear_weight, offset);
MPTsys.u.penalty = QuadFunction(control_weights.input_weight);

% Ensure that the final state is within the target set.
MPTsys.x.with('terminalSet');
MPTsys.x.terminalSet = end_region;

% Evaluate Controller
controller = MPCController(MPTsys, horizon);
u = controller.evaluate(continuous_state);

% Remove terminal set filter so that this function doesn't print a warning
% (and slowing down the simulation) next time it's called
MPTsys.x.without('terminalSet');

end
