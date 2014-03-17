function u = get_input(continuous_state, MPTsys, end_loc, regions, ...
    control_weights, horizon)

% Get Chebyshev center of end region
end_region = regions(end_loc).region;
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

end