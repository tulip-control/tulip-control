% Takes a continuous state and a list of Polyhedra and then returns the
% index of the region that the continuous state is in. This gets us the
% variable "loc" when working with a continuous system.

% Function throws an error if a point is outside the specified domain.

function discrete_state = cont_to_disc(state)

% Get regions struct from base workspace
regions = evalin('base','regions');

% Initialize value of discrete variable
discrete_state = -1;

% Find what region we're in
for i = 1:length(regions)
    if regions{i}.region.contains(state, 1)
        discrete_state = regions{i}.index;
        break
    end
end

% Throw error if we didn't land in any region
if discrete_state == -1
    error(['State ', num2str(state'), ' not in operating domain'])
end

end
