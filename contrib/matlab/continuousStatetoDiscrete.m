% Takes a continuous state and a list of PolyUnions and then returns the
% index of the region that the continuous state is in. This gets us the
% variable "loc" in continuous systems. Throws an error if a point is
% outside the domain covered by the PolyUnions in the list regions.

function discrete_state = continuousStatetoDiscrete(state, regions)

discrete_state = -1;

for i = 1:length(regions)
    [isin, ~, ~] = regions{i}.region.contains(state, 1);
    if isin
        discrete_state = regions{i}.index;
        break
    end
end

if discrete_state == -1
    error(['State ', num2str(state'), ' not in operating domain'])
end

end