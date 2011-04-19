function [done, ret] = unionWithTimeout(poly, timeout)
% UNIONWITHTIMEOUT compute the union of polytopes given the timeout.
% If MPT doesn't finish computing the union within timeout, then an empty
% polytope will be returned.
%
% USAGE:
%   [done, ret] = unionWithTimeout(poly, timeout)
%
% INPUT:
% * poly: MPT polytope object
% * timeout: timeout in seconds
%
% OUTPUT:
% * done: whether MPT finishes computing the union within the timeout
% * ret: the resulting polytope


ret = polytope;
done = false;
j = createJob();
t = createTask(j, @unionN, 1, {poly});
submit(j)

starttime = clock;
% lastprint = clock;
while (~done && etime(clock, starttime) < timeout)
    pause(5);
    if (strcmp(j.State, 'finished'))
        done = true;
        out = getAllOutputArguments(j);
        ret = out{1};
    end
%     if (etime(clock, lastprint) > 1)
%         display('Timer running');
%         display([j.State ' ' num2str(etime(clock, starttime))])
%         lastprint = clock;
%     end
end
destroy(t);
destroy(j);


function ret = unionN(poly)
	ret = union(poly);
end

end