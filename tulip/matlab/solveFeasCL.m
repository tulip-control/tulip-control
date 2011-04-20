function X0 = solveFeasCL(probStruct, Options)
% SOLVEFEASCL Return the set X0 such that for any x_0 \in X0, there exists 
% a sequence u_0, u_1, ..., u_{N-1} such that for any w_0,...,w_{N-1} \in Wset, 
% the following problem is feasible:
%
%   x_k \in Sset, i.e. L1*x_k <= M1 for all k = 0,...,N-1
%   u_k \in Uset, i.e. L2*u_k <= M2 for all k = 0,...,N-1
%   x_N \in Tset, i.e. L3*x_N <= M3
%                    k-1
%                    ---
%                    \
%   x_k = A^k*x_0 +  /   (A^j*B*u_{k-1-j} + A^j*E*w_{k-1-j})
%                    ---
%                    j=0
%
% This function uses closed loop algorithm. For the difference between the
% closed loop and the open loop algorithm, see Borrelli, F. Constrained
% Optimal Control of Linear and Hybrid Systems, volume 290 of Lecture Notes
% in Control and Information Sciences. Springer. 2003.
%
% USAGE:
%   X0 = solveFeasCL(probStruct)
%   X0 = solveFeasCL(probStruct, Options)
%
% INPUT:
% * probStruct contains the following fields: A, B, E, Uset, Wset, N, Tset,
%   Sset.
% * Options.useAllHorizonLength specifies whether all the horizon length up to
%   probStruct.N can be used.
% * Options.timeout specifies the timeout (in seconds) for polytope union 
%   operation. If negative, the timeout won't be used. Note that using timeout 
%   requires MATLAB parallel computing toolbox. The default value is -1.
% * Options.maxNumPoly specifies the maximum number of polytopes in a
%   region used in computing reachability. The default value is 5.
% * Options.verbose: level of verbosity of the algorithms. The default
%   value is 0.

if (nargin < 2)
    Options = [];
end
useAllHorizonLength = true;
if (isfield(Options,'useAllHorizonLength'))
    useAllHorizonLength = Options.useAllHorizonLength;
end

X0 = probStruct.Tset;
tempProbStruct = probStruct;

E = zeros(size(probStruct.A,1), 1);
Wset = [];
if (isfield(probStruct, 'Wset') && ~isempty(probStruct.Wset) && isfield(probStruct, 'E') && ~isempty(probStruct.E))
    E = probStruct.E;
    Wset = probStruct.Wset;
end

Wextremes = zeros(1,size(E,2));
if (~isempty(Wset))
    [Wextremes,temp,Wset]=extreme(Wset);
end
Wextremes = Wextremes';

for j = probStruct.N:-1:1
    tempProbStruct.Tset = X0;
    tempProbStruct.N = 1;
    tempX0 = solveFeas(tempProbStruct, Wextremes, Options);
    if (~useAllHorizonLength)
        X0 = tempX0;
    else
        try
            X0 = union([X0 tempX0]);
        catch
            display('solveFeasCL: union failed.')
            X0 = tempX0;
        end
    end
end