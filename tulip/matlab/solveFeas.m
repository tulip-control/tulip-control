function X0 = solveFeas(probStruct, W, Options)
% SOLVEFEAS Return the set X0 such that for any x_0 \in X0, there exists 
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
% This function uses open loop algorithm. For the difference between the
% closed loop and the open loop algorithm, see Borrelli, F. Constrained
% Optimal Control of Linear and Hybrid Systems, volume 290 of Lecture Notes
% in Control and Information Sciences. Springer. 2003.
%
% USAGE:
%   X0 = solveFeas(probStruct)
%   X0 = solveFeas(probStruct, W)
%   X0 = solveFeas(probStruct, W, Options)
%
% INPUT:
% * probStruct contains the following fields: A, B, E, Uset, Wset, N, Tset,
%   Sset.
% * W is the set of all the extreme points of Wset^N.
%   If W = [], the set of all the extreme points of Wset^N 
%   will be automatically computed.
% * Options.timeout specifies the timeout (in seconds) for polytope union 
%   operation. If negative, the timeout won't be used. Note that using timeout 
%   requires MATLAB parallel computing toolbox. The default value is -1.
% * Options.maxNumPoly specifies the maximum number of polytopes in a
%   region used in computing reachability. The default value is 5.
% * Options.verbose: level of verbosity of the algorithms. The default
%   value is 0.

if (nargin < 3)
    Options = [];
end

verbose = 0;
maxNumPoly = 5;
timeout = -1;
if (isfield(Options, 'verbose'))
    verbose = Options.verbose;
end
if (isfield(Options,'maxNumPoly'))
    maxNumPoly = Options.maxNumPoly;
end
if (isfield(Options,'timeout'))
    timeout = Options.timeout;
end

A = probStruct.A;
B = probStruct.B;
E = zeros(size(A,1), 1);
Uset = probStruct.Uset;
Wset = [];
N = probStruct.N;
Tset = probStruct.Tset;
Sset = probStruct.Sset;

% Get rid of small cells in Sset and Tset
if (length(Sset) > maxNumPoly)
    SsetVol = zeros(size(Sset));
    for i = 1:length(Sset)
        SsetVol(i) = volumeN(Sset(i), [], false);
    end
    [sortSsetVol,IX] = sort(SsetVol, 'descend');
    Sset = Sset(IX(1:maxNumPoly));
    if (verbose > 0)
        display(['Remove polytopes of volume ' ...
            num2str(sum(sortSsetVol(maxNumPoly+1:end))) ...
            ' from Sset.']);
    end
end

if (length(Tset) > maxNumPoly)
    TsetVol = zeros(size(Tset));
    for i = 1:length(Tset)
        TsetVol(i) = volumeN(Tset(i), [], false);
    end
    [sortSsetVol,IX] = sort(TsetVol, 'descend');
    Tset = Tset(IX(1:maxNumPoly));
    if (verbose > 0)
        display(['Remove polytopes of volume ' ...
            num2str(sum(sortSsetVol(maxNumPoly+1:end))) ...
            ' from Tset.']);
    end
end

if (isfield(probStruct, 'Wset') && ~isempty(probStruct.Wset) && ...
        isfield(probStruct, 'E') && ~isempty(probStruct.E))
    E = probStruct.E;
    Wset = probStruct.Wset;
end

if (length(Uset) > 1)
    error('Uset must be convex');
end
if (length(Wset) > 1)
    error('Wset must be convex');
end

% Check the size of the matrices
if (size(A,1) ~= size(A,2))
    error('probStruct.A must be a square matrix');
end
if (size(A,1) ~= size(E,1))
    error('The number of rows of A must match that of E');
end
if (size(A,1) ~= size(B,1))
    error('The number of rows of A must match that of B');
end

[L1vect, M1vect] = double(Sset);
L1 = L1vect;
M1 = M1vect;
if (length(Sset) > 1)
    L1 = L1vect{1};
    M1 = M1vect{1};
end
L2 = zeros(0,size(B,2));
M2 = zeros(0,size(M1,2));
if (~isempty(Uset))
    [L2, M2] = double(Uset);
end
[L3vect, M3vect] = double(Tset);
L3 = L3vect;
M3 = M3vect;
if (length(Tset) > 1)
    L3 = L3vect{1};
    M3 = M3vect{1};
end

if (size(L1,2) ~= size(A,2))
    error('The number of columns of L1 must match that of A');
end
if (size(L2,2) ~= size(B,2))
    error('The number of columns of L2 must match that of B');
end
if (size(L3,2) ~= size(A,2))
    error('The number of columns of L3 must match that of A');
end
if (size(M2,2) ~= size(M1,2))
    error('The number of columns of M2 must match that of M1');
end
if (size(M3,2) ~= size(M1,2))
    error('The number of columns of M3 must match that of M1');
end

if (nargin < 2 || isempty(W))
    Wextremes = zeros(1,size(E,2));
    if (~isempty(Wset))
        [Wextremes,temp,Wset] = extreme(Wset);
    end
    numExtremePoints = size(Wextremes,1);

    % Construct the set W of all the extreme points of Wextremes^N
    W = zeros(size(Wextremes,2)*N, numExtremePoints^N);
    for i = 1:numExtremePoints^N
        str = '0';
        for j = 2:N
            str = [str '0'];
        end
        if (numExtremePoints > 1)
            str = dec2base(i-1,numExtremePoints,N);
        end
        for j = 1:N
            W(size(Wextremes,2)*(j-1)+1:size(Wextremes,2)*j,i) = Wextremes(str2num(str(j))+1,:)';
        end
    end
end

% Rewrite the problem so it is of the form
%   L*[x_0; u_0; u_1; ...; u_{N-1}] <= M
X0 = polytope;

if (verbose > 1)
    display(['numloops = ' num2str(length(Sset)*length(Tset))])
end
for k = 1:length(Sset)
    for m = 1:length(Tset)
        L1 = L1vect;
        M1 = M1vect;    
        if (length(Sset) > 1)
            L1 = L1vect{k};
            M1 = M1vect{k};
        end
        L3 = L3vect;
        M3 = M3vect;    
        if (length(Tset) > 1)
            L3 = L3vect{m};
            M3 = M3vect{m};
        end

        % Construct L
        L = zeros(size(L1,1)*N + size(L3,1) + size(L2,1)*N, size(L1,2) + size(B,2)*N);
        for i = 0:N-1
            L((size(L1,1)*i)+1:size(L1,1)*(i+1), 1:size(L1,2)) = L1*A^i;
            for j = 0:N-1
                if (i-1-j >= 0)
                    L((size(L1,1)*i)+1:size(L1,1)*(i+1), size(L1,2)+size(B,2)*j+1:size(L1,2)+size(B,2)*(j+1)) = L1*A^(i-1-j)*B;
                else
                    L((size(L1,1)*i)+1:size(L1,1)*(i+1), size(L1,2)+size(B,2)*j+1:size(L1,2)+size(B,2)*(j+1)) = zeros(size(L1,1),size(B,2));
                end
            end
        end
        if (size(L2,1) > 0)
            for i = 0:N-1
                L(size(L1,1)*N+size(L3,1)+size(L2,1)*i+1:size(L1,1)*N+size(L3,1)+size(L2,1)*(i+1), size(L1,2)+size(B,2)*i+1:size(L1,2)+size(B,2)*(i+1)) = L2;
            end
        end

        % Construct G
        G = zeros(size(M1,1)*N + size(M3,1), size(E,2)*N);
        for i = 0:N-1
            for j = 0:N-1
                if (i-1-j >= 0)
                    G((size(L1,1)*i)+1:size(L1,1)*(i+1), size(E,2)*j+1:size(E,2)*(j+1)) = L1*A^(i-1-j)*E;
                else
                    G((size(L1,1)*i)+1:size(L1,1)*(i+1), size(E,2)*j+1:size(E,2)*(j+1)) = zeros(size(L1,1),size(E,2));
                end
            end
        end

        % Construct M = min_{w \in W} (F - GW)
        M = zeros(size(M1,1)*N + size(M3,1) + size(M2,1)*N, size(M1,2));
        if (size(M2,1) > 0)
            for i = 0:N-1
                M(size(M1,1)*N+size(M3,1)+size(M2,1)*i+1:size(M1,1)*N+size(M3,1)+size(M2,1)*(i+1), :) = M2;
            end
        end

        % Update L
        L((size(L1,1)*N)+1:size(L1,1)*N+size(L3,1), 1:size(L1,2)) = L3*A^N;
        for j = 0:N-1
            L((size(L1,1)*N)+1:size(L1,1)*N+size(L3,1), size(L1,2)+size(B,2)*j+1:size(L1,2)+size(B,2)*(j+1)) = L3*A^(N-1-j)*B;
        end

        % Update G
        for j = 0:N-1
            G((size(L1,1)*N)+1:size(L1,1)*N+size(L3,1), size(E,2)*j+1:size(E,2)*(j+1)) = L3*A^(N-1-j)*E;
        end
        
        % Find max_{w \in W} Gw
        maxGW = max(G*W, [], 2);

        % Update M = min_{w \in W} (F - GW)
        for i = 0:N-1
            M((size(M1,1)*i)+1:size(M1,1)*(i+1), :) = M1 - maxGW((size(M1,1)*i)+1:size(M1,1)*(i+1), :);
        end
        M((size(M1,1)*N)+1:size(M1,1)*N+size(M3,1), :) = M3 - maxGW((size(M1,1)*N)+1:size(M1,1)*N+size(M3,1), :);

        % Get the projection
        try
            P = polytope(L,M);
%         Options.projection = [0, 2, 3, 4];
            Options.iterhull_maxiter = 3;
            tempX0 = projection(P,1:size(A,2), Options);
            tempX0vol = volumeN(tempX0, [], false);
        catch
            tempX0 = polytope;
            tempX0vol = -1;
        end
        if (verbose > 1)
            display(['tempX0vol = ' num2str(tempX0vol)]);
        end
        if (tempX0vol > 0.1 || timeout <= 0)
            X0 = union([X0 tempX0]);
        elseif (tempX0vol > 0)
            if (verbose > 1)
                display('Calling unionWithTimeout')
            end
            [done, ret] = unionWithTimeout([X0 tempX0], timeout);
            if (done)
                X0 = ret;
            else
                display('Union computation timeout!!!')
            end
        end
    end
end