function [newPartition, trans, numNewCells, newCellVol] = ...
    discretize(origPartition, adj, probStruct, Options)
% DISCRETIZE Discretize the continuous state space based on the dynamics of
% the system.
%
% USAGE:
%   [newPartition, trans, numNewCells, newCellVol] =
%   partition(origPartition, adj, probStruct)
%
%   [newPartition, trans, numNewCells, newCellVol] =
%   partition(origPartition, adj, probStruct, Options)
%
% INPUT:
% * origParition: a cell array of MPT polytope objects that represents the  
%   proposition preserving partition of the state space.
% * adj: the adjacency between cells in origParition. Use adj = [] if
%   the adjacency matrix is not known. 
% * probStruct contains the following fields: A, B, E, Uset, Wset, N.
% * Options.minCellVolume specifies the minimum volume of cells in the 
%   resulting partition. The default value is 0.1.
% * Options.maxNumIterations specifies the maximum number of iterations.
%   The default value is 5.
% * Options.useClosedLoopAlg specifies whether to use the closed loop algorithm.
%   The default value is true.
% * Options.useAllHorizonLength specifies whether all the horizon length up
%   to probStruct.N can be used. This option is relevant only when the closed 
%   loop algorithm is used, The default value is true.
% * Options.useLargeSset specifies whether when solving the reachability
%   problem between subcells of the original partition, the cell of the
%   original partition should be used for the safe set. The default value
%   is true.
% * Options.timeout specifies the timeout (in seconds) for polytope union 
%   operation. If negative, the timeout won't be used. Note that using timeout 
%   requires MATLAB parallel computing toolbox. The default value is -1.
% * Options.maxNumPoly specifies the maximum number of polytopes in a
%   region used in computing reachability. The default value is 5.
% * Options.verbose: level of verbosity of the algorithms. The default
%   value is 0.
% 
% OUTPUT:
% * newPartition: a 2d cell array of MPT polytope objects that represents 
%   the new partition. 
%   newPartition{i, 1:numNewCells(i)} are the new cells that contain in 
%   the i^{th} cell of the original partition.
% * trans: the transition matrix. trans(i,j) specifies whether subcell i is
%   reachable from subcell j.
% * numNewCells: a vector that specifies the number of new cells in the 
%   original partition. numNewCells(i) is the number of new cells in the
%   i^{th} cell of the original partition.
% * newCellVol: a matrix that specifies the volume of the new cell. 
%   newCellVol(i,j) is the volume of j^{th} subcell of the i^{th} cell 
%   in the original partition.

if (nargin < 3)
    error('origPartition, adj and probStruct need to be specified')
elseif (nargin < 4)
    Options = [];
end

if (isempty(adj))
    adj = ones(length(origPartition));
end
adj = adj';

tol = 1e-03;
verbose = 0;
maxNumPoly = 5;
timeout = -1;
useLargeSset = true;
useAllHorizonLength = true;
useClosedLoopAlg = true;
maxNumIterations = 5;
minCellVolume = 0.1;

if (isfield(Options, 'verbose'))
    verbose = Options.verbose;
end
if (isfield(Options,'maxNumPoly'))
    maxNumPoly = Options.maxNumPoly;
end
if (isfield(Options,'timeout'))
    timeout = Options.timeout;
end
if (isfield(Options,'useLargeSset'))
    useLargeSset = Options.useLargeSset;
end
if (isfield(Options,'useAllHorizonLength'))
    useAllHorizonLength = Options.useAllHorizonLength;
end
if (isfield(Options,'useClosedLoopAlg'))
    useClosedLoopAlg = Options.useClosedLoopAlg;
end
if (isfield(Options,'maxNumIterations'))
    maxNumIterations = Options.maxNumIterations;
end
if (isfield(Options,'minCellVolume'))
    minCellVolume = Options.minCellVolume;
end

if (verbose > 0)
    display('Options:');
    display(['\tminCellVolume:       ' num2str(minCellVolume)]);
    display(['\tmaxNumIterations:    ' num2str(maxNumIterations)]);
    display(['\tuseClosedLoopAlg:    ' num2str(useClosedLoopAlg)]);
    display(['\tuseAllHorizonLength: ' num2str(useAllHorizonLength)]);
    display(['\tuseLargeSset:        ' num2str(useLargeSset)]);
    display(['\ttimeout:             ' num2str(timeout)]);
    display(['\tmaxNumPoly:          ' num2str(maxNumPoly)]);
end

numOrigCells = length(origPartition);
origCellVol = zeros(length(origPartition),1);
for i = 1:numOrigCells
    origCellVol(i) = volumeN(origPartition{i});
end


% Initialize newPartition
numNewCells = ones(numOrigCells, 1);
newPartition = cell(numOrigCells, 1);
newAdj = adj;
trans = zeros(size(adj));
newCellVol = origCellVol;
for i = 1:numOrigCells
    newPartition{i,1} = origPartition{i};
end

itCounter = 0;
done = 0;

% while ( itCounter < maxNumIterations & sum(sum(newCellVol >= 2*minCellVolume)) > 0 & (~done | sum(sum(1-newAdj))>0) )
while ( itCounter < maxNumIterations && sum(sum(newCellVol >= 2*minCellVolume)) > 0 && ~done )
    done = 1;
     for initCell = 1:numOrigCells
         if (sum(newCellVol(initCell,:) >= 2*minCellVolume) > 0)
            for finalCell = 1:numOrigCells
                % To speed things up a little bit, partitioning cell happens only if initCell and finalCell are adjacent
                if (adj(initCell,finalCell)) 
                    for subFCell = 1:numNewCells(finalCell)
                        if (verbose > 0)
                            display(' ');
                            display(['Partitioning ' num2str(itCounter+1) '...']);
                            display(['initCell: ' num2str(initCell)]);
                            display(['finalCell: ' num2str(finalCell)]);
                            display(['subFCell: ' num2str(subFCell)]);
                        end
                        startIndex = sum(numNewCells(1:initCell-1));
                        
                        if (sum(newAdj(startIndex+1:startIndex+numNewCells(initCell), sum(numNewCells(1:finalCell-1))+subFCell)) > 0)
                            % Safe set
                            try
                                probStruct.Sset = union([origPartition{initCell} origPartition{finalCell}]);
                            catch
                                probStruct.Sset = origPartition{initCell};
                            end
                            if (verbose > 1)
                                numPolyInSset = size(probStruct.Sset,2);
                                display(['safe set: ' num2str(numPolyInSset) ' polytope(s):']);
                                for temp = 1:size(probStruct.Sset,2)
                                    probStruct.Sset(temp)
                                end
                            end
                            
                            % Terminal set
                            probStruct.Tset = newPartition{finalCell,subFCell};
                            
                            % Feasible initial set
                            try
                                if (useClosedLoopAlg)
                                    allFeasInitSet = solveFeasCL(probStruct, Options);
                                else
                                    allFeasInitSet = solveFeas(probStruct, [], Options);
                                end
                            catch
                                display('Computing feasible initial set failed. Setting feasInitSet to empty polytope.');
                                allFeasInitSet = polytope;
                            end
                            
                            allFeasSetVol = 0;
                            for n = 1:length(allFeasInitSet)
                                tmpvol = volumeN(allFeasInitSet(n));
                                allFeasSetVol = allFeasSetVol + tmpvol;
                            end
                            if (verbose > 1)
                                display(['allFeasSetVol: ' num2str(allFeasSetVol)])
                            end
                            
                            % partition all the cells in initCell based on allFeasInitSet if
                            % allFeasInitSet is not empty
                            if (allFeasSetVol > tol)
                                numInitCells = numNewCells(initCell);
                                for subICell = 1:numInitCells
                                    if (verbose > 0)
                                        display(' ');
                                        display(['subICell: ' num2str(subICell)]);
                                    end
                                    subICellIDInAdj = sum(numNewCells(1:initCell-1))+subICell;
                                    subFCellIDInAdj = sum(numNewCells(1:finalCell-1))+subFCell;
                                    if (newAdj(subICellIDInAdj, subFCellIDInAdj) && ~trans(subICellIDInAdj, subFCellIDInAdj))
                                        if (~useLargeSset)
                                            try
                                                probStruct.Sset = newPartition{initCell,subICell};
                                                if (useClosedLoopAlg)
                                                    feasInitSet = solveFeasCL(probStruct, Options);
                                                else
                                                    feasInitSet = solveFeas(probStruct, [], Options);
                                                end
                                            catch
                                                display('Computing feasible initial set failed. Setting feasInitSet to empty polytope.');
                                                feasInitSet = polytope;
                                            end
                                        else
                                            feasInitSet = polytope;
                                            for n = 1:length(newPartition{initCell,subICell})
                                                for m = 1:length(allFeasInitSet)
                                                    try
                                                        feasInitSet = union([feasInitSet intersect(newPartition{initCell,subICell}(n), allFeasInitSet(m))]);
                                                    end
                                                end
                                            end
                                        end
                                        feasSetVol = 0;
                                        infeasInitSet = newPartition{initCell,subICell};
                                        failed = false;
                                        if (length(feasInitSet) > 0)
                                            try
                                                infeasInitSet = newPartition{initCell,subICell} \ feasInitSet;
                                                for n = 1:length(feasInitSet)
                                                    tmpvol = volumeN(feasInitSet(n));
                                                    feasSetVol = feasSetVol + tmpvol;
                                                end
                                            catch
                                                display('Computing infeasInitSet failed.')
                                            end
                                        end
                                        if (verbose > 1)
                                            display(['feasSetVol: ' num2str(feasSetVol)])
                                        end
                                        infeasSetVol = newCellVol(initCell,subICell) - feasSetVol;
                                        
                                        % The case where this cell needs to be partitioned into two cells
                                        if (~failed && feasSetVol >= minCellVolume && infeasSetVol >= minCellVolume)
                                            done = 0;
                                            newPartition{initCell,subICell} = feasInitSet;
                                            newPartition{initCell,numNewCells(initCell)+1} = infeasInitSet;
                                            
                                            newAdj(subICellIDInAdj, subFCellIDInAdj) = 1;
                                            trans(subICellIDInAdj, subFCellIDInAdj) = 1;
                                            
                                            indexToInsert = sum(numNewCells(1:initCell));
                                            newAdj(:, indexToInsert+1:end+1) = newAdj(:, indexToInsert:end);
                                            trans(:, indexToInsert+1:end+1) = trans(:, indexToInsert:end);
                                            
                                            newAdj(:, indexToInsert) = 1;
                                            trans(:, indexToInsert) = 0;
                                            
                                            newAdj(indexToInsert+1:end+1, :) = newAdj(indexToInsert:end, :);
                                            trans(indexToInsert+1:end+1, :) = trans(indexToInsert:end, :);
                                            
                                            newAdj(indexToInsert, :) = 1;
                                            trans(indexToInsert, :) = trans(subICellIDInAdj, :);
                                            
                                            newAdj(indexToInsert, subFCellIDInAdj) = 0;
                                            trans(indexToInsert, subFCellIDInAdj) = 0;
                                            
                                            newCellVol(initCell,subICell) = feasSetVol;
                                            newCellVol(initCell,numNewCells(initCell)+1) = infeasSetVol;
                                            numNewCells(initCell) = numNewCells(initCell)+1;
                                            
                                        % The case where there exists a control law which takes the
                                        % system from anywhere in initCell to finalCell
                                        elseif (~failed && infeasSetVol < tol)
                                            newAdj(subICellIDInAdj, subFCellIDInAdj) = 1;
                                            trans(subICellIDInAdj, subFCellIDInAdj) = 1;
                                        else
                                            if (verbose > 1)
                                                display('feasSetVol < minCellVolume || infeasSetVol < minCellVolume')
                                            end
                                            newAdj(subICellIDInAdj, subFCellIDInAdj) = 0;
                                            trans(subICellIDInAdj, subFCellIDInAdj) = 0;
                                        end
                                    end
                                end
                            else
                                if (verbose > 1)
                                    display('allFeasSetVol < tol')
                                end
                                subFCellIDInAdj = sum(numNewCells(1:finalCell-1))+subFCell;
                                newAdj(sum(numNewCells(1:initCell-1))+1:sum(numNewCells(1:initCell)), subFCellIDInAdj) = 0;
                                trans(sum(numNewCells(1:initCell-1))+1:sum(numNewCells(1:initCell)), subFCellIDInAdj) = 0;
                            end
                        end
                    end
                else
                    if (verbose > 1)
                        display(['initCell: ' num2str(initCell)]);
                        display(['finalCell: ' num2str(finalCell)]);
                        display('~adj(initCell,finalCell)')
                    end
                    newAdj(sum(numNewCells(1:initCell-1))+1:sum(numNewCells(1:initCell)), sum(numNewCells(1:finalCell-1))+1:sum(numNewCells(1:finalCell))) = 0;
                    trans(sum(numNewCells(1:initCell-1))+1:sum(numNewCells(1:initCell)), sum(numNewCells(1:finalCell-1))+1:sum(numNewCells(1:finalCell))) = 0;
                end
            end
         end
    end
    itCounter = itCounter + 1;
end


if (~done)
    display(' ');
    display('Done partitioning. Start checking adjacency...');
    display(' ');
    for initCell = 1:numOrigCells
        adjFinalCells = find(adj(initCell,:) > 0);
        for finalCell = adjFinalCells
            for subFCell = 1:numNewCells(finalCell)
                startIndex = sum(numNewCells(1:initCell-1));
                if (sum(newAdj(startIndex+1:startIndex+numNewCells(initCell), sum(numNewCells(1:finalCell-1))+subFCell)) > 0)
                    if (verbose > 0)
                        display(' ');
                        display('Checking... ');
                        display(['initCell: ' num2str(initCell)]);
                        display(['finalCell: ' num2str(finalCell)]);
                        display(['subFCell: ' num2str(subFCell)]);
                    end
                    % Safe set
                    try
                        probStruct.Sset = union([origPartition{initCell} origPartition{finalCell}]);
                    catch
                        probStruct.Sset = origPartition{initCell};
                    end
                    if (verbose > 1)
                        numPolyInSset = size(probStruct.Sset,2);
                        display(['safe set: ' num2str(numPolyInSset) ' polytope(s):']);
                        for temp = 1:size(probStruct.Sset,2)
                            probStruct.Sset(temp)
                        end
                    end

                    % Terminal set
                    probStruct.Tset = newPartition{finalCell,subFCell};

                    % Feasible initial set
                    try
                        if (useClosedLoopAlg)
                            allFeasInitSet = solveFeasCL(probStruct, Options);
                        else
                            allFeasInitSet = solveFeas(probStruct, [], Options);
                        end
                    catch
                        display('Computing feasible initial set failed.');
                        allFeasInitSet = polytope;
                    end
                    
                    allFeasSetVol = 0;
                    for n = 1:length(allFeasInitSet)
                        tmpvol = volumeN(allFeasInitSet(n));
                        allFeasSetVol = allFeasSetVol + tmpvol;
                    end
                    if (verbose > 1)
                        display(['allFeasSetVol: ' num2str(allFeasSetVol)])
                    end
                    
                    % partition all the cells in initCell based on allFeasInitSet if
                    % allFeasInitSet is not empty
                    if (allFeasSetVol > tol)                                
                        numInitCells = numNewCells(initCell);
                        for subICell = 1:numInitCells
                            if (verbose > 0)
                                display(' ');
                                display(['subICell: ' num2str(subICell)]);
                            end
                            subICellIDInAdj = sum(numNewCells(1:initCell-1))+subICell;
                            subFCellIDInAdj = sum(numNewCells(1:finalCell-1))+subFCell;
                            if (newAdj(subICellIDInAdj, subFCellIDInAdj) && ~trans(subICellIDInAdj, subFCellIDInAdj))           
                                if (~useLargeSset)
                                    try
                                        probStruct.Sset = newPartition{initCell,subICell};
                                        if (useClosedLoopAlg)
                                            feasInitSet = solveFeasCL(probStruct, Options);
                                        else
                                            feasInitSet = solveFeas(probStruct, [], Options);
                                        end
                                    catch
                                        display('Computing feasible initial set failed.');
                                        feasInitSet = polytope;
                                    end
                                else
                                    feasInitSet = polytope;
                                    for n = 1:length(newPartition{initCell,subICell})
                                        for m = 1:length(allFeasInitSet)
                                            try
                                                feasInitSet = union([feasInitSet intersect(newPartition{initCell,subICell}(n), allFeasInitSet(m))]);
                                            end
                                        end
                                    end
                                end
                                feasSetVol = 0;
                                for n = 1:length(feasInitSet)
                                    tmpvol = volumeN(feasInitSet(n));
                                    feasSetVol = feasSetVol + tmpvol;
                                end
                                if (verbose > 1)
                                    display(['feasSetVol: ' num2str(feasSetVol)])
                                end
                                if (newCellVol(initCell,subICell) - feasSetVol < tol)
                                    trans(subICellIDInAdj, subFCellIDInAdj) = 1;
                                else
                                    trans(subICellIDInAdj, subFCellIDInAdj) = 0;
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

trans = trans';