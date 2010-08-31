load dataToMatlab.mat

adj = double(adj);
minCellVolume = double(minCellVolume);
maxNumIterations = double(maxNumIterations);
useClosedLoopAlg = double(useClosedLoopAlg);
useAllHorizonLength = double(useAllHorizonLength);
useLargeSset = double(useLargeSset);
probStruct.A = double(A);
probStruct.B = double(B);
probStruct.E = double(E);
probStruct.Uset = polytope(double(Uset.A),double(Uset.b));
probStruct.Wset = polytope(double(Wset.A),double(Wset.b));
numpolyvec = double(numpolyvec);

for i1 = 1:length(numpolyvec)
    for i2 = 1:numpolyvec(i1)
        eval(['HK = double(Reg' int2str(i1) 'Poly' int2str(i2) 'Ab);']);
        origPartition{i1}(i2) = polytope(HK(:,1:end-1),HK(:,end));
    end
end

%% at this point all input to discratize should have been created
%% Nok: Put your discratize here.
%% now, save a mat file back to python
%% Everything until the line of % signs will be coming from discretize.m
%% call.

trans = ones(3,5);
numNewCells = [3;2;1;2];
newCellVol = rand(4,3);
newPartition{1,1} = origPartition{1};
newPartition{1,2} = origPartition{2};
newPartition{1,3} = origPartition{3};
newPartition{2,1} = origPartition{4};
newPartition{2,2} = origPartition{5};
newPartition{3,1} = origPartition{6};
newPartition{4,1} = origPartition{7};
newPartition{4,2} = origPartition{8};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


aux = [', '];
num_cells = length(newPartition);
for i1 = 1:length(newPartition)
    for i2 = 1:numNewCells(i1)
        numpoly(i1,i2) = length(newPartition{i1,i2});
        for i3 = 1:length(newPartition{i1,i2})
            te = ['Cell' int2str(i1-1) 'Reg' int2str(i2-1) 'Poly' int2str(i3-1) 'Ab'];
            [H,K] = double(newPartition{i1,i2}(i3));
            eval([te ' = [H K];']);
            aux = [aux char(39) te char(39) ', '];
        end
    end
end

aux = [aux char(39) 'trans' char(39) ', ' ];
aux = [aux char(39) 'numNewCells' char(39) ', '];
aux = [aux char(39) 'num_cells' char(39) ', '];
aux = [aux char(39) 'numpoly' char(39) ', '];
aux = [aux char(39) 'newCellVol' char(39)];

aux = ['save(' char(39) 'dataFromMatlab.mat' char(39) aux ');'];
eval(aux);
        





