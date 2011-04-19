%fname = 'runDiscretizeMatlab.m';
%p = which(fname);
%p = p(1:end-length(fname));
%matfilein = [p 'tmpmat' filesep 'dataToMatlab.mat'];
%matfileout = [p 'tmpmat' filesep 'dataFromMatlab.mat'];
%donefile = [p 'tmpmat' filesep 'done.txt'];
matfilein = [p filesep 'dataToMatlab.mat'];
matfileout = [p filesep 'dataFromMatlab.mat'];
donefile = [p filesep 'done.txt'];

load(matfilein)

% mpt_init('rescueLP', true, 'rescueQP', true, 'lpsolver', 'sedumi')

adj = double(adj);
Options.minCellVolume = double(minCellVolume);
Options.maxNumIterations = double(maxNumIterations);
Options.useClosedLoopAlg = double(useClosedLoopAlg);
Options.useAllHorizonLength = double(useAllHorizonLength);
Options.useLargeSset = double(useLargeSset);
Options.timeout = double(timeout);
Options.maxNumPoly = double(maxNumPoly);
Options.verbose = double(verbose);
probStruct.A = double(A);
probStruct.B = double(B);
probStruct.E = double(E);
probStruct.N = double(N);
probStruct.Uset = polytope(double(UsetA),double(Usetb));
% probStruct.Uset = polytope(double(Uset.A),double(Uset.b));
if (~isempty(WsetA) && ~isempty(Wsetb))
    probStruct.Wset = polytope(double(WsetA),double(Wsetb));
else
    probStruct.Wset = [];
end
numpolyvec = double(numpolyvec);

for i1 = 1:length(numpolyvec)
    for i2 = 1:numpolyvec(i1)
        eval(['HK = double(Reg' int2str(i1) 'Poly' int2str(i2) 'Ab);']);
        origPartition{i1}(i2) = polytope(HK(:,1:end-1),HK(:,end));
    end
end

%% at this point all input to discratize should have been created
[newPartition, trans, numNewCells, newCellVol] = ...
    discretize(origPartition, adj, probStruct, Options)

%% now, save a mat file back to python
aux = [', '];
num_cells = size(newPartition,1);
for i1 = 1:num_cells
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

aux = ['save(' char(39) matfileout char(39) aux ');'];
eval(aux);
        

f = fopen(donefile,'w');
fprintf(f, 'done \n');
fclose(f);




