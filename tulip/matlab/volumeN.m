function vol = volumeN(poly, Nsample, useMPT)
% VOLUMEN Compute volume of polytope.
% 
% USAGE:
%   vol = volumeN(poly)
%   vol = volumeN(poly, Nsample)
%   vol = volumeN(poly, Nsample, useMPT)
%
% INPUT:
% * poly: MPT polytope object
% * Nsample: the number of samples to be used. The default value depends on
%   poly.
% * useMPT: whether to use MPT volume computation and only resort to this
%   algorithm when MPT fails. The default value is true.

if (nargin < 3)
    useMPT = true;
end
if (nargin < 2)
    Nsample = [];
end

vol = inf;
if (useMPT)
%     try
        vol = volume(poly);
%     catch
%         display('MPT volume failed');
%     end
end

if (isinf(vol))
    vol = 0;
    [n,m] = size(get(poly(1),'H'));
    
    if (isempty(Nsample))
        if m == 1
            Nsample = 500;
        elseif m ==2
            Nsample = 5000;
        elseif m == 3
            Nsample = 10000;
        else
            Nsample = 20000;
        end
    end
    
    Options.noPolyOutput=1;
    Options.bboxvertices = 0;
    Options.Voutput = 1;
    for i1 = 1:length(poly)
        E = bounding_box(poly(i1), Options);
        lb = E(:,1);
        ub = E(:,2);
        x = repmat(lb,1,Nsample) + rand(m,Nsample).*repmat(ub-lb,1,Nsample);
        [H,K] = double(poly(i1));
        aux = H*x - repmat(K,1,Nsample);
        vol = vol + prod(ub-lb)*length(find(all(aux<0,1)))/Nsample;
    end
end
