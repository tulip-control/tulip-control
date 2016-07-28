function mpt2python(model, filename)
% Takes a MPT system data structure (either LTISystem or PWASystem "model") and
% exports it to a .mat file "filename". This file can then be read back
% into Python to create a LtiSysDyn or PwaSysDyn in Tulip.

    if isa(model,'LTISystem')
        [A, B, K, domainA, domainB, UsetA, UsetB] = lti2python(model);
        islti = 1;
        ispwa = 0;
        save(filename, 'A', 'B', 'K', 'domainA', 'domainB', 'UsetA', ...
            'UsetB', 'islti', 'ispwa');
    elseif isa(model, 'PWASystem')
        [A, B, K, domainA, domainB, UsetA, UsetB, ctsA, ctsB] = ...
            pwa2python(model);
        islti = 0;
        ispwa = 1;
        save(filename, 'A', 'B', 'K', 'domainA', 'domainB', 'UsetA', ...
            'UsetB', 'ctsA', 'ctsB', 'islti', 'ispwa');
    end


    function [A,B,K,domainA,domainB,UsetA,UsetB,ctsA,ctsB] = pwa2python(pwasys)

        % Dynamics matrices already stored conveniently
        A = pwasys.A;
        B = pwasys.B;
        K = pwasys.f;

        % Get domain matrices for each of the lti systems
        domainA = cell(1,pwasys.ndyn);
        domainB = cell(1,pwasys.ndyn);
        UsetA = cell(1,pwasys.ndyn);
        UsetB = cell(1,pwasys.ndyn);
        cts_ss = cell(1,pwasys.ndyn);
        for i = 1:pwasys.ndyn
            ltisys = pwasys.toLTI(i);
            [~,~,~,dA,dB,uA,uB] = lti2python(ltisys);
            domainA{i} = dA;
            domainB{i} = dB;
            UsetA{i} = uA;
            UsetB{i} = uB;
            cts_ss{i} = Polyhedron(dA, dB);
        end

        % Get domain of the whole state
        cts_ss = PolyUnion('Set', [cts_ss{:}]);
        cts_ss.merge;
        ctsA = cts_ss.Set.A;
        ctsB = cts_ss.Set.b;
    end


    function [A,B,K,domainA,domainB,UsetA,UsetB] = lti2python(ltisys)

        % Make sure that the domain was set with 'xu' and not just 'x'
        if (ltisys.domain.Dim ~= (ltisys.nx + ltisys.nu))
            error('Domain of LTISystem must be set with xu option');
        end

        % Dynamics matrices
        A = ltisys.A;
        B = ltisys.B;
        K = ltisys.f;

        % State domain
        xdomain = projection(ltisys.domain, 1:1:ltisys.nx);
        domainA = xdomain.A;
        domainB = xdomain.b;

        % Input domain - rearrange order of cols so that domain is
        % [u;x] instead of [x;u]
        UsetA = ltisys.domain.A;
        UsetB = ltisys.domain.b;
        UsetA = [UsetA(:,ltisys.nx+1:end) UsetA(:,1:ltisys.nx)];
    end

end
