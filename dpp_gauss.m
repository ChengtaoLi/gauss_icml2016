%% sampling subsets from (k-)DPP with Gauss quadrature
%
% -input
%   L: data kernel matrix, N*N where N is number of samples
%   rangeFun: range function for bounding eigenspectrum of matrices
%   mixStep: number of burn-in iterations
%   k: if sampling from k-DPP, this spcifies the size of sampled subset
%
% -output
%   C: sampled subset
%
% sample usage:
%   C = dpp_gauss(L,@gershgorin,1000,20)

function C = dpp_gauss(L, rangeFun, mixStep, k)
    if nargin >= 4
        % sample from k-DPP
        C = gauss_kdpp(L, k, rangeFun, mixStep);
    else
        % sample from DPP
        C = gauss_dpp(L, rangeFun, mixStep);
    end 
end

%% sample from k-DPP
function C = gauss_kdpp(L, k, rangeFun, mixStep)
    n = length(L);
    C = sort(randperm(n, k));
    A = L(C,C);
    ticLen = ceil(mixStep / 5);
    
    for i = 1:mixStep
        if mod(i, ticLen) == 0
            disp([num2str(i) '-th iteration.']);
        end
        
        delInd = randi(k);
        v = C(delInd); % one to remove
        u = randi(n); % one to add
        while any(C == u)
            u = randi(n);
        end
        tmpC = C; tmpA = A;
        tmpC(delInd) = []; tmpA(delInd,:) = []; tmpA(:,delInd) = [];
        bu = L(tmpC, u); bv = L(tmpC, v);
        
        [lambdaMin, lambdaMax] = rangeFun(tmpA);
        lambdaMin = max(lambdaMin, 1e-5);
        
        prob = rand;
        tar = full(prob * L(v,v) - L(u,u));
        
        flag = gauss_kdpp_judge(tmpA, bu, bv, prob, tar, lambdaMin, lambdaMax);

        if flag % accept move
            C = [tmpC u];
            A = [tmpA bu; bu' L(u,u)];
        end
    end
end

%% sample from DPP
function C = gauss_dpp(L, rangeFun, mixStep)
    n = length(L);
    C = sort(randperm(n, ceil(n / 3)));
    A = L(C,C);
    ticLen = ceil(mixStep / 5);
    
    col = ceil(sqrt(mixStep));
    row = mixStep / col;
    
    for i = 1:mixStep
        if mod(i, ticLen) == 0
            disp([num2str(i) '-th iteration. Size = ' num2str(length(C))]);
        end
        
        u = randi(n, 1);
        bu = L(C, u); 
        cu = L(u, u);
        
        ind = find(C == u);
        if isempty(ind) % try to add
            [lambdaMin, lambdaMax] = rangeFun(A);
            lambdaMin = max(lambdaMin, 1e-5);
            [flag, ~] = gauss_dpp_judge(A, bu, cu - rand, lambdaMin, lambdaMax);
            if ~flag
                C = [C,u];
                A = [A, bu; bu', cu];
            end
            
        else % try to remove
            tmpC = C; tmpA = A;
            tmpC(ind) = []; tmpA(ind,:) = []; tmpA(:,ind) = [];
            bu(ind) = [];
            
            [lambdaMin, lambdaMax] = rangeFun(tmpA);
            lambdaMin = max(lambdaMin, 1e-5);
            [flag, ~] = gauss_dpp_judge(tmpA, bu, cu - 1 / rand, lambdaMin, lambdaMax);
            if flag
                C = tmpC;
                A = tmpA;
            end 
        end
    end
end



