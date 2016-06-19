%%  function flag = gaussJudge(A, u, prob, lambdaMin, lambdaMax)
%   judge if prob < u^T A^{-1} u
function [flag, gauss] = gauss_dpp_judge(A, u, prob, lambdaMin, lambdaMax)

%% Gauss Quadrature results
% Gauss                     -> gauss(1,:)
% Gauss Radau Lower Bound   -> gauss(2,:)
% Gauss Radau Upper Bound   -> gauss(3,:)
% Gauss Lobatto             -> gauss(4,:)

% Initialization
K = size(A, 1);
gauss = zeros(4,1);
g = 0;
len = u' * u;

% case that the vector is too small
if len < 1e-10
    if prob < 0
        flag = true;
    else
        flag = false;
    end
    
    return;
end

p = u;
beta = 0;
gamma = 1;
c = 1;

f = 0;
fU = 0;
fL = 0;
fT = 0;

delta = 0;
deltaU = 0;
deltaL = 0;

eta = 0;
etaT = 0;

alpha = 0;
alphaU = 0;
alphaL = 0;
alphaT = 0;

truth = 0;%ori'*pinv(full(A))*ori;

% CGQL Main Iteration
for k = 1:2*K
    newGamma = u' * u / (p' * A * p);
    alpha = 1 / newGamma + beta / gamma;
    gamma = newGamma;
    
    if k == 1
        f = 1 / alpha;
        delta = alpha;
        deltaU = alpha - lambdaMin;
        deltaL = alpha - lambdaMax;
    else
        c = c * eta / (delta^2);
        delta = 1 / gamma;
        f = gamma * c;
        deltaU = alpha - alphaU;
        deltaL = alpha - alphaL;
    end
    
    beta = u' * u;
    u = u - gamma * A * p;
    beta = u' * u / beta;
    eta = beta / (gamma^2);
    p = u + beta * p;
    
    alphaU = lambdaMin + eta / deltaU;
    alphaL = lambdaMax + eta / deltaL;
    alphaT = deltaU * deltaL / (deltaL - deltaU);
    etaT = alphaT * (lambdaMax - lambdaMin);
    alphaT = alphaT * (lambdaMax / deltaU - lambdaMin / deltaL);
    
    fU = eta * c / (delta * (alphaU * delta - eta));
    fL = eta * c / (delta * (alphaL * delta - eta));
    fT = etaT * c / (delta * (alphaT * delta - etaT));
    
    g = g + f;
    gauss(1) = len * g;
    gauss(2) = len * (g + fL);
    gauss(3) = len * (g + fU);
    gauss(4) = len * (g + fT);
    
    %fprintf('prob:%f\t, truth:%f\t, l1: %f\t, l2:%f\t, u1:%f\t, u2:%f\n', prob, truth, gauss(1,k), gauss(2,k), gauss(3,k), gauss(4,k));
    
    % approximation is exact
    if eta < 1e-10
        if prob < gauss(1)
            flag = true;
        else
            flag = false;
        end
        
        return;
    end
    
    if prob < max(gauss(1:2))
        flag = true;
        return;
    elseif prob > min(gauss(3:4))
        flag = false;  
        return;
    end
    
    
end

if prob < (max(gauss(1:2)) + min(gauss(3:4))) / 2
    flag = true;
else
    flag = false;
end