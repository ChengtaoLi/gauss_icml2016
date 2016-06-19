%%  function flag = gauss_kdpp_judge(A, u, v, prob, tar, lambdaMin, lambdaMax)
%   judge if tar < prob * v' * inv(L) * v - u' * inv(L) * u
function [flag] = gauss_kdpp_judge(A, u, v, prob, tar, lambdaMin, lambdaMax)

%% Gauss Quadrature results
% Gauss                     -> gauss(1,:)
% Gauss Radau Lower Bound   -> gauss(2,:)
% Gauss Radau Upper Bound   -> gauss(3,:)
% Gauss Lobatto             -> gauss(4,:)

% Initialization
K = size(A, 1);
gauss_U = zeros(4,1);
gauss_V = zeros(4,1);
g_U = 0;
g_V = 0;
len_U = u' * u;
len_V = v' * v;

% case that the vector is too small
if len_U < 1e-10
    [flag,~] = gauss_dpp_judge(A, v, tar/prob, lambdaMin, lambdaMax);
    return;
    
elseif len_V < 1e-10
    [flag,~] = gauss_dpp_judge(A, u, -tar, lambdaMin, lambdaMax);
    flag = ~flag;
    return;
end

p_U = u;
p_V = v;
beta_U = 0;
beta_V = 0;
gamma_U = 1;
gamma_V = 1;
c_U = 1;
c_V = 1;

f_U = 0;
fU_U = 0;
fL_U = 0;
fT_U = 0;
f_V = 0;
fU_V = 0;
fL_V = 0;
fT_V = 0;

delta_U = 0;
deltaU_U = 0;
deltaL_U = 0;
delta_V = 0;
deltaU_V = 0;
deltaL_V = 0;

eta_U = 0;
etaT_U = 0;
eta_V = 0;
etaT_V = 0;

alpha_U = 0;
alphaU_U = 0;
alphaL_U = 0;
alphaT_U = 0;
alpha_V = 0;
alphaU_V = 0;
alphaL_V = 0;
alphaT_V = 0;

iter_U = 1;
iter_V = 1;
gap_U = -1;
gap_V = -1;

proceed_u();
proceed_v();

%debug_info();
if tar <= prob * max(gauss_V(1:2)) - min(gauss_U(3:4))
    flag = true;
    return;
elseif tar >= prob * min(gauss_V(3:4)) - max(gauss_U(1:2))
    flag = false;
    return;
end
    

while true
    if gap_U >= prob * gap_V
        proceed_u();
    else
        proceed_v();
    end
    
    if tar <= prob * max(gauss_V(1:2)) - min(gauss_U(3:4))
        flag = true;
        return;
    elseif tar >= prob * min(gauss_V(3:4)) - max(gauss_U(1:2))
        flag = false;
        return;
    end
    
    %debug_info();
end

function proceed_u()
    newGamma_U = u' * u / (p_U' * A * p_U);
    alpha_U = 1 / newGamma_U + beta_U / gamma_U;
    gamma_U = newGamma_U;
    
    if iter_U == 1
        f_U = 1 / alpha_U;
        delta_U = alpha_U;
        deltaU_U = alpha_U - lambdaMin;
        deltaL_U = alpha_U - lambdaMax;
    else
        c_U = c_U * eta_U / (delta_U^2);
        delta_U = 1 / gamma_U;
        f_U = gamma_U * c_U;
        deltaU_U = alpha_U - alphaU_U;
        deltaL_U = alpha_U - alphaL_U;
    end
    
    beta_U = u' * u;
    u = u - gamma_U * A * p_U;
    beta_U = u' * u / beta_U;
    eta_U = beta_U / (gamma_U^2);
    p_U = u + beta_U * p_U;
    
    alphaU_U = lambdaMin + eta_U / deltaU_U;
    alphaL_U = lambdaMax + eta_U / deltaL_U;
    alphaT_U = deltaU_U * deltaL_U / (deltaL_U - deltaU_U);
    etaT_U = alphaT_U * (lambdaMax - lambdaMin);
    alphaT_U = alphaT_U * (lambdaMax / deltaU_U - lambdaMin / deltaL_U);
    
    fU_U = eta_U * c_U / (delta_U * (alphaU_U * delta_U - eta_U));
    fL_U = eta_U * c_U / (delta_U * (alphaL_U * delta_U - eta_U));
    fT_U = etaT_U * c_U / (delta_U * (alphaT_U * delta_U - etaT_U));
    
    g_U = g_U + f_U;
    gauss_U(1) = len_U * g_U;
    gauss_U(2) = len_U * (g_U + fL_U);
    gauss_U(3) = len_U * (g_U + fU_U);
    gauss_U(4) = len_U * (g_U + fT_U);
    gap_U = min(gauss_U(3:4)) - max(gauss_U(1:2));
    
    iter_U = iter_U + 1;
end

function proceed_v()
    newGamma_V = v' * v / (p_V' * A * p_V);
    alpha_V = 1 / newGamma_V + beta_V / gamma_V;
    gamma_V = newGamma_V;
    
    if iter_V == 1
        f_V = 1 / alpha_V;
        delta_V = alpha_V;
        deltaU_V = alpha_V - lambdaMin;
        deltaL_V = alpha_V - lambdaMax;
    else
        c_V = c_V * eta_V / (delta_V^2);
        delta_V = 1 / gamma_V;
        f_V = gamma_V * c_V;
        deltaU_V = alpha_V - alphaU_V;
        deltaL_V = alpha_V - alphaL_V;
    end
    
    beta_V = v' * v;
    v = v - gamma_V * A * p_V;
    beta_V = v' * v / beta_V;
    eta_V = beta_V / (gamma_V^2);
    p_V = v + beta_V * p_V;
    
    alphaU_V = lambdaMin + eta_V / deltaU_V;
    alphaL_V = lambdaMax + eta_V / deltaL_V;
    alphaT_V = deltaU_V * deltaL_V / (deltaL_V - deltaU_V);
    etaT_V = alphaT_V * (lambdaMax - lambdaMin);
    alphaT_V = alphaT_V * (lambdaMax / deltaU_V - lambdaMin / deltaL_V);
    
    fU_V = eta_V * c_V / (delta_V * (alphaU_V * delta_V - eta_V));
    fL_V = eta_V * c_V / (delta_V * (alphaL_V * delta_V - eta_V));
    fT_V = etaT_V * c_V / (delta_V * (alphaT_V * delta_V - etaT_V));
    
    g_V = g_V + f_V;
    gauss_V(1) = len_V * g_V;
    gauss_V(2) = len_V * (g_V + fL_V);
    gauss_V(3) = len_V * (g_V + fU_V);
    gauss_V(4) = len_V * (g_V + fT_V);
    gap_V = min(gauss_V(3:4)) - max(gauss_V(1:2));
    
    iter_V = iter_V + 1;
end

function debug_info()
    fprintf('total iter: %d, u: %d, v:%d\n', iter_U + iter_V, iter_U, iter_V);
    fprintf('prob:%f\t, tar:%f\t, l_U: %f\t, u_U:%f\t, l_V:%f\t, u_V:%f\n', prob, tar, max(gauss_U(1:2)), min(gauss_U(3:4)), max(gauss_V(1:2)), min(gauss_V(3:4)));
    fprintf('gap_U:%f\t, gap_V:%f\t, currL:%f\t, currU:%f\n\n', gap_U, gap_V, ...
        prob * max(gauss_V(1:2)) - min(gauss_U(3:4)), prob * min(gauss_V(3:4)) - max(gauss_U(1:2)));
    
end
end