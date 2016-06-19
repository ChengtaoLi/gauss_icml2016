function [flag] = gauss_dg_judge(A, B, u, v, tar, prob, rangeFun)

[lambdaMin_A, lambdaMax_A] = rangeFun(A);
[lambdaMin_B, lambdaMax_B] = rangeFun(B);
lambdaMin_A = max(lambdaMin_A,1e-2);
lambdaMin_B = max(lambdaMin_B,1e-2);

size_U = size(A,1);
size_V = size(B,1);
gauss_U = zeros(2,1);
gauss_V = zeros(2,1);
g_U = 0;
g_V = 0;
len_U = u' * u;
len_V = v' * v;

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
f_V = 0;
fU_V = 0;
fL_V = 0;

delta_U = 0;
deltaU_U = 0;
deltaL_U = 0;
delta_V = 0;
deltaU_V = 0;
deltaL_V = 0;

eta_U = 0;
eta_V = 0;

alpha_U = 0;
alphaU_U = 0;
alphaL_U = 0;
alpha_V = 0;
alphaU_V = 0;
alphaL_V = 0;

iter_U = 1;
iter_V = 1;
flag_U = true;
flag_V = true;

if len_U < 1e-10
    flag_U = false;
    lower_U = full(max(log(tar), 1e-10));
    upper_U = lower_U;
end
if len_V < 1e-10
    flag_V = false;
    lower_V = full(max(-log(tar), 1e-10));
    upper_V = lower_V;
end
    
if prob <= 0.5
    flag = false;
else
    flag = true;
end

while flag_U || flag_V
    if flag_U
        proceed_u();
        [tmp_lower_U, tmp_upper_U] = get_gauss_dg_bounds(gauss_U(1), gauss_U(2), tar);
        
        lower_U = max(tmp_lower_U, 1e-10);
        upper_U = max(tmp_upper_U, 1e-10);
        if upper_U <= 1e-10
            lower_U = 1e-10;
            upper_U = 1e-10;
            flag_U = false;
        end
    end
    if flag_V
        proceed_v();
        [tmp_lower_V, tmp_upper_V] = get_gauss_dg_bounds(gauss_V(1), gauss_V(2), tar);
    
        lower_V = max(-tmp_upper_V, 1e-10);
        upper_V = max(-tmp_lower_V, 1e-10);
        
        if upper_V <= 1e-10
            lower_V = 1e-10;
            upper_U = 1e-10;
            flag_V = false;
        end
    end
    
    if prob * upper_V <= (1-prob) * lower_U
        flag = true;
        return;
    elseif prob * lower_V >= (1-prob) * upper_U
        flag = false;
        return;
    end    
end

function proceed_u()
    newGamma_U = u' * u / (p_U' * A * p_U);
    alpha_U = 1 / newGamma_U + beta_U / gamma_U;
    gamma_U = newGamma_U;
    
    if iter_U == 1
        f_U = 1 / alpha_U;
        delta_U = alpha_U;
        deltaU_U = alpha_U - lambdaMin_A;
        deltaL_U = alpha_U - lambdaMax_A;
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
    
    alphaU_U = lambdaMin_A + eta_U / deltaU_U;
    alphaL_U = lambdaMax_A + eta_U / deltaL_U;
     
    fU_U = eta_U * c_U / (delta_U * (alphaU_U * delta_U - eta_U));
    fL_U = eta_U * c_U / (delta_U * (alphaL_U * delta_U - eta_U));
    
    g_U = g_U + f_U;
    gauss_U(1) = len_U * (g_U + fL_U);
    gauss_U(2) = len_U * (g_U + fU_U);
    
    iter_U = iter_U + 1;
    if iter_U > size_U
        flag_U = false;
    end
        
end

function proceed_v()
    newGamma_V = v' * v / (p_V' * B * p_V);
    alpha_V = 1 / newGamma_V + beta_V / gamma_V;
    gamma_V = newGamma_V;
    
    if iter_V == 1
        f_V = 1 / alpha_V;
        delta_V = alpha_V;
        deltaU_V = alpha_V - lambdaMin_B;
        deltaL_V = alpha_V - lambdaMax_B;
    else
        c_V = c_V * eta_V / (delta_V^2);
        delta_V = 1 / gamma_V;
        f_V = gamma_V * c_V;
        deltaU_V = alpha_V - alphaU_V;
        deltaL_V = alpha_V - alphaL_V;
    end
    
    beta_V = v' * v;
    v = v - gamma_V * B * p_V;
    beta_V = v' * v / beta_V;
    eta_V = beta_V / (gamma_V^2);
    p_V = v + beta_V * p_V;
    
    alphaU_V = lambdaMin_B + eta_V / deltaU_V;
    alphaL_V = lambdaMax_B + eta_V / deltaL_V;
    
    fU_V = eta_V * c_V / (delta_V * (alphaU_V * delta_V - eta_V));
    fL_V = eta_V * c_V / (delta_V * (alphaL_V * delta_V - eta_V));
    
    g_V = g_V + f_V;
    gauss_V(1) = len_V * (g_V + fL_V);
    gauss_V(2) = len_V * (g_V + fU_V);
    
    iter_V = iter_V + 1;
    if iter_V > size_V
        flag_V = false;
    end
end
end

