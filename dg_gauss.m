%% sampling subsets for Double Greedy algorithm with Gauss quadrature
%
% -input
%   K: data kernel matrix, N*N where N is number of samples
%
% -output
%   C: selected elements by Double Greedy algorithm
%
% sample usage:
%   C = dg_gauss(K)

function C = dg_gauss(K)

n = size(K,1);

C = [];
C_bar = 1:n;

curr_mat = [];
curr_mat_bar = K;

acc = 0;
rej = 0;
iterlen = n / 5;

for i = 1:n
    if mod(i,iterlen) == 0
        fprintf('%d/%d\n', i,n);
    end
    
    tmp_C_bar = C_bar;
    tmp_C_bar(length(C)+1) = [];
    
    tmp_mat_bar = curr_mat_bar;
    tmp_mat_bar(length(C)+1,:) = [];
    tmp_mat_bar(:,length(C)+1) = [];
    
    prob = rand;
    
    bu = K(C,i);
    bv = K(tmp_C_bar,i);
    
    if gauss_dg_judge(curr_mat,tmp_mat_bar,bu,bv,K(i,i),prob,@gershgorin);
        curr_mat = [curr_mat K(C,i); K(i,C) K(i,i)];
        C = [C i];
        acc = acc + 1;
    else
        curr_mat_bar = tmp_mat_bar;
        C_bar = tmp_C_bar;
        rej = rej + 1;
    end
end
end
