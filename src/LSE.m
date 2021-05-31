function w = LSE(x, y, eta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sequential Domain Adaptation by Synthesizing Distributionally Robust
% Experts
% ICML 2021 Submission
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function computes the value of w that solve the ridge regression problem
% min \| xw - y\|_2^2 + eta \|w\|_2^2
% INPUT:
% x : feature vectors
% y : labels
% eta : regularization parameter 
% OUTPUT:
% w : LSE regressor
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%     yalmip('clear')
%     [~, d] = size(x);
%     w_s = sdpvar(d, 1);
%     % Objective function
%     objective_s = sum((x * w_s - y).^2);
%     % Specify solver settings and run solver
%     ops = sdpsettings('solver', 'mosek', 'verbose', 0);
%     optimize([], objective_s , ops);
%     w = value(w_s);
    

     if nargin < 3
         eta = 1e-6;
     end
     w = inv(x' * x/size(x, 1) + eta*eye(size(x, 2))) * (x' * y/size(x, 1)) ;
end