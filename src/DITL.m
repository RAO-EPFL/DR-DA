function beta_ditl = DITL(domain, h, reg)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sequential Domain Adaptation by Synthesizing Distributionally Robust
% Experts
% ICML 2021 Submission
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function creates RWS experts 
% INPUT:
% domain : contains the target and the source data 
% h>0 : Gaussian kernel bandwidth
% reg : regularization parameter Reweighthing Strategy explained in
% Section~2 in the paper.
% OUTPUT:
% beta_ditl : regressor
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    x_source = domain.source.x;
    y_source = domain.source.y;

    x_target = domain.target.x;
    y_target = domain.target.y;

    N_target = length(domain.target.y);
    N_source = length(domain.source.y);
    yalmip('clear')
    theta = sdpvar(N_target, 1);
    alpha = sdpvar(N_source, 1);
    obj = sum(log(theta));

    cons = [sum(alpha' * exp(-(pdist2(x_source, x_source, 'euclidean') .^ 2 +...
        pdist2(y_source, y_source, 'euclidean') .^ 2) / (h^2))) == N_source];
    cons = [cons; alpha' * exp(-(pdist2(x_target, x_source, 'euclidean') .^ 2 +...
        pdist2(y_target, y_source, 'euclidean') .^ 2) / (h^2))' >= theta'];
    cons = [cons; alpha >= 0];



    ops = sdpsettings('solver', 'mosek', 'verbose', 0);
    diagnose = optimize(cons, -obj , ops);

    fprintf(diagnose.info + "\n")
    w = (value(alpha)' * exp(-(pdist2(x_source, x_source, 'euclidean') .^ 2 +...
    pdist2(y_source, y_source, 'euclidean') .^ 2) / (h^2)))';
%     reg = 1e-5;

   
    beta_ditl = inv(x_target' * x_target + (x_source' * diag(w) * x_source) + reg*eye(domain.dimension)) * ...
        (x_target' * y_target + x_source' * diag(w) * y_source);
end