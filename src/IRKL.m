function beta = IRKL(mean_xi, cov_xi, rho)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sequential Domain Adaptation by Synthesizing Distributionally Robust
% Experts
% ICML 2021 Submission
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function creates IR-KL expert for a given radius value 
% INPUT:
% mean_xi : corresponds to barycenter interpolation \widehat \mean_\lambda
% cov_xi : corresponds to barycenter interpolation \widehat \Sigma_\lambda
% rho : radius of the ambiguity ball
% OUTPUT:
% beta : the regressor
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    yalmip('clear')
    icov_xi = inv(cov_xi);
    p = length(mean_xi);
    d = p - 1;
    tau = sdpvar(1, 1);
    mu = sdpvar(p, 1);
    t = sdpvar(1, 1);
    M = sdpvar(p, p);
    gamma = sdpvar(1, 1);
    Z = sdpvar(p, p, 'full');

    cons = [tau >= 0, t >= 0, M >= 0];
    cons = [cons, mean_xi' * icov_xi * mean_xi - 2 * mean_xi' * icov_xi * mu + ... 
        trace(M * icov_xi) - log(1 - t) - p - rho + log_det(cov_xi) <= gamma];

%     cons = [cons; [M mu; mu' 1] >= 0];

    cons = [cons; triu(Z) - diag(diag(Z)) == 0]; % Lower triangular Z
    cons = [cons; [M Z; Z' diag(diag(Z))] >= 0];
    cons = [cons; gamma <= sum(log(diag(Z)))];

    cons = [cons; [M mu; mu' t] >= 0, M - [[zeros(d, d) zeros(d, 1)]; [zeros(1, d) tau]] >= 0];
    
    obj = tau;  
    ops = sdpsettings('solver', 'mosek', 'verbose', 0);
%     ops.mosek.MSK_DPAR_SEMIDEFINITE_TOL_APPROX = 1e-5;
%     ops.mosek.MSK_DPAR_INTPNT_CO_TOL_DFEAS = 1e-10;
%     ops.mosek.MSK_DPAR_INTPNT_CO_TOL_INFEAS = 1e-10;
%     ops.mosek.MSK_DPAR_INTPNT_CO_TOL_MU_RED = 1e-10;
%     ops.mosek.MSK_DPAR_INTPNT_CO_TOL_NEAR_REL = 1;
%     ops.mosek.MSK_DPAR_INTPNT_CO_TOL_PFEAS = 1e-10;
%     ops.mosek.MSK_DPAR_INTPNT_CO_TOL_REL_GAP = 1e-10;
    diagnoise = optimize(cons, -obj , ops);
    
    fprintf(diagnoise.info + "\n")
    if diagnoise.problem > 0
        keyboard
    end
    
    M_opt = value(M);
    beta = inv(M_opt(1:d, 1:d)) * M_opt(1:d, p);
end