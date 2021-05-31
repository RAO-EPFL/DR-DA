function beta = SIKL(domain)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sequential Domain Adaptation by Synthesizing Distributionally Robust
% Experts
% ICML 2021 Submission
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function creates SI-KL experts 
% INPUT:
% domain : contains the target and the source mean&covariance, source and 
% target radius
% OUTPUT:
% beta : regressor
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    yalmip('clear')
    mean_xi_source = domain.source.mean;
    cov_xi_source = domain.source.cov;
    icov_xi_source = domain.source.icov;
    rho_source = domain.source.radius;

    mean_xi_target = domain.target.mean;
    cov_xi_target = domain.target.cov;
    icov_xi_target = domain.target.icov;
    rho_target = domain.target.radius;

    p = length(mean_xi_target);
    d = p - 1;
    tau = sdpvar(1, 1);
    mu = sdpvar(p, 1);
    t = sdpvar(1, 1);
    M = sdpvar(p, p);
    epig_var = sdpvar(p, 1);
    gamma = sdpvar(1, 1);
    Z = sdpvar(p, p, 'full');
    
    cons = [tau >= 0, t >= 0, M >= 0];
    cons = [cons, mean_xi_source' * icov_xi_source * mean_xi_source - 2 * mean_xi_source' * icov_xi_source * mu + ...
        trace(M * icov_xi_source) - log(1 - t) - p - rho_source + log_det(cov_xi_source) <= gamma];

    cons = [cons, mean_xi_target' * icov_xi_target * mean_xi_target - 2 * mean_xi_target' * icov_xi_target * mu + ... 
        trace(M * icov_xi_target) - log(1 - t) - p - rho_target + log_det(cov_xi_target) <= gamma];
%     cons = [cons; [M - 1e-4 * eye(p) mu; mu' 1] >= 0];
    cons = [cons; [M mu; mu' 1] >= 0];

    cons = [cons; triu(Z, 1) == 0]; % Lower triangular Z
    cons = [cons; [M Z; Z' diag(diag(Z))] >= 0];
    cons = [cons; gamma <= sum(log(diag(Z)))];
%     cons = [cons; epig_var <= 0];
%     for i = 1 : p
%         cons = [cons; Z(i, i) >= pexp([1; epig_var(i)]), Z(i, i) >= 0];
%     end
    cons = [cons; [M mu; mu' t] >= 0, M - [[zeros(d, d) zeros(d, 1)]; [zeros(1, d) tau]] >= 0];

    obj = tau;  
    ops = sdpsettings('solver', 'mosek', 'verbose', 0);
%     ops.mosek.MSK_DPAR_SEMIDEFINITE_TOL_APPROX = 1e-15;
%     ops.mosek.MSK_DPAR_INTPNT_CO_TOL_DFEAS = 1e-10;
%     ops.mosek.MSK_DPAR_INTPNT_CO_TOL_INFEAS = 1e-10;
%     ops.mosek.MSK_DPAR_INTPNT_CO_TOL_MU_RED = 1e-10;
%     ops.mosek.MSK_DPAR_INTPNT_CO_TOL_NEAR_REL = 1;
%     ops.mosek.MSK_DPAR_INTPNT_CO_TOL_PFEAS = 1e-10;
%     ops.mosek.MSK_DPAR_INTPNT_CO_TOL_REL_GAP = 1e-10;
    diagnoise = optimize(cons, -obj , ops);

    fprintf(diagnoise.info + "\n" + "SI-KL" + "\n")
    problem_error = diagnoise.problem;
    if problem_error > 0
        keyboard
    end     
    M_opt = value(M);
    beta = inv(M_opt(1:d, 1:d)) * M_opt(1:d, p);
end