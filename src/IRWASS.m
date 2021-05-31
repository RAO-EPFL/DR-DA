function beta = IRWASS(mean_xi_hat, cov_xi_hat, rho)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sequential Domain Adaptation by Synthesizing Distributionally Robust
% Experts
% ICML 2021 Submission
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function creates IR-WASS expert for a given radius value 
% Distributionally robust linear regression with an ambiguity set
% around the given mean_xi_hat and cov_xi_hat with the radius of rho
% the divergence is Wasserstein-type
% Inputs:
% mean_xi : corresponds to barycenter interpolation \widehat \mean_\lambda
% cov_xi : corresponds to barycenter interpolation \widehat \Sigma_\lambda
% rho : radius of the ambiguity ball

    yalmip('clear')

    p = length(mean_xi_hat);
    d = p - 1;
    tau = sdpvar(1, 1);
    mu = sdpvar(p, 1);
    M = sdpvar(p, p);
    C = sdpvar(p, p, 'full');
    T = sdpvar(p, p);

    cons = [tau >= 0, M >= 0, T >= 0];
    cons = [cons, norm(mean_xi_hat) ^ 2 - 2 * mean_xi_hat' * mu + trace(M + cov_xi_hat - 2 * C) <= rho];
%     cons = [cons; [M mu; mu' 1] >= 0];
    cons = [cons, [T C; C' cov_xi_hat] >= 0];
    cons = [cons; [M mu; mu' 1] >= 0, [M - 1e-10 * eye(p) mu; mu' 1] >= 0, M - [[zeros(d, d) zeros(d, 1)]; [zeros(1, d) tau]] >= 0];
    cons = [cons; [M-T mu; mu' 1] >= 0];
    obj = tau;
    
%     % debug by adding trivial constraints
%     cons = [cons; M == mean_xi_source*mean_xi_source' + cov_xi_source];
%     cons = [cons; mu == mean_xi_source];
    
    
    ops = sdpsettings('solver', 'mosek', 'verbose', 0);
%     ops.sedumi.eps = 1e-15;
%     ops.sedumi.bigeps = 1e-10;
%     ops.mosek.MSK_DPAR_SEMIDEFINITE_TOL_APPROX = 1e-10;
%     ops.mosek.MSK_DPAR_INTPNT_CO_TOL_DFEAS = 1e-15;
%     ops.mosek.MSK_DPAR_INTPNT_CO_TOL_INFEAS = 1e-15;
%     ops.mosek.MSK_DPAR_INTPNT_CO_TOL_MU_RED = 1e-15;
%     ops.mosek.MSK_DPAR_INTPNT_CO_TOL_NEAR_REL = 1;
%     ops.mosek.MSK_DPAR_INTPNT_CO_TOL_PFEAS = 1e-15;
%     ops.mosek.MSK_DPAR_INTPNT_CO_TOL_REL_GAP = 1e-15;
    diagnose = optimize(cons, -obj , ops);
    
    fprintf(diagnose.info + "\n")
    
    if diagnose.problem > 0 
        keyboard
    end

    M_opt = value(M);
    beta = inv(M_opt(1:d, 1:d)) * M_opt(1:d, p);
    
    %for debugging
    %M_xi_target = mean_xi_target*mean_xi_target' + cov_xi_target;
    %b = inv(M_xi_target(1:d, 1:d))*M_xi_target(1:d, p);
end