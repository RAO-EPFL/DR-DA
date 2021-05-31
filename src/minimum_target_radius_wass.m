function rho_target_min = minimum_target_radius_wass(domain);
    yalmip('clear')
    
    mean_xi_source = domain.source.mean;
    cov_xi_source = domain.source.cov;
    rho_source = domain.source.radius; % domain.source.radius;

    mean_xi_target = domain.target.mean;
    cov_xi_target = domain.target.cov;


    p = length(mean_xi_target);
    d = p - 1;
    tau = sdpvar(1, 1);
    mu = sdpvar(p, 1);
    M = sdpvar(p, p);
    C1 = sdpvar(p, p, 'full');
    C2 = sdpvar(p, p, 'full');
    T = sdpvar(p, p);

    cons = [tau >= 0, M >= 0, T >= 0];
    cons = [cons, norm(mean_xi_source) ^ 2 - 2 * mean_xi_source' * mu + trace(M + cov_xi_source - 2 * C1) <= rho_source];
%     cons = [cons, norm(mean_xi_target) ^ 2 - 2 * mean_xi_target' * mu + trace(M + cov_xi_target - 2 * C2) <= rho_target];
%     cons = [cons; [M mu; mu' 1] >= 0];
    cons = [cons, [T C1; C1' cov_xi_source] >= 0];
    cons = [cons, [T C2; C2' cov_xi_target] >= 0];
    cons = [cons; [M mu; mu' 1] >= 0, [M - 1e-10 * eye(p) mu; mu' 1] >= 0, M - [[zeros(d, d) zeros(d, 1)]; [zeros(1, d) tau]] >= 0];
    cons = [cons; [M-T mu; mu' 1] >= 0];
    obj = norm(mean_xi_target) ^ 2 - 2 * mean_xi_target' * mu + trace(M + cov_xi_target - 2 * C2);
    
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
    diagnose = optimize(cons, obj , ops);
   
    rho_target_min = value(obj);
    %for debugging
    %M_xi_target = mean_xi_target*mean_xi_target' + cov_xi_target;
    %b = inv(M_xi_target(1:d, 1:d))*M_xi_target(1:d, p);
end
