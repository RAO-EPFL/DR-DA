function [val, min_radius] = minimum_source_radius_kl_function(gamma, domain)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sequential Domain Adaptation by Synthesizing Distributionally Robust
% Experts
% ICML 2021 Submission
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function computes the value of the objective function of the
% optimization problem in Proposition~5.1.
% The minimum source radius is given by a one dimensional maximization
% problem that can be solved using a bisection algorithm
% INPUT:
% domain : source and target data, target radius
% OUTPUT:
% min_radius : minimum source radius for given mean and covarinace
% information
% val : value of the objective function of the optimization problem in 
% Proposition~5.1.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    mean_xi_source = domain.source.mean;
    cov_xi_source = domain.source.cov;
    icov_xi_source = domain.source.icov;
    rho_target = domain.target.radius;

    mean_xi_target = domain.target.mean;
    cov_xi_target = domain.target.cov;
    icov_xi_target = domain.target.icov;
    
    cov_gamma = inv(icov_xi_source / (gamma + 1) + gamma * icov_xi_target / (gamma + 1));
    mean_gamma = cov_gamma / (gamma + 1) * (icov_xi_source * mean_xi_source + ...
        gamma * icov_xi_target * mean_xi_target);
    min_radius = kl_divergence(mean_gamma, cov_gamma, mean_xi_source, cov_xi_source);
    val =  min_radius + ...
        gamma * kl_divergence(mean_gamma, cov_gamma, mean_xi_target, cov_xi_target) - ...
        gamma * rho_target;
end