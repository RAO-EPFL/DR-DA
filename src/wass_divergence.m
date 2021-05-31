function diver_wass_s2t = wass_divergence(mean_xi_source, cov_xi_source, mean_xi_target, cov_xi_target)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sequential Domain Adaptation by Synthesizing Distributionally Robust
% Experts
% ICML 2021 Submission
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function computes the Wasserstein-type divergence between the given
% mean-covariance pairs
% INPUT:
% mean_xi_source : mean of source domain data (X, Y)
% cov_xi_source : covariance of source domain data (X, Y)
% mean_xi_target : mean of target domain data (X, Y)
% cov_xi_target : covariance of target domain data (X, Y)
% OUTPUT:
% diver_wass_s2t : Wasserstein-type divergence
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    diver_wass_s2t = norm(mean_xi_source - mean_xi_target) ^ 2 + ...
        trace(cov_xi_source + cov_xi_target - 2 * sqrtm(sqrtm(cov_xi_target) *...
        cov_xi_source * sqrtm(cov_xi_target)));
end