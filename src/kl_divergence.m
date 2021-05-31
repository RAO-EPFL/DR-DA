function divergence_kl = kl_divergence(mu1, Sigma1, mu2, Sigma2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sequential Domain Adaptation by Synthesizing Distributionally Robust
% Experts
% ICML 2021 Submission
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function computes the Kullback-Leibler-type divergence between the given
% mean-covariance pairs
% INPUT:
% mu1 : mean of source domain data (X, Y)
% Sigma1 : covariance of source domain data (X, Y)
% mu2 : mean of target domain data (X, Y)
% Sigma2 : covariance of target domain data (X, Y)
% OUTPUT:
% divergence_kl : Kullback-Leibler divergence
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    isigma2 = inv(Sigma2);
    divergence_kl = (mu2 - mu1)' * isigma2 * (mu2 - mu1) + trace(Sigma1 * isigma2) - ... 
        log_det(Sigma1) - log_det(isigma2) - length(mu1);
end

