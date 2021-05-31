function experts_SIKL = create_SIKL_experts(domain, K)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sequential Domain Adaptation by Synthesizing Distributionally Robust
% Experts
% ICML 2021 Submission
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function creates SI-KL experts 
% INPUT:
% domain : contains the target and the source data 
% K : number of experts
% OUTPUT:
% experts_SIKL : experts of SI-KL with different source and target radius
% pairs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    rhos_source = zeros(K, 1);
    rhos_target = zeros(K, 1);
    d = domain.dimension;
    diver_kullback_t2s = kl_divergence(domain.target.mean, domain.target.cov, domain.source.mean, domain.source.cov);
    experts_SIKL = zeros(d, K);
    rho_source_max = diver_kullback_t2s - 1;
    rho_source_min = 1e-3;
    if domain.dimension > 15
        rho_source_min = min(diver_kullback_t2s, 5);
    end
    domain.source.radius = rho_source_min;
    for k = 1 : K
        domain.source.radius = domain.source.radius + (rho_source_max - rho_source_min) / (sum(exp([0 : K-1])) - (K-1)) * (exp(k-1)-1);
        rhos_source(k, 1) = domain.source.radius;
        min_target = minimum_target_radius_kullback(domain);
        domain.target.radius = min_target + domain.source.radius / 2;
        rhos_target(k, 1) = domain.target.radius;
        experts_SIKL(:, k) = SIKL(domain);
    end
    
end