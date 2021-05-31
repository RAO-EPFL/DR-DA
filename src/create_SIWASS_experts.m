function experts_SIWASS = create_SIWASS_experts(domain, K)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sequential Domain Adaptation by Synthesizing Distributionally Robust
% Experts
% ICML 2021 Submission
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function creates SI-WASS experts 
% INPUT:
% domain : contains the target and the source data 
% K : number of experts
% OUTPUT:
% experts_SIWASS : experts of SI-WASS with different source and target
% radius pairs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    rhos_source = zeros(K, 1);
    rhos_target = zeros(K, 1);
    d = domain.dimension;
    diver_wass = wass_divergence(domain.source.mean, domain.source.cov, ...
        domain.target.mean, domain.target.cov);

    experts_SIWASS = zeros(d, K);
    rho_source_max = diver_wass;
    rho_source_min = 1e-4;
    
    for k = 1 : K
        domain.source.radius = domain.source.radius + (rho_source_max - rho_source_min) / sum(exp((0 : K-1)) - (K-1)) * (exp(k - 1) - 1);
%         domain.source.radius = rho_source_min + (rho_source_max - rho_source_min) * (k - 1) / (K - 1);

        rhos_source(k, 1) = domain.source.radius;
        
        domain.target.radius = (sqrt(diver_wass) - sqrt(domain.source.radius)) ^ 2 + domain.source.radius / 2;
        rhos_target(k, 1) = domain.target.radius;
%             fprintf('Target raidus %d, Source radius %d \n', domain.target.radius, domain.source.radius)
        experts_SIWASS(:, k) = SIWASS(domain);

    end
end