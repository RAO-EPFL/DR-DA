function [val, min_radius] = minimum_target_radius_kl_function(gamma, domain)
    mean_xi_source = domain.source.mean;
    cov_xi_source = domain.source.cov;
    icov_xi_source = domain.source.icov;
    rho_source = domain.source.radius;

    mean_xi_target = domain.target.mean;
    cov_xi_target = domain.target.cov;
    icov_xi_target = domain.target.icov;
    
    cov_gamma = inv(icov_xi_target / (gamma + 1) + gamma * icov_xi_source / (gamma + 1));
    mean_gamma = cov_gamma / (gamma + 1) * (icov_xi_target * mean_xi_target + ...
        gamma * icov_xi_source * mean_xi_source);
    min_radius = kl_divergence(mean_gamma, cov_gamma, mean_xi_target, cov_xi_target);
    val =  min_radius + ...
        gamma * kl_divergence(mean_gamma, cov_gamma, mean_xi_source, cov_xi_source) - ...
        gamma * rho_source;
end