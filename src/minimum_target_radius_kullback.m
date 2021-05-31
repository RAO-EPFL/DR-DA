function min_target_radius = minimum_target_radius_kullback(domain)

% gammas = [0 : 0.01: 500];
% clear obj
% % rho_source = 1e-2;
% for i = 1 : length(gammas)
%     i
%     gamma = gammas(i);
%     cov_gamma = (gamma + 1) * inv(icov_xi_target + gamma * icov_xi_source);
%     mean_gamma = inv(icov_xi_target + gamma * icov_xi_source) * (icov_xi_target * mean_xi_target + gamma * icov_xi_source * mean_xi_source);
%     obj(i) = kl_divergence(mean_gamma, cov_gamma, mean_xi_target, cov_xi_target) + ...
%         gamma * kl_divergence(mean_gamma, cov_gamma, mean_xi_source, cov_xi_source) - ...
%         rho_source * gamma;
% end
% figure
% plot(gammas, obj)

% cons = [gamma >= 0];
% 
% ops = sdpsettings('solver', 'mosek', 'verbose', 0);
% diagnoise = optimize(cons, obj , ops);
    lower_ = 0;
    upper_ = 1e15; 
    err = 1e-8;
    while (upper_ - lower_) > err
        [temp_up, ~] = minimum_target_radius_kl_function((lower_ + (lower_ + upper_) / 2) / 2, domain) ;
        [temp_low, ~] = minimum_target_radius_kl_function((upper_ + (lower_ + upper_) / 2) / 2, domain);
        if temp_up > temp_low 
            upper_ = (lower_ + upper_) / 2;
        else
            lower_ = (lower_ + upper_) / 2;
        end
    end
    max_gamma = (upper_ + lower_) / 2;

    [~, min_target_radius] = minimum_target_radius_kl_function(max_gamma, domain);
end

