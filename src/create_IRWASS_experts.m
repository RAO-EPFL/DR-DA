function experts_IRWASS = create_IRWASS_experts(domain, K)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sequential Domain Adaptation by Synthesizing Distributionally Robust
% Experts
% ICML 2021 Submission
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function creates IR-WASS experts 
% INPUT:
% domain : contains the target and the source data 
% K : number of experts
% OUTPUT:
% experts_IRWASS : experts of IR-WASS with different lambda values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    mean_source = domain.source.mean;
    cov_source = domain.source.cov;


    mean_target = domain.target.mean;
    cov_target = domain.target.cov;

    L = sqrtm(cov_target) * sqrtm(inv(sqrtm(cov_target) * ...
        cov_source * sqrtm(cov_target))) * sqrtm(cov_target); 
    [p, ~] = size(cov_target);
    lambdas = (0 : 1 / (K-1) : 1);
    cov_lambda = zeros(p, p, K);
    mean_lambda = zeros(p, K);

    cov_lambda(:, :, 1) = cov_source;
    cov_lambda(:, :, K) = cov_target;

    mean_lambda(:, 1) = mean_source;
    mean_lambda(:, K) = mean_target;
    experts_IRWASS = zeros(p-1, K);
    wass_dist_total = sqrt(wass_divergence(mean_lambda(:, K), ...
        cov_lambda(:, :, K), mean_lambda(:, 1), cov_lambda(: , :, 1)));
    rho = wass_dist_total / (K - 1) / 3;
    experts_IRWASS(:, 1) = IRWASS(mean_lambda(:, 1), cov_lambda(:, :, 1), rho .^ 2);
    experts_IRWASS(:, K) = IRWASS(mean_lambda(:, K), cov_lambda(:, :, K), rho .^ 2);
    for i = 2 : (K-1)
        lambdas(i) = lambdas(i-1) + 1 * (exp(i-1)-1) / (sum(exp((0 : K-1))) - (K-1));

        mean_lambda(:, i) = (1 - lambdas(i)) * mean_source + lambdas(i) * mean_target;
        temp = (1 - lambdas(i)) * eye(p) + lambdas(i) * L;
        cov_lambda(:, :, i) = temp * cov_source * temp';
        experts_IRWASS(:, i) = IRWASS(mean_lambda(:, i), (cov_lambda(:, :, i) + ...
            cov_lambda(:, :, i)') / 2, rho .^ 2);
    %     %%%% To debug one can check the distances between two distributions
    %     kkkk = sqrt( wass_divergence(mean_lambda(:, i), ...
    %             cov_lambda(:, :, i), mean_lambda(:, i-1), cov_lambda(: , :, i-1)));
    %     fprintf('Distance between two distributions '+ string(kkkk) + '\n')
    %     sum = sum + kkkk;
    end

    % %%%% For debugging
    % sum = sum + sqrt( wass_divergence(mean_lambda(:, K), ...
    %             cov_lambda(:, :, K), mean_lambda(:, K-1), cov_lambda(: , :, K-1)))
    % wass_dist_total = sqrt(wass_divergence(mean_lambda(:, end), ...
    %             cov_lambda(:, :, end), mean_lambda(:, 1), cov_lambda(: , :, 1)))
    %  %%%% wass_dist should be equal to sum
    
end


