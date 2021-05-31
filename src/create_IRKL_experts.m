function experts_IRKL = create_IRKL_experts(domain, K)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sequential Domain Adaptation by Synthesizing Distributionally Robust
% Experts
% ICML 2021 Submission
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function creates IR-KL experts 
% INPUT:
% domain: contains the target and the source data 
% K : number of experts
% OUTPUT: 
% experts_IRKL : experts of IR-KL with different lambda values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    mean_source = domain.source.mean;
    cov_source = domain.source.cov;
    icov_source = domain.source.icov;

    mean_target = domain.target.mean;
    cov_target = domain.target.cov;
    icov_target = domain.target.icov;

    [p, ~] = size(cov_target);
    lambdas = (0 : 1 / (K-1) : 1);
    cov_lambda = zeros(p, p, K);
    mean_lambda = zeros(p, K);

    cov_lambda(:, :, 1) = cov_source;
    cov_lambda(:, :, K) = cov_target;

    mean_lambda(:, 1) = mean_source;
    mean_lambda(:, K) = mean_target;
    diver_kullback_t2s = kl_divergence(domain.target.mean, domain.target.cov, domain.source.mean, domain.source.cov);

    rho = diver_kullback_t2s / (K - 1) / 3;

    experts_IRKL = zeros(p-1, K);
    % kullback_diver_t2s = sqrt(kl_divergence(mean_lambda(:, K), ...
    %     cov_lambda(:, :, K), mean_lambda(:, 1), cov_lambda(: , :, 1)));
    % rho = kullback_diver_t2s / (K - 1) / 3;
    % experts_kullback_wass(:, 1) = DRO_kullback(mean_lambda(:, 1), cov_lambda(:, :, 1), rho .^ 2);
    %  = DRO_kullback(mean_lambda(:, K), cov_lambda(:, :, K), rho .^ 2);
%     sum = 0;
    experts_IRKL(:, 1) = inv(cov_lambda(1:p-1, 1:p-1, 1)) * cov_lambda(1:p-1, p, 1);
    experts_IRKL(:, K) = inv(cov_lambda(1:p-1, 1:p-1, K)) * cov_lambda(1:p-1, p, K);

    for i = 2 : (K-1)
    %     lambdas(i)
        lambdas(i) = lambdas(i-1) + 1 * (exp(i-1)-1) / (sum(exp((0 : K-1))) - (K-1));

        cov_lambda(:, :, i) = inv((1 - lambdas(i)) * icov_source + lambdas(i) * icov_target);
        mean_lambda(:, i) = cov_lambda(:, :, i) * ((1 - lambdas(i)) * icov_source * mean_source + ... 
            lambdas(i) * icov_target * mean_target);
%         %%%% To debug one can check the distances between two distributions
%         kkkk = kl_divergence(mean_lambda(:, i), ...
%                 cov_lambda(:, :, i), mean_lambda(:, i-1), cov_lambda(: , :, i-1)) ;
%         fprintf('Divergence between two distributions with given mean and covariance'+ string(kkkk) + '\n')
%         sum = sum + kkkk;
%         experts_dro_kullback(:, i) = inv(cov_lambda(1:p-1, 1:p-1, i)) * cov_lambda(1:p-1, p, i);
%         experts_IRKL(:, i) = IRKL(mean_lambda(:, i), ...
%             cov_lambda(:, :, i), rho);
        [experts_IRKL(:, i), ~] = AGD(mean_lambda(:, i), ...
            cov_lambda(:, :, i), rho, 1e5);
    end

    

%     %%%% For debugging
%     sum = sum +  kl_divergence(mean_lambda(:, K), ...
%                 cov_lambda(:, :, K), mean_lambda(:, K-1), cov_lambda(: , :, K-1));
%     divergence_total = kl_divergence(mean_lambda(:, end), ...
%                 cov_lambda(:, :, end), mean_lambda(:, 1), cov_lambda(: , :, 1));
%      %%%% wass_dist should be equal to sum
end



