function emp_experts = create_emp_experts(beta_source, beta_target, K)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sequential Domain Adaptation by Synthesizing Distributionally Robust
% Experts
% ICML 2021 Submission
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function creates Convex Combination (CC) experts explained in 
% Section~2 of the paper 
% INPUT:
% beta_source : empirical regressor trained only with the source domain data
% beta_target : empirical regressor trained only with the target domain
% data
% K : number of experts
% OUTPUT:
% emp_experts : experts of Convex Combination 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    d = length(beta_source);
    emp_experts = zeros(d, K, 5);

    %%%%%%%%%%%%%% Create Convex Combination Experts
    %% CC-L
    lambdas = [0 : 1 / K : 1];
    for k = 1 : K
        emp_experts(:, k, 1) = beta_target * lambdas(k) + (1 - lambdas(k)) * beta_source;
    end
    %% CC-SL
    
    lambdas = [0 : 1 / 2 / (K-1) : 1/2];
    for k = 1 : K
        emp_experts(:, k, 2) = beta_target * lambdas(k) + (1 - lambdas(k)) * beta_source;
    end
    %% CC-TL
    lambdas = 1 - [0 : 1 / 2 / (K-1) : 1/2];
    for k = 1 : K
        emp_experts(:, k, 3) = beta_target * lambdas(k) + (1 - lambdas(k)) * beta_source;
    end
    
    %% CC-TE
    lambda = 1;
    for k = 1 : K
        emp_experts(:, k, 4) = beta_target * lambda + (1 - lambda) * beta_source;
        lambda = lambda - 1 * (exp(k-1)-1) / (sum(exp([0 : K-1])) - (K-1));
    end
    %% CC-SE
    lambda = 0;
    for k = 1 : K
        emp_experts(:, k, 5) = beta_target * lambda + (1 - lambda) * beta_source;
        lambda = lambda + 1 * (exp(k-1)-1) / (sum(exp([0 : K-1])) - (K-1));
    end
end