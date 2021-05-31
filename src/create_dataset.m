function domain = create_dataset(domain_org, eta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sequential Domain Adaptation by Synthesizing Distributionally Robust
% Experts
% ICML 2021 Submission
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function computes first&second moment of a given data
% INPUT:
% domain_org : domain information source&target feature vectors and labels
% eta is the amount of regularization
% OUTPUT:
% domain : source&target domain information (mean, covariance)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    if nargin < 2
        eta = 1e-3;
    end
   
    x_source_org = domain_org.source.x;
    y_source_org = domain_org.source.y;
    x_target_org = domain_org.target.x;
    y_target_org = domain_org.target.y;
    N_target = domain_org.N_target;
    
    % Choose samples to create the initial target set
    indices_target_sampling = randperm(N_target);
    x_target = x_target_org(indices_target_sampling, :);
    y_target = y_target_org(indices_target_sampling, :);
    
    % Remove the selected samples from the target test set
    x_target_org(indices_target_sampling, :) = []; 
    y_target_org(indices_target_sampling, :) = [];
    
    % Recalculate number of samples in the target test set
    [N_target_test, d] = size(x_target_org);
    
    % Change the order of samples in the target test set randomly 
    % From 1 till N_target_test this will create one stochastic sequence
    rand_inds = randperm(N_target_test);
    x_target_org = x_target_org(rand_inds, :);
    y_target_org = y_target_org(rand_inds, :);
    
    % Create new random variable \xi both for source and target domain
    xi_source = [x_source_org, y_source_org];
    xi_target = [x_target, y_target]; 
    p = d + 1;

    mean_xi_source = mean(xi_source)';
    % call covariance, normalize by 1/N
    cov_xi_source = cov(xi_source, 1);
    % The source usually has many samples, no regularization is needed
    if sum(eig(cov_xi_source) <= 1e-6) >= 1
        cov_xi_source = cov_xi_source + 1e-6 * eye(p);
    end
    icov_xi_source = inv(cov_xi_source);

    mean_xi_target = mean(xi_target)';
    cov_xi_target = cov(xi_target, 1);
    
    
%     if sum(eig(cov_xi_target) <= 1e-6) >= 1
%         cov_xi_target = cov_xi_target + 1e-6 * eye(p);
%     end
   
    
    %for debugging
%     M_xi_target = mean_xi_target*mean_xi_target' + cov_xi_target;
%     b = inv(M_xi_target(1:d, 1:d) + eta*eye(d))*M_xi_target(1:d, p);
    % b should be equal to LSE(x_target, y_target, eta)
%     
%     % M_xi_target should be equal xi_target'*xi_target/size(xi_target, 1)
%     norm(M_xi_target - xi_target'*xi_target/size(xi_target, 1), 'fro')
%     norm(M_xi_target(1:d, p) - x_target'*y_target/size(x_target, 1), 'fro')
%     
%     % the below calculation shows that we have to be careful with the scale
%     % of eta
%     norm(inv(M_xi_target(1:d, 1:d) + eta*eye(d)) - inv((x_target'*x_target)/size(x_target, 1) + eta*eye(d)), 'fro')
%     
    
    % regularize the moment matrix by adding eta*I to the x-component
    temp = eta * eye(p);
    temp(end, end) = 0;
    cov_xi_target = cov_xi_target + temp;
    icov_xi_target = inv(cov_xi_target);
    cov_xi_source = cov_xi_source + temp;
    icov_xi_source = inv(cov_xi_source);
    
    
    
    domain = {};
    domain.dimension = d;
    domain.source = {};
    domain.source.mean = mean_xi_source;
    domain.source.cov = cov_xi_source;
    domain.source.icov = icov_xi_source;
    domain.source.radius = 0;
    domain.source.x = x_source_org;
    domain.source.y = y_source_org;
    
    domain.target = {};
    domain.target.mean = mean_xi_target;
    domain.target.cov = cov_xi_target;
    domain.target.icov = icov_xi_target;
    domain.target.radius = 0;
    domain.target.x = x_target;
    domain.target.y = y_target;
    domain.target.test = {};
    domain.target.test.x = x_target_org;
    domain.target.test.y = y_target_org;
    
end