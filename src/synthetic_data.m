%% Domain adaptation Dataset Preparation
clear;
close all
clc
rng(97768)  % For reproducibility
d = 50;
% Source distribution
N_source = 10000;
mean_source = zeros(d, 1);
beta_source = 1 * ones(d, 1); 
a1 = abs(randn(d));
cov_source = a1 * a1';
x_source = mvnrnd(mean_source, cov_source, N_source);
temp = abs(randn(d));
y_source = x_source * beta_source + 1 * randn(N_source, 1);
% Target distribution
N_target = 150;
[V,D,W] = eig(cov_source);
D_new = D + diag(abs(randn(d, 1)));
cov_target = V * D_new * W';
mean_target = mean_source + abs(randn(d, 1)) * .1;
beta_target = beta_source + randn(d, 1) * .03; 
% a2 = abs(randn(d)) * 1;
% cov_target = a2 * a2';
x_target = mvnrnd(mean_target, cov_target, N_target);
y_target = x_target * beta_target + 1 * randn(N_target, 1);

% Target test distribution
N_target_test = 10000;
x_target_test = mvnrnd(mean_target, cov_target, N_target_test);
y_target_test = x_target_test * beta_target + 5 * randn(N_target_test, 1) ;
x_source_test = mvnrnd(mean_source, cov_source, 10000);
temp = abs(randn(d));
y_source_test = x_source_test * beta_source + 1 * randn(10000, 1);

xi_source = [x_source, y_source];
xi_target = [x_target, y_target]; 
p = d + 1; 

mean_xi_source = mean(xi_source)';
cov_xi_source = cov(xi_source);
icov_xi_source = inv(cov_xi_source);

mean_xi_target = mean(xi_target)';
cov_xi_target= cov(xi_target);
icov_xi_target = inv(cov_xi_target);


diver_true_s2t = kl_divergence(mean([x_source_test, y_source_test])', ...
    cov([x_source_test, y_source_test]), ...
    mean([x_target_test, y_target_test])', cov([x_target_test, y_target_test]));

diver_true_t2s = kl_divergence(mean([x_target_test, y_target_test])', ...
    cov([x_target_test, y_target_test]), mean([x_source_test, y_source_test])', ...
    cov([x_source_test, y_source_test]));

diver_t2t_hat = kl_divergence(mean([x_target_test, y_target_test])', ...
    cov([x_target_test, y_target_test]), mean_xi_target, cov_xi_target);

diver_s2s_hat = kl_divergence( mean([x_source_test, y_source_test])', ...
    cov([x_source_test, y_source_test]), mean_xi_source, cov_xi_source);

diver_s2t = kl_divergence(mean_xi_source, cov_xi_source, mean_xi_target, cov_xi_target);
diver_t2s = kl_divergence(mean_xi_target, cov_xi_target, mean_xi_source, cov_xi_source);


domain = {};
domain.source = {};
domain.source.mean = mean_xi_source;
domain.source.cov = cov_xi_source;
domain.source.icov = icov_xi_source;
domain.source.radius = 0;
domain.target = {};
domain.target.mean = mean_xi_target;
domain.target.cov = cov_xi_target;
domain.target.icov = icov_xi_target;
domain.target.radius = 0;
%% Empirical LSE on Source and Target data
beta_lse_source = LSE(x_source, y_source);
lse_error_source = lse_loss(x_source, y_source, beta_lse_source);
fprintf('Error of LSE trained on Source data = %d \n', lse_error_source); 

beta_lse_target = LSE(x_target, y_target);
lse_error_target = lse_loss(x_target, y_target, beta_lse_target);
fprintf('Error of LSE trained on Target data = %d \n', lse_error_target); 

%% Create Experts 
%%%%%%%%%%%%%% Create DRO Experts
K = 10; % number of experts
rho_target_max = diver_s2t;
rho_target_min = 0;
rhos_source = zeros(K, 1);
rhos_target = zeros(K, 1);

experts_dro = zeros(d, K);
experts_emp = zeros(d, K);

for k = 1 : K
    domain.target.radius = rho_target_min + (rho_target_max - rho_target_min) / (K - 1) * (k - 1);
    rhos_target(k, 1) = domain.target.radius;
    domain.source.radius = minimum_source_radius(domain) + domain.target.radius / 2;
    rhos_source(k, 1) = domain.source.radius;
    fprintf('Target raidus %d, Source radius %d \n', domain.target.radius, domain.source.radius)
    experts_dro(:, k) = worst_case_domain_adapt(domain);
end


%%%%%%%%%%%%%% Create Empirical Experts %%%%%%%%%%%%%%
lambda = 1;
for k = 1 : K
    experts_emp(:, k) = beta_lse_target * lambda + (1 - lambda) * beta_lse_source;
    lambda = lambda - lambda / K * k / 10;
end
%% Choose the best expert
% expert = [beta_dro_worst1'; beta_dro_worst2'; w_s1'; w_s2'];
sequential_data = {};
sequential_data.x = x_target_test;
sequential_data.y = y_target_test;
eta = 1 * 1e-2;
M = 100;
expert_dist_DRO = BOA(sequential_data, experts_dro', M, eta);
expert_dist_EMP = BOA(sequential_data, experts_emp', M, eta);
%%
loss_track_dro = zeros(M+1, 1);
loss_track_emp = zeros(M+1, 1);

for t = 1 : M + 1
    x_t = x_target_test(t, :);
    y_t = y_target_test(t);
    loss_track_emp(t) = lse_loss(x_target_test(t, :), y_target_test(t), experts_emp * expert_dist_EMP(:, t));
    loss_track_dro(t) = lse_loss(x_target_test(t, :), y_target_test(t), experts_dro * expert_dist_DRO(:, t));

end

figure; 
hold on; 
plt_emp = plot(cumsum(loss_track_emp), 'linewidth', 3, 'color', 'b');
hold on; 
plt_dro = plot(cumsum(loss_track_dro), 'linewidth', 3, 'color', 'r'); 
grid on;
lgd = legend([plt_emp, plt_dro], 'EMP', 'DRO', 'Location', 'southeast', 'interpreter','latex');
lgd.FontSize = 16;