function pi_dist = BOA(data, expert, M, eta_c)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sequential Domain Adaptation by Synthesizing Distributionally Robust
% Experts
% ICML 2021 Submission
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function computes the least squares loss for a given regressor
% INPUT:
% data : squential feature vectors
% expert : regressor
% M : time horizon
% eta_c : step size
% OUTPUT:
% pi_dist = distribution of the weight vector over the experts (K) throughout
% the time horizon (M) (size : K x (M+1))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    data_x = data.x;
    data_y = data.y;
    % expert = [beta_dro_worst1'; beta_dro_worst2'; w_s1'; w_s2'];
    [K, ~] = size(expert); % number of experts
    L = zeros(K, M+1);
    ell = zeros(K, M+1);
    E = zeros(K, M+1);
    pi_dist = ones(K, 1) / K;
    for t = 1 : M
        for j = 1 : K
            eta = eta_c / 1;
            x_t = data_x(t, :);
            y_t = data_y(t);
            ell(j, t+1) = lse_loss(x_t, y_t, expert(j, :)') - sum(sum((expert * x_t' - y_t) .^ 2, 2) .* pi_dist(:, t));
            pi_dist(j, t+1) = exp(- eta * ell(j, t + 1) * (1 + eta * ell(j, t + 1))) * pi_dist(j, t) / ...
                sum(pi_dist(:, t) .* exp(- eta * ell(:, t + 1) .* (1 + eta * ell(:, t + 1))));
        end
    end
end