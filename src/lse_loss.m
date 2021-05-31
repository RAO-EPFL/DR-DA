function loss = lse_loss(x, y, beta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sequential Domain Adaptation by Synthesizing Distributionally Robust
% Experts
% ICML 2021 Submission
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function computes the least squares loss for a given regressor
% INPUT:
% x : feature vectors
% y : label information
% beta : regressor
% OUTPUT:
% min_source_radius : minimum source radius such that the ambiguity set is
% non-empty.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    loss = sum((x * beta - y) .^ 2);
end