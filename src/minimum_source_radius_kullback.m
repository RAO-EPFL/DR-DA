function min_source_radius = minimum_source_radius_kullback(domain)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sequential Domain Adaptation by Synthesizing Distributionally Robust
% Experts
% ICML 2021 Submission
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function computes the minimum source radius when the target radius
% is given. 
% The minimum source radius is given by a one dimensional maximization
% problem that can be solved using a bisection algorithm
% INPUT:
% domain : source and target data, target radius
% OUTPUT:
% min_source_radius : minimum source radius such that the ambiguity set is
% non-empty.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    lower_ = 0;
    upper_ = 1e15; 
    err = 1e-15;
    while (upper_ - lower_) > err
        [temp_up, ~] = minimum_source_radius_kl_function((lower_ + (lower_ + upper_) / 2) / 2, domain) ;
        [temp_low, ~] = minimum_source_radius_kl_function((upper_ + (lower_ + upper_) / 2) / 2, domain);
        if temp_up > temp_low 
            upper_ = (lower_ + upper_) / 2;
        else
            lower_ = (lower_ + upper_) / 2;
        end
    end
    max_gamma = (upper_ + lower_) / 2;

    [~, min_source_radius] = minimum_source_radius_kl_function(max_gamma, domain);
end

