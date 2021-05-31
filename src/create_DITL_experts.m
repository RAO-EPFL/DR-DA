function experts_DITL = create_DITL_experts(domain, K, reg)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sequential Domain Adaptation by Synthesizing Distributionally Robust
% Experts
% ICML 2021 Submission
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function creates RWS experts 
% Inputs:
% domain : contains the target and the source data 
% K : number of experts
% reg : regularization parameter Reweighthing Strategy explained in
% Section~2 in the paper.
% OUTPUT: 
% experts_DITL : Experts of RWS with different h values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    experts_DITL = zeros(domain.dimension, K);
    hmin = 0.5;
    hmax= 10;
    for k = 1 : K
        h = hmin + (hmax - hmin) * (k-1) / (K-1);
        experts_DITL(:, k) = DITL(domain, h, reg);
    end
    
end