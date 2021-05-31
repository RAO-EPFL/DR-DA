%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sequential Domain Adaptation by Synthesizing Distributionally Robust
% Experts
% ICML 2021 Submission
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script creates the cumulative loss of the CC-L, CC-TL, CC-SL, CC-TE,
% CC-SE, RWS, IR-KL, IR-WASS, SI-KL, SI-WASS experts
% The experiment is replicated for "replications" times and the cumulative
% loss is averaged 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialize the cumulative loss matrices
cumloss_track_SIKL = zeros(replications, M+1);
cumloss_track_SIWASS = zeros(replications, M+1);
cumloss_track_IRWASS = zeros(replications, M+1);
cumloss_track_IRKL = zeros(replications, M+1);
cumloss_track_DITL = zeros(replications, M+1);
cumloss_track_emp = zeros(replications, M+1, 5); % CC-L, CC-TL, CC-SL, CC-TE, CC-SE
cumloss_track_lse_alltarget = zeros(replications, M+1);
cumloss_track_lse_alltarget_source = zeros(replications, M+1);
%% Specify Source and Target
parfor rep = 1 : replications
    rng(rep)
    fprintf('Replication ==> ' + string(rep) + '\n')
    domain = create_dataset(domain_org, reg);
    sequential_data = {};
    sequential_data.x = domain.target.test.x;
    sequential_data.y = domain.target.test.y;
    %% Empirical LSE on Source and Target data
    %%% LSE trained on the source data
    domain.source.lse_beta = LSE(domain.source.x, domain.source.y, reg);
    lse_error_source = lse_loss(domain.source.x, domain.source.y, domain.source.lse_beta);
%     fprintf('Error of LSE trained on Source data = %d \n', lse_error_source); 
    %%% LSE trained on the target data
    domain.target.lse_beta = LSE(domain.target.x, domain.target.y, reg);
    lse_error_target = lse_loss(domain.target.x, domain.target.y, domain.target.lse_beta);
%     fprintf('Error of LSE trained on Target data = %d \n', lse_error_target); 
    %% Create Experts 
    %%%%%%%%%%%%%% Create Experts
    experts_SIKL = create_SIKL_experts(domain, K);
    experts_SIWASS = create_SIWASS_experts(domain, K);
    experts_IRWASS = create_IRWASS_experts(domain, K);
    experts_IRKL = create_IRKL_experts(domain, K);
    experts_DITL = create_DITL_experts(domain, K, reg);

    if sum(sum(isnan(experts_SIKL))) | sum(sum(isnan(experts_IRKL)))
        fprintf('Problem with IRKL or SIKL')
        keyboard
    end

    emp_experts = create_emp_experts(domain.source.lse_beta, domain.target.lse_beta, K);
    [~, ~, num_emp_experts] = size(emp_experts);
    %% Choose the best expert combination
    expert_dist_EMP = zeros(K, M+1, num_emp_experts);
    temp_isnan  = 0;
    expert_dist_SIKL = BOA(sequential_data, experts_SIKL', M, eta);
    temp_isnan = temp_isnan | sum(sum(isnan(expert_dist_SIKL)));
    expert_dist_SIWASS = BOA(sequential_data, experts_SIWASS', M, eta);
    temp_isnan = temp_isnan | sum(sum(isnan(expert_dist_SIWASS)));

    for i = 1 : num_emp_experts
        expert_dist_EMP(:, : , i) = BOA(sequential_data, emp_experts(:, :, i)', M, eta);
        temp_isnan = temp_isnan | sum(sum(isnan(expert_dist_EMP(:, :, i))));
    end
    expert_dist_DITL = BOA(sequential_data, experts_DITL', M, eta);
    expert_dist_DRO_wass = BOA(sequential_data, experts_IRWASS', M, eta);

    expert_dist_DRO_kl = BOA(sequential_data, experts_IRKL', M, eta);
    temp_isnan = temp_isnan | sum(sum(isnan(expert_dist_DRO_kl))) | ...
        sum(sum(isnan(expert_dist_DRO_wass))) | sum(sum(isnan(expert_dist_DITL))) ;
    cnt = 1;
    %%% Check if the step size of BOA algorithm is large that for some
    %%% experts resulting in NaN values for the distribution over experts
    %%% If this is the case we decrease the step size at each iteration
    %%% until all experts can be aggregated in a meaningful way
    while temp_isnan
        fprintf('Reduce the learning rate...\n')
        cnt = cnt + 1;
        temp_isnan  = 0;
        expert_dist_SIKL = BOA(sequential_data, experts_SIKL', M, eta / cnt);
        temp_isnan = temp_isnan | sum(sum(isnan(expert_dist_SIKL)));
        expert_dist_SIWASS = BOA(sequential_data, experts_SIWASS', M, eta / cnt);
        temp_isnan = temp_isnan | sum(sum(isnan(expert_dist_SIWASS)));
        for i = 1 : num_emp_experts
            expert_dist_EMP(:, : , i) = BOA(sequential_data, emp_experts(:, :, i)', M, eta / cnt);
            temp_isnan = temp_isnan | sum(sum(isnan(expert_dist_EMP(:, :, i))));
        end
        expert_dist_DRO_wass = BOA(sequential_data, experts_IRWASS', M, eta / cnt);
        expert_dist_DRO_kl = BOA(sequential_data, experts_IRKL', M, eta / cnt);
        expert_dist_DITL = BOA(sequential_data, experts_DITL', M, eta / cnt);

        temp_isnan = temp_isnan | sum(sum(isnan(expert_dist_DRO_kl)));
        temp_isnan = temp_isnan | sum(sum(isnan(expert_dist_DRO_wass)));
        temp_isnan = temp_isnan | sum(sum(isnan(expert_dist_DITL)));

    end

    loss_track_SIKL = zeros(M+1, 1);
    loss_track_SIWASS = zeros(M+1, 1);
    loss_track_IRWASS = zeros(M+1, 1);
    loss_track_IRKL = zeros(M+1, 1);
    loss_track_DITL = zeros(M+1, 1);
    loss_track_emp = zeros(M+1, num_emp_experts);

    for t = 1 : M + 1
        x_t = sequential_data.x(t, :);
        y_t = sequential_data.y(t);
        for i = 1 : num_emp_experts
            loss_track_emp(t, i) = lse_loss(x_t, y_t, emp_experts(:, :, i) * expert_dist_EMP(:, t, i));
        end
        loss_track_SIWASS(t) = lse_loss(x_t, y_t, experts_SIWASS(:, :) * expert_dist_SIWASS(:, t));
        loss_track_SIKL(t) = lse_loss(x_t, y_t, experts_SIKL * expert_dist_SIKL(:, t));
        loss_track_IRWASS(t) = lse_loss(x_t, y_t, experts_IRWASS * expert_dist_DRO_wass(:, t));
        loss_track_IRKL(t) = lse_loss(x_t, y_t, experts_IRKL * expert_dist_DRO_kl(:, t));
        loss_track_DITL(t) = lse_loss(x_t, y_t, experts_DITL * expert_dist_DITL(:, t));

    end
    
    %% LSE training on the aggregated training set
    lse_all_target_loss = LSE_training_alltarget(domain, sequential_data, M, reg);
    lse_all_target_source_loss = LSE_training_alltarget_source(domain, sequential_data, M, reg);

    %%
    cumloss_track_lse_alltarget(rep, :) = cumsum(lse_all_target_loss);
    cumloss_track_lse_alltarget_source(rep, :) = cumsum(lse_all_target_source_loss);

    cumloss_track_SIKL(rep, :) = cumsum(loss_track_SIKL);
    cumloss_track_SIWASS(rep, :) = cumsum(loss_track_SIWASS);
    cumloss_track_IRWASS(rep, :) = cumsum(loss_track_IRWASS);
    cumloss_track_IRKL(rep, :) = cumsum(loss_track_IRKL);
    cumloss_track_emp(rep, :, :) = cumsum(loss_track_emp, 1);
    cumloss_track_DITL(rep, :) = cumsum(loss_track_DITL);

end


% %% VISUALIZE
% close all
% fig = figure;
% % set(fig, 'Units', 'normalized', 'Position', [0.35, 0.25, 0.4, 0.55])
% hold on
% ind = [1 : M+1];
% 
% % p_dro2 = plot_with_shade(ind, cumloss_track_dro2', 5, 0.1, 'y');
% lgnd_string = [];
% num_emp_experts = 5;
% p = [];
% lw = 3;
% 
% for i = 1 : num_emp_experts
%    p(i) = plot_with_shade(ind, cumloss_track_emp(:, :, i)', ...
%        0.1, 0.1, [0.7 - 0.7 / num_emp_experts * i , 0.1 + ...
%        0.5 / num_emp_experts * i, 0.1 + 0.8 / num_emp_experts * i], lw);
%    lgnd_string = [lgnd_string, "EMP-" + string(i)]; 
% end
% 
% 
% for i = 1 : length(additional_radius_kl)
%    p(i + num_emp_experts) = plot_with_shade(ind, ...
%        cumloss_track_SIKL(:, :, i)', 0.1, 0.1, ...
%        [0.1  + 0.5 / length(additional_radius_kl) * (i-1), ...
%        1 - 0.5 / length(additional_radius_kl) * i, 0.1], lw);
%    lgnd_string = [lgnd_string,  "SI-KL"];
% end
% 
% for i = 1 : length(additional_radius_wass)
%    p(i + num_emp_experts + length(additional_radius_kl)) = plot_with_shade(ind, ...
%        cumloss_track_SIWASS(:, :, i)', 0.1, 0.1, ...
%        [1, 0.5 + 0.3 / length(additional_radius_wass) * i, ...
%        0.3 - 0.2 / length(additional_radius_wass) * i], lw);
%    lgnd_string = [lgnd_string, "SI-WASS"];
% end
% p_dro_kl = plot_with_shade(ind, cumloss_track_IRKL', 0.1, 0.1, [0.9, 0.6, 0.1], lw);
% 
% p_dro_wass = plot_with_shade(ind, cumloss_track_IRWASS', 0.1, 0.1, 'k', lw);
% p_emp_alltarget = plot_with_shade(ind, cumloss_track_lse_alltarget', 0.1, 0.1, 'r', lw);
% p_emp_alltarget_source = plot_with_shade(ind, cumloss_track_lse_alltarget_source', 0.1, 0.1, 'c', lw);
% 
% p_ditl = plot_with_shade(ind, cumloss_track_DITL', 0.1, 0.1, 'g', lw);
% 
% set(gca, 'XScale', 'linear', 'YScale', 'linear');
% set(gca, 'FontSize', 16);
% xlabel('Time horizon', 'FontSize', 20, 'interpreter','latex');
% ylabel('Cumulative loss','FontSize', 20, 'interpreter','latex')
% title(data_set, 'FontSize', 20, 'interpreter','latex')
% grid on
% xlim([1, M])
% %ylim([0, 200])
% lgd = legend([p,p_dro_kl,  p_dro_wass, p_ditl, p_emp_alltarget, p_emp_alltarget_source], [lgnd_string, "IR-KL", "IR-WASS", "DITL" ,"OPT-LSE-Target", "OPT-LSE-Target-Source"], 'Location', 'best', 'interpreter','latex');
% lgd.FontSize = 16;
% remove_border()
% saveas(gcf, [data_set + "cumloss"], 'svg')
% % saveas(gcf, 'convergence', 'png')
