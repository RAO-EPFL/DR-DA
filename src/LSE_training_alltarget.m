function loss = LSE_training_alltarget(domain, sequential_data, M, reg)

    x_target_set = domain.target.x;
    y_target_set = domain.target.y;
    seq_data_x = sequential_data.x;
    seq_data_y = sequential_data.y;
    lse_experts_all_target = zeros(domain.dimension, M+1);
    loss_lse_experts_all_target = zeros(M + 1);
    lse_experts_all_target(:, 1) =  domain.target.lse_beta;
    for t = 1 : M + 1
        x_t = seq_data_x(t, :);
        y_t = seq_data_y(t);
        loss_lse_experts_all(t) = lse_loss(x_t, y_t, lse_experts_all_target(:, t));
        x_target_set = [x_target_set; x_t];
        y_target_set = [y_target_set; y_t];
        lse_experts_all_target(:, t+1) = LSE(x_target_set, y_target_set, reg);
    end
%     figure
%     plot(cumsum(loss_lse_experts_all))
    loss = loss_lse_experts_all;
end