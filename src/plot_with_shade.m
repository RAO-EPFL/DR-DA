function p = plot_with_shade(x, Y, color, lt)
    Y = mean(Y, 2);
    p = plot(x, Y(x) , lt ,'linewidth',4, 'color', color,...
    'MarkerSize', 5);
%     hold on;
%     x2 = [x(:)', flip(x(:)')];
%     fill(x2, [prctile(Y, 75, 2)', flip(prctile(Y, 100-75, 2))'], color, 'LineStyle', 'none');
%     alpha(0.2)
end