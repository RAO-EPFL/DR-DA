%%

clc
clear all
data_set = "life_expectancy";
load([data_set + ".mat"])
close all
fig = figure;
% set(fig, 'Units', 'normalized', 'Position', [0.35, 0.25, 0.4, 0.55])
hold on
ind = [1 : M+1];
% ind = (1:9)' * logspace(0, log10(M)-1, log10(M)); 
% ind = [ind(:)', M];


% p_dro2 = plot_with_shade(ind, cumloss_track_dro2', 5, 0.1, 'y');
lgnd_string = [];
num_emp_experts = 5;
p = [];
lw = 3;

p = [p, plot_with_shade(ind, cumloss_track_emp(:, :, 1)', ...
        [201/255, 165/255, 236/255], '-')];
lgnd_string = [lgnd_string, "CC-L"]; 
p = [p, plot_with_shade(ind, cumloss_track_emp(:, :, 2)', ...
         [151/255, 87/255, 223/255], '-')];
   lgnd_string = [lgnd_string, "CC-SL"]; 
   
p = [p, plot_with_shade(ind, cumloss_track_emp(:, :, 3)', ...
         [97/255, 28/255, 166/255], '-')];
   lgnd_string = [lgnd_string, "CC-TL"]; 
   
   
p = [p, plot_with_shade(ind, cumloss_track_emp(:, :, 4)', ...
         [0.7 - 0.7 / num_emp_experts * 4 , 0.1 + ...
       0.5 / num_emp_experts * 4, 0.1 + 0.8 / num_emp_experts * 4], '-')];
   lgnd_string = [lgnd_string, "CC-TE"]; 
   
p = [p, plot_with_shade(ind, cumloss_track_emp(:, :, 5)', ...
         [0, 0, 1], '-' )];
lgnd_string = [lgnd_string, "CC-SE"]; 

p = [p, plot_with_shade(ind, cumloss_track_lse_alltarget', ...
    'r', '--' )];
lgnd_string = [lgnd_string, "LSE-T"];

p = [p, plot_with_shade(ind, cumloss_track_lse_alltarget_source', ...
    [12/255, 90/255, 25/255], '--' )];
lgnd_string = [lgnd_string, "LSE-T{$\&$}S"];

p = [p, plot_with_shade(ind, cumloss_track_DITL', ...
    [100/255, 1, 160/255], '-' )];
lgnd_string = [lgnd_string, "RWS"];


p = [p, plot_with_shade(ind, cumloss_track_IRKL', ...
    [1, 128/255, 0], '-' )];
lgnd_string = [lgnd_string, "IR-KL"];

plot_with_shade(ind, cumloss_track_IRWASS', 'k', '-');
hold on;
[xmarkers,ymarkers, p] = evenMarkers(ind, mean(cumloss_track_IRWASS', 2), 20, p);
lgnd_string = [lgnd_string, "IR-WASS"];

hold on;


p = [p, plot_with_shade(ind, ...
   cumloss_track_SIKL(:, :, 1)',   ...
   [0.6, 0.8, 0.5], '-' )];
lgnd_string = [lgnd_string,  "SI-KL"];


p = [p, plot_with_shade(ind, ...
   cumloss_track_SIWASS(:, :, 1)',   ...
   [239/255, 88/255, 234/255], '-')];
lgnd_string = [lgnd_string, "SI-WASS"];

set(gca, 'XScale', 'log', 'YScale', 'log');
set(gca, 'FontSize', 16);
xlabel('Time horizon', 'FontSize', 20, 'interpreter','latex');
ylabel('Cumulative loss','FontSize', 20, 'interpreter','latex')
% title(data_set, 'FontSize', 20, 'interpreter','latex')
grid on
% xlim([1, 500])
ylim([0, 1e5])
lgd = legend(p, lgnd_string, 'Location', 'bestoutside','interpreter','latex');
lgd.FontSize = 16;
remove_border()
saveas(gcf, [data_set + "cumloss"], 'svg')
% saveas(gcf, 'convergence', 'png')
