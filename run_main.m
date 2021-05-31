%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sequential Domain Adaptation by Synthesizing Distributionally Robust
% Experts
% ICML 2021 Submission
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script runs the main.m script for different datasets.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear 
close all
clc
data_sets = ["birth_USA", "uber", "houses", "california_housing", "life_expectancy"];
addpath('src')
addpath('data')

for ds_i = 1 : length(data_sets)
    rng('default');
    data_set = data_sets(ds_i);
    domain_org = read_data(data_set); % Create the original domain
    replications = 100; % Number of replications for the experiment
    reg = 1e-6; % Regularization of the ridge regression problem
    N_target = domain_org.dimension; % Number of target samples
    M = min(1000, size(domain_org.target.x,   1) - N_target - 1); % Time horizon
    K = 10; % number of experts 
    eta = 0.05; % Learning rate of BOA algorithm
    domain_org.N_target = N_target;
    num_emp_method = 5; 
    run main.m
    save(data_set + ".mat")
    clearvars -except data_sets ds_i
end 