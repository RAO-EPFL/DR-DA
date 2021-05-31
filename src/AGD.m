function [beta_hat, val_f_D] = AGD(mu_hat, Sigma_hat, rho, K)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sequential Domain Adaptation by Synthesizing Distributionally Robust
% Experts
% ICML 2021 Submission
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solving the KL regression with 1 ball using Adaptive Gradient Descent
% INPUT:
% mu_hat, Sigma_hat: center of the ball (mean, covariance)
% rho: radius of the ball
% K: number of iterations
% OUTPUT:
% beta_hat: the regressor
% val_f_D: objective value over iterations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    p = length(mu_hat);
    d = p - 1;

    
    val_f_D = zeros(K,1);
    beta_last = (Sigma_hat(1:d,1:d) + mu_hat(1:d)*mu_hat(1:d)')\(Sigma_hat(1:d,p) + mu_hat(1:d)*mu_hat(p));
    tau = Inf; 
    alpha = 1;
    grad_f_D_last = Grad_f_D(beta_last, rho, mu_hat, Sigma_hat);
    original_val = Val_f_D(beta_last, rho, mu_hat, Sigma_hat); % for debugging
    beta = beta_last - 1e-5*grad_f_D_last/norm(grad_f_D_last);
    k = 1;

    % Main loop of Adaptive Gradient Descent
    while (k<=K)
        alpha_last = alpha;
        grad_f_D = Grad_f_D(beta, rho, mu_hat, Sigma_hat);
        alpha = min(sqrt(1+tau)*alpha, norm(beta - beta_last, 2)/(2*norm(grad_f_D - grad_f_D_last, 2)) );
        %fprintf('alpha = %f,  alpha_last = %f \n', alpha,alpha_last);
        beta_last = beta;
        grad_f_D_last = grad_f_D;
        beta = beta - alpha*grad_f_D;
        tau = alpha/alpha_last;
        if (k == 1)
            alpha_hat = alpha*(1 + tau);
            beta_hat = beta;
            sum_weight = alpha_hat;
        else
            alpha_hat = alpha;
            beta_hat = (beta_hat + (alpha_hat*beta)/sum_weight)/(1+ alpha_hat/sum_weight);
            sum_weight = sum_weight + alpha_hat;
        end
        val_f_D(k) = Val_f_D(beta_hat, rho, mu_hat, Sigma_hat);
        k = k+1;
    end
    
    temp = norm(grad_f_D);
    fprintf('For rho = %f, %d iterations finished and norm(grad) = %s.\n',rho,K,temp);
    if temp > 1e-3
        disp('Gradient norm too large!');
        keyboard
    end
end


function val = Val_f_D(beta, rho, mu_hat, Sigma_hat)

% % Compute g(gamma) by gamma_stat
% [gamma_star, omega1, omega2] = Gamma(beta, rho, mu_hat, Sigma_hat);
% val = gamma_star*rho + (gamma_star*omega2)/(gamma_star - omega1) - gamma_star*log(1 - (omega1/gamma_star));

% Compute g(gamma) by omega_star
[omega_star, omega1, omega2] = Omega(beta, rho, mu_hat, Sigma_hat);
val = (rho*omega1)/omega_star + omega2/(1-omega_star) - (omega1/omega_star)*log(1 - omega_star);
end

function grad = Grad_f_D(beta, rho, mu_hat, Sigma_hat)

d = length(beta);
Cov_xx_hat = Sigma_hat(1:d, 1:d);
Cov_xy_hat = Sigma_hat(1:d, d+1);
mu_x_hat = mu_hat(1:d);
mu_y_hat = mu_hat(d+1);

% % gradient by gamma
%[gamma_star, omega1, omega2] = Gamma(beta, rho, mu_hat, Sigma_hat);
%grad = (2*omega2/(gamma_star - 2*omega1 + (omega1/gamma_star)))*(Cov_xx_hat*beta - Cov_xy_hat) ...
%         + (2/(1 - (omega1/gamma_star)))*(mu_x_hat*mu_x_hat'*beta - mu_x_hat*mu_y_hat + Cov_xx_hat*beta - Cov_xy_hat);

% gradient by omega
[omega_star, omega1, omega2] = Omega(beta, rho, mu_hat, Sigma_hat);
% fprintf('1-omega_star = %e\n', 1-omega_star);
grad = (2*omega2/omega1)*(omega_star/(1-omega_star))*((Cov_xx_hat*beta - Cov_xy_hat)) ...
        + (2*omega_star/(1-omega_star))*(mu_x_hat*mu_x_hat'*beta - mu_x_hat*mu_y_hat + Cov_xx_hat*beta - Cov_xy_hat);

end
    
function [omega_star, omega1, omega2] = Omega(beta, rho, mu_hat, Sigma_hat)

w = [beta; -1];
omega1 = w'*Sigma_hat*w;
omega2 = (mu_hat'*w)^2;

func = @(omega) (omega2/omega1)*(omega^2/((1 - omega)^2)) + (omega/(1 - omega)) + log(1 - omega) - rho;
% omega0 = [2*rho/(1+2*rho+sqrt(1+4*rho*omega2)), sqrt(rho*omega1)/(sqrt(omega2) + sqrt(rho*omega1))];
range = [2*rho/(1+2*rho+sqrt(1+4*rho*omega2)), 1]; %using only lower bound as initial point
% fprintf('omega1 = %f; omega2 = %f; omega0 = [%f , %f]\n', omega1, omega2, 2*rho/(1+2*rho+sqrt(1+4*rho*omega2)), sqrt(rho*omega1)/(sqrt(omega2) + sqrt(rho*omega1)));

omega_star = wise_bisect( func, range, 10^(-10) );

% omega_star = fzero(func,omega0);
%fprintf('omega_star = %e, nonlinear equation residual = %e \n',omega_star, feval(func,omega_star));
end

function [gamma_star, omega1, omega2] = Gamma(beta, rho, mu_hat, Sigma_hat)

w = [beta; -1];
omega1 = w'*Sigma_hat*w;
omega2 = (mu_hat'*w)^2;

func = @(x) (omega1*omega2)/((x - omega1)^2) + (omega1)/(x - omega1) + log(1 - (omega1/x)) - rho;
range = [omega1*((sqrt(rho*omega1) + sqrt(omega2))/(sqrt(rho*omega1))), omega1/(2*rho)*(1 + 2*rho + sqrt(1 + 4*rho*omega2))];
gamma_star = wise_bisect(func, range, 1e-10);
end

function [ mid ] = wise_bisect( func_name, range, tol )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Wasserstein Inverse covariance Shrinkage Estimator
% Viet Anh NGUYEN, Daniel KUHN, Peyman MOHAJERIN ESFAHANI
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Bisection function (used to find gamma)
%
% Input:
% func_name: the name of function to bisect
% range: a vector containing the initial lower and upper bound for bisection
% tol: tolerance 

    done = 0;
    lower = range(1);
    upper = range(2);
    mid = (lower+upper)/2;
    while done < 1
        
        if func_name(mid) > 0
            upper = mid;
        else
            lower = mid;
        end
        mid = (lower+upper)/2;
        
        
        if upper - lower < tol
            done = 2;
        end
    end

end




% for debugging
%(inv(Cov_xx_hat) - inv(Cov_xx_hat)*mu_x_hat*mu_x_hat'*inv(Cov_xx_hat)/(1 + mu_x_hat'*inv(Cov_xx_hat)*mu_x_hat))*(Cov_xy_hat + mu_x_hat*mu_y_hat)
%inv(Cov_xx_hat + mu_x_hat*mu_x_hat')