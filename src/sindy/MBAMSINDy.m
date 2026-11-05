%% MBAM + SINDy Pipeline for SMIB System
% This script runs the single-generator (SMIB) baseline simulation,
% builds the candidate function library, and applies sparse regression (SINDy).

clc; clear; close all;

%% 1. Baseline Simulation Setup & Execution
p = sg4_parameters();
tspan = [0 1];

% Define scalar baseline inputs for the ODE wrapper
u.V = 1.0; 
u.theta = 0.0;
u.Pm = 0.206517;
u.vf = 1.011699;

% Initial conditions at equilibrium
x0 = [0.1; 1.0; 1.0; 0.0459]; 

% Run solver using a robust wrapper function
[t_all, x_all] = ode15s(@(t,x) sg4_rhs_wrapper(t, x, p, u), tspan, x0);

%% 2. Compute Derivatives & Outputs
m = length(t_all);
X = x_all;
X_dot = zeros(m, 4);
for i = 1:m
    X_dot(i, :) = sg4_rhs_wrapper(t_all(i), X(i,:)', p, u)';
end

%% 3. Construct Candidate Function Library (Theta)
% Polynomial library up to quadratic order for states [delta, omega, eqp, edp]
Theta = [ones(m,1), X, X(:,1).^2, X(:,2).^2, X(:,3).^2, X(:,4).^2, ...
         X(:,1).*X(:,2), X(:,1).*X(:,3), X(:,1).*X(:,4)];

%% 4. Sparse Identification (SINDy via Sequential Thresholded Least Squares)
lambda = 0.05; % Sparsity threshold
Xi = pinv(Theta) * X_dot;

for k = 1:10
    small_inds = (abs(Xi) < lambda);
    Xi(small_inds) = 0;
    for ind = 1:4
        big_inds = ~small_inds(:, ind);
        if any(big_inds)
            Xi(big_inds, ind) = pinv(Theta(:, big_inds)) * X_dot(:, ind);
        end
    end
end

disp('Identified Sparse Coefficient Matrix (Xi):');
disp(Xi);

%% Local Wrapper Function to handle scalar time-dependent inputs safely
function dxdt = sg4_rhs_wrapper(t, x, p, u)
    u_local = u;
    % Apply the voltage disturbance window: 0.8 between t = 0.3 and 0.5, else 1.0
    if t >= 0.3 && t <= 0.5
        u_local.V = 0.8;
    else
        u_local.V = 1.0;
    end
    
    % If sg4_rhs expects u to have field 't', provide a dummy vector to prevent interp1 crashes
    u_local.t = [0, 1];
    u_local.theta_vec = [0, 0]; 
    
    % Call original model dynamics
    dxdt = sg4_rhs(t, x, p, u_local);
end