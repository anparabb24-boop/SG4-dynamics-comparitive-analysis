% RUN_MBAM_GEODESIC (Reduced 2-Parameter Version: D and Td0_prime)
clear; clc;

% Load baseline parameters and initial conditions
p_nominal = sg4_parameters();
load('results/sg4_disturbance_baseline.mat', 't_all', 'x0', 'Pm0', 'vf0');

% Reconstruct voltage disturbance matching baseline simulation
V_all = ones(size(t_all));
V_all(t_all >= 0.3 & t_all <= 0.5) = 0.8;
theta_all = zeros(size(t_all));

u_data.t = t_all;
u_data.V = V_all;
u_data.theta = theta_all;
u_data.Pm = Pm0 * ones(size(t_all));
u_data.vf = vf0 * ones(size(t_all));
u_data.x0 = x0;

t_span = [t_all(1), t_all(end)];

% --- FOCUS ONLY ON D AND Td0_prime ---
theta_0 = [p_nominal.D; p_nominal.Td0_prime];
phi_0 = log(theta_0);

% Compute initial FIM for these 2 parameters and find sloppiest direction
[~, J_0, ~] = compute_geodesic_derivatives_2param(phi_0, p_nominal, u_data, t_span);
I_0 = J_0' * J_0;
[V_eig, D_eig] = eig(I_0);
[~, min_idx] = min(diag(D_eig));

v_0 = V_eig(:, min_idx);
v_0 = v_0 / norm(v_0); % Normalize initial velocity vector

y_0 = [phi_0; v_0];

% Integrate geodesic using ode15s
options = odeset('RelTol', 1e-4, 'AbsTol', 1e-5, 'Events', @geodesic_stop_event);
[tau_traj, y_traj] = ode15s(@(tau, y) geodesic_rhs_2param(tau, y, p_nominal, u_data, t_span), [0, 0.1], y_0, options);

% Extract parameter trajectories along arc-length tau
phi_traj = y_traj(:, 1:2);
theta_traj = exp(phi_traj);

% Plot results
figure;
plot(tau_traj, theta_traj, 'LineWidth', 1.5);
legend({'D', 'T_{d0}\prime'}, 'Location', 'best');
xlabel('Arc-length \tau');
ylabel('Parameter values \theta(\tau)');
title('MBAM Geodesic (D and T_{d0}\prime)');
grid on;

disp('2-Parameter Geodesic MBAM execution completed.');

function [value, isterminal, direction] = geodesic_stop_event(~, y)
    phi = y(1:length(y)/2);
    value = max(abs(phi)) - 15; 
    isterminal = 1;
    direction = 1;
end

% RUN_MBAM_GEODESIC (Reduced 2-Parameter Version: D and Td0_prime)
clear; clc;

% Load baseline parameters and initial conditions
p_nominal = sg4_parameters();
load('results/sg4_disturbance_baseline.mat', 't_all', 'x0', 'Pm0', 'vf0');

% Reconstruct voltage disturbance matching baseline simulation
V_all = ones(size(t_all));
V_all(t_all >= 0.3 & t_all <= 0.5) = 0.8;
theta_all = zeros(size(t_all));

u_data.t = t_all;
u_data.V = V_all;
u_data.theta = theta_all;
u_data.Pm = Pm0 * ones(size(t_all));
u_data.vf = vf0 * ones(size(t_all));
u_data.x0 = x0;

t_span = [t_all(1), t_all(end)];

% --- FOCUS ONLY ON D AND Td0_prime ---
theta_0 = [p_nominal.D; p_nominal.Td0_prime];
phi_0 = log(theta_0);

% Compute initial FIM for these 2 parameters and find sloppiest direction
[~, J_0, ~] = compute_geodesic_derivatives_2param(phi_0, p_nominal, u_data, t_span);
I_0 = J_0' * J_0;
[V_eig, D_eig] = eig(I_0);
[~, min_idx] = min(diag(D_eig));

v_0 = V_eig(:, min_idx);
v_0 = v_0 / norm(v_0); % Normalize initial velocity vector

y_0 = [phi_0; v_0];

% Integrate geodesic using ode15s
options = odeset('RelTol', 1e-4, 'AbsTol', 1e-5, 'Events', @geodesic_stop_event);
[tau_traj, y_traj] = ode15s(@(tau, y) geodesic_rhs_2param(tau, y, p_nominal, u_data, t_span), [0, 0.1], y_0, options);

% Extract parameter trajectories along arc-length tau
phi_traj = y_traj(:, 1:2);
theta_traj = exp(phi_traj);

% Plot results
figure;
plot(tau_traj, theta_traj, 'LineWidth', 1.5);
legend({'D', 'T_{d0}\prime'}, 'Location', 'best');
xlabel('Arc-length \tau');
ylabel('Parameter values \theta(\tau)');
title('MBAM Geodesic (D and T_{d0}\prime)');
grid on;

disp('2-Parameter Geodesic MBAM execution completed.');