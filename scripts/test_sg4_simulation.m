
clc;
clear;
close all;

%% Load generator parameters

p = sg4_parameters();

%% Initial equilibrium conditions

delta0 = 0.1;
omega0 = 1.0;
eqp0 = 1.0;

V0 = 1.0;
theta0 = 0.0;

% Calculate equilibrium currents
iq0 = V0*sin(delta0-theta0)/p.xq;
edp0 = (p.xq-p.xq_prime)*iq0;
id0 = (eqp0-V0*cos(delta0-theta0))/p.xd_prime;

% Initial state
x0 = [delta0; omega0; eqp0; edp0];

% Calculate equilibrium power and field voltage
y0 = sg4_outputs(x0,V0,theta0,p);

Pm0 = y0.Pg;
vf0 = eqp0 + (p.xd-p.xd_prime)*id0;

%% Define 10-second simulation intervals

% Original voltage disturbance retained:
% 0.0-0.3 s: Normal voltage
% 0.3-0.5 s: Voltage drops to 0.8 pu
% 0.5-10.0 s: Voltage returns to 1.0 pu

intervals = [
    0.0   0.3   1.0;
    0.3   0.5   0.8;
    0.5  10.0   1.0
];

% Uniform sampling interval
dt = 0.01;

%% Storage for simulation results

t_all = [];
x_all = [];

x_current = x0;

%% Simulate each interval

for k = 1:size(intervals,1)

    t_start = intervals(k,1);
    t_end = intervals(k,2);
    V_current = intervals(k,3);

    % Constant inputs during this interval
    u.t = [t_start t_end];
    u.V = [V_current V_current];
    u.theta = [theta0 theta0];
    u.Pm = [Pm0 Pm0];
    u.vf = [vf0 vf0];

    % Generate time points at 0.01-second intervals
    nPoints = round((t_end-t_start)/dt) + 1;
    t_segment = linspace(t_start,t_end,nPoints)';

    % Integrate differential equations
    [t_segment,x_segment] = ode15s( ...
        @(t,x) sg4_rhs(t,x,p,u), ...
        t_segment,x_current);

    % Store results without duplicating boundary points
    if k == 1
        t_all = t_segment;
        x_all = x_segment;
    else
        t_all = [t_all; t_segment(2:end)];
        x_all = [x_all; x_segment(2:end,:)];
    end

    % Carry final state into next interval
    x_current = x_segment(end,:)';

end

disp('10-second disturbance simulation completed.');

%% Calculate generator electrical outputs and currents

N = length(t_all);

Pg_all = zeros(N,1);
Qg_all = zeros(N,1);
id_all = zeros(N,1);
iq_all = zeros(N,1);

for k = 1:N

    if t_all(k) < 0.3
        V_current = 1.0;
    elseif t_all(k) < 0.5
        V_current = 0.8;
    else
        V_current = 1.0;
    end

    x_current = x_all(k,:)';

    y = sg4_outputs(x_current,V_current,theta0,p);

    Pg_all(k) = y.Pg;
    Qg_all(k) = y.Qg;
    id_all(k) = y.id;
    iq_all(k) = y.iq;

end

%% Rotor angle and speed unit conversion

% Rotor angle: radians to degrees
delta_rad = x_all(:,1);
delta_deg = rad2deg(delta_rad);

% Rotor speed: per unit to actual mechanical/electrical angular speed
omega_pu = x_all(:,2);
omega_actual = omega_pu * p.Omega_b;

% Rotor speed deviation from synchronous speed
omega_deviation_pu = omega_pu - p.omega0;
omega_deviation_actual = omega_deviation_pu * p.Omega_b;

% Display initial and final values
fprintf('\nROTOR ANGLE AND SPEED CONVERSION\n');
fprintf('--------------------------------\n');

fprintf('Initial rotor angle: %.6f rad = %.6f deg\n', ...
    delta_rad(1),delta_deg(1));

fprintf('Final rotor angle: %.6f rad = %.6f deg\n', ...
    delta_rad(end),delta_deg(end));

fprintf('Initial rotor speed: %.6f pu = %.6f rad/s\n', ...
    omega_pu(1),omega_actual(1));

fprintf('Final rotor speed: %.6f pu = %.6f rad/s\n', ...
    omega_pu(end),omega_actual(end));

fprintf('Synchronous angular speed: %.6f rad/s\n', ...
    p.Omega_b);

%% Plot generator currents

figure;
subplot(2,1,1);
plot(t_all,id_all,"LineWidth",1.5);
ylabel('i_d (p.u.)');
xlabel('Time (s)');
title('Direct-Axis Current');
xline(0.3,'--r','LineWidth',1.5);
xline(0.5,'--r','LineWidth',1.5);
grid on;

subplot(2,1,2);
plot(t_all,iq_all,"LineWidth",1.5);
ylabel('i_q (p.u.)');
xlabel('Time (s)');
title('Quadrature-Axis Current');
xline(0.3,'--r','LineWidth',1.5);
xline(0.5,'--r','LineWidth',1.5);
grid on;

%% Plot active and reactive power

figure;

subplot(2,1,1);
plot(t_all,Pg_all,'LineWidth',1.5);
ylabel('P_g (p.u.)');
xlabel('Time (s)');
title('Generator Active Power');
xline(0.3,'--r','LineWidth',1.5);
xline(0.5,'--r','LineWidth',1.5);
grid on;

subplot(2,1,2);
plot(t_all,Qg_all,'LineWidth',1.5);
ylabel('Q_g (p.u.)');
xlabel('Time (s)');
title('Generator Reactive Power');
xline(0.3,'--r','LineWidth',1.5);
xline(0.5,'--r','LineWidth',1.5);
grid on;

%% Plot generator states

figure;

stateNames = {'\delta','\omega','E_q''','E_d'''};

for k = 1:4
    subplot(4,1,k);
    plot(t_all,x_all(:,k),'LineWidth',1.5);
    ylabel(stateNames{k});
    xline(0.3,'--r','LineWidth',1.5);
    xline(0.5,'--r','LineWidth',1.5);
    grid on;
end

xlabel('Time (s)');
sgtitle('SG4 State Trajectories - 10 Seconds');

%% Prepare input trajectories

V_all = ones(N,1);
V_all(t_all >= 0.3 & t_all < 0.5) = 0.8;

theta_all = zeros(N,1);
Pm_all = Pm0*ones(N,1);
vf_all = vf0*ones(N,1);

%% Save 10-second baseline separately

if ~exist('results','dir')
    mkdir('results');
end

save('results/sg4_disturbance_10s.mat', ...
    't_all','x_all', ...
    'id_all','iq_all', ...
    'Pg_all','Qg_all', ...
    'p','x0','Pm0','vf0', ...
    'theta_all','V_all','Pm_all','vf_all');

disp('10-second baseline results saved.');
disp('File: results/sg4_disturbance_10s.mat');
disp(['Total samples: ',num2str(N)]);
