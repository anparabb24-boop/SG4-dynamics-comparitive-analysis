clear; clc; close all;

addpath(genpath(pwd));

%% Load baseline data

load('results/sg4_disturbance_baseline.mat');

t = t_all;
N = length(t);

%% Reconstruct inputs

V = ones(N,1);
V(t >= 0.3 & t <= 0.5) = 0.8;

theta = zeros(N,1);
Pm = Pm0 * ones(N,1);
vf = vf0 * ones(N,1);

u.t = t;
u.V = V;
u.theta = theta;
u.Pm = Pm;
u.vf = vf;

%% SG3 reference simulation

x03 = [x0(1); x0(2); x0(4)];

options = odeset('RelTol',1e-8,'AbsTol',1e-10);

[t3,x3] = ode15s( ...
    @(time,state) sg3_reduced_rhs(time,state,p,u), ...
    t,x03,options);

delta3 = x3(:,1);
omega3 = x3(:,2);
edp3 = x3(:,3);

eqp3 = (p.xd_prime/p.xd).*vf + ...
    ((p.xd-p.xd_prime)/p.xd).*V.* ...
    cos(delta3-theta);

%% Test progressively smaller Td0 values

Td_values = [6.5 1 0.5 0.1 0.05 0.01];

errors = zeros(length(Td_values),4);

for k = 1:length(Td_values)

    p_test = p;
    p_test.Td0_prime = Td_values(k);

    [~,x_test] = ode15s( ...
        @(time,state) sg4_rhs(time,state,p_test,u), ...
        t,x0,options);

    errors(k,1) = norm(x_test(:,1)-delta3)/norm(delta3);
    errors(k,2) = norm(x_test(:,2)-omega3)/norm(omega3);
    errors(k,3) = norm(x_test(:,4)-edp3)/norm(edp3);
    errors(k,4) = norm(x_test(:,3)-eqp3)/norm(eqp3);

    fprintf('Td0 = %.4f | delta = %.6f | omega = %.6f | edp = %.6f | eqp = %.6f\n', ...
        Td_values(k),errors(k,1),errors(k,2),errors(k,3),errors(k,4));

end

%% Plot convergence

figure;

semilogx(Td_values,errors(:,1),'-o','LineWidth',1.5);
hold on;
semilogx(Td_values,errors(:,2),'-s','LineWidth',1.5);
semilogx(Td_values,errors(:,3),'-^','LineWidth',1.5);
semilogx(Td_values,errors(:,4),'-d','LineWidth',1.5);

set(gca,'XDir','reverse');

xlabel('T''_{d0} (seconds)');
ylabel('Relative RMS error');
title('SG4 convergence toward SG3 boundary model');

legend('\delta','\omega','e''_d','e''_q','Location','best');
grid on;