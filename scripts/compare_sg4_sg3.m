clear; clc; close all;

addpath(genpath(pwd));

%% 1. Load SG4 baseline results

load('results/sg4_disturbance_baseline.mat');

t = t_all;
N = length(t);

% SG4 states:
% x_all(:,1) = delta
% x_all(:,2) = omega
% x_all(:,3) = eqp
% x_all(:,4) = edp

%% 2. Reconstruct the disturbance inputs

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

%% 3. Initialise the SG3 model

% SG3 states: delta, omega, edp

x03 = [
    x0(1)
    x0(2)
    x0(4)
];

% Check algebraic initial value of eqp

eqp0_sg3 = (p.xd_prime/p.xd)*vf0 + ...
    ((p.xd-p.xd_prime)/p.xd)*V(1)* ...
    cos(x03(1)-theta(1));

fprintf('SG4 initial eqp = %.8f\n', x0(3));
fprintf('SG3 initial eqp = %.8f\n', eqp0_sg3);

%% 4. Simulate SG3

options = odeset('RelTol',1e-8,'AbsTol',1e-10);

[t3,x3] = ode45( ...
    @(time,state) sg3_reduced_rhs(time,state,p,u), ...
    t, x03, options);

%% 5. Calculate algebraic eqp for SG3

delta3 = x3(:,1);
omega3 = x3(:,2);
edp3 = x3(:,3);

eqp3 = (p.xd_prime/p.xd).*vf + ...
    ((p.xd-p.xd_prime)/p.xd).*V.* ...
    cos(delta3-theta);

%% 6. Calculate SG3 electrical outputs

Pg3 = zeros(N,1);
Qg3 = zeros(N,1);

for k = 1:N

    x_temp = [
        delta3(k)
        omega3(k)
        eqp3(k)
        edp3(k)
    ];

    out = sg4_outputs(x_temp,V(k),theta(k),p);

    Pg3(k) = out.Pg;
    Qg3(k) = out.Qg;

end

%% 7. Compare SG4 and SG3

figure;

subplot(3,2,1);
plot(t,x_all(:,1),'b','LineWidth',1.4);
hold on;
plot(t3,delta3,'r--','LineWidth',1.4);
title('Rotor angle \delta');
legend('SG4','SG3');
grid on;

subplot(3,2,2);
plot(t,x_all(:,2),'b','LineWidth',1.4);
hold on;
plot(t3,omega3,'r--','LineWidth',1.4);
title('Rotor speed \omega');
legend('SG4','SG3');
grid on;

subplot(3,2,3);
plot(t,x_all(:,4),'b','LineWidth',1.4);
hold on;
plot(t3,edp3,'r--','LineWidth',1.4);
title('Transient voltage e''_d');
legend('SG4','SG3');
grid on;

subplot(3,2,4);
plot(t,x_all(:,3),'b','LineWidth',1.4);
hold on;
plot(t3,eqp3,'r--','LineWidth',1.4);
title('Transient voltage e''_q');
legend('SG4','SG3');
grid on;

subplot(3,2,5);
plot(t,Pg_all,'b','LineWidth',1.4);
hold on;
plot(t3,Pg3,'r--','LineWidth',1.4);
title('Generator active power P_g');
legend('SG4','SG3');
grid on;

subplot(3,2,6);
plot(t,Qg_all,'b','LineWidth',1.4);
hold on;
plot(t3,Qg3,'r--','LineWidth',1.4);
title('Generator reactive power Q_g');
legend('SG4','SG3');
grid on;

sgtitle('SG4 Full Model vs SG3 Reduced Model');

%% 8. Calculate relative errors

fprintf('\nRelative RMS errors:\n');

fprintf('delta : %.8f\n', ...
    norm(x_all(:,1)-delta3)/norm(x_all(:,1)));

fprintf('omega : %.8f\n', ...
    norm(x_all(:,2)-omega3)/norm(x_all(:,2)));

fprintf('edp   : %.8f\n', ...
    norm(x_all(:,4)-edp3)/norm(x_all(:,4)));

fprintf('eqp   : %.8f\n', ...
    norm(x_all(:,3)-eqp3)/norm(x_all(:,3)));

fprintf('Pg    : %.8f\n', ...
    norm(Pg_all-Pg3)/norm(Pg_all));

fprintf('Qg    : %.8f\n', ...
    norm(Qg_all-Qg3)/norm(Qg_all));