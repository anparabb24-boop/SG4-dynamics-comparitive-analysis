clc;
clear;
close all;

%% Load baseline experiment

load('results/sg4_disturbance_baseline.mat');

%% Baseline model outputs

[t,Y_baseline] = sg4_sensitivity_simulate( ...
    p,x0,Pm0,vf0);

output_scales = sqrt(mean(Y_baseline.^2,1));
output_scales = max(output_scales,1e-3);

%% Define increasing Td0 values

Td0_values = [
    6.5
    13
    26
    65
    130
    650
    1300
];

n = length(Td0_values);

%% Storage

normalized_error = zeros(n,1);
eqp_trajectories = zeros(length(t),n);

%% Simulate each parameter value

for k = 1:n

    p_test = p;

    p_test.Td0_prime = Td0_values(k);

    [t_test,Y_test] = sg4_sensitivity_simulate( ...
        p_test,x0,Pm0,vf0);

    % Calculate normalized output difference
    difference = (Y_test-Y_baseline)./output_scales;

    normalized_error(k) = ...
        sqrt(mean(difference(:).^2));

    % Recover eq' trajectories separately
    [~,x_test] = simulate_states( ...
        p_test,x0,Pm0,vf0);

    eqp_trajectories(:,k) = x_test(:,3);

    fprintf('Td0 = %8.2f | Error = %.8f\n', ...
        Td0_values(k),normalized_error(k));

end

%% Plot eq' trajectories

figure;

plot(t,eqp_trajectories,'LineWidth',1.3);

xlabel('Time (s)');
ylabel('e''_q (p.u.)');
title('Effect of Increasing T''_{d0}');
legend(string(Td0_values),'Location','best');
grid on;

%% Plot output differences

figure;

loglog(Td0_values,normalized_error,'o-', ...
    'LineWidth',1.5);

xlabel('T''_{d0} (s)');
ylabel('Normalized RMS Output Difference');
title('Approach to the T''_{d0} Limiting Model');
grid on;

%% Save results

save('results/sg4_Td0_boundary_test.mat', ...
    'Td0_values','normalized_error', ...
    'eqp_trajectories','t');

disp('Td0 boundary experiment completed.');