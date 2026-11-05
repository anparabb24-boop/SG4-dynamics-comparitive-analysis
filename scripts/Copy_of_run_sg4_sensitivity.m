clc;
clear;
close all;

%% 1. Load baseline experiment

load('results/sg4_disturbance_baseline.mat');

disp('Baseline experiment loaded.');

%% 2. Define parameter names

parameter_names = {
    'H'
    'D'
    'Td0_prime'
    'Tq0_prime'
    'xd'
    'xq'
    'xd_prime'
    'xq_prime'
};

n_parameters = length(parameter_names);

%% 3. Calculate baseline outputs

[t,Y0] = sg4_sensitivity_simulate(p,x0,Pm0,vf0);

output_names = {'Pg','Qg','id','iq'};

% Characteristic scale for each output
output_scales = sqrt(mean(Y0.^2,1));

% Avoid division by very small values
output_scales = max(output_scales,1e-3);

Y0_normalized = Y0 ./ output_scales;

%% 4. Calculate finite-difference sensitivities

J = zeros(numel(Y0_normalized),n_parameters);

% Small logarithmic parameter perturbation
h = 1e-3;

for j = 1:n_parameters

    name = parameter_names{j};

    % Positive perturbation
    p_plus = p;
    p_plus.(name) = p.(name)*exp(h);

    % Negative perturbation
    p_minus = p;
    p_minus.(name) = p.(name)*exp(-h);

    % Simulate both perturbed models
    [~,Y_plus] = sg4_sensitivity_simulate( ...
        p_plus,x0,Pm0,vf0);

    [~,Y_minus] = sg4_sensitivity_simulate( ...
        p_minus,x0,Pm0,vf0);

    % Normalize outputs
    Y_plus = Y_plus ./ output_scales;
    Y_minus = Y_minus ./ output_scales;

    % Central finite difference
    sensitivity = (Y_plus-Y_minus)/(2*h);

    % Store as one Jacobian column
    J(:,j) = sensitivity(:);

    fprintf('Calculated sensitivity for %s\n',name);

end

%% Individual parameter sensitivity

parameter_sensitivity = vecnorm(J,2,1);

disp(' ');
disp('Individual parameter sensitivity:');

for j = 1:n_parameters
    fprintf('%-12s : %.8f\n', ...
        parameter_names{j}, ...
        parameter_sensitivity(j));
end

figure;

bar(parameter_sensitivity);

set(gca,'XTick',1:n_parameters);
set(gca,'XTickLabel',parameter_names);
xtickangle(45);

ylabel('Sensitivity Norm');
title('Individual Parameter Sensitivity');
grid on;

%% 5. Construct Fisher Information Matrix

FIM = J' * J;

% Symmetrize against numerical round-off
FIM = (FIM + FIM')/2;

%% 6. Calculate singular values

[~,S,V] = svd(J,'econ');

singular_values = diag(S);

%% 7. Calculate FIM eigenvalues

[eigenvectors,D] = eig(FIM);

[eigenvalues,index] = sort(diag(D),'ascend');

eigenvectors = eigenvectors(:,index);

%% 8. Display results

disp(' ');
disp('FIM Eigenvalues:');
disp(eigenvalues);

disp('Singular Values of Sensitivity Jacobian:');
disp(singular_values);

disp('Least identifiable parameter direction:');

weakest_direction = eigenvectors(:,1);

for j = 1:n_parameters
    fprintf('%-12s : % .6f\n', ...
        parameter_names{j},weakest_direction(j));
end

%% 9. Plot singular values

figure;

semilogy(1:length(singular_values), ...
    singular_values,'o-','LineWidth',1.5);

xlabel('Singular Value Index');
ylabel('Singular Value');
title('SG4 Sensitivity Jacobian Singular Values');
grid on;

%% 10. Save results

if ~exist('results','dir')
    mkdir('results');
end

save('results/sg4_sensitivity_results.mat', ...
    'J','FIM','eigenvalues','eigenvectors', ...
    'singular_values','weakest_direction', ...
    'parameter_names','output_scales','t','Y0');

disp(' ');
disp('Sensitivity analysis completed.');
disp('Results saved successfully.');