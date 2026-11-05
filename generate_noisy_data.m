
clc;
clear;
close all;

%% CONFIGURATION

experimentFiles = {
    'results/experiment_A_voltage_10s.mat'
    'results/experiment_B_mechanical_10s.mat'
    'results/experiment_C_excitation_10s.mat'
    'results/experiment_D_combined_10s.mat'
};

experimentNames = {
    'A_voltage'
    'B_mechanical'
    'C_excitation'
    'D_combined'
};

outputRoot = fullfile('data', 'noisy_experiments_10s');

noiseTypes = {'gaussian', 'uniform', 'impulsive'};
noiseLevels = [0.01, 0.05, 0.10];

nRealizations = 10;

baseSeed = 20261105;

impulseProbability = 0.01;
impulseAmplitudeSigma = 10;

expectedSamples = 1001;

stateNames = {'delta', 'omega', 'eq_prime', 'ed_prime'};
outputNames = {'id', 'iq', 'Pg', 'Qg'};

%% INITIALIZATION

if ~exist(outputRoot, 'dir')
    mkdir(outputRoot);
end

totalGenerated = 0;

fprintf('\n====================================\n');
fprintf('10-SECOND EXCITATION NOISE GENERATION\n');
fprintf('====================================\n');

%% PROCESS EACH EXPERIMENT

for e = 1:length(experimentFiles)

    fprintf('\n------------------------------------\n');
    fprintf('Experiment: %s\n', experimentNames{e});
    fprintf('------------------------------------\n');

    %% Load clean experiment

    assert(isfile(experimentFiles{e}), ...
        'Experiment file not found: %s', experimentFiles{e});

    S = load(experimentFiles{e});

    requiredVariables = {
        't_all', 'x_all', ...
        'id_all', 'iq_all', 'Pg_all', 'Qg_all', ...
        'V_all', 'theta_all', 'Pm_all', 'vf_all', ...
        'x0', 'p', 'Pm0', 'vf0'
    };

    for q = 1:length(requiredVariables)

        assert(isfield(S, requiredVariables{q}), ...
            'Missing variable %s in %s', ...
            requiredVariables{q}, experimentFiles{e});

    end

    t = S.t_all(:);

    X_clean = S.x_all;

    Y_clean = [ ...
        S.id_all(:), ...
        S.iq_all(:), ...
        S.Pg_all(:), ...
        S.Qg_all(:)];

    V = S.V_all(:);
    theta = S.theta_all(:);
    Pm = S.Pm_all(:);
    vf = S.vf_all(:);

    %% Validate clean data

    assert(size(X_clean,1) == expectedSamples);
    assert(size(X_clean,2) == 4);

    assert(isequal(size(Y_clean), [expectedSamples 4]));

    assert(numel(t) == expectedSamples);

    assert(all(isfinite(X_clean(:))));
    assert(all(isfinite(Y_clean(:))));

    assert(all(diff(t) > 0));

    assert(abs(t(end) - 10) < 1e-8);

    %% Generate noisy realizations

    for a = 1:length(noiseTypes)

        noiseType = noiseTypes{a};

        for b = 1:length(noiseLevels)

            relativeLevel = noiseLevels(b);

            levelPercent = round(100*relativeLevel);

            folder = fullfile(outputRoot, ...
                experimentNames{e}, ...
                noiseType, ...
                sprintf('level_%02d_percent', levelPercent));

            if ~exist(folder, 'dir')
                mkdir(folder);
            end

            for r = 1:nRealizations

                %% Reproducible random seed

                seed = baseSeed ...
                    + e*100000 ...
                    + a*10000 ...
                    + b*1000 ...
                    + r;

                rng(seed, 'twister');

                %% Add noise to states

                X_noisy = add_noise( ...
                    X_clean, ...
                    relativeLevel, ...
                    noiseType, ...
                    impulseProbability, ...
                    impulseAmplitudeSigma);

                %% Add noise to outputs

                Y_noisy = add_noise( ...
                    Y_clean, ...
                    relativeLevel, ...
                    noiseType, ...
                    impulseProbability, ...
                    impulseAmplitudeSigma);

                %% Construct dataset

                dataset = struct();

                dataset.t = t;

                dataset.X_clean = X_clean;
                dataset.X_noisy = X_noisy;

                dataset.Y_clean = Y_clean;
                dataset.Y_noisy = Y_noisy;

                % Known inputs remain uncorrupted.
                dataset.V = V;
                dataset.theta = theta;
                dataset.Pm = Pm;
                dataset.vf = vf;

                dataset.state_names = stateNames;
                dataset.output_names = outputNames;

                dataset.parameters = S.p;
                dataset.initial_state = S.x0;

                dataset.Pm0 = S.Pm0;
                dataset.vf0 = S.vf0;

                dataset.duration_seconds = t(end) - t(1);
                dataset.sample_count = length(t);
                dataset.time_step = mean(diff(t));

                %% Noise metadata

                dataset.noise = struct();

                dataset.noise.type = noiseType;
                dataset.noise.relative_level = relativeLevel;
                dataset.noise.seed = seed;
                dataset.noise.realization = r;

                dataset.noise.signal_scale_method = ...
                    'RMS magnitude per channel';

                dataset.noise.impulse_probability = ...
                    impulseProbability;

                dataset.noise.impulse_amplitude_sigma = ...
                    impulseAmplitudeSigma;

                dataset.experiment = experimentNames{e};

                %% Save dataset

                filename = sprintf( ...
                    'experiment_%s_level_%02d_realization_%02d_10s.mat', ...
                    experimentNames{e}, levelPercent, r);

                filepath = fullfile(folder, filename);

                save(filepath, 'dataset');

                totalGenerated = totalGenerated + 1;

            end

            fprintf('%-10s %2d%%: %d realizations generated\n', ...
                noiseType, levelPercent, nRealizations);

        end

    end

end

%% FINAL REPORT

expectedTotal = length(experimentFiles) ...
    * length(noiseTypes) ...
    * length(noiseLevels) ...
    * nRealizations;

fprintf('\n====================================\n');
fprintf('NOISY EXPERIMENT GENERATION COMPLETED\n');
fprintf('====================================\n');

fprintf('Experiments:             %d\n', length(experimentFiles));
fprintf('Noise types:             %d\n', length(noiseTypes));
fprintf('Noise levels:            %d\n', length(noiseLevels));
fprintf('Realizations per condition: %d\n', nRealizations);

fprintf('Total datasets generated: %d\n', totalGenerated);

fprintf('Samples per dataset:     %d\n', expectedSamples);

fprintf('Duration per dataset:    10 seconds\n');

fprintf('Output directory: %s\n', outputRoot);

assert(totalGenerated == expectedTotal, ...
    'Generated dataset count does not match expectation.');

fprintf('\nExpected dataset count verified: %d\n', expectedTotal);


%% LOCAL NOISE GENERATION FUNCTION

function noisy = add_noise(clean, relativeLevel, ...
    noiseType, impulseProbability, impulseAmplitudeSigma)

    noisy = zeros(size(clean));

    for j = 1:size(clean,2)

        signal = clean(:,j);

        % RMS scale of the clean signal.
        signalRMS = sqrt(mean(signal.^2));

        % Target noise standard deviation.
        sigma = relativeLevel * signalRMS;

        switch lower(noiseType)

            case 'gaussian'

                noise = sigma * randn(size(signal));

            case 'uniform'

                % Variance-matched uniform distribution.
                noise = sqrt(3) * sigma * ...
                    (2*rand(size(signal)) - 1);

            case 'impulsive'

                % Sparse impulses with random positive/negative signs.
                impulseMask = rand(size(signal)) < impulseProbability;

                impulseSigns = 2*(rand(size(signal)) > 0.5) - 1;

                noise = impulseMask .* impulseSigns .* ...
                    impulseAmplitudeSigma .* sigma;

            otherwise

                error('Unknown noise type: %s', noiseType);

        end

        noisy(:,j) = signal + noise;

    end

end
