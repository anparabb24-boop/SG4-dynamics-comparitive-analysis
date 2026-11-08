%% BUILD_DATASET_20S
% Build unified SG4 20-second noisy-dynamics reconstruction dataset.
%
% Source files:
%   results/sg4_disturbance_20s.mat
%   results/experiment_A_voltage_20s.mat
%   results/experiment_B_mechanical_20s.mat
%   results/experiment_C_excitation_20s.mat
%   results/experiment_D_combined_20s.mat
%
% Output:
%   data/dataset_20s/sg4_dataset_20s.mat
%   data/dataset_20s/split_manifest.csv
%   data/dataset_20s/noise_audit.csv
%
% Dimensions:
%   X_clean, X_noisy: [time x 4 states x trajectories]
%   Y_clean, Y_noisy: [time x 4 outputs x trajectories]
%   U:                [time x 5 inputs x trajectories]
%
% State order:
%   delta, omega, eq_prime, ed_prime
%
% Output order:
%   id, iq, Pg, Qg
%
% Input order:
%   V, theta, Pm, vf, kPg
%
% Split:
%   Realizations 1:6  -> Training
%   Realizations 7:8  -> Validation
%   Realizations 9:10 -> Testing

clc;
clear;
close all;

fprintf('\n============================================\n');
fprintf('BUILDING SG4 20-SECOND DATASET\n');
fprintf('============================================\n\n');

%% 1. CONFIGURATION

sourceFiles = {
    fullfile('results','sg4_disturbance_20s.mat')
    fullfile('results','experiment_A_voltage_20s.mat')
    fullfile('results','experiment_B_mechanical_20s.mat')
    fullfile('results','experiment_C_excitation_20s.mat')
    fullfile('results','experiment_D_combined_20s.mat')
};

experimentNames = {
    'baseline'
    'A_voltage'
    'B_mechanical'
    'C_excitation'
    'D_combined'
};

outputDir = fullfile('data','dataset_20s');

outputFile = fullfile(outputDir,'sg4_dataset_20s.mat');

expectedSamples = 2001;
expectedDuration = 20;
expectedDt = 0.01;
timeTolerance = 1e-9;

stateNames = {
    'delta'
    'omega'
    'eq_prime'
    'ed_prime'
};

outputNames = {
    'id'
    'iq'
    'Pg'
    'Qg'
};

inputNames = {
    'V'
    'theta'
    'Pm'
    'vf'
    'kPg'
};

noiseTypes = {
    'gaussian'
    'uniform'
    'impulsive'
};

noiseLevels = [1 5 10];

nRealizations = 10;

baseSeed = 20261105;

impulseProbability = 0.01;
impulseAmplitudeSigma = 10;

trainRealizations = 1:6;
validationRealizations = 7:8;
testRealizations = 9:10;

%% 2. CREATE OUTPUT DIRECTORY

if ~exist(outputDir,'dir')
    mkdir(outputDir);
end

%% 3. CHECK SOURCE FILES

fprintf('Checking source files...\n');

for k = 1:numel(sourceFiles)

    if ~isfile(sourceFiles{k})

        error('Missing source file: %s',sourceFiles{k});

    end

    fprintf('  Found: %s\n',sourceFiles{k});

end

nExperiments = numel(sourceFiles);
nNoiseTypes = numel(noiseTypes);
nNoiseLevels = numel(noiseLevels);

expectedTotal = ...
    nExperiments*nNoiseTypes*nNoiseLevels*nRealizations;

fprintf('\nExpected trajectories: %d\n',expectedTotal);

%% 4. LOAD AND VALIDATE CLEAN DATA

fprintf('\nLoading clean simulations...\n');

cleanX = cell(nExperiments,1);
cleanY = cell(nExperiments,1);
cleanU = cell(nExperiments,1);

commonTime = [];

requiredFields = {
    't_all'
    'x_all'
    'id_all'
    'iq_all'
    'Pg_all'
    'Qg_all'
    'V_all'
    'theta_all'
    'Pm_all'
    'vf_all'
};

for e = 1:nExperiments

    fprintf('\nExperiment: %s\n',experimentNames{e});

    S = load(sourceFiles{e});

    %% Required fields

    missingFields = requiredFields(~isfield(S,requiredFields));

    if ~isempty(missingFields)

        error('Missing fields in %s: %s', ...
            sourceFiles{e},strjoin(missingFields,', '));

    end

    %% Time vector

    t = double(S.t_all(:));

    if numel(t) ~= expectedSamples

        error('%s must contain exactly %d time samples.', ...
            sourceFiles{e},expectedSamples);

    end

    if abs(t(1)) > timeTolerance || ...
       abs(t(end)-expectedDuration) > timeTolerance

        error('Time vector in %s must span 0 to 20 seconds.', ...
            sourceFiles{e});

    end

    if any(abs(diff(t)-expectedDt) > timeTolerance)

        error('Incorrect time step in %s. Expected 0.01 seconds.', ...
            sourceFiles{e});

    end

    if isempty(commonTime)

        commonTime = t;

    elseif any(abs(t-commonTime) > timeTolerance)

        error('Time vectors do not match across experiments.');

    end

    %% States

    X = double(S.x_all);

    if ~isequal(size(X),[expectedSamples 4])

        error('x_all in %s must be 2001 x 4. Found %s.', ...
            sourceFiles{e},mat2str(size(X)));

    end

    if any(~isfinite(X(:)))

        error('State data contains NaN or Inf.');

    end

    %% Outputs

    id = double(S.id_all(:));
    iq = double(S.iq_all(:));
    Pg = double(S.Pg_all(:));
    Qg = double(S.Qg_all(:));

    Y = [id iq Pg Qg];

    if ~isequal(size(Y),[expectedSamples 4])

        error('Output dimensions are incorrect in %s.', ...
            sourceFiles{e});

    end

    if any(~isfinite(Y(:)))

        error('Output data contains NaN or Inf.');

    end

    %% Inputs

    V = double(S.V_all(:));
    theta = double(S.theta_all(:));
    Pm = double(S.Pm_all(:));
    vf = double(S.vf_all(:));

    if any([numel(V),numel(theta),numel(Pm),numel(vf)] ...
            ~= expectedSamples)

        error('Input dimensions are incorrect in %s.', ...
            sourceFiles{e});

    end

    %% kPg input

    if isfield(S,'kPg_all')

        kPg = double(S.kPg_all(:));

        if numel(kPg) ~= expectedSamples

            error('kPg_all must contain 2001 samples.');

        end

    else

        if e == 1

            warning(['Baseline simulation does not contain kPg_all. ' ...
                'Assuming kPg = 1 throughout.']);

        end

        kPg = ones(expectedSamples,1);

    end

    %% Construct input matrix

    U = [V theta Pm vf kPg];

    if any(~isfinite(U(:)))

        error('Input data contains NaN or Inf.');

    end

    %% Store clean data

    cleanX{e} = X;
    cleanY{e} = Y;
    cleanU{e} = U;

    fprintf('  Samples: %d\n',size(X,1));
    fprintf('  States:  %d\n',size(X,2));
    fprintf('  Outputs: %d\n',size(Y,2));
    fprintf('  Inputs:  %d\n',size(U,2));

end

fprintf('\nAll clean simulations passed validation.\n');

%% 5. PREALLOCATE DATA ARRAYS

fprintf('\nPreallocating dataset arrays...\n');

nTime = expectedSamples;
nStates = numel(stateNames);
nOutputs = numel(outputNames);
nInputs = numel(inputNames);

X_clean = zeros(nTime,nStates,expectedTotal,'single');
X_noisy = zeros(nTime,nStates,expectedTotal,'single');

Y_clean = zeros(nTime,nOutputs,expectedTotal,'single');
Y_noisy = zeros(nTime,nOutputs,expectedTotal,'single');

U_all = zeros(nTime,nInputs,expectedTotal,'single');

experimentIndex = zeros(expectedTotal,1,'uint8');
noiseTypeIndex = zeros(expectedTotal,1,'uint8');

noiseLevelPercent = zeros(expectedTotal,1,'single');

realizationIndex = zeros(expectedTotal,1,'uint8');

randomSeed = zeros(expectedTotal,1,'uint32');

%% 6. GENERATE NOISY DATA

fprintf('\nGenerating noisy trajectories...\n');

trajectory = 0;

auditRows = cell(expectedTotal*(nStates+nOutputs),1);

auditCounter = 0;

for e = 1:nExperiments

    Xc = cleanX{e};
    Yc = cleanY{e};
    Uc = cleanU{e};

    for nt = 1:nNoiseTypes

        noiseName = noiseTypes{nt};

        for nl = 1:nNoiseLevels

            levelPercent = noiseLevels(nl);

            levelFraction = levelPercent/100;

            for r = 1:nRealizations

                trajectory = trajectory+1;

                %% Deterministic random seed

                seed = baseSeed + ...
                    e*100000 + ...
                    nt*10000 + ...
                    nl*1000 + ...
                    r;

                rng(double(seed),'twister');

                Xn = zeros(size(Xc));
                Yn = zeros(size(Yc));

                %% Add noise to states

                for c = 1:nStates

                    signal = Xc(:,c);

                    signalRMS = sqrt(mean(signal.^2));

                    if signalRMS < eps
                        signalRMS = 1;
                    end

                    sigma = levelFraction*signalRMS;

                    noise = generate_noise( ...
                        noiseName, ...
                        numel(signal), ...
                        sigma, ...
                        impulseProbability, ...
                        impulseAmplitudeSigma);

                    Xn(:,c) = signal+noise;

                    auditCounter = auditCounter+1;

                    auditRows{auditCounter} = make_audit_row( ...
                        e,experimentNames{e},noiseName, ...
                        levelPercent,r,seed,'state', ...
                        stateNames{c},signal,noise);

                end

                %% Add noise to outputs

                for c = 1:nOutputs

                    signal = Yc(:,c);

                    signalRMS = sqrt(mean(signal.^2));

                    if signalRMS < eps
                        signalRMS = 1;
                    end

                    sigma = levelFraction*signalRMS;

                    noise = generate_noise( ...
                        noiseName, ...
                        numel(signal), ...
                        sigma, ...
                        impulseProbability, ...
                        impulseAmplitudeSigma);

                    Yn(:,c) = signal+noise;

                    auditCounter = auditCounter+1;

                    auditRows{auditCounter} = make_audit_row( ...
                        e,experimentNames{e},noiseName, ...
                        levelPercent,r,seed,'output', ...
                        outputNames{c},signal,noise);

                end

                %% Store trajectory

                X_clean(:,:,trajectory) = single(Xc);
                X_noisy(:,:,trajectory) = single(Xn);

                Y_clean(:,:,trajectory) = single(Yc);
                Y_noisy(:,:,trajectory) = single(Yn);

                % Inputs are not corrupted.
                U_all(:,:,trajectory) = single(Uc);

                experimentIndex(trajectory) = uint8(e);
                noiseTypeIndex(trajectory) = uint8(nt);

                noiseLevelPercent(trajectory) = ...
                    single(levelPercent);

                realizationIndex(trajectory) = uint8(r);

                randomSeed(trajectory) = uint32(seed);

            end

        end

    end

end

if trajectory ~= expectedTotal

    error('Incorrect number of trajectories generated.');

end

fprintf('Generated %d trajectories.\n',trajectory);

%% 7. NOISE AUDIT TABLE

auditRows = auditRows(1:auditCounter);

noiseAudit = cell2table( ...
    vertcat(auditRows{:}), ...
    'VariableNames',{ ...
        'ExperimentIndex', ...
        'Experiment', ...
        'NoiseType', ...
        'NoiseLevelPercent', ...
        'Realization', ...
        'Seed', ...
        'SignalGroup', ...
        'Channel', ...
        'SignalRMS', ...
        'NoiseRMS', ...
        'RelativeNoiseRMSPercent', ...
        'NoiseMean', ...
        'NoiseStd', ...
        'OutlierRatePercent'});

%% 8. TRAIN / VALIDATION / TEST SPLIT

fprintf('\nCreating dataset splits...\n');

trainIdx = find(ismember( ...
    double(realizationIndex),trainRealizations));

validationIdx = find(ismember( ...
    double(realizationIndex),validationRealizations));

testIdx = find(ismember( ...
    double(realizationIndex),testRealizations));

if numel(trainIdx) ~= 270

    error('Training split should contain 270 trajectories.');

end

if numel(validationIdx) ~= 90

    error('Validation split should contain 90 trajectories.');

end

if numel(testIdx) ~= 90

    error('Test split should contain 90 trajectories.');

end

fprintf('\nSPLIT SUMMARY\n');
fprintf('---------------------------\n');

fprintf('Training:   %d\n',numel(trainIdx));
fprintf('Validation: %d\n',numel(validationIdx));
fprintf('Testing:    %d\n',numel(testIdx));
fprintf('Total:      %d\n',expectedTotal);

%% 9. METADATA

metadata = table( ...
    (1:expectedTotal)', ...
    double(experimentIndex(:)), ...
    string(experimentNames(double(experimentIndex(:)))), ...
    double(noiseTypeIndex(:)), ...
    string(noiseTypes(double(noiseTypeIndex(:)))), ...
    double(noiseLevelPercent(:)), ...
    double(realizationIndex(:)), ...
    double(randomSeed(:)), ...
    'VariableNames',{ ...
        'Trajectory', ...
        'ExperimentIndex', ...
        'Experiment', ...
        'NoiseTypeIndex', ...
        'NoiseType', ...
        'NoiseLevelPercent', ...
        'Realization', ...
        'Seed'});
%% 10. ASSEMBLE DATASET STRUCTURE

dataset = struct();

dataset.description = ...
    'SG4 20-second noisy dynamics reconstruction dataset';

dataset.time = commonTime;

dataset.stateNames = stateNames;
dataset.outputNames = outputNames;
dataset.inputNames = inputNames;

dataset.X_clean = X_clean;
dataset.X_noisy = X_noisy;

dataset.Y_clean = Y_clean;
dataset.Y_noisy = Y_noisy;

dataset.U = U_all;

dataset.metadata = metadata;

dataset.trainIdx = trainIdx(:);
dataset.validationIdx = validationIdx(:);
dataset.testIdx = testIdx(:);

%% Individual split structures

dataset.train = make_split( ...
    X_clean,X_noisy,Y_clean,Y_noisy,U_all,trainIdx);

dataset.validation = make_split( ...
    X_clean,X_noisy,Y_clean,Y_noisy,U_all,validationIdx);

dataset.test = make_split( ...
    X_clean,X_noisy,Y_clean,Y_noisy,U_all,testIdx);

%% Configuration metadata

dataset.configuration.expectedSamples = expectedSamples;
dataset.configuration.expectedDuration = expectedDuration;
dataset.configuration.expectedDt = expectedDt;

dataset.configuration.noiseTypes = noiseTypes;
dataset.configuration.noiseLevelsPercent = noiseLevels;

dataset.configuration.nRealizations = nRealizations;

dataset.configuration.baseSeed = baseSeed;

dataset.configuration.impulseProbability = impulseProbability;
dataset.configuration.impulseAmplitudeSigma = ...
    impulseAmplitudeSigma;

dataset.configuration.trainRealizations = trainRealizations;
dataset.configuration.validationRealizations = ...
    validationRealizations;
dataset.configuration.testRealizations = testRealizations;

%% 11. FINAL STRUCTURAL VALIDATION

fprintf('\nPerforming final validation...\n');

assert(isequal(size(X_clean), ...
    [expectedSamples 4 expectedTotal]));

assert(isequal(size(X_noisy), ...
    [expectedSamples 4 expectedTotal]));

assert(isequal(size(Y_clean), ...
    [expectedSamples 4 expectedTotal]));

assert(isequal(size(Y_noisy), ...
    [expectedSamples 4 expectedTotal]));

assert(isequal(size(U_all), ...
    [expectedSamples 5 expectedTotal]));

assert(all(isfinite(X_clean(:))));
assert(all(isfinite(X_noisy(:))));

assert(all(isfinite(Y_clean(:))));
assert(all(isfinite(Y_noisy(:))));

assert(all(isfinite(U_all(:))));

fprintf('All array dimensions are correct.\n');
fprintf('All dataset values are finite.\n');

%% 12. SAVE DATASET

fprintf('\nSaving dataset...\n');

save(outputFile,'dataset','-v7.3');

%% 13. SAVE SPLIT MANIFEST

splitManifest = metadata;

splitName = strings(expectedTotal,1);

splitName(trainIdx) = "train";

splitName(validationIdx) = "validation";

splitName(testIdx) = "test";

splitManifest.Split = splitName;

writetable( ...
    splitManifest, ...
    fullfile(outputDir,'split_manifest.csv'));

%% 14. SAVE NOISE AUDIT

writetable( ...
    noiseAudit, ...
    fullfile(outputDir,'noise_audit.csv'));

%% 15. FINAL REPORT

fprintf('\n============================================\n');
fprintf('20-SECOND DATASET BUILD COMPLETED\n');
fprintf('============================================\n');

fprintf('Dataset: %s\n',outputFile);

fprintf('Time samples:          %d\n',expectedSamples);
fprintf('Duration:              %.2f seconds\n',expectedDuration);
fprintf('Time step:             %.4f seconds\n',expectedDt);

fprintf('State channels:        %d\n',nStates);
fprintf('Output channels:       %d\n',nOutputs);
fprintf('Known input channels:  %d\n',nInputs);

fprintf('Total trajectories:    %d\n',expectedTotal);

fprintf('Training trajectories: %d\n',numel(trainIdx));
fprintf('Validation trajectories: %d\n',numel(validationIdx));
fprintf('Testing trajectories:  %d\n',numel(testIdx));

fprintf('\nInput order:\n');
fprintf('[V, theta, Pm, vf, kPg]\n');

fprintf('\nState order:\n');
fprintf('[delta, omega, eq_prime, ed_prime]\n');

fprintf('\nOutput order:\n');
fprintf('[id, iq, Pg, Qg]\n');

fprintf('\nSaved files:\n');
fprintf('  %s\n',outputFile);
fprintf('  %s\n',fullfile(outputDir,'split_manifest.csv'));
fprintf('  %s\n',fullfile(outputDir,'noise_audit.csv'));

fprintf('\n============================================\n');

fprintf(['IMPORTANT: The current split distributes all five experiments ' ...
    'across train, validation and test. It does not independently test ' ...
    'generalization to an unseen excitation experiment.\n']);

%% LOCAL FUNCTIONS

function noise = generate_noise( ...
    noiseName,n,sigma,impulseProbability,impulseAmplitudeSigma)

    switch lower(noiseName)

        case 'gaussian'

            noise = sigma*randn(n,1);

        case 'uniform'

            % Uniform noise with standard deviation sigma.

            noise = sqrt(3)*sigma*(2*rand(n,1)-1);

        case 'impulsive'

            % Sparse impulses with specified probability.

            noise = zeros(n,1);

            mask = rand(n,1) < impulseProbability;

            signs = 2*(rand(n,1)>0.5)-1;

            amplitudes = ...
                impulseAmplitudeSigma*sigma.*signs;

            noise(mask) = amplitudes(mask);

        case 'laplace'

            % Laplace noise with standard deviation sigma.

            b = sigma/sqrt(2);

            u = rand(n,1)-0.5;

            noise = -b*sign(u).*log(1-2*abs(u));

        case 'student_t'

            % Student-t noise with five degrees of freedom.

            nu = 5;

            raw = randn(n,1)./ ...
                sqrt(2*randg(nu/2,n,1)/nu);

            noise = raw*(sigma/sqrt(nu/(nu-2)));

        otherwise

            error('Unsupported noise type: %s',noiseName);

    end

end

function row = make_audit_row( ...
    experimentIndex,experimentName,noiseName, ...
    levelPercent,realization,seed,signalGroup, ...
    channel,signal,noise)

    signalRMS = sqrt(mean(double(signal(:)).^2));

    noiseRMS = sqrt(mean(double(noise(:)).^2));

    if signalRMS < eps

        relativeRMS = NaN;

    else

        relativeRMS = 100*noiseRMS/signalRMS;

    end

    noiseMean = mean(double(noise(:)));

    noiseStd = std(double(noise(:)));

    if strcmpi(noiseName,'impulsive')

        outlierRate = 100*mean(abs(noise(:))>0);

    else

        outlierRate = NaN;

    end

    row = { ...
        experimentIndex, ...
        experimentName, ...
        noiseName, ...
        levelPercent, ...
        realization, ...
        double(seed), ...
        signalGroup, ...
        channel, ...
        signalRMS, ...
        noiseRMS, ...
        relativeRMS, ...
        noiseMean, ...
        noiseStd, ...
        outlierRate};

end

function S = make_split( ...
    Xclean,Xnoisy,Yclean,Ynoisy,U,indices)

    S = struct();

    S.X_clean = Xclean(:,:,indices);

    S.X_noisy = Xnoisy(:,:,indices);

    S.Y_clean = Yclean(:,:,indices);

    S.Y_noisy = Ynoisy(:,:,indices);

    S.U = U(:,:,indices);

    S.indices = indices(:);

end