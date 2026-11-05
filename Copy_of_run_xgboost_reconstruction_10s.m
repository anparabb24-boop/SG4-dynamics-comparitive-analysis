clc;
close all;

fprintf('\n============================================\n');
fprintf('XGBOOST RECONSTRUCTION - 10 SECOND SG4\n');
fprintf('NATIVE XGBOOST API\n');
fprintf('============================================\n\n');

%% 1. CONFIGURATION

dataFile = 'data/modeling_data_10s/model_data_10s.mat';
evalFile = 'data/evaluation_10s/evaluation_config_10s.mat';

outputDir = 'results/xgboost_10s';
modelDir = fullfile(outputDir, 'models');

if ~isfolder(outputDir)
    mkdir(outputDir);
end
if ~isfolder(modelDir)
    mkdir(modelDir);
end

assert(isfile(dataFile), 'Unified dataset not found.');
assert(isfile(evalFile), 'Evaluation configuration not found.');

%% 2. VERIFY PYTHON AND NATIVE XGBOOST

fprintf('Checking Python and native XGBoost...\n');

pye = pyenv;
if ~contains(char(pye.Executable), 'xgboost_env312')
    error(['MATLAB is not connected to xgboost_env312. ', ...
           'Configure pyenv before running this script.']);
end

np = py.importlib.import_module('numpy');
xgb = py.importlib.import_module('xgboost');

fprintf('Python environment: %s\n', char(pye.Executable));
fprintf('Native XGBoost imported successfully.\n');

%% 3. LOAD DATA

fprintf('\nLoading datasets...\n');

S = load(dataFile);
E = load(evalFile);

modelData = S.modelData;
evaluationConfig = E.evaluationConfig;

trainData = modelData.train;
valData = modelData.validation;
testData = modelData.test;

stateNames = modelData.state_names;
outputNames = modelData.output_names;
channelNames = [stateNames, outputNames];

numChannels = 8;

assert(numel(channelNames) == numChannels, ...
    'Expected eight reconstruction channels.');

fprintf('Training trajectories: %d\n', numel(trainData.X_noisy));
fprintf('Validation trajectories: %d\n', numel(valData.X_noisy));
fprintf('Testing trajectories: %d\n', numel(testData.X_noisy));

%% 4. NORMALIZATION PARAMETERS

N = evaluationConfig.normalization;

normStd = [double(N.state_std(:)); double(N.output_std(:))];
normRange = [double(N.state_range(:)); double(N.output_range(:))];

assert(numel(normStd) == numChannels);
assert(numel(normRange) == numChannels);

%% 5. PREPARE FEATURES AND TARGETS

fprintf('\nPreparing modeling matrices...\n');

featureNames = { ...
    'delta_noisy', 'omega_noisy', 'eq_prime_noisy', 'ed_prime_noisy', ...
    'id_noisy', 'iq_noisy', 'Pg_noisy', 'Qg_noisy', ...
    'V', 'theta', 'Pm', 'vf', 'time'};

[XTrain, YTrain] = prepareSplit(trainData);
[XVal, YVal, valCleanCells, valNoisyCells, valTimeCells] = prepareSplit(valData);
[XTest, YTest, testCleanCells, testNoisyCells, testTimeCells] = prepareSplit(testData);

fprintf('Number of features: %d\n', size(XTrain,2));
fprintf('Training samples: %d\n', size(XTrain,1));
fprintf('Validation samples: %d\n', size(XVal,1));
fprintf('Testing samples: %d\n', size(XTest,1));

assert(size(XTrain,2) == 13);
assert(size(YTrain,2) == 8);
assert(size(XVal,2) == 13 && size(YVal,2) == 8);
assert(size(XTest,2) == 13 && size(YTest,2) == 8);

%% 6. HYPERPARAMETER CONFIGURATIONS

configs = struct([]);

configs(1).n_estimators = 250;
configs(1).max_depth = 4;
configs(1).learning_rate = 0.05;

configs(2).n_estimators = 400;
configs(2).max_depth = 4;
configs(2).learning_rate = 0.05;

configs(3).n_estimators = 300;
configs(3).max_depth = 6;
configs(3).learning_rate = 0.05;

numConfigs = numel(configs);

subsample = 0.8;
colsample = 0.8;
minChildWeight = 5;
regLambda = 1;

%% 7. CONVERT MATRICES TO NUMPY AND VERIFY ORIENTATION

fprintf('\nConverting data for native XGBoost...\n');

pyXTrain = makePythonMatrix(np, xgb, XTrain);
pyXVal = makePythonMatrix(np, xgb, XVal);
pyXTrainVal = makePythonMatrix(np, xgb, [XTrain; XVal]);
pyXTest = makePythonMatrix(np, xgb, XTest);

fprintf('Verified Python matrix dimensions:\n');
fprintf('  Training:   %d x %d\n', ...
    size(XTrain,1), size(XTrain,2));
fprintf('  Validation: %d x %d\n', ...
    size(XVal,1), size(XVal,2));
fprintf('  Test:       %d x %d\n', ...
    size(XTest,1), size(XTest,2));

%% 8. HYPERPARAMETER TUNING AND MODEL SELECTION

fprintf('\n============================================\n');
fprintf('HYPERPARAMETER TUNING\n');
fprintf('============================================\n');

bestModels = cell(1,numChannels);
bestValPredictions = zeros(size(YVal));

bestConfigIndex = zeros(1,numChannels);
bestValidationRMSE = inf(1,numChannels);

tuningRows = cell(0,7);

for ch = 1:numChannels

    fprintf('\nChannel %d/%d: %s\n', ...
        ch, numChannels, channelNames{ch});

    pyYTrain = np.ravel(np.array(single(YTrain(:,ch))));
    pyYVal = np.ravel(np.array(single(YVal(:,ch))));

    dtrain = xgb.DMatrix(pyXTrain, pyargs('label',pyYTrain));
    dval = xgb.DMatrix(pyXVal, pyargs('label',pyYVal));

    for cfg = 1:numConfigs

        fprintf('  Configuration %d/%d...\n', cfg, numConfigs);

        params = makeXGBoostParams(configs(cfg), ...
            subsample, colsample, minChildWeight, regLambda);

        % Train with the native XGBoost API.
        model = xgb.train(params, dtrain, ...
            int32(configs(cfg).n_estimators));

        % Predict validation data.
        pyPred = model.predict(dval);
        valPrediction = pythonVectorToDouble(pyPred);

        assert(numel(valPrediction) == size(YVal,1), ...
            'Validation prediction count does not match validation samples.');

        valRMSE = sqrt(mean((valPrediction - YVal(:,ch)).^2));

        fprintf('    Validation RMSE: %.8g\n', valRMSE);

        selected = valRMSE < bestValidationRMSE(ch);

        tuningRows(end+1,:) = { ...
            channelNames{ch}, cfg, ...
            configs(cfg).n_estimators, ...
            configs(cfg).max_depth, ...
            configs(cfg).learning_rate, ...
            valRMSE, selected};

        if selected
            bestValidationRMSE(ch) = valRMSE;
            bestConfigIndex(ch) = cfg;
            bestModels{ch} = model;
            bestValPredictions(:,ch) = valPrediction;
        end
    end

    fprintf('  Selected configuration: %d\n', bestConfigIndex(ch));
    fprintf('  Best validation RMSE: %.8g\n', bestValidationRMSE(ch));
end

tuningTable = cell2table(tuningRows, ...
    'VariableNames', {'Channel','Configuration','NumTrees', ...
    'MaxDepth','LearningRate','ValidationRMSE','Selected'});

writetable(tuningTable, ...
    fullfile(outputDir,'xgboost_tuning_results.csv'));

%% 9. VALIDATION METRICS

fprintf('\n============================================\n');
fprintf('VALIDATION EVALUATION\n');
fprintf('============================================\n');

validationRows = cell(0,7);

for ch = 1:numChannels

    [rmse,mae,nrmseStd,nrmseRange] = calculateMetrics( ...
        bestValPredictions(:,ch), YVal(:,ch), ...
        normStd(ch), normRange(ch));

    [baseRMSE,baseMAE] = calculateMetrics( ...
        XVal(:,ch), YVal(:,ch), normStd(ch), normRange(ch));

    validationRows(end+1,:) = { ...
        channelNames{ch}, rmse, mae, nrmseStd, nrmseRange, ...
        baseRMSE, baseMAE};

    fprintf('%-12s XGBoost RMSE: %.8g | Noisy RMSE: %.8g\n', ...
        channelNames{ch}, rmse, baseRMSE);
end

validationTable = cell2table(validationRows, ...
    'VariableNames', {'Channel','XGBoost_RMSE','XGBoost_MAE', ...
    'XGBoost_NRMSE_std','XGBoost_NRMSE_range', ...
    'Noisy_RMSE','Noisy_MAE'});

writetable(validationTable, ...
    fullfile(outputDir,'xgboost_validation_metrics.csv'));

%% 10. FINAL TRAINING USING TRAIN + VALIDATION

fprintf('\n============================================\n');
fprintf('FINAL MODEL TRAINING\n');
fprintf('============================================\n');

YTrainVal = [YTrain; YVal];

finalModels = cell(1,numChannels);
modelPaths = cell(1,numChannels);

for ch = 1:numChannels

    cfg = bestConfigIndex(ch);
    fprintf('Training final model: %s\n', channelNames{ch});

    pyYFinal = np.ravel(np.array(single(YTrainVal(:,ch))));
    dtrainFinal = xgb.DMatrix(pyXTrainVal, ...
        pyargs('label',pyYFinal));

    params = makeXGBoostParams(configs(cfg), ...
        subsample, colsample, minChildWeight, regLambda);

    finalModel = xgb.train(params, dtrainFinal, ...
        int32(configs(cfg).n_estimators));

    finalModels{ch} = finalModel;

    modelPath = fullfile(modelDir, ...
        sprintf('xgb_%s.json',channelNames{ch}));

    finalModel.save_model(modelPath);
    modelPaths{ch} = modelPath;
end

%% 11. TEST SET PREDICTIONS

fprintf('\n============================================\n');
fprintf('FINAL TEST EVALUATION\n');
fprintf('============================================\n');

dtest = xgb.DMatrix(pyXTest);
testPredictions = zeros(size(YTest));

for ch = 1:numChannels

    fprintf('Predicting channel: %s\n', channelNames{ch});

    pyPred = finalModels{ch}.predict(dtest);
    prediction = pythonVectorToDouble(pyPred);

    assert(numel(prediction) == size(YTest,1), ...
        'Test prediction count does not match test samples.');

    testPredictions(:,ch) = prediction;
end

%% 12. RESTORE INDIVIDUAL TRAJECTORIES

numTestTrajectories = numel(testData.X_noisy);
testPredCells = cell(1,numTestTrajectories);

trajectoryLength = size(testData.X_noisy{1},1);

assert(size(testPredictions,1) == ...
    numTestTrajectories * trajectoryLength, ...
    'Test matrix cannot be divided into equal trajectories.');

for k = 1:numTestTrajectories
    first = (k-1)*trajectoryLength + 1;
    last = k*trajectoryLength;
    testPredCells{k} = testPredictions(first:last,:);
end

%% 13. OVERALL TEST METRICS

fprintf('\n============================================\n');
fprintf('OVERALL TEST METRICS\n');
fprintf('============================================\n');

overallRows = cell(0,8);

for ch = 1:numChannels

    [rmse,mae,nrmseStd,nrmseRange] = calculateMetrics( ...
        testPredictions(:,ch), YTest(:,ch), ...
        normStd(ch), normRange(ch));

    [baseRMSE,baseMAE,baseNRMSEStd,baseNRMSERange] = ...
        calculateMetrics(XTest(:,ch), YTest(:,ch), ...
        normStd(ch), normRange(ch));

    overallRows(end+1,:) = { ...
        'XGBoost', channelNames{ch}, ...
        rmse, mae, nrmseStd, nrmseRange, ...
        size(YTest,1), numTestTrajectories};

    overallRows(end+1,:) = { ...
        'NoisyReference', channelNames{ch}, ...
        baseRMSE, baseMAE, baseNRMSEStd, baseNRMSERange, ...
        size(YTest,1), numTestTrajectories};

    fprintf('%-12s XGBoost RMSE: %.8g | Noisy RMSE: %.8g\n', ...
        channelNames{ch}, rmse, baseRMSE);
end

overallTestTable = cell2table(overallRows, ...
    'VariableNames', {'Method','Channel','RMSE','MAE', ...
    'NRMSE_std','NRMSE_range','SampleCount','TrajectoryCount'});

writetable(overallTestTable, ...
    fullfile(outputDir,'xgboost_overall_test_metrics.csv'));

%% 14. CONDITION-WISE TEST METRICS

fprintf('\n============================================\n');
fprintf('CONDITION-WISE TEST EVALUATION\n');
fprintf('============================================\n');

testMeta = testData.metadata;

experiments = unique({testMeta.experiment});
noiseTypes = unique({testMeta.noise_type});
noiseLevels = unique([testMeta.noise_level]);

metricRows = cell(0,12);

for ei = 1:numel(experiments)
    for ni = 1:numel(noiseTypes)
        for li = 1:numel(noiseLevels)

            idx = find(arrayfun(@(m) ...
                strcmp(m.experiment,experiments{ei}) && ...
                strcmp(m.noise_type,noiseTypes{ni}) && ...
                abs(m.noise_level-noiseLevels(li)) < 1e-12, ...
                testMeta));

            if isempty(idx)
                continue;
            end

            predCondition = vertcat(testPredCells{idx});
            cleanCondition = vertcat(testCleanCells{idx});
            noisyCondition = vertcat(testNoisyCells{idx});

            for ch = 1:numChannels

                [rmse,mae,nrmseStd,nrmseRange] = calculateMetrics( ...
                    predCondition(:,ch), cleanCondition(:,ch), ...
                    normStd(ch), normRange(ch));

                metricRows(end+1,:) = { ...
                    'XGBoost', experiments{ei}, noiseTypes{ni}, ...
                    noiseLevels(li), channelNames{ch}, ...
                    numel(idx), size(cleanCondition,1), ...
                    rmse, mae, nrmseStd, nrmseRange, ...
                    bestConfigIndex(ch)};

                [baseRMSE,baseMAE,baseNRMSEStd,baseNRMSERange] = ...
                    calculateMetrics(noisyCondition(:,ch), ...
                    cleanCondition(:,ch), normStd(ch), normRange(ch));

                metricRows(end+1,:) = { ...
                    'NoisyReference', experiments{ei}, noiseTypes{ni}, ...
                    noiseLevels(li), channelNames{ch}, ...
                    numel(idx), size(cleanCondition,1), ...
                    baseRMSE, baseMAE, baseNRMSEStd, baseNRMSERange, NaN};
            end
        end
    end
end

conditionTestTable = cell2table(metricRows, ...
    'VariableNames', {'Method','Experiment','NoiseType','NoiseLevel', ...
    'Channel','TrajectoryCount','SampleCount','RMSE','MAE', ...
    'NRMSE_std','NRMSE_range','SelectedConfiguration'});

writetable(conditionTestTable, ...
    fullfile(outputDir,'xgboost_condition_test_metrics.csv'));

fprintf('Condition-wise metric rows: %d\n', height(conditionTestTable));

%% 15. RECONSTRUCTION PLOT

fprintf('\nGenerating reconstruction plot...\n');

figure('Name','XGBoost SG4 Reconstruction', ...
    'Position',[100 100 1200 850]);

tiledlayout(4,2,'TileSpacing','compact');

sampleIndex = 1;
time = testTimeCells{sampleIndex};
clean = testCleanCells{sampleIndex};
noisy = testNoisyCells{sampleIndex};
predicted = testPredCells{sampleIndex};

for ch = 1:numChannels
    nexttile;

    plot(time,clean(:,ch),'k','LineWidth',1.3);
    hold on;
    plot(time,noisy(:,ch),'Color',[0.75 0.75 0.75]);
    plot(time,predicted(:,ch),'r','LineWidth',1.1);

    xlabel('Time (s)');
    ylabel(channelNames{ch});
    title(channelNames{ch});
    grid on;

    if ch == 1
        legend('Clean reference','Noisy measurement', ...
            'XGBoost reconstruction','Location','best');
    end
end

sgtitle('XGBoost Reconstruction - First Test Trajectory');

exportgraphics(gcf, ...
    fullfile(outputDir,'xgboost_reconstruction_example.png'), ...
    'Resolution',300);

%% 16. SAVE RESULTS

fprintf('\nSaving results...\n');

xgboostResults = struct();

xgboostResults.channel_names = channelNames;
xgboostResults.feature_names = featureNames;

xgboostResults.best_config_index = bestConfigIndex;
xgboostResults.best_validation_rmse = bestValidationRMSE;

xgboostResults.tuning_table = tuningTable;
xgboostResults.validation_table = validationTable;
xgboostResults.overall_test_table = overallTestTable;
xgboostResults.condition_test_table = conditionTestTable;

xgboostResults.test_predictions = testPredCells;
xgboostResults.test_clean = testCleanCells;
xgboostResults.test_noisy = testNoisyCells;
xgboostResults.test_time = testTimeCells;
xgboostResults.test_metadata = testMeta;

xgboostResults.model_paths = modelPaths;
xgboostResults.normalization_std = normStd;
xgboostResults.normalization_range = normRange;
xgboostResults.hyperparameter_configurations = configs;

xgboostResults.feature_count = size(XTrain,2);
xgboostResults.training_sample_count = size(XTrain,1);
xgboostResults.validation_sample_count = size(XVal,1);
xgboostResults.test_sample_count = size(XTest,1);
xgboostResults.python_api = 'Native XGBoost DMatrix/train';

save(fullfile(outputDir,'xgboost_results_10s.mat'), ...
    'xgboostResults','-v7.3');

%% 17. FINAL SUMMARY

fprintf('\n============================================\n');
fprintf('XGBOOST RECONSTRUCTION COMPLETED\n');
fprintf('============================================\n');

fprintf('Channels trained: %d\n',numChannels);
fprintf('Hyperparameter configurations: %d\n',numConfigs);
fprintf('Final models: %d\n',numel(finalModels));

fprintf('\nOutput directory:\n%s\n',outputDir);

fprintf('\nSaved files:\n');
fprintf('1. xgboost_tuning_results.csv\n');
fprintf('2. xgboost_validation_metrics.csv\n');
fprintf('3. xgboost_overall_test_metrics.csv\n');
fprintf('4. xgboost_condition_test_metrics.csv\n');
fprintf('5. xgboost_results_10s.mat\n');
fprintf('6. xgboost_reconstruction_example.png\n');
fprintf('7. Eight native XGBoost JSON models\n');

fprintf('\nALL OPERATIONS COMPLETED.\n');

%% LOCAL FUNCTION 1: PREPARE DATA SPLIT

function [X,Y,cleanCells,noisyCells,timeCells] = prepareSplit(D)

    n = numel(D.X_noisy);

    X = [];
    Y = [];

    cleanCells = cell(1,n);
    noisyCells = cell(1,n);
    timeCells = cell(1,n);

    for k = 1:n

        Xn = double(D.X_noisy{k});
        Xc = double(D.X_clean{k});
        Yn = double(D.Y_noisy{k});
        Yc = double(D.Y_clean{k});

        V = double(D.V{k}(:));
        theta = double(D.theta{k}(:));
        Pm = double(D.Pm{k}(:));
        vf = double(D.vf{k}(:));
        t = double(D.t{k}(:));

        features = [Xn,Yn,V,theta,Pm,vf,t];
        targets = [Xc,Yc];

        assert(size(features,1) == size(targets,1));
        assert(size(features,2) == 13);
        assert(size(targets,2) == 8);
        assert(all(isfinite(features(:))));
        assert(all(isfinite(targets(:))));

        X = [X;features]; %#ok<AGROW>
        Y = [Y;targets]; %#ok<AGROW>

        cleanCells{k} = targets;
        noisyCells{k} = [Xn,Yn];
        timeCells{k} = t;
    end
end

%% LOCAL FUNCTION 2: CONVERT MATLAB MATRIX TO VERIFIED NUMPY MATRIX

function pyArray = makePythonMatrix(np,xgb,A)

    % Convert to single precision to reduce memory use.
    pyArray = np.array(single(A));

    % Verify orientation using XGBoost's own DMatrix dimensions.
    dcheck = xgb.DMatrix(pyArray);

    nRows = double(dcheck.num_row());
    nCols = double(dcheck.num_col());

    if nRows == size(A,1) && nCols == size(A,2)
        return;
    end

    % MATLAB/Python conversion can differ in dimension ordering.
    % Transpose once if the converted matrix is reversed.
    if nRows == size(A,2) && nCols == size(A,1)
        pyArray = np.transpose(pyArray);
        dcheck = xgb.DMatrix(pyArray);

        nRows = double(dcheck.num_row());
        nCols = double(dcheck.num_col());
    end

    assert(nRows == size(A,1) && nCols == size(A,2), ...
        'NumPy matrix dimensions do not match the MATLAB feature matrix.');
end

%% LOCAL FUNCTION 3: BUILD NATIVE XGBOOST PARAMETERS

function params = makeXGBoostParams(cfg,subsample,colsample, ...
    minChildWeight,regLambda)

    params = py.dict(pyargs( ...
        'objective','reg:squarederror', ...
        'eval_metric','rmse', ...
        'max_depth',int32(cfg.max_depth), ...
        'eta',cfg.learning_rate, ...
        'subsample',subsample, ...
        'colsample_bytree',colsample, ...
        'min_child_weight',minChildWeight, ...
        'lambda',regLambda, ...
        'tree_method','hist', ...
        'nthread',int32(-1), ...
        'seed',int32(42), ...
        'verbosity',int32(0)));
end

%% LOCAL FUNCTION 4: CONVERT PYTHON PREDICTION TO MATLAB VECTOR

function y = pythonVectorToDouble(pyPrediction)

    % XGBoost returns a one-dimensional NumPy array.
    % MATLAB's double conversion produces a MATLAB numeric array.
    y = double(pyPrediction);
    y = y(:);
end

%% LOCAL FUNCTION 5: CALCULATE METRICS

function [rmse,mae,nrmseStd,nrmseRange] = ...
    calculateMetrics(predicted,reference,scaleStd,scaleRange)

    predicted = double(predicted(:));
    reference = double(reference(:));

    errorValues = predicted-reference;

    rmse = sqrt(mean(errorValues.^2));
    mae = mean(abs(errorValues));

    if scaleStd > 0
        nrmseStd = rmse/scaleStd;
    else
        nrmseStd = NaN;
    end

    if scaleRange > 0
        nrmseRange = rmse/scaleRange;
    else
        nrmseRange = NaN;
    end
end
