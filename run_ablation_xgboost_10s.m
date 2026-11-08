%% RUN_XGBOOST_ABLATION_V2  (single self-contained file, plots shown)
% Loads the dataset written by BUILD_DATASET_10s.m / BUILD_DATASET_20s.m and runs
% the XGBoost reconstruction ablation + leave-one-excitation-group-out evaluation.
% Data converter, splitting, noise selection, metrics, plotting and the XGBoost
% experiment are all local functions at the bottom of this file.
%
% Needs: MATLAB with the Python environment that has xgboost and numpy
% (the same ~/xgboost_env312 you used before). Native XGBoost API only.
% Run from the project root (the folder that contains data/ and results/).
%
% Variants (all pointwise; "inputs" = V, theta, Pm, vf and kPg when present):
%   full_withTime_all_inputs : all 8 noisy channels + inputs + time
%   noTime_all_inputs        : removes the time feature
%   noTime_all_noInputs      : only the 8 noisy channels
%   noTime_single_inputs     : target's own noisy channel + inputs
%   noTime_single_noInputs   : target's own noisy channel only (floor)
% Protocols: 'standard' and 'loeo' for groups 1..4
%   (1 voltage = baseline + A_voltage, 2 mechanical, 3 excitation, 4 combined).

clear; clc; close all;

%% ---- settings -----------------------------------------------------------
RUN_LOEO        = true;      % false = standard split only (quick smoke test)
TUNE            = false;     % false = fixed config 3 for speed
RETRAIN         = false;     % false = train on the train split only
VARIANTS_TO_RUN = 1:5;

% Plots: reconstruction figure of every variant (standard split) saved as PNG
% and, with SHOW_PLOTS = true, also kept open on screen.
SHOW_PLOTS       = true;
PLOT_EXPERIMENT  = 'combined';       % e.g. 'mechanical', 'combined', 'baseline'
PLOT_NOISE_TYPE  = 'uniform';       % e.g. 'impulsive'
PLOT_NOISE_LEVEL = 0.01;       % e.g. 0.10

% Dataset file. '' = auto-detect (tries the paths below in order).
datasetFile = '';
candidateFiles = { ...
    fullfile('data','dataset_10s','sg4_dataset_10s.mat'), ...
    fullfile('data','dataset_20s','sg4_dataset_20s.mat')};

%% ---- noise selection ----------------------------------------------------
% {} or [] = use everything (all noise types, all levels).
% Examples:  TRAIN_NOISE_TYPES = {'gaussian'};   TRAIN_NOISE_LEVELS = [0.01 0.05];
% Levels are fractions (0.01 = 1 %). Each run prints the mix it trained on.
TRAIN_NOISE_TYPES  = {'uniform'};
TRAIN_NOISE_LEVELS = 0.01;
TEST_NOISE_TYPES   = {'uniform'};
TEST_NOISE_LEVELS  = 0.01;

%% ---- python environment -------------------------------------------------
pe = pyenv;
if pe.Status == "NotLoaded"
    pyenv(Version = fullfile(getenv('HOME'), 'xgboost_env312', 'bin', 'python'));
end
xgbCheck = py.importlib.import_module('xgboost');
pe = pyenv;
fprintf('XGBoost %s, Python %s\n', char(py.getattr(xgbCheck, '__version__')), char(pe.Version));

%% ---- data ---------------------------------------------------------------
if isempty(datasetFile)
    for q = 1:numel(candidateFiles)
        if isfile(candidateFiles{q}), datasetFile = candidateFiles{q}; break; end
    end
end
if isempty(datasetFile) || ~isfile(datasetFile)
    error(['Dataset not found. Tried:\n  %s\nRun BUILD_DATASET_*s.m first, or set ' ...
           '"datasetFile" at the top of this script.'], strjoin(candidateFiles, sprintf('\n  ')));
end
fprintf('Loading %s ...\n', datasetFile);
Sd = load(datasetFile);
if isfield(Sd, 'dataset')
    ds = Sd.dataset;
else
    fn = fieldnames(Sd);
    ds = Sd.(fn{1});
end
clear Sd;
modelData = dataset_to_modelData(ds);
clear ds;

durationSec = round(modelData.train.t{1}(end));
outRoot = fullfile('results', sprintf('xgboost_ablation_%ds', durationSec));

tagParts = "";
if ~isempty(TRAIN_NOISE_TYPES),  tagParts = tagParts + "_trainT-" + strjoin(string(TRAIN_NOISE_TYPES), '+'); end
if ~isempty(TRAIN_NOISE_LEVELS), tagParts = tagParts + "_trainL-" + strjoin(string(TRAIN_NOISE_LEVELS*100), '+') + "pct"; end
if ~isempty(TEST_NOISE_TYPES),   tagParts = tagParts + "_testT-"  + strjoin(string(TEST_NOISE_TYPES), '+'); end
if ~isempty(TEST_NOISE_LEVELS),  tagParts = tagParts + "_testL-"  + strjoin(string(TEST_NOISE_LEVELS*100), '+') + "pct"; end
outRoot = [outRoot char(tagParts)];

variants = {
    % tag                        useTime  channelMode  useInputs
    'full_withTime_all_inputs',  true,    'all',       true
    'noTime_all_inputs',         false,   'all',       true
    'noTime_all_noInputs',       false,   'all',       false
    'noTime_single_inputs',      false,   'single',    true
    'noTime_single_noInputs',    false,   'single',    false
    };

protocols = {'standard', 0};
if RUN_LOEO
    for g = 1:4
        protocols(end+1,:) = {'loeo', g}; %#ok<SAGROW>
    end
end

%% ---- run ----------------------------------------------------------------
S = table();
for v = VARIANTS_TO_RUN
    for p = 1:size(protocols,1)
        cfg = struct();
        cfg.tag          = variants{v,1};
        cfg.useTime      = variants{v,2};
        cfg.channelMode  = variants{v,3};
        cfg.useInputs    = variants{v,4};
        cfg.protocol     = protocols{p,1};
        cfg.holdoutGroup = protocols{p,2};
        cfg.trainNoiseTypes  = TRAIN_NOISE_TYPES;
        cfg.trainNoiseLevels = TRAIN_NOISE_LEVELS;
        cfg.testNoiseTypes   = TEST_NOISE_TYPES;
        cfg.testNoiseLevels  = TEST_NOISE_LEVELS;
        cfg.tune         = TUNE;
        cfg.retrain      = RETRAIN;
        cfg.saveModels   = false;
        cfg.makePlot     = (p == 1);
        cfg.showPlot     = SHOW_PLOTS;
        cfg.plotExperiment = PLOT_EXPERIMENT;
        cfg.plotNoiseType  = PLOT_NOISE_TYPE;
        cfg.plotNoiseLevel = PLOT_NOISE_LEVEL;
        cfg.outDir       = outRoot;

        try
            R = xgb_run_experiment(cfg, modelData);
        catch ME
            fprintf(2, '  FAILED: %s | %s : %s\n', cfg.tag, cfg.protocol, ME.message);
            continue;
        end

        T = R.overall;
        T.Variant  = repmat(string(cfg.tag), height(T), 1);
        T.Protocol = repmat(string(R.protocol), height(T), 1);
        T = movevars(T, {'Variant','Protocol'}, 'Before', 'Channel');
        if isempty(S), S = T; else, S = [S; T]; end %#ok<AGROW>

        fprintf('  -> mean NRMSE_std %.4f | mean improvement %.1f %%\n', ...
            mean(T.NRMSE_std), mean(T.Improvement_pct));
    end
end

%% ---- summary ------------------------------------------------------------
if ~exist(outRoot,'dir'), mkdir(outRoot); end
writetable(S, fullfile(outRoot,'ablation_summary.csv'));
G = groupsummary(S, {'Variant','Protocol'}, 'mean', {'NRMSE_std','Improvement_pct'});
disp(G);
writetable(G, fullfile(outRoot,'ablation_summary_by_variant.csv'));

fprintf('\nHow to read this:\n');
fprintf(' * standard vs noTime_*: a big rise in NRMSE_std when time is removed = time lookup was doing the work.\n');
fprintf(' * all vs single: gap = benefit of cross-channel information.\n');
fprintf(' * loeo_* rows are the honest generalisation numbers; Improvement_pct <= 0 means no better than the noisy input.\n');
fprintf('Full per-channel table: %s\n', fullfile(outRoot,'ablation_summary.csv'));

%% =========================================================================
%  LOCAL FUNCTIONS (everything below this line is used by the script above)
%% =========================================================================

function modelData = dataset_to_modelData(ds)
% Convert the struct saved by BUILD_DATASET_*s.m into per-trajectory cell
% arrays (modelData.train / .validation / .test) used by the functions below.
% Noise level is converted from percent (1, 5, 10) to fraction (0.01, 0.05, 0.10).

need = {'X_noisy','X_clean','Y_noisy','Y_clean','U','time','metadata', ...
        'trainIdx','validationIdx','testIdx','inputNames'};
missing = need(~isfield(ds, need));
if ~isempty(missing)
    error('Dataset is missing field(s): %s', strjoin(missing, ', '));
end
mdVars = {'Experiment','NoiseType','NoiseLevelPercent','Realization','Seed'};
mdMissing = mdVars(~ismember(mdVars, ds.metadata.Properties.VariableNames));
if ~isempty(mdMissing)
    error('dataset.metadata is missing column(s): %s (found: %s)', ...
        strjoin(mdMissing, ', '), strjoin(ds.metadata.Properties.VariableNames, ', '));
end

inNames = cellstr(string(ds.inputNames));
col = @(n) find(strcmp(inNames, n), 1);
for n = {'V','theta','Pm','vf'}
    assert(~isempty(col(n{1})), 'Input "%s" not found in dataset.inputNames.', n{1});
end
hasKPg = ~isempty(col('kPg'));

tvec = double(ds.time(:));
md   = ds.metadata;
lvlScale = 1;
if max(double(md.NoiseLevelPercent)) > 1, lvlScale = 1/100; end

splitName = {'train','validation','test'};
splitIdx  = {ds.trainIdx(:), ds.validationIdx(:), ds.testIdx(:)};

modelData = struct();
for si = 1:3
    idx = splitIdx{si};
    n   = numel(idx);

    T = struct();
    for f = {'X_noisy','X_clean','Y_noisy','Y_clean','t','V','theta','Pm','vf'}
        T.(f{1}) = cell(n,1);
    end
    if hasKPg, T.kPg = cell(n,1); end
    meta = repmat(struct('source','dataset','experiment','','noise_type','', ...
        'noise_level',0,'realization',0,'seed',0), 1, n);

    for j = 1:n
        k = idx(j);
        T.X_noisy{j} = double(ds.X_noisy(:,:,k));
        T.X_clean{j} = double(ds.X_clean(:,:,k));
        T.Y_noisy{j} = double(ds.Y_noisy(:,:,k));
        T.Y_clean{j} = double(ds.Y_clean(:,:,k));
        T.t{j}       = tvec;

        U = double(ds.U(:,:,k));
        T.V{j}     = U(:, col('V'));
        T.theta{j} = U(:, col('theta'));
        T.Pm{j}    = U(:, col('Pm'));
        T.vf{j}    = U(:, col('vf'));
        if hasKPg, T.kPg{j} = U(:, col('kPg')); end

        meta(j).experiment  = char(md.Experiment(k));
        meta(j).noise_type  = char(md.NoiseType(k));
        meta(j).noise_level = double(md.NoiseLevelPercent(k)) * lvlScale;
        meta(j).realization = double(md.Realization(k));
        meta(j).seed        = double(md.Seed(k));
    end
    T.metadata = meta;
    modelData.(splitName{si}) = T;
end

fprintf('Converted dataset: %d samples per trajectory (%.1f s), %d known inputs | train %d, validation %d, test %d\n', ...
    numel(tvec), tvec(end), 4 + hasKPg, numel(modelData.train.X_noisy), ...
    numel(modelData.validation.X_noisy), numel(modelData.test.X_noisy));
end

% ------------------------------------------------------------------------
% DATA SELECTION: protocols, groups, noise filters
% ------------------------------------------------------------------------
function sel = sg4_select(M, cfg)
%SG4_SELECT Train / validation / test subsets for the chosen protocol.
%   M is the modelData struct saved by prepare_modeling_data_10s.m
%   (data/modeling_data_10s/model_data_10s.mat). Nothing else is loaded.
%
%   cfg.protocol = 'standard' : your realization-based split, unchanged
%   cfg.protocol = 'loeo'     : leave-one-excitation-group-out. Train and
%       validation exclude group cfg.holdoutGroup; test holds only that group
%       (from the test split).
%
%   OPTIONAL noise selection (all default to "use everything"):
%   cfg.trainNoiseTypes  : e.g. {'gaussian'}            -> train + validation
%   cfg.trainNoiseLevels : e.g. [0.01 0.05]             -> train + validation
%   cfg.testNoiseTypes   : e.g. {'impulsive'}           -> test
%   cfg.testNoiseLevels  : e.g. 0.10                    -> test
%   Empty ({} or []) means no filtering. Type names are matched
%   case-insensitively against metadata.noise_type.
%
%   Groups come from metadata.experiment:
%       1 voltage (baseline + A_voltage), 2 mechanical, 3 excitation, 4 combined
%
%   Output sel has .train .val .test (cell arrays + labels), .nrm
%   (training-only std/range of the 8 clean channels from the full standard
%   training split, fixed for every protocol and noise selection) and .groupNames.

P = prepare(M);

switch cfg.protocol
    case 'standard'
        sel.train = P.train;
        sel.val   = P.validation;
        sel.test  = P.test;
    case 'loeo'
        g = cfg.holdoutGroup;
        sel.train = subset_traj(P.train,      P.train.group      ~= g);
        sel.val   = subset_traj(P.validation, P.validation.group ~= g);
        sel.test  = subset_traj(P.test,       P.test.group       == g);
    otherwise
        error('Unknown protocol "%s".', cfg.protocol);
end

% ---- optional noise filters ---------------------------------------------
trT = get_opt(cfg, 'trainNoiseTypes',  {});
trL = get_opt(cfg, 'trainNoiseLevels', []);
teT = get_opt(cfg, 'testNoiseTypes',   {});
teL = get_opt(cfg, 'testNoiseLevels',  []);

if ~isempty(trT) || ~isempty(trL)
    sel.train = subset_traj(sel.train, noise_mask(sel.train, trT, trL, 'training'));
    sel.val   = subset_traj(sel.val,   noise_mask(sel.val,   trT, trL, 'validation'));
end
if ~isempty(teT) || ~isempty(teL)
    sel.test  = subset_traj(sel.test,  noise_mask(sel.test,  teT, teL, 'test'));
end

fprintf('  [noise] train: %s\n', mix_str(sel.train));
fprintf('  [noise] test : %s\n', mix_str(sel.test));

sel.nrm        = P.nrm;
sel.groupNames = P.groupNames;
end

% ------------------------------------------------------------------------
function v = get_opt(cfg, name, default)
if isfield(cfg, name) && ~isempty(cfg.(name))
    v = cfg.(name);
else
    v = default;
end
end

function m = noise_mask(S, types, levels, which)
m = true(numel(S.group), 1);
if ~isempty(types)
    m = m & ismember(lower(S.noiseType(:)), lower(string(types)));
end
if ~isempty(levels)
    lv = false(numel(S.group), 1);
    for q = 1:numel(levels)
        lv = lv | abs(S.noiseLevel(:) - levels(q)) < 1e-9;
    end
    m = m & lv;
end
if ~any(m)
    error('sg4_select:noMatch', ...
        'No %s trajectories match the noise filter. Available types: %s | levels: %s', ...
        which, strjoin(unique(S.noiseType(:))', ', '), mat2str(unique(S.noiseLevel(:))'));
end
end

function s = mix_str(S)
ty = unique(S.noiseType(:));
parts = strings(1, numel(ty));
for q = 1:numel(ty)
    parts(q) = sprintf('%s x%d', ty(q), sum(S.noiseType(:) == ty(q)));
end
s = sprintf('%s | levels %s | %d trajectories', strjoin(parts, ', '), ...
    mat2str(unique(S.noiseLevel(:))'), numel(S.group));
end

% ------------------------------------------------------------------------
function P = prepare(M)
P.groupNames = ["voltage","mechanical","excitation","combined"];
splits     = {'train','validation','test'};
trajFields = {'X_noisy','X_clean','Y_noisy','Y_clean','t','V','theta','Pm','vf'};

for s = 1:numel(splits)
    T = M.(splits{s});
    n = numel(T.X_noisy);
    out = struct();
    for f = trajFields
        out.(f{1}) = T.(f{1})(:);
    end
    if isfield(T, 'kPg')
        out.kPg = T.kPg(:);
    end
    grp = zeros(n,1); nt = strings(n,1); nl = zeros(n,1);
    rl  = zeros(n,1); ex = strings(n,1);
    for i = 1:n
        md = T.metadata(i);
        ex(i)  = string(md.experiment);
        grp(i) = group_of(ex(i));
        nt(i)  = string(md.noise_type);
        nl(i)  = double(md.noise_level);
        rl(i)  = double(md.realization);
    end
    out.group       = grp;
    out.noiseType   = nt;
    out.noiseLevel  = nl;
    out.realization = rl;
    out.experiment  = ex;
    out.groupNames  = P.groupNames;
    P.(splits{s})   = out;
end

Yc = cellfun(@(a,b) [a b], P.train.X_clean, P.train.Y_clean, 'UniformOutput', false);
Yc = vertcat(Yc{:});
P.nrm.names = {'delta','omega','eq_prime','ed_prime','id','iq','Pg','Qg'};
P.nrm.mean  = mean(Yc,1);
P.nrm.std   = std(Yc,0,1);
P.nrm.range = max(Yc,[],1) - min(Yc,[],1);
end

function g = group_of(e)
e = lower(e);
if contains(e,'baseline') || contains(e,'voltage')
    g = 1;
elseif contains(e,'mechanical')
    g = 2;
elseif contains(e,'excitation')
    g = 3;
elseif contains(e,'combined')
    g = 4;
else
    error('Unrecognised experiment name "%s". Edit group_of() in sg4_select.m.', e);
end
end

function out = subset_traj(S, mask)
fields = {'X_noisy','X_clean','Y_noisy','Y_clean','t','V','theta','Pm','vf', ...
          'group','noiseType','noiseLevel','realization','experiment'};
if isfield(S, 'kPg'), fields{end+1} = 'kPg'; end
out = struct();
for f = fields
    out.(f{1}) = S.(f{1})(mask);
end
out.groupNames = S.groupNames;
end

% ------------------------------------------------------------------------
% XGBOOST EXPERIMENT
% ------------------------------------------------------------------------
function R = xgb_run_experiment(cfg, modelData)
%XGB_RUN_EXPERIMENT Native-API XGBoost reconstruction with ablation options.
%   cfg.tag, cfg.protocol ('standard'|'loeo'), cfg.holdoutGroup (1..4)
%   cfg.useTime     : include t as a feature
%   cfg.useInputs   : include all known inputs (V, theta, Pm, vf and kPg when present)
%   cfg.channelMode : 'all'    -> all 8 noisy channels as features
%                     'single' -> only the target's own noisy channel
%   cfg.tune        : true = choose among 3 hyperparameter configs on validation;
%                     false = use config 3 (300 trees, depth 6, eta 0.05)
%   cfg.retrain     : retrain the chosen config on train+validation before testing
%   cfg.saveModels, cfg.makePlot, cfg.showPlot, cfg.outDir, noise-filter and plot-pick fields

np  = py.importlib.import_module('numpy');
xgb = py.importlib.import_module('xgboost');

sel   = sg4_select(modelData, cfg);
nrm   = sel.nrm;
names = nrm.names;

[Ftr, Ytr] = stack_samples(sel.train);
[Fva, Yva] = stack_samples(sel.val);
[Fte, ~]   = stack_samples(sel.test);
nIn = size(Ftr,2) - 9;                 % known inputs; last column is time
nTe = cellfun(@(a) size(a,1), sel.test.X_noisy);

hp = struct('rounds', {250, 400, 300}, 'depth', {4, 4, 6}, 'eta', {0.05, 0.05, 0.05});
if cfg.tune, hpList = 1:3; else, hpList = 3; end

ptag   = protocol_tag(cfg, sel.groupNames);
outDir = fullfile(cfg.outDir, cfg.tag, ptag);
if ~exist(outDir,'dir'), mkdir(outDir); end
if cfg.saveModels
    modelDir = fullfile(outDir,'models');
    if ~exist(modelDir,'dir'), mkdir(modelDir); end
end

Pte      = zeros(size(Fte,1), 8);
selCfg   = zeros(1,8);
valRmse  = zeros(1,8);
tic;
for k = 1:8
    cols = select_cols(k, cfg, nIn);
    dtr  = make_dmatrix(np, xgb, Ftr(:,cols), Ytr(:,k));
    dva  = make_dmatrix(np, xgb, Fva(:,cols), Yva(:,k));

    best = inf; bestH = hpList(1); bestBooster = [];
    for h = hpList
        b  = train_booster(xgb, dtr, hp(h));
        pv = predict_vec(b, dva);
        r  = sqrt(mean((pv - Yva(:,k)).^2));
        if r < best
            best = r; bestH = h; bestBooster = b;
        end
    end
    selCfg(k) = bestH; valRmse(k) = best;

    if cfg.retrain
        dall   = make_dmatrix(np, xgb, [Ftr(:,cols); Fva(:,cols)], [Ytr(:,k); Yva(:,k)]);
        finalB = train_booster(xgb, dall, hp(bestH));
    else
        finalB = bestBooster;
    end

    dte = make_dmatrix(np, xgb, Fte(:,cols), []);
    Pte(:,k) = predict_vec(finalB, dte);

    if cfg.saveModels
        finalB.save_model(char(fullfile(modelDir, ['xgb_' names{k} '.json'])));
    end
end
fprintf('  [%s | %s] 8 channels done in %.1f s\n', cfg.tag, ptag, toc);

P = mat2cell(Pte, nTe(:), 8);
[overall, cond] = sg4_metrics(P, sel.test, nrm);

writetable(overall, fullfile(outDir,'overall_test_metrics.csv'));
writetable(cond,    fullfile(outDir,'condition_test_metrics.csv'));
save(fullfile(outDir,'results.mat'), 'P', 'overall', 'cond', 'cfg', 'selCfg', 'valRmse');
if cfg.makePlot
    pidx = pick_plot_index(sel.test, cfg);
    sg4_plot_example(sel.test, P, pidx, sprintf('XGBoost reconstruction (%s, %s)', cfg.tag, ptag), ...
        fullfile(outDir,'xgboost_reconstruction_example.png'), isfield(cfg,'showPlot') && cfg.showPlot);
end

R = struct('overall', overall, 'cond', cond, 'tag', cfg.tag, 'protocol', ptag);
end

% ------------------------------------------------------------------------
function ptag = protocol_tag(cfg, groupNames)
if strcmp(cfg.protocol, 'standard')
    ptag = 'standard';
else
    ptag = ['loeo_' char(groupNames(cfg.holdoutGroup))];
end
end

function u = input_matrix(S, i)
% Known inputs [V theta Pm vf], plus kPg when the dataset provides it.
u = [S.V{i}(:), S.theta{i}(:), S.Pm{i}(:), S.vf{i}(:)];
if isfield(S, 'kPg')
    u = [u, S.kPg{i}(:)];
end
end

function [F, Y] = stack_samples(S)
% Columns of F: 1-8 noisy [delta omega eq' ed' id iq Pg Qg], 9..8+nIn known inputs, last = time.
n   = numel(S.X_noisy);
len = cellfun(@(a) size(a,1), S.X_noisy);
nIn = size(input_matrix(S,1), 2);
F = zeros(sum(len), 8 + nIn + 1);
Y = zeros(sum(len), 8);
r = 0;
for i = 1:n
    idx = r + (1:len(i));
    F(idx,:) = [S.X_noisy{i}, S.Y_noisy{i}, input_matrix(S,i), S.t{i}(:)];
    Y(idx,:) = [S.X_clean{i}, S.Y_clean{i}];
    r = r + len(i);
end
end

function cols = select_cols(k, cfg, nIn)
if strcmp(cfg.channelMode, 'all'), cols = 1:8; else, cols = k; end
if cfg.useInputs, cols = [cols 9:(8+nIn)]; end
if cfg.useTime,   cols = [cols 9+nIn];     end
end

function dm = make_dmatrix(np, xgb, X, y)
Xn = np.array(X);
if isempty(y)
    dm = xgb.DMatrix(Xn);
else
    yn = np.ravel(np.array(y));
    dm = xgb.DMatrix(Xn, pyargs('label', yn));
end
end

function booster = train_booster(xgb, dtrain, hp)
params = py.dict(pyargs( ...
    'objective',        'reg:squarederror', ...
    'eval_metric',      'rmse', ...
    'max_depth',        int32(hp.depth), ...
    'eta',              hp.eta, ...
    'subsample',        0.8, ...
    'colsample_bytree', 0.8, ...
    'min_child_weight', 5, ...
    'reg_lambda',       1, ...
    'tree_method',      'hist', ...
    'seed',             int32(42)));
booster = xgb.train(params, dtrain, int32(hp.rounds));
end

function p = predict_vec(booster, dm)
raw = booster.predict(dm);
try
    p = double(raw);
catch
    p = double(py.array.array('d', raw.astype('float64').tolist()));
end
p = p(:);
end

% ------------------------------------------------------------------------
% METRICS
% ------------------------------------------------------------------------
function [overall, cond] = sg4_metrics(P, S, nrm)
%SG4_METRICS Overall and condition-wise reconstruction metrics.
%   P   : cell array of predicted trajectories (n x 8), order
%         [delta omega eq_prime ed_prime id iq Pg Qg]
%   S   : trajectory struct (e.g. sel.test) with clean / noisy data and labels
%   nrm : sel.nrm from sg4_select (training-only std and range, fixed for all protocols)
%
%   Columns: RMSE, MAE, NRMSE_std, NRMSE_range, NoisyRMSE (noisy input used
%   as estimate, same trajectories) and Improvement_pct.

C = cellfun(@(a,b) [a b], S.X_clean, S.Y_clean, 'UniformOutput', false);
N = cellfun(@(a,b) [a b], S.X_noisy, S.Y_noisy, 'UniformOutput', false);
n = numel(P);

overall = to_table(calc(P, C, N, true(n,1), nrm), nrm.names);

rows = {};
for g = unique(S.group(:))'
    for nt = unique(S.noiseType(:))'
        for nl = unique(S.noiseLevel(:))'
            idx = S.group(:) == g & S.noiseType(:) == nt & S.noiseLevel(:) == nl;
            if ~any(idx), continue; end
            T = to_table(calc(P, C, N, idx, nrm), nrm.names);
            H = table(repmat(S.groupNames(g),8,1), repmat(nt,8,1), repmat(nl,8,1), ...
                'VariableNames', {'Experiment','NoiseType','NoiseLevel'});
            rows{end+1} = [H, T]; %#ok<AGROW>
        end
    end
end
cond = vertcat(rows{:});
end

function m = calc(P, C, N, idx, nrm)
Pa = vertcat(P{idx});
Ca = vertcat(C{idx});
Na = vertcat(N{idx});
E  = Pa - Ca;
m.rmse  = sqrt(mean(E.^2, 1));
m.mae   = mean(abs(E), 1);
m.nstd  = m.rmse ./ nrm.std;
m.nrng  = m.rmse ./ nrm.range;
m.noisy = sqrt(mean((Na - Ca).^2, 1));
m.impr  = 100 * (1 - m.rmse ./ m.noisy);
end

function T = to_table(m, names)
T = table(string(names(:)), m.rmse(:), m.mae(:), m.nstd(:), m.nrng(:), m.noisy(:), m.impr(:), ...
    'VariableNames', {'Channel','RMSE','MAE','NRMSE_std','NRMSE_range','NoisyRMSE','Improvement_pct'});
end

% ------------------------------------------------------------------------
% PLOTTING
% ------------------------------------------------------------------------
function sg4_plot_example(S, P, idx, ttl, pngPath, showFig)
%SG4_PLOT_EXAMPLE Plot noisy / clean / reconstructed signals for trajectory idx.
%   showFig = true  -> the figure stays open on screen (and the PNG is saved)
%   showFig = false -> figure is created hidden, saved, then closed
if nargin < 6, showFig = false; end
names = {'delta','omega','eq_prime','ed_prime','id','iq','Pg','Qg'};
Cl = [S.X_clean{idx}, S.Y_clean{idx}];
Nz = [S.X_noisy{idx}, S.Y_noisy{idx}];
t  = S.t{idx}(:);

vis = 'off';
if showFig, vis = 'on'; end
fig = figure('Visible', vis, 'Color', 'w', 'Position', [60 60 1500 1050], 'Name', ttl);
tl = tiledlayout(4,2,'TileSpacing','compact');
title(tl, sprintf('%s  |  %s, %s %.0f%%', ttl, S.experiment(idx), S.noiseType(idx), 100*S.noiseLevel(idx)), ...
    'Interpreter','none');
for k = 1:8
    nexttile;
    plot(t, Nz(:,k), 'Color', [0.75 0.75 0.75]); hold on;
    plot(t, Cl(:,k), 'k', 'LineWidth', 1.6);
    plot(t, P{idx}(:,k), 'r', 'LineWidth', 1.0);
    title(names{k}, 'Interpreter','none');
    xlabel('Time (s)'); grid on;
    if k == 1
        legend({'Noisy measurement','Clean reference','Reconstruction'}, 'Location','northeast');
    end
end
exportgraphics(fig, pngPath, 'Resolution', 150);
if showFig
    drawnow;
else
    close(fig);
end
end

function idx = pick_plot_index(S, cfg)
%PICK_PLOT_INDEX First test trajectory matching cfg.plotExperiment / plotNoiseType / plotNoiseLevel.
%   Falls back to trajectory 1 when a field is empty or nothing matches.
m = true(numel(S.group), 1);
if isfield(cfg,'plotExperiment') && ~isempty(cfg.plotExperiment)
    m = m & contains(lower(S.experiment(:)), lower(string(cfg.plotExperiment)));
end
if isfield(cfg,'plotNoiseType') && ~isempty(cfg.plotNoiseType)
    m = m & lower(S.noiseType(:)) == lower(string(cfg.plotNoiseType));
end
if isfield(cfg,'plotNoiseLevel') && ~isempty(cfg.plotNoiseLevel)
    m = m & abs(S.noiseLevel(:) - cfg.plotNoiseLevel) < 1e-9;
end
idx = find(m, 1);
if isempty(idx), idx = 1; end
end