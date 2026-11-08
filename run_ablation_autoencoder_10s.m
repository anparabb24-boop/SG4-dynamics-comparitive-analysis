%% RUN_AUTOENCODER_ABLATION_V2  (single self-contained file, plots shown)
% Loads the dataset written by BUILD_DATASET_10s.m / BUILD_DATASET_20s.m and
% trains the windowed denoising autoencoder with ablation + leave-one-
% excitation-group-out evaluation. No other project files are needed: the
% data converter, splitting, noise selection, metrics, plotting and the
% autoencoder itself are all local functions at the bottom of this file.
%
% Needs: MATLAB R2024a+ with Deep Learning Toolbox (trainnet, minibatchpredict).
% Run from the project root (the folder that contains data/ and results/).
%
% Expected dataset layout (as built by BUILD_DATASET_*s.m):
%   dataset.X_noisy / X_clean / Y_noisy / Y_clean : [time x 4 x trajectories]
%   dataset.U            : [time x inputs x trajectories], order in dataset.inputNames
%   dataset.time         : time vector
%   dataset.metadata     : table with Experiment, NoiseType, NoiseLevelPercent, Realization, Seed
%   dataset.trainIdx / validationIdx / testIdx
%
% Variants:
%   withTime_all_inputs    : absolute time added as input (the leak-prone setup)
%   noTime_all_inputs      : baseline (no time)
%   noTime_all_noInputs    : noisy channels only
%   noTime_single_inputs   : one network per channel, own noisy channel + known inputs
%   noTime_all_inputs_W16  : 0.16 s context
%   noTime_all_inputs_W128 : 1.28 s context
% Protocols: 'standard' and 'loeo' for groups 1..4
%   (1 voltage = baseline + A_voltage, 2 mechanical, 3 excitation, 4 combined).
%
% TIP: smoke test first with EPOCHS = 2, RUN_LOEO = false, VARIANTS_TO_RUN = 2.

clear; clc; close all;

%% ---- settings -----------------------------------------------------------
EPOCHS          = 10;
RUN_LOEO        = true;
VARIANTS_TO_RUN = 1:6;

% Plots: the reconstruction figure of every variant (standard split) is saved as
% PNG and, with SHOW_PLOTS = true, also kept open on screen.
SHOW_PLOTS      = false;
% Which TEST trajectory to plot ('' / [] = first one, a clean 1 % Gaussian case):
PLOT_EXPERIMENT = 'combined';       % e.g. 'mechanical', 'combined', 'baseline'
PLOT_NOISE_TYPE = 'uniform';       % e.g. 'impulsive'
PLOT_NOISE_LEVEL = 0.1;      % e.g. 0.10

% Dataset file. '' = auto-detect (tries the paths below in order).
datasetFile = '';
candidateFiles = { ...
    fullfile('data','dataset_10s','sg4_dataset_10s.mat'), ...
    fullfile('data','dataset_20s','sg4_dataset_20s.mat')};

%% ---- noise selection ----------------------------------------------------
% {} or [] = use everything (all noise types, all levels).
% Examples:  TRAIN_NOISE_TYPES = {'gaussian'};   TRAIN_NOISE_LEVELS = [0.01 0.05];
% Levels are fractions (0.01 = 1 %). The printed "[noise] train:" line shows
% exactly what each run trained on.
TRAIN_NOISE_TYPES  = {'uniform'};
TRAIN_NOISE_LEVELS = 0.1;
TEST_NOISE_TYPES   = {'uniform'};
TEST_NOISE_LEVELS  = 0.1;

%% ---- checks and data ----------------------------------------------------
if ~exist('trainnet','file') || ~exist('minibatchpredict','file')
    error('Deep Learning Toolbox (trainnet / minibatchpredict) not found.');
end

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
outRoot = fullfile('results', sprintf('autoencoder_ablation_%ds', durationSec));

tagParts = "";
if ~isempty(TRAIN_NOISE_TYPES),  tagParts = tagParts + "_trainT-" + strjoin(string(TRAIN_NOISE_TYPES), '+'); end
if ~isempty(TRAIN_NOISE_LEVELS), tagParts = tagParts + "_trainL-" + strjoin(string(TRAIN_NOISE_LEVELS*100), '+') + "pct"; end
if ~isempty(TEST_NOISE_TYPES),   tagParts = tagParts + "_testT-"  + strjoin(string(TEST_NOISE_TYPES), '+'); end
if ~isempty(TEST_NOISE_LEVELS),  tagParts = tagParts + "_testL-"  + strjoin(string(TEST_NOISE_LEVELS*100), '+') + "pct"; end
outRoot = [outRoot char(tagParts)];

variants = {
    % tag                       useTime  channelMode  useInputs  window
    'withTime_all_inputs',      true,    'all',       true,      64
    'noTime_all_inputs',        false,   'all',       true,      64
    'noTime_all_noInputs',      false,   'all',       false,     64
    'noTime_single_inputs',     false,   'single',    true,      64
    'noTime_all_inputs_W16',    false,   'all',       true,      16
    'noTime_all_inputs_W128',   false,   'all',       true,      128
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
        cfg = ae_default_cfg();
        cfg.tag          = variants{v,1};
        cfg.useTime      = variants{v,2};
        cfg.channelMode  = variants{v,3};
        cfg.useInputs    = variants{v,4};
        cfg.window       = variants{v,5};
        cfg.trainStride  = max(1, cfg.window/8);
        cfg.predStride   = max(1, cfg.window/16);
        cfg.protocol     = protocols{p,1};
        cfg.holdoutGroup = protocols{p,2};
        cfg.trainNoiseTypes  = TRAIN_NOISE_TYPES;
        cfg.trainNoiseLevels = TRAIN_NOISE_LEVELS;
        cfg.testNoiseTypes   = TEST_NOISE_TYPES;
        cfg.testNoiseLevels  = TEST_NOISE_LEVELS;
        cfg.epochs       = EPOCHS;
        cfg.verbose      = false;
        cfg.saveModels   = false;
        cfg.makePlot     = (p == 1);
        cfg.showPlot     = SHOW_PLOTS;
        cfg.plotExperiment = PLOT_EXPERIMENT;
        cfg.plotNoiseType  = PLOT_NOISE_TYPE;
        cfg.plotNoiseLevel = PLOT_NOISE_LEVEL;
        cfg.outDir       = outRoot;

        try
            R = ae_run_experiment(cfg, modelData);
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
fprintf(' * withTime vs noTime: effect of the absolute-time shortcut.\n');
fprintf(' * W16 / W64 / W128: effect of temporal context.\n');
fprintf(' * loeo_* rows are the honest generalisation numbers.\n');
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
% AUTOENCODER CONFIG
% ------------------------------------------------------------------------
function cfg = ae_default_cfg()
%AE_DEFAULT_CFG Default configuration of the windowed denoising autoencoder.
cfg.tag          = 'ae_baseline';
cfg.protocol     = 'standard';     % 'standard' | 'loeo'
cfg.holdoutGroup = 0;              % 1..4 when protocol = 'loeo'
cfg.trainNoiseTypes  = {};         % {} = all, e.g. {'gaussian'}
cfg.trainNoiseLevels = [];         % [] = all, e.g. [0.01 0.05]
cfg.testNoiseTypes   = {};
cfg.testNoiseLevels  = [];
cfg.useTime      = false;          % absolute time as an extra input (leak-prone)
cfg.useInputs    = true;           % known inputs V, theta, Pm, vf
cfg.channelMode  = 'all';          % 'all' = 8 noisy channels -> 8 clean; 'single' = one net per channel
cfg.window       = 64;             % samples per window (0.64 s)
cfg.trainStride  = 8;              % window stride when training
cfg.predStride   = 4;              % window stride when reconstructing (overlap-averaged)
cfg.encoder      = [512 256 64];   % hidden sizes; last one is the latent layer
cfg.epochs       = 40;
cfg.miniBatch    = 256;
cfg.learnRate    = 1e-3;
cfg.patience     = 6;              % validation patience (in validation checks)
cfg.seed         = 42;
cfg.verbose      = false;
cfg.saveModels   = false;
cfg.makePlot     = false;
cfg.showPlot     = false;         % true = keep the figure open on screen
cfg.plotExperiment = '';         % e.g. 'mechanical' (substring match), '' = first test trajectory
cfg.plotNoiseType  = '';         % e.g. 'impulsive'
cfg.plotNoiseLevel = [];         % e.g. 0.10
cfg.outDir       = fullfile('results','autoencoder_10s');
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
% AUTOENCODER TRAINING AND RECONSTRUCTION
% ------------------------------------------------------------------------
function R = ae_run_experiment(cfg, modelData)
%AE_RUN_EXPERIMENT Train and evaluate a windowed denoising autoencoder.
%   Input  : window of noisy channels (+ optional known inputs, + optional time)
%   Output : window of clean channels
%   Windows are overlap-averaged to rebuild full 1001-sample trajectories.
%   Requires Deep Learning Toolbox (trainnet, minibatchpredict).

rng(cfg.seed);
sel = sg4_select(modelData, cfg);
nrm = sel.nrm;
nz  = make_norm(sel.train);

if strcmp(cfg.channelMode, 'all')
groups = {1:8};
else
groups = num2cell(1:8);
end

ptag   = protocol_tag(cfg, sel.groupNames);
outDir = fullfile(cfg.outDir, cfg.tag, ptag);
if ~exist(outDir,'dir'), mkdir(outDir); end

nTe  = numel(sel.test.X_noisy);
P    = cellfun(@(a) zeros(size(a,1),8), sel.test.X_noisy, 'UniformOutput', false);
nets = cell(numel(groups),1);

for gi = 1:numel(groups)
chs = groups{gi};
tic;
[Xtr, Ytr] = make_windows(sel.train, cfg, nz, chs, cfg.trainStride);
[Xva, Yva] = make_windows(sel.val,   cfg, nz, chs, 2*cfg.trainStride);

layers  = build_layers(size(Xtr,2), size(Ytr,2), cfg.encoder);
valFreq = max(1, floor(size(Xtr,1) / cfg.miniBatch));

opts = trainingOptions('adam', ...
'InitialLearnRate',    cfg.learnRate, ...
'MaxEpochs',           cfg.epochs, ...
'MiniBatchSize',       cfg.miniBatch, ...
'Shuffle',             'every-epoch', ...
'LearnRateSchedule',   'piecewise', ...
'LearnRateDropFactor', 0.5, ...
'LearnRateDropPeriod', 10, ...
'ValidationData',      {Xva, Yva}, ...
'ValidationFrequency', valFreq, ...
'ValidationPatience',  cfg.patience, ...
'OutputNetwork',       'best-validation', ...
'ExecutionEnvironment','cpu', ...
'Plots',               'none', ...
'Verbose',             cfg.verbose, ...
'VerboseFrequency',    valFreq);

net = trainnet(Xtr, Ytr, layers, "mse", opts);
nets{gi} = net;

for i = 1:nTe
Fin = make_input_matrix(sel.test, i, cfg, nz, chs);
Zp  = reconstruct(net, Fin, cfg.window, cfg.predStride, numel(chs));
P{i}(:,chs) = Zp .* nz.cleanStd(chs) + nz.cleanMean(chs);
end
fprintf('  [%s | %s] network %d/%d (channels %s) done in %.1f s, %d train windows\n', ...
cfg.tag, ptag, gi, numel(groups), mat2str(chs), toc, size(Xtr,1));
end

[overall, cond] = sg4_metrics(P, sel.test, nrm);
writetable(overall, fullfile(outDir,'overall_test_metrics.csv'));
writetable(cond,    fullfile(outDir,'condition_test_metrics.csv'));
save(fullfile(outDir,'results.mat'), 'P', 'overall', 'cond', 'cfg');
if cfg.saveModels
save(fullfile(outDir,'ae_networks.mat'), 'nets', 'nz', 'cfg');
end
if cfg.makePlot
pidx = pick_plot_index(sel.test, cfg);
sg4_plot_example(sel.test, P, pidx, sprintf('Autoencoder reconstruction (%s, %s)', cfg.tag, ptag), ...
fullfile(outDir,'autoencoder_reconstruction_example.png'), isfield(cfg,'showPlot') && cfg.showPlot);
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

function nz = make_norm(S)
% Normalisation from the selected TRAINING set only.
noisy = cellfun(@(a,b) [a b], S.X_noisy, S.Y_noisy, 'UniformOutput', false);
noisy = vertcat(noisy{:});
clean = cellfun(@(a,b) [a b], S.X_clean, S.Y_clean, 'UniformOutput', false);
clean = vertcat(clean{:});
u = cell(numel(S.V), 1);
for i = 1:numel(S.V)
u{i} = input_matrix(S, i);
end
u = vertcat(u{:});

nz.noisyMean = mean(noisy,1);  nz.noisyStd = std(noisy,0,1);
nz.cleanMean = mean(clean,1);  nz.cleanStd = std(clean,0,1);
nz.inMean    = mean(u,1);      nz.inStd    = std(u,0,1);
nz.noisyStd(nz.noisyStd < 1e-12) = 1;
nz.cleanStd(nz.cleanStd < 1e-12) = 1;
nz.inStd(nz.inStd       < 1e-12) = 1;
end

function F = make_input_matrix(S, i, cfg, nz, chs)
noisy = [S.X_noisy{i}, S.Y_noisy{i}];
F = (noisy(:,chs) - nz.noisyMean(chs)) ./ nz.noisyStd(chs);
if cfg.useInputs
u = input_matrix(S, i);
F = [F, (u - nz.inMean) ./ nz.inStd];
end
if cfg.useTime
F = [F, S.t{i}(:) / max(S.t{i}(:))];
end
end

function u = input_matrix(S, i)
% Known inputs [V theta Pm vf], plus kPg when the dataset provides it.
u = [S.V{i}(:), S.theta{i}(:), S.Pm{i}(:), S.vf{i}(:)];
if isfield(S, 'kPg')
u = [u, S.kPg{i}(:)];
end
end

function st = win_starts(n, W, stride)
st = 1:stride:(n - W + 1);
if st(end) ~= n - W + 1
st(end+1) = n - W + 1;
end
end

function [X, Y] = make_windows(S, cfg, nz, chs, stride)
W = cfg.window;
n = numel(S.X_noisy);
cnt = 0;
for i = 1:n
cnt = cnt + numel(win_starts(size(S.X_noisy{i},1), W, stride));
end
K = numel(chs) + size(input_matrix(S,1),2)*cfg.useInputs + 1*cfg.useTime;
X = zeros(cnt, W*K,         'single');
Y = zeros(cnt, W*numel(chs),'single');
r = 0;
for i = 1:n
F = make_input_matrix(S, i, cfg, nz, chs);
clean = [S.X_clean{i}, S.Y_clean{i}];
T = (clean(:,chs) - nz.cleanMean(chs)) ./ nz.cleanStd(chs);
st = win_starts(size(F,1), W, stride);
for j = 1:numel(st)
a = st(j); b = a + W - 1;
r = r + 1;
X(r,:) = reshape(F(a:b,:), 1, []);
Y(r,:) = reshape(T(a:b,:), 1, []);
end
end
end

function layers = build_layers(inDim, outDim, enc)
layers = featureInputLayer(inDim, 'Name', 'in');
for h = enc
layers = [layers; fullyConnectedLayer(h); reluLayer]; %#ok<AGROW>
end
for h = fliplr(enc(1:end-1))
layers = [layers; fullyConnectedLayer(h); reluLayer]; %#ok<AGROW>
end
layers = [layers; fullyConnectedLayer(outDim, 'Name', 'out')];
end

function Z = reconstruct(net, F, W, stride, c)
n  = size(F,1);
K  = size(F,2);
st = win_starts(n, W, stride);
X  = zeros(numel(st), W*K, 'single');
for j = 1:numel(st)
a = st(j); b = a + W - 1;
X(j,:) = reshape(F(a:b,:), 1, []);
end
Yw  = minibatchpredict(net, X, 'MiniBatchSize', 1024);
Z   = zeros(n, c);
cnt = zeros(n, 1);
for j = 1:numel(st)
a = st(j); b = a + W - 1;
Z(a:b,:) = Z(a:b,:) + double(reshape(Yw(j,:), W, c));
cnt(a:b) = cnt(a:b) + 1;
end
Z = Z ./ cnt;
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

