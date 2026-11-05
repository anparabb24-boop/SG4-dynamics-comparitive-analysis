%% RUN_ABLATION_XGBOOST_10S
% Ablation + leave-one-excitation-group-out evaluation for XGBoost.
%
% Variants (all pointwise):
%   full_withTime_all_inputs : reproduces the original setup (13 features, but no tuning)
%   noTime_all_inputs        : removes the time feature
%   noTime_all_noInputs      : only the 8 noisy channels (no t, no known inputs)
%   noTime_single_inputs     : target's own noisy channel + known inputs
%   noTime_single_noInputs   : target's own noisy channel only (floor)
%
% Protocols: 'standard' (unseen noise realizations) and 'loeo' for each group
%   1 voltage (baseline + A), 2 mechanical, 3 excitation, 4 combined.
%
% Run from the project root. Needs the Python env used earlier.

clear; clc; close all;

%% ---- settings -----------------------------------------------------------
RUN_LOEO       = true;       % false = standard split only (quick smoke test)
TUNE           = false;      % false = fixed config 3 for speed
RETRAIN        = false;      % false = train on train split only
VARIANTS_TO_RUN = 1:5;
outRoot        = fullfile('results','ablation_10s','xgboost');

%% ---- python environment -------------------------------------------------
pe = pyenv;
if pe.Status == "NotLoaded"
    pyenv(Version = fullfile(getenv('HOME'), 'xgboost_env312', 'bin', 'python'));
end
xgbCheck = py.importlib.import_module('xgboost');
pe = pyenv;
fprintf('XGBoost %s, Python %s\n', char(py.getattr(xgbCheck, '__version__')), char(pe.Version));

%% ---- data ---------------------------------------------------------------
Mf = load(fullfile('data','modeling_data_10s','model_data_10s.mat'));
modelData = Mf.modelData;

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
        cfg.tune         = TUNE;
        cfg.retrain      = RETRAIN;
        cfg.saveModels   = false;
        cfg.makePlot     = (p == 1);
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