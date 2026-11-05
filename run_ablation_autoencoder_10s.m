%% RUN_ABLATION_AUTOENCODER_10S
% Ablation + leave-one-excitation-group-out evaluation for the windowed
% denoising autoencoder.
%
% Variants:
%   withTime_all_inputs    : absolute time added as input (the leak-prone setup)
%   noTime_all_inputs      : baseline (no time)
%   noTime_all_noInputs    : noisy channels only
%   noTime_single_inputs   : one network per channel, own noisy channel + known inputs
%   noTime_all_inputs_W16  : 0.16 s context
%   noTime_all_inputs_W128 : 1.28 s context
%
% Protocols: 'standard' and 'loeo' for groups 1..4.
% Run from the project root. Needs Deep Learning Toolbox.
%
% TIP: do a smoke test first with EPOCHS = 2, RUN_LOEO = false, VARIANTS_TO_RUN = 2.

clear; clc; close all;

%% ---- settings -----------------------------------------------------------
EPOCHS          = 25;
RUN_LOEO        = true;
VARIANTS_TO_RUN = 1:6;
outRoot         = fullfile('results','ablation_10s','autoencoder');

if ~exist('trainnet','file') || ~exist('minibatchpredict','file')
    error('Deep Learning Toolbox (trainnet / minibatchpredict) not found.');
end

Mf = load(fullfile('data','modeling_data_10s','model_data_10s.mat'));
modelData = Mf.modelData;

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
        cfg.epochs       = EPOCHS;
        cfg.verbose      = false;
        cfg.saveModels   = false;
        cfg.makePlot     = (p == 1);
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
