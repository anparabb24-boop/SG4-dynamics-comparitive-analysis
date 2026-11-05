%% RUN_CLASSICAL_BASELINES_10S
% Training-free reference points for the standard test split, plus a data check.
%
%  1. Classical filters per channel (movmean, movmedian, Savitzky-Golay);
%     window length chosen per channel on the validation set.
%  2. "Lookup" reference: for each excitation group, the mean clean training
%     curve. It uses the experiment label, so it is NOT a fair denoiser. It
%     shows how much of the standard split is solvable by memorising the curves.
%  3. Data check: how different are the clean curves of baseline vs A_voltage,
%     and do clean curves vary within a group?
%
% Uses the file written by prepare_modeling_data_10s.m. Run from the project root.

clear; clc; close all;

Mf = load(fullfile('data','modeling_data_10s','model_data_10s.mat'));
modelData = Mf.modelData;

sel = sg4_select(modelData, struct('protocol','standard','holdoutGroup',0));
tr = sel.train;  va = sel.val;  te = sel.test;
nrm = sel.nrm;   groupNames = sel.groupNames;

fprintf('Trajectories: train %d, validation %d, test %d\n', ...
    numel(tr.group), numel(va.group), numel(te.group));
for g = 1:4
    fprintf('  group %d (%-10s): train %3d | val %3d | test %3d\n', g, groupNames(g), ...
        sum(tr.group==g), sum(va.group==g), sum(te.group==g));
end
fprintf('  sanity: training std of omega = %.4g (report value 3.63e-05)\n', nrm.std(2));

outDir = fullfile('results','classical_baselines_10s');
if ~exist(outDir,'dir'), mkdir(outDir); end

%% ---- 3. data check ------------------------------------------------------
fprintf('\n=== Data check: clean-curve spread within groups (training set) ===\n');
Ctr = cellfun(@(a,b) [a b], tr.X_clean, tr.Y_clean, 'UniformOutput', false);
curves = cell(1,4);
for g = 1:4
    idx = find(tr.group == g);
    A = cat(3, Ctr{idx});                    % n x 8 x trajectories
    curves{g} = mean(A, 3);
    dev = max(abs(A - curves{g}), [], 'all');
    fprintf('  group %d (%-10s): max |clean_i - mean| = %.3g\n', g, groupNames(g), dev);
end
ib = find(tr.group == 1 & tr.experiment == "baseline", 1);
ia = find(tr.group == 1 & tr.experiment ~= "baseline", 1);
if ~isempty(ib) && ~isempty(ia)
    fprintf('  baseline vs %s: max |clean difference| = %.3g\n', tr.experiment(ia), ...
        max(abs(Ctr{ib} - Ctr{ia}), [], 'all'));
end
fprintf('  (values near 0 => identical clean curves; baseline and A are then one distinct scenario)\n');
for g = 2:4
    fprintf('  distance group1 vs group%d: %.3g\n', g, max(abs(curves{1} - curves{g}), [], 'all'));
end

%% ---- 1. classical filters ----------------------------------------------
methods = {'movmean','movmedian','sgolay'};
wins    = [5 11 21 41];

Nval = cellfun(@(a,b) [a b], va.X_noisy, va.Y_noisy, 'UniformOutput', false);
Cval = cellfun(@(a,b) [a b], va.X_clean, va.Y_clean, 'UniformOutput', false);

bestW = zeros(numel(methods), 8);
for m = 1:numel(methods)
    sse = zeros(numel(wins), 8);
    for i = 1:numel(Nval)
        for wi = 1:numel(wins)
            for k = 1:8
                z = apply_filter(Nval{i}(:,k), methods{m}, wins(wi));
                sse(wi,k) = sse(wi,k) + sum((z - Cval{i}(:,k)).^2);
            end
        end
    end
    [~, wbest] = min(sse, [], 1);
    bestW(m,:) = wins(wbest);
end

Ntest = cellfun(@(a,b) [a b], te.X_noisy, te.Y_noisy, 'UniformOutput', false);
allTables = table();
for m = 1:numel(methods)
    P = cell(numel(Ntest),1);
    for i = 1:numel(Ntest)
        Z = zeros(size(Ntest{i}));
        for k = 1:8
            Z(:,k) = apply_filter(Ntest{i}(:,k), methods{m}, bestW(m,k));
        end
        P{i} = Z;
    end
    [ov, cd] = sg4_metrics(P, te, nrm);
    ov.Method = repmat(string(methods{m}), height(ov), 1);
    allTables = [allTables; ov]; %#ok<AGROW>
    writetable(cd, fullfile(outDir, ['condition_' methods{m} '.csv']));
    fprintf('\n=== %s (best window per channel: %s) ===\n', methods{m}, mat2str(bestW(m,:)));
    disp(ov(:, {'Channel','RMSE','NRMSE_std','NoisyRMSE','Improvement_pct'}));
end

%% ---- 2. lookup reference -----------------------------------------------
P = cell(numel(te.group),1);
for i = 1:numel(P)
    P{i} = curves{te.group(i)};
end
[ov, cd] = sg4_metrics(P, te, nrm);
ov.Method = repmat("lookup_oracle", height(ov), 1);
allTables = [allTables; ov];
writetable(cd, fullfile(outDir, 'condition_lookup_oracle.csv'));
fprintf('\n=== lookup oracle (uses experiment label; memorises the clean curves) ===\n');
disp(ov(:, {'Channel','RMSE','NRMSE_std','NoisyRMSE','Improvement_pct'}));
fprintf(['If this is ~0, the standard split cannot distinguish denoising from memorisation:\n' ...
         'XGBoost''s ~3e-4 RMSE sits between the classical filters and a pure lookup.\n']);

writetable(allTables, fullfile(outDir, 'classical_baselines_overall.csv'));
fprintf('\nSaved to %s\n', outDir);

%% ---- helpers ------------------------------------------------------------
function z = apply_filter(x, method, w)
switch method
    case 'movmean',   z = movmean(x, w);
    case 'movmedian', z = movmedian(x, w);
    case 'sgolay',    z = smoothdata(x, 'sgolay', w);
end
end
