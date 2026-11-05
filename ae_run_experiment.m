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
    sg4_plot_example(sel.test, P, 1, sprintf('Autoencoder reconstruction (%s, %s)', cfg.tag, ptag), ...
        fullfile(outDir,'autoencoder_reconstruction_example.png'));
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
u = cellfun(@(v,th,pm,vf) [v(:) th(:) pm(:) vf(:)], S.V, S.theta, S.Pm, S.vf, 'UniformOutput', false);
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
    u = [S.V{i}(:), S.theta{i}(:), S.Pm{i}(:), S.vf{i}(:)];
    F = [F, (u - nz.inMean) ./ nz.inStd];
end
if cfg.useTime
    F = [F, S.t{i}(:) / 10];
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
K = numel(chs) + 4*cfg.useInputs + 1*cfg.useTime;
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
