function cfg = ae_default_cfg()
%AE_DEFAULT_CFG Default configuration of the windowed denoising autoencoder.
cfg.tag          = 'ae_baseline';
cfg.protocol     = 'standard';     % 'standard' | 'loeo'
cfg.holdoutGroup = 0;              % 1..4 when protocol = 'loeo'
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
cfg.outDir       = fullfile('results','autoencoder_10s');
end
