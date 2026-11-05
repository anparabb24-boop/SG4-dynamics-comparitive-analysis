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
%   Groups come from metadata.experiment:
%       1 voltage (baseline + A_voltage), 2 mechanical, 3 excitation, 4 combined
%
%   Output sel has .train .val .test (cell arrays + labels), .nrm
%   (training-only std/range of the 8 clean channels from the standard
%   training split, fixed for all protocols) and .groupNames.

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
sel.nrm        = P.nrm;
sel.groupNames = P.groupNames;
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
out = struct();
for f = fields
    out.(f{1}) = S.(f{1})(mask);
end
out.groupNames = S.groupNames;
end
