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
