%% refit_sg3_to_sg4.m
% Refit the reduced SG3 model to SG4 "measurements" at the NOMINAL T'd0 = 6.5 s.
%
% Question answered: after taking the T'd0 -> 0 boundary, can re-tuning the
% remaining SG3 parameters recover the SG4 response (Pg, Qg in particular),
% or is a structural correction (the paper's data-driven SINDy terms) needed?
%
% Design choices
%   * Target data: SG4 simulated with nominal parameters (T'd0 = 6.5 s),
%     same voltage disturbance and time grid as the baseline run.
%   * Fitted parameters (log-multipliers of nominal): H, D, T'q0, xd, xq, x'q.
%   * x'd is NOT fitted: with eqp algebraic, id = (vf - V cos(delta-theta))/xd,
%     so x'd cancels out of every output (non-identifiable from outputs).
%   * SG3 initial state is recomputed as the SG3 equilibrium for each trial
%     parameter set (V, theta, Pm0, vf0 fixed), so the pre-fault operating
%     point is parameter-consistent and cannot be fitted by free initial
%     conditions. If the parameters move the operating point away from the
%     SG4 data, the residual penalises it.
%   * Two fits: (A) Pg only, (B) Pg + Qg. Residuals are scaled by the RMS
%     deviation of each SG4 signal from its pre-fault value.
%   * Multistart (8 starts) with bounds 0.1x to 10x nominal.
%
% Requires: results/sg4_disturbance_baseline.mat, src/models/sg4_rhs.m,
%           src/models/sg4_outputs.m.  Uses lsqnonlin if available,
%           otherwise falls back to fminsearch.

clear; clc; close all;

%% Paths
root = fileparts(fileparts(mfilename('fullpath')));
if isempty(root) || ~exist(fullfile(root,'src'),'dir')
    root = pwd;
end
addpath(genpath(fullfile(root,'src')));
if ~exist(fullfile(root,'results','figures'),'dir'), mkdir(fullfile(root,'results','figures')); end
if ~exist(fullfile(root,'results','tables'),'dir'),  mkdir(fullfile(root,'results','tables'));  end

%% Baseline data and inputs
S   = load(fullfile(root,'results','sg4_disturbance_baseline.mat'));
p0  = S.p;
x0  = S.x0(:);
Pm0 = S.Pm0;
vf0 = S.vf0;
t   = S.t_all(:);
N   = numel(t);

Vin = ones(N,1);
Vin(t >= 0.3 & t <= 0.5) = 0.8;
u.t = t; u.V = Vin; u.theta = zeros(N,1); u.Pm = Pm0*ones(N,1); u.vf = vf0*ones(N,1);

Fi.V  = griddedInterpolant(t, u.V,     'linear');
Fi.th = griddedInterpolant(t, u.theta, 'linear');
Fi.Pm = griddedInterpolant(t, u.Pm,    'linear');
Fi.vf = griddedInterpolant(t, u.vf,    'linear');

opts = odeset('RelTol',1e-9,'AbsTol',1e-11);
ctx = struct('t',t,'N',N,'u',u,'Fi',Fi,'Pm0',Pm0,'vf0',vf0, ...
             'dguess',x0(1),'opts',opts);

%% SG4 target ("measurements")
[~, X4] = ode15s(@(tt,x) sg4_rhs(tt,x,p0,u), t, x0, opts);
tgt.delta = X4(:,1); tgt.omega = X4(:,2); tgt.eqp = X4(:,3); tgt.edp = X4(:,4);
tgt.id = zeros(N,1); tgt.iq = zeros(N,1); tgt.Pg = zeros(N,1); tgt.Qg = zeros(N,1);
for i = 1:N
    o = sg4_outputs(X4(i,:).', u.V(i), u.theta(i), p0);
    tgt.id(i) = o.id; tgt.iq(i) = o.iq; tgt.Pg(i) = o.Pg; tgt.Qg(i) = o.Qg;
end

allv = {'delta','omega','edp','eqp','id','iq','Pg','Qg'};
scale = struct();
for j = 1:numel(allv)
    y = tgt.(allv{j});
    scale.(allv{j}) = max(sqrt(mean((y - y(1)).^2)), 1e-3);
end

%% Nominal SG3 (sanity check against the previous extended test)
pnames = {'H','D','Tq0_prime','xd','xq','xq_prime'};
np = numel(pnames);
tr_nom = sg3_sim(p0, ctx);
if ~tr_nom.ok, error('Nominal SG3 simulation failed.'); end

fprintf('Nominal SG3 equilibrium: delta0 = %.6f (SG4 data %.6f), edp0 = %.6f (SG4 data %.6f)\n', ...
        tr_nom.delta(1), x0(1), tr_nom.edp(1), x0(4));
Mn = metrics(tr_nom, tgt, allv);
fprintf('Nominal SG3 absolute RMS vs SG4: delta=%.6f Pg=%.6f Qg=%.6f\n', ...
        Mn.abs(1), Mn.abs(7), Mn.abs(8));
fprintf('  (previous run at Td0=6.5 gave delta=0.013295, Pg=0.040563, Qg=0.31517)\n\n');

%% Fits
variants(1).name = 'Pg only'; variants(1).vars = {'Pg'};
variants(2).name = 'Pg+Qg';   variants(2).vars = {'Pg','Qg'};

rng(1);
nStart = 8;
fit = struct('name',{},'m',{},'p',{},'tr',{},'rn',{},'J',{},'nclose',{});
for c = 1:numel(variants)
    fprintf('Fitting variant %d: %s ...\n', c, variants(c).name);
    f = fit_variant(variants(c).vars, pnames, p0, tgt, scale, ctx, nStart);
    f.name = variants(c).name;
    fit(c) = f;
    fprintf('  best cost = %.4e | starts within 1%% of best: %d / %d\n', f.rn, f.nclose, nStart);
end
fprintf('\n');

%% Fitted parameters
vals = zeros(np,3);
for k = 1:np
    vals(k,1) = p0.(pnames{k});
    vals(k,2) = fit(1).p.(pnames{k});
    vals(k,3) = fit(2).p.(pnames{k});
end
fprintf('%-12s %12s %12s %8s %12s %8s\n','param','nominal','fit Pg','ratio','fit Pg+Qg','ratio');
for k = 1:np
    fprintf('%-12s %12.5f %12.5f %8.3f %12.5f %8.3f\n', pnames{k}, vals(k,1), ...
        vals(k,2), vals(k,2)/vals(k,1), vals(k,3), vals(k,3)/vals(k,1));
end
lb = log(0.1); ub = log(10);
for c = 1:2
    if any(abs(fit(c).m - lb) < 1e-3 | abs(fit(c).m - ub) < 1e-3)
        fprintf('WARNING: fit "%s" has parameters at a bound (0.1x or 10x).\n', fit(c).name);
    end
end
fprintf('\n');

%% Error comparison
cases = {'Nominal SG3', ['Fit: ' fit(1).name], ['Fit: ' fit(2).name]};
trs   = {tr_nom, fit(1).tr, fit(2).tr};
Mabs = zeros(3,numel(allv)); Mrel = zeros(3,numel(allv));
for c = 1:3
    M = metrics(trs{c}, tgt, allv);
    Mabs(c,:) = M.abs; Mrel(c,:) = M.rel;
end

fprintf('--- Absolute RMS error vs SG4 data (pu) ---\n');
fprintf('%-16s', 'case'); fprintf('%11s', allv{:}); fprintf('\n');
for c = 1:3
    fprintf('%-16s', cases{c}); fprintf('%11.5f', Mabs(c,:)); fprintf('\n');
end
fprintf('\n--- Relative RMS error vs SG4 data (denominator = norm of SG4 signal) ---\n');
fprintf('%-16s', 'case'); fprintf('%11s', allv{:}); fprintf('\n');
for c = 1:3
    fprintf('%-16s', cases{c}); fprintf('%11.5f', Mrel(c,:)); fprintf('\n');
end
fprintf('\n');

%% Local conditioning at the optimum (feeds Phase 2)
for c = 1:2
    J = fit(c).J;
    if ~isempty(J)
        [~, Ssv, Vsv] = svd(full(J), 'econ');
        s = diag(Ssv);
        fprintf('Fit "%s": normalised singular values of the scaled Jacobian (log-parameters):\n  ', fit(c).name);
        fprintf('%.2e ', s/s(1)); fprintf('\n');
        fprintf('  Weakest (least identifiable) direction in log-parameters:\n');
        for k = 1:np
            fprintf('    %-12s % .3f\n', pnames{k}, Vsv(k,end));
        end
    end
end
fprintf('\n');

%% Tables
Tp = table(pnames(:), vals(:,1), vals(:,2), vals(:,3), ...
           'VariableNames', {'param','nominal','fit_Pg','fit_PgQg'});
writetable(Tp, fullfile(root,'results','tables','SG3_refit_parameters.csv'));

Te = array2table(Mabs, 'VariableNames', allv);
Te.case_name = cases(:);
Te = movevars(Te, 'case_name', 'Before', 1);
writetable(Te, fullfile(root,'results','tables','SG3_refit_abs_errors.csv'));

Tr = array2table(Mrel, 'VariableNames', allv);
Tr.case_name = cases(:);
Tr = movevars(Tr, 'case_name', 'Before', 1);
writetable(Tr, fullfile(root,'results','tables','SG3_refit_rel_errors.csv'));

%% Figure: trajectories
figure('Color','w','Position',[100 100 1200 800]);
show = {'Pg','Qg','delta','eqp'};
lbls = {'P_g (pu)','Q_g (pu)','\delta (rad)','e''_q (pu)'};
for s = 1:4
    subplot(2,2,s); hold on; grid on;
    plot(t, tgt.(show{s}), 'k-', 'LineWidth',2);
    plot(t, tr_nom.(show{s}), '--', 'LineWidth',1.3);
    plot(t, fit(1).tr.(show{s}), '--', 'LineWidth',1.3);
    plot(t, fit(2).tr.(show{s}), '--', 'LineWidth',1.3);
    xlabel('t (s)'); ylabel(lbls{s});
    title([lbls{s} ': SG4 data vs SG3']);
    if s == 1
        legend({'SG4 data','SG3 nominal','SG3 fit (Pg)','SG3 fit (Pg+Qg)'},'Location','best');
    end
end
saveas(gcf, fullfile(root,'results','figures','SG3_refit_trajectories.png'));

save(fullfile(root,'results','SG3_refit.mat'), 'pnames','vals','Mabs','Mrel','cases','allv');

%% ---------------- Local functions ----------------

function p = apply_params(p0, m, pnames)
    p = p0;
    for k = 1:numel(pnames)
        p.(pnames{k}) = p0.(pnames{k})*exp(m(k));
    end
end

function dx = sg3_rhs(t, x, p, Fi)
    delta = x(1); omega = x(2); edp = x(3);
    V = Fi.V(t); th = Fi.th(t); Pm = Fi.Pm(t); vf = Fi.vf(t);
    a   = delta - th;
    eqp = (p.xd_prime/p.xd)*vf + ((p.xd-p.xd_prime)/p.xd)*V*cos(a);
    id  = (eqp - V*cos(a))/p.xd_prime;
    iq  = (V*sin(a) - edp)/p.xq_prime;
    Pg  = V*sin(a)*id + V*cos(a)*iq;
    ddelta = p.Omega_b*(omega - p.omega0);
    domega = (Pm - Pg - p.D*(omega - p.omega0))/(2*p.H);
    dedp   = (-(p.xq/p.xq_prime)*edp + ((p.xq-p.xq_prime)/p.xq_prime)*V*sin(a))/p.Tq0_prime;
    dx = [ddelta; domega; dedp];
end

function tr = sg3_sim(p, ctx)
% Simulate SG3 from its own equilibrium for parameter set p.
    tr.ok = false;
    V0 = ctx.u.V(1); th0 = ctx.u.theta(1);
    f = @(a) V0*sin(a).*(ctx.vf0 - V0*cos(a))/p.xd + V0^2*sin(a).*cos(a)/p.xq - ctx.Pm0;
    try
        a0 = fzero(f, ctx.dguess - th0);
    catch
        return
    end
    if ~isfinite(a0) || abs(f(a0)) > 1e-8, return; end
    d0   = a0 + th0;
    edp0 = ((p.xq - p.xq_prime)/p.xq)*V0*sin(a0);
    try
        [~, X] = ode15s(@(tt,x) sg3_rhs(tt,x,p,ctx.Fi), ctx.t, [d0; p.omega0; edp0], ctx.opts);
    catch
        return
    end
    if size(X,1) ~= ctx.N || any(~isfinite(X(:))), return; end

    a   = X(:,1) - ctx.u.theta;
    Vv  = ctx.u.V;
    eqp = (p.xd_prime/p.xd)*ctx.u.vf + ((p.xd-p.xd_prime)/p.xd)*Vv.*cos(a);
    id  = (eqp - Vv.*cos(a))/p.xd_prime;
    iq  = (Vv.*sin(a) - X(:,3))/p.xq_prime;
    tr.delta = X(:,1); tr.omega = X(:,2); tr.edp = X(:,3); tr.eqp = eqp;
    tr.id = id; tr.iq = iq;
    tr.Pg = Vv.*sin(a).*id + Vv.*cos(a).*iq;
    tr.Qg = Vv.*cos(a).*id - Vv.*sin(a).*iq;
    tr.ok = true;
end

function r = fit_resid(m, vars, pnames, p0, tgt, scale, ctx)
    p  = apply_params(p0, m, pnames);
    tr = sg3_sim(p, ctx);
    nres = numel(vars)*ctx.N;
    if ~tr.ok
        r = 1e3*ones(nres,1);
        return
    end
    r = zeros(nres,1);
    for j = 1:numel(vars)
        v = vars{j};
        idx = (j-1)*ctx.N + (1:ctx.N);
        r(idx) = (tr.(v) - tgt.(v))/(scale.(v)*sqrt(ctx.N));
    end
end

function best = fit_variant(vars, pnames, p0, tgt, scale, ctx, nStart)
    np = numel(pnames);
    lb = log(0.1)*ones(1,np);
    ub = log(10)*ones(1,np);
    fun = @(m) fit_resid(m, vars, pnames, p0, tgt, scale, ctx);

    rn_all = inf(nStart,1);
    m_all  = zeros(nStart,np);
    J_all  = cell(nStart,1);
    for s = 1:nStart
        if s == 1
            m0 = zeros(1,np);
        else
            m0 = min(max(0.6*randn(1,np), lb), ub);
        end
        try
            o = optimoptions('lsqnonlin','Display','off','MaxFunctionEvaluations',4000, ...
                             'MaxIterations',200,'FunctionTolerance',1e-12,'StepTolerance',1e-10);
            [x, rn, ~, ~, ~, ~, J] = lsqnonlin(fun, m0, lb, ub, o);
        catch
            obj = @(m) sum(fun(min(max(m,lb),ub)).^2);
            x = fminsearch(obj, m0, optimset('MaxFunEvals',4000,'MaxIter',4000, ...
                                             'TolFun',1e-14,'TolX',1e-8));
            x = min(max(x,lb),ub);
            rn = obj(x);
            J = [];
        end
        rn_all(s) = rn; m_all(s,:) = x; J_all{s} = J;
    end
    [rnb, ib] = min(rn_all);
    best.m  = m_all(ib,:);
    best.p  = apply_params(p0, best.m, pnames);
    best.tr = sg3_sim(best.p, ctx);
    best.rn = rnb;
    best.J  = J_all{ib};
    best.nclose = sum(rn_all <= 1.01*rnb);
end

function M = metrics(tr, tgt, vars)
    n = numel(vars);
    M.abs = zeros(1,n); M.rel = zeros(1,n);
    for j = 1:n
        a = tr.(vars{j}); b = tgt.(vars{j});
        M.abs(j) = sqrt(mean((a-b).^2));
        M.rel(j) = norm(a-b)/norm(b);
    end
end