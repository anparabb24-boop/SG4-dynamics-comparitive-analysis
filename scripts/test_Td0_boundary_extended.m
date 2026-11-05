%% test_Td0_boundary_extended.m
% Extended T'd0 -> 0 boundary test: SG4 versus the SG3 quasi-steady model.
%
% Extends test_Td0_small_boundary.m by:
%   * adding T'd0 = 0.005 and 0.001 s
%   * adding Pg, Qg, id, iq to the error comparison
%   * reporting BOTH relative and absolute RMS errors (Qg is small, so its
%     relative error is inflated by the denominator)
%   * estimating the empirical convergence order from log-log slopes
%
% Run from the MBAMSINDy_PSA root (or any folder; paths are added below).
% Requires: results/sg4_disturbance_baseline.mat, src/models/sg4_parameters.m,
%           src/models/sg4_rhs.m, src/models/sg4_outputs.m

clear; clc; close all;

%% Paths
root = fileparts(fileparts(mfilename('fullpath')));
if isempty(root) || ~exist(fullfile(root,'src'),'dir')
    root = pwd;            % fallback if the script is run from the project root
end
addpath(genpath(fullfile(root,'src')));
if ~exist(fullfile(root,'results','figures'),'dir'), mkdir(fullfile(root,'results','figures')); end
if ~exist(fullfile(root,'results','tables'),'dir'),  mkdir(fullfile(root,'results','tables'));  end

%% Load baseline data (initial condition and inputs)
S = load(fullfile(root,'results','sg4_disturbance_baseline.mat'));
p     = S.p;
x0    = S.x0(:);      % [delta; omega; eqp; edp]
Pm0   = S.Pm0;
vf0   = S.vf0;
t_all = S.t_all(:);

%% Inputs (same voltage disturbance as the baseline run)
V = ones(size(t_all));
V(t_all >= 0.3 & t_all <= 0.5) = 0.8;
u.t     = t_all;
u.V     = V;
u.theta = zeros(size(t_all));
u.Pm    = Pm0*ones(size(t_all));
u.vf    = vf0*ones(size(t_all));

%% Solver options (tight tolerances; ode15s because small T'd0 is stiff)
opts = odeset('RelTol',1e-9,'AbsTol',1e-11);

%% SG3 reference (quasi-steady limit, eqp algebraic)
x0_sg3 = x0([1 2 4]);                                  % [delta; omega; edp]
[~, X3] = ode15s(@(t,x) sg3_local_rhs(t,x,p,u), t_all, x0_sg3, opts);
ref = reconstruct_sg3(t_all, X3, p, u);                % struct of trajectories

% Consistency check on the algebraic eqp
fprintf('SG4 initial eqp = %.8f | SG3 initial eqp = %.8f\n', x0(3), ref.eqp(1));

%% Sweep over T'd0
Td0_list = [6.5 1 0.5 0.1 0.05 0.01 0.005 0.001];
nT = numel(Td0_list);

vars = {'delta','omega','edp','eqp','id','iq','Pg','Qg'};
nV = numel(vars);
relErr = zeros(nT,nV);
absErr = zeros(nT,nV);

for k = 1:nT
    pk = p;
    pk.Td0_prime = Td0_list(k);

    [~, X4] = ode15s(@(t,x) sg4_rhs(t,x,pk,u), t_all, x0, opts);
    tr = reconstruct_sg4(t_all, X4, pk, u);

    for j = 1:nV
        a = tr.(vars{j});
        b = ref.(vars{j});
        relErr(k,j) = norm(a-b)/norm(b);
        absErr(k,j) = sqrt(mean((a-b).^2));
    end

    fprintf('Td0 = %7.4f | rel: delta=%.6f omega=%.6f edp=%.6f eqp=%.6f Pg=%.6f Qg=%.6f\n', ...
        Td0_list(k), relErr(k,1), relErr(k,2), relErr(k,3), relErr(k,4), relErr(k,7), relErr(k,8));
end

%% Tables
Trel = array2table(relErr, 'VariableNames', vars);
Trel.Td0 = Td0_list(:);
Trel = movevars(Trel,'Td0','Before',1);

Tabs = array2table(absErr, 'VariableNames', vars);
Tabs.Td0 = Td0_list(:);
Tabs = movevars(Tabs,'Td0','Before',1);

disp('--- Relative RMS error (vs SG3) ---');  disp(Trel);
disp('--- Absolute RMS error (vs SG3) ---');  disp(Tabs);

writetable(Trel, fullfile(root,'results','tables','Td0_boundary_relative_errors.csv'));
writetable(Tabs, fullfile(root,'results','tables','Td0_boundary_absolute_errors.csv'));

%% Empirical convergence order (small-T'd0 region only)
% Fit log(err) = a + n*log(Td0) for Td0 <= 0.1; n ~ 1 means first-order decay.
idx = Td0_list <= 0.1;
fprintf('\nEmpirical convergence order n (err ~ Td0^n), using Td0 <= 0.1 s:\n');
for j = 1:nV
    c = polyfit(log(Td0_list(idx)), log(absErr(idx,j)'), 1);
    fprintf('  %-6s  n = %.2f\n', vars{j}, c(1));
end

%% Figure 1: state errors (log-log) with first-order reference
figure('Color','w','Position',[100 100 900 600]);
styles = {'o-','s-','^-','d-'};
for j = 1:4
    loglog(Td0_list, relErr(:,j), styles{j}, 'LineWidth',1.5, 'MarkerSize',7); hold on;
end
ref_line = relErr(idx,1);  Tref = Td0_list(idx);
loglog(Tref, ref_line(1)*(Tref/Tref(1)), 'k--', 'LineWidth',1);   % slope-1 guide
set(gca,'XDir','reverse'); grid on;
xlabel('T''_{d0} (seconds)'); ylabel('Relative RMS error vs SG3');
title('SG4 convergence toward SG3: states');
legend({'\delta','\omega','e''_d','e''_q','slope 1 guide'},'Location','southwest');
saveas(gcf, fullfile(root,'results','figures','Td0_boundary_states.png'));

%% Figure 2: electrical outputs, relative and absolute
figure('Color','w','Position',[100 100 1100 500]);
subplot(1,2,1);
loglog(Td0_list, relErr(:,5), 'o-', Td0_list, relErr(:,6), 's-', ...
       Td0_list, relErr(:,7), '^-', Td0_list, relErr(:,8), 'd-', 'LineWidth',1.5);
set(gca,'XDir','reverse'); grid on;
xlabel('T''_{d0} (s)'); ylabel('Relative RMS error vs SG3');
title('Outputs: relative error');
legend({'i_d','i_q','P_g','Q_g'},'Location','southwest');

subplot(1,2,2);
loglog(Td0_list, absErr(:,5), 'o-', Td0_list, absErr(:,6), 's-', ...
       Td0_list, absErr(:,7), '^-', Td0_list, absErr(:,8), 'd-', 'LineWidth',1.5);
set(gca,'XDir','reverse'); grid on;
xlabel('T''_{d0} (s)'); ylabel('Absolute RMS error vs SG3 (pu)');
title('Outputs: absolute error');
legend({'i_d','i_q','P_g','Q_g'},'Location','southwest');
saveas(gcf, fullfile(root,'results','figures','Td0_boundary_outputs.png'));

%% Figure 3: Pg and Qg trajectories, SG3 vs SG4 at selected T'd0
figure('Color','w','Position',[100 100 1100 500]);
Tshow = [6.5 0.1 0.01];
cols  = lines(numel(Tshow));
for s = 1:2
    subplot(1,2,s); hold on; grid on;
    plot(t_all, ref.(vars{6+s}), 'k-', 'LineWidth',2);
    for q = 1:numel(Tshow)
        pk = p; pk.Td0_prime = Tshow(q);
        [~, X4] = ode15s(@(t,x) sg4_rhs(t,x,pk,u), t_all, x0, opts);
        tr = reconstruct_sg4(t_all, X4, pk, u);
        plot(t_all, tr.(vars{6+s}), '--', 'Color',cols(q,:), 'LineWidth',1.3);
    end
    xlabel('t (s)');
    ylabel(vars{6+s});
    title([vars{6+s} ': SG3 (black) vs SG4']);
    legend([{'SG3'}, arrayfun(@(v) sprintf('SG4, T''_{d0}=%g',v), Tshow, 'UniformOutput',false)], ...
           'Location','best');
end
saveas(gcf, fullfile(root,'results','figures','Td0_boundary_PgQg_trajectories.png'));

save(fullfile(root,'results','Td0_boundary_extended.mat'), ...
     'Td0_list','vars','relErr','absErr');

%% ---------------- Local functions ----------------

function dx = sg3_local_rhs(t, x, p, u)
% SG3: states [delta; omega; edp]; eqp is algebraic.
    delta = x(1); omega = x(2); edp = x(3);
    V     = interp1(u.t, u.V,     t, 'linear', 'extrap');
    theta = interp1(u.t, u.theta, t, 'linear', 'extrap');
    Pm    = interp1(u.t, u.Pm,    t, 'linear', 'extrap');
    vf    = interp1(u.t, u.vf,    t, 'linear', 'extrap');

    eqp = (p.xd_prime/p.xd)*vf + ((p.xd-p.xd_prime)/p.xd)*V*cos(delta-theta);

    out = sg4_outputs([delta; omega; eqp; edp], V, theta, p);
    Pg  = out.Pg;

    ddelta = p.Omega_b*(omega - p.omega0);
    domega = (Pm - Pg - p.D*(omega - p.omega0)) / (2*p.H);
    dedp   = (-(p.xq/p.xq_prime)*edp + ...
              ((p.xq-p.xq_prime)/p.xq_prime)*V*sin(delta-theta)) / p.Tq0_prime;

    dx = [ddelta; domega; dedp];
end

function tr = reconstruct_sg3(t, X, p, u)
% Rebuild eqp (algebraic) and outputs along an SG3 trajectory.
    n = numel(t);
    tr.delta = X(:,1); tr.omega = X(:,2); tr.edp = X(:,3);
    tr.eqp = zeros(n,1); tr.id = zeros(n,1); tr.iq = zeros(n,1);
    tr.Pg = zeros(n,1);  tr.Qg = zeros(n,1);
    for i = 1:n
        V = u.V(i); th = u.theta(i); vf = u.vf(i);
        eqp = (p.xd_prime/p.xd)*vf + ((p.xd-p.xd_prime)/p.xd)*V*cos(X(i,1)-th);
        o = sg4_outputs([X(i,1); X(i,2); eqp; X(i,3)], V, th, p);
        tr.eqp(i) = eqp; tr.id(i) = o.id; tr.iq(i) = o.iq;
        tr.Pg(i) = o.Pg; tr.Qg(i) = o.Qg;
    end
end

function tr = reconstruct_sg4(t, X, p, u)
% Outputs along an SG4 trajectory.
    n = numel(t);
    tr.delta = X(:,1); tr.omega = X(:,2); tr.eqp = X(:,3); tr.edp = X(:,4);
    tr.id = zeros(n,1); tr.iq = zeros(n,1); tr.Pg = zeros(n,1); tr.Qg = zeros(n,1);
    for i = 1:n
        o = sg4_outputs(X(i,:).', u.V(i), u.theta(i), p);
        tr.id(i) = o.id; tr.iq(i) = o.iq; tr.Pg(i) = o.Pg; tr.Qg(i) = o.Qg;
    end
end