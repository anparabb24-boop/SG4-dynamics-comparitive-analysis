clc; clear; close all;

%% SG4 20-second disturbance simulation
% x = [delta(rad); omega(pu); eq_prime(pu); ed_prime(pu)].
% Rotor angle is kept in radians in all saved data and converted to
% degrees only in plots.
%
% kPg is an imposed electrical-power scaling event, not a detailed
% network-fault model. It is applied consistently in the swing equation
% and in the reported Pg output.

%% 1. Parameters and initial equilibrium
p = sg4_parameters();

delta0 = 0.6301;      % rad
omega0 = 1.0;         % pu
eqp0 = 1.0150;        % pu
V0 = 1.0;             % pu
theta0 = 0.0;         % rad

iq0 = V0*sin(delta0-theta0)/p.xq;
edp0 = (p.xq-p.xq_prime)*iq0;
id0 = (eqp0-V0*cos(delta0-theta0))/p.xd_prime;

x0 = [delta0; omega0; eqp0; edp0];

y0 = sg4_outputs(x0,V0,theta0,p);
Pm0 = y0.Pg;
vf0 = eqp0 + (p.xd-p.xd_prime)*id0;

fprintf('Initial equilibrium: delta=%.6f rad (%.4f deg), Pm0=%.6f, vf0=%.6f\\n', ...
    delta0,rad2deg(delta0),Pm0,vf0);

%% 2. Simulation and disturbance schedule
dt = 0.01;
simulationEndTime = 20.0;

% Columns: [t_start t_end V_factor kPg Pm_factor]
% 0-1 s:    normal
% 1-1.2 s:  20% terminal-voltage dip
% 1.2-5 s:  recovery
% 5-6 s:    20% electrical Pg drop
% 6-20 s:   recovery
%
% Change the factors/times here to test other disturbance magnitudes.
intervals = [
     0.0   1.0   1.00   1.00   1.00
     1.0   1.2   0.80   1.0   1.00
     1.2   5.0   1.00   1.00   1.00
     5.0   6.0   1.00   1.0   1.00
     6.0  20.0   1.00   1.00   1.00
];

event_times = intervals(2:end,1);
assert(abs(intervals(1,1))<1e-12 && ...
       abs(intervals(end,2)-simulationEndTime)<1e-12, ...
       'Intervals must cover 0 to 20 seconds.');
assert(all(abs(intervals(2:end,1)-intervals(1:end-1,2))<1e-12), ...
       'Intervals must be contiguous.');
assert(all(abs(diff(intervals(:,1))/dt-round(diff(intervals(:,1))/dt))<1e-9), ...
       'Interval boundaries must align with dt.');

opts = odeset('RelTol',1e-8,'AbsTol',1e-10);

%% 3. Integrate interval by interval
t_all = [];
x_all = [];
x_current = x0;

for k = 1:size(intervals,1)
    ta = intervals(k,1);
    tb = intervals(k,2);

    % Inputs are constant within each interval.
    u.t = [ta tb];
    u.V = intervals(k,3)*[1 1];
    u.theta = theta0*[1 1];
    u.Pm = Pm0*intervals(k,5)*[1 1];
    u.vf = vf0*[1 1];
    u.kPg = intervals(k,4)*[1 1];

    nPoints = round((tb-ta)/dt)+1;
    t_segment = linspace(ta,tb,nPoints)';

    [t_segment,x_segment] = ode15s( ...
        @(t,x) sg4_rhs_with_pg_drop(t,x,p,u), ...
        t_segment,x_current,opts);

    if k==1
        t_all = t_segment;
        x_all = x_segment;
    else
        % Avoid duplicate samples at interval boundaries.
        t_all = [t_all; t_segment(2:end)];
        x_all = [x_all; x_segment(2:end,:)];
    end
    x_current = x_segment(end,:)';
end

fprintf('Simulation completed. Samples: %d, duration: %.2f s, dt: %.4f s\\n', ...
    numel(t_all),t_all(end)-t_all(1),mean(diff(t_all)));

%% 4. Construct input trajectories
N = numel(t_all);
V_all = zeros(N,1);
kPg_all = zeros(N,1);
Pm_all = zeros(N,1);
vf_all = vf0*ones(N,1);
theta_all = theta0*ones(N,1);

for k = 1:N
    r = rowAt(t_all(k),intervals);
    V_all(k) = intervals(r,3);
    kPg_all(k) = intervals(r,4);
    Pm_all(k) = Pm0*intervals(r,5);
end

%% 5. Calculate outputs
Pg_all = zeros(N,1);
Qg_all = zeros(N,1);
id_all = zeros(N,1);
iq_all = zeros(N,1);

for k = 1:N
    y = sg4_outputs(x_all(k,:)',V_all(k),theta_all(k),p);
    Pg_all(k) = kPg_all(k)*y.Pg;
    Qg_all(k) = y.Qg;
    id_all(k) = y.id;
    iq_all(k) = y.iq;
end

%% 6. State units and summary
% Do not overwrite x_all(:,1): it must remain in radians.
delta_rad = x_all(:,1);
delta_deg = rad2deg(delta_rad);
omega_pu = x_all(:,2);
omega_actual = omega_pu*p.Omega_b;

fprintf('\\nSIMULATION SUMMARY\\n');
fprintf('Initial delta: %.6f rad = %.6f deg\\n',delta_rad(1),delta_deg(1));
fprintf('Final delta:   %.6f rad = %.6f deg\\n',delta_rad(end),delta_deg(end));
fprintf('Initial speed: %.6f pu = %.6f rad/s\\n',omega_pu(1),omega_actual(1));
fprintf('Final speed:   %.6f pu = %.6f rad/s\\n',omega_pu(end),omega_actual(end));
fprintf('Synchronous angular speed: %.6f rad/s\\n',p.Omega_b);

%% 7. Plot currents
figure('Name','SG4 Currents','Color','w');
subplot(2,1,1);
plot(t_all,id_all,'LineWidth',1.5);
ylabel('i_d (p.u.)'); xlabel('Time (s)');
title('Direct-Axis Current'); markEvents(event_times); grid on;
subplot(2,1,2);
plot(t_all,iq_all,'LineWidth',1.5);
ylabel('i_q (p.u.)'); xlabel('Time (s)');
title('Quadrature-Axis Current'); markEvents(event_times); grid on;
sgtitle('SG4 Generator Currents');

%% 8. Plot active and reactive power
figure('Name','SG4 Power','Color','w');
subplot(2,1,1);
plot(t_all,Pg_all,'LineWidth',1.5); hold on;
plot(t_all,Pm_all,'--','LineWidth',1.2);
ylabel('Power (p.u.)'); xlabel('Time (s)');
title('Generator Active Power');
legend('P_g (effective)','P_m','Location','best');
markEvents(event_times); grid on;
subplot(2,1,2);
plot(t_all,Qg_all,'LineWidth',1.5);
ylabel('Q_g (p.u.)'); xlabel('Time (s)');
title('Generator Reactive Power'); markEvents(event_times); grid on;
sgtitle('SG4 Generator Power');

%% 9. Plot states (delta shown in degrees; x_all remains in radians)
figure('Name','SG4 States','Color','w');
statePlotData = x_all;
statePlotData(:,1) = delta_deg;
stateNames = {'\delta (deg)','\omega (p.u.)','E_q'' (p.u.)','E_d'' (p.u.)'};
for k = 1:4
    subplot(4,1,k);
    plot(t_all,statePlotData(:,k),'LineWidth',1.5);
    ylabel(stateNames{k}); markEvents(event_times); grid on;
end
xlabel('Time (s)');
sgtitle('SG4 State Trajectories');

%% 10. Plot disturbance inputs
figure('Name','SG4 Disturbance Inputs','Color','w');
subplot(3,1,1);
plot(t_all,V_all,'LineWidth',1.5);
ylabel('V (p.u.)'); title('Terminal Voltage'); markEvents(event_times); grid on;
subplot(3,1,2);
plot(t_all,kPg_all,'LineWidth',1.5);
ylabel('k_{Pg}'); title('Electrical Power Scaling'); markEvents(event_times); grid on;
subplot(3,1,3);
plot(t_all,Pm_all,'LineWidth',1.5);
ylabel('P_m (p.u.)'); xlabel('Time (s)');
title('Mechanical Power Input'); markEvents(event_times); grid on;
sgtitle('Applied Disturbances');

%% 11. Save results: 20 s, delta in radians
if ~exist('results','dir')
    mkdir('results');
end
save('results/sg4_disturbance_20s.mat', ...
    't_all','x_all','id_all','iq_all','Pg_all','Qg_all', ...
    'p','x0','Pm0','vf0','theta_all','V_all','Pm_all','vf_all', ...
    'kPg_all','intervals','event_times','dt');

fprintf('\\nSaved: results/sg4_disturbance_20s.mat\\n');
fprintf('Saved delta unit: radians. Plot delta unit: degrees.\\n');

%% 12. Generator electrical frequency
freq_t = omega_pu*p.f;
figure('Name','SG4 Frequency','Color','w');
plot(t_all,freq_t,'LineWidth',1.5);
markEvents(event_times);
xlabel('Time (s)'); ylabel('Frequency (Hz)');
title('Generator Electrical Frequency'); grid on;

%% 13. FFT of rotor angle (degrees)
Ts = mean(diff(t_all));
Fs = 1/Ts;
Nf = numel(delta_deg);
delta_ac = delta_deg-mean(delta_deg);

% Hann window without requiring hann() from the Signal Processing Toolbox.
w = 0.5*(1-cos(2*pi*(0:Nf-1)'/(Nf-1)));
Y = fft(delta_ac.*w);
P2 = abs(Y/Nf);
P1 = P2(1:floor(Nf/2)+1);
if numel(P1)>2
    P1(2:end-1) = 2*P1(2:end-1);
end
P1 = P1/mean(w);
fax = Fs*(0:floor(Nf/2))/Nf;

figure('Name','Rotor Angle FFT','Color','w');
plot(fax,P1,'LineWidth',1.5);
xlim([0 5]); xlabel('Frequency (Hz)'); ylabel('Amplitude (deg)');
title('Single-Sided Amplitude Spectrum of Rotor Angle'); grid on;
if numel(P1)>1
    [~,idx] = max(P1(2:end));
    fprintf('Dominant frequency of delta: %.4f Hz\\n',fax(idx+1));
end

%% 14. Spectrogram (requires Signal Processing Toolbox)
win_len = 400;
overlap = win_len-10;
nfft = 2048;
if Nf>=win_len
    specWindow = 0.5*(1-cos(2*pi*(0:win_len-1)'/(win_len-1)));
    [S,F,T] = spectrogram(delta_ac,specWindow,overlap,nfft,Fs);
    figure('Name','Rotor Angle Spectrogram','Color','w');
    imagesc(T+t_all(1),F,20*log10(abs(S)+eps));
    axis xy; ylim([0 5]); colorbar; colormap turbo;
    xlabel('Time (s)'); ylabel('Frequency (Hz)');
    title('Spectrogram of Rotor Angle'); hold on;
    for te = event_times'
        xline(te,'--w');
    end
else
    warning('Not enough samples for the selected spectrogram window.');
end

%% Local functions
function dx = sg4_rhs_with_pg_drop(~,x,p,u)
    % Start with the standard SG4 model.
    dx = sg4_rhs(0,x,p,u);

    % Apply the Pg scaling correction to the swing equation:
    % P_e,eff = kPg*P_e,nominal.
    kPgNow = u.kPg(1);
    if abs(kPgNow-1)>1e-12
        y = sg4_outputs(x,u.V(1),u.theta(1),p);
        dx(2) = dx(2) - (1-kPgNow)*y.Pg/(2*p.H);
    end
end

function row = rowAt(t,intervals)
    % At a boundary, select the later interval.
    tol = 1e-10;
    row = find(t>=intervals(:,1)-tol & ...
               t<=intervals(:,2)+tol,1,'last');
    if isempty(row)
        error('No disturbance interval found for t = %.12g.',t);
    end
end

function markEvents(event_times)
    for te = event_times'
        xline(te,'--r','LineWidth',1.2);
    end
end
