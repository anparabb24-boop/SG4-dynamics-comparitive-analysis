
%% RUN_MBAM_SG4_REPOSITORY
% Repository-based MBAM geodesic implementation for SG4 / SMIB.
%
% Implements:
%   1. Physical SG4 parameter fitting
%   2. Model manifold r(q)
%   3. Numerical Jacobian J(q)
%   4. Second directional derivative A_vv
%   5. Sloppiest singular-vector initialization
%   6. MBAM geodesic integration
%   7. Held-out physical-model reconstruction
%
% Parameter coordinates:
%   q = log(p)
%
% IMPORTANT:
% This implements a repository-style geodesic calculation.
% It does not establish an asymptotic reduction unless a genuine
% boundary and limiting model are subsequently demonstrated.
%
% State order: [delta, omega, eq_prime, ed_prime]
% Input order: [V, theta, Pm, vf, kPg]
%
% Requires Optimization Toolbox and Signal Processing Toolbox.

clear;
clc;
close all;

fprintf('\n============================================\n');
fprintf(' REPOSITORY-BASED MBAM: SG4 / SMIB\n');
fprintf('============================================\n\n');

%% 1. CONFIGURATION

cfg.dataFile = fullfile('data','dataset_20s','sg4_dataset_20s.mat');

cfg.nTrainTraj = 15;
cfg.sampleStride = 5;

cfg.sgolayOrder = 3;
cfg.sgolayFrame = 11;

cfg.jacobianStep = 1e-4;
cfg.directionalStep = 1e-3;

cfg.initialGeodesicSpeed = 0.05;
cfg.geodesicEndTime = 2.0;

cfg.relTol = 1e-5;
cfg.absTol = 1e-7;

% Select noise condition for MBAM experiment
cfg.noiseType = "gaussian";       % gaussian, uniform, impulsive
cfg.noiseLevelPercent = 10;        % 1, 5, or 10

% Generate dynamic filename tag (e.g., uniform10pct)
noiseTag = sprintf('%s%dpct', lower(string(cfg.noiseType)), cfg.noiseLevelPercent);

cfg.outputDir = fullfile('results','mbam_sg4_repository');
if ~exist(cfg.outputDir,'dir')
    mkdir(cfg.outputDir);
end

%% 2. LOAD DATASET

fprintf('Loading unified 20-second SG4 dataset...\n');

if ~isfile(cfg.dataFile)
    error('Dataset not found: %s',cfg.dataFile);
end

S = load(cfg.dataFile);

if isfield(S,'dataset')
    ds = S.dataset;
else
    ds = S;
end

required = {'X_noisy','X_clean','Y_clean','Y_noisy', ...
    'U','time','trainIdx','validationIdx','testIdx','inputNames'};

for k = 1:numel(required)
    if ~isfield(ds,required{k})
        error('Missing dataset field: %s',required{k});
    end
end

t = double(ds.time(:));
dt = median(diff(t));

inputNames = cellstr(string(ds.inputNames));

idxV = find(strcmp(inputNames,'V'),1);
idxTheta = find(strcmp(inputNames,'theta'),1);
idxPm = find(strcmp(inputNames,'Pm'),1);
idxVf = find(strcmp(inputNames,'vf'),1);
idxKPg = find(strcmp(inputNames,'kPg'),1);

if any(cellfun(@isempty,{idxV,idxTheta,idxPm,idxVf}))
    error('Required SG4 inputs are missing.');
end

if isempty(idxKPg)
    warning('kPg is absent. Using kPg = 1.');
end

trainIdx = ds.trainIdx(:);
testIdx = ds.testIdx(:);

% Filter trajectories by selected noise condition
metadata = ds.metadata;

conditionMask = ...
    strcmpi(string(metadata.NoiseType),cfg.noiseType) & ...
    metadata.NoiseLevelPercent == cfg.noiseLevelPercent;

selectedTrainIdx = trainIdx(conditionMask(trainIdx));
selectedTestIdx = testIdx(conditionMask(testIdx));

fprintf('\nSelected noise condition: %s, %d%%\n', ...
    cfg.noiseType,cfg.noiseLevelPercent);

fprintf('Matching training trajectories: %d\n', ...
    numel(selectedTrainIdx));

fprintf('Matching testing trajectories: %d\n\n', ...
    numel(selectedTestIdx));

if isempty(selectedTrainIdx) || isempty(selectedTestIdx)
    error('No trajectories found for the selected noise condition.');
end

fprintf('Duration: %.2f seconds\n',t(end));
fprintf('Samples per trajectory: %d\n',numel(t));
fprintf('Training trajectories: %d\n',numel(trainIdx));
fprintf('Testing trajectories: %d\n\n',numel(testIdx));

%% 3. PREPARE TRAINING OBSERVATIONS

nUse = min(cfg.nTrainTraj,numel(selectedTrainIdx));

fprintf('Preparing %d training trajectories...\n',nUse);

D.X = [];
D.U = [];
D.dX = [];
D.Xclean = [];
D.dXclean = [];

for k = 1:nUse

    trajectory = selectedTrainIdx(k);

    Xraw = double(ds.X_noisy(:,:,trajectory));
    XcleanRaw = double(ds.X_clean(:,:,trajectory));
    Uraw = double(ds.U(:,:,trajectory));

    Xsmooth = zeros(size(Xraw));

    for j = 1:4
        Xsmooth(:,j) = sgolayfilt( ...
            Xraw(:,j),cfg.sgolayOrder,cfg.sgolayFrame);
    end

    dX = zeros(size(Xsmooth));
    dXclean = zeros(size(XcleanRaw));
    for j = 1:4
        dXclean(:,j) = gradient(XcleanRaw(:,j),dt);
    end
    
    for j = 1:4
        dX(:,j) = gradient(Xsmooth(:,j),dt);
    end

    % Canonical input order: [V, theta, Pm, vf, kPg]
    Ucanonical = [ ...
        Uraw(:,idxV), ...
        Uraw(:,idxTheta), ...
        Uraw(:,idxPm), ...
        Uraw(:,idxVf)];

    if ~isempty(idxKPg)
        Ucanonical = [Ucanonical,Uraw(:,idxKPg)];
    else
        Ucanonical = [Ucanonical,ones(size(Uraw,1),1)];
    end

    stride = cfg.sampleStride;
    sampleIdx = 1:stride:numel(t);

    D.X = [D.X;Xsmooth(sampleIdx,:)];
    D.dX = [D.dX;dX(sampleIdx,:)];
    D.Xclean = [D.Xclean; XcleanRaw(sampleIdx,:)];
    D.dXclean = [D.dXclean; dXclean(sampleIdx,:)];
    D.U = [D.U;Ucanonical(sampleIdx,:)];

end

% D.U is already canonicalized.
D.V = D.U(:,1);
D.theta = D.U(:,2);
D.Pm = D.U(:,3);
D.vf = D.U(:,4);
D.kPg = D.U(:,5);

fprintf('Training observations: %d\n',size(D.X,1));


%% 4. GROUND-TRUTH PARAMETER REFERENCE

% Parameter order:
% [2H, D, Td0_prime, Tq0_prime, xd, xq, xd_prime, xq_prime]

p0 = [ ...
    12.467, ...
    1.9, ...
    6.5, ...
    0.2, ...
    0.655, ...
    0.487, ...
    0.196, ...
    0.263];

paramNames = {'2H','D','Td0_prime','Tq0_prime', ...
    'xd','xq','xd_prime','xq_prime'};

q0 = log(p0);

% No parameter fitting in this experiment.
% Both conditions start at the known physical parameters.
pFit = p0;
qFit = q0;

% Construct two model-manifold datasets.
Dclean.X = D.Xclean;
Dclean.U = D.U;

Dnoisy.X = D.X;
Dnoisy.U = D.U;

% Use one common normalization for both conditions.
% The scale is estimated from clean-state derivatives.
derivativeScale = std(D.dXclean,0,1);
derivativeScale = max(derivativeScale,1e-6);

fprintf('\nGround-truth parameters:\n');

for k = 1:numel(p0)
    fprintf('%-15s : %.8g\n',paramNames{k},p0(k));
end

fprintf('\nNo nonlinear parameter fitting will be performed.\n');
fprintf('Clean and noisy analyses use identical parameter references.\n');


%% 5. CLEAN MODEL MANIFOLD

fprintf('\n============================================\n');
fprintf(' CLEAN-DATA MODEL MANIFOLD\n');
fprintf('============================================\n');

rClean = model_manifold(q0,Dclean,derivativeScale);

Jclean = numerical_jacobian( ...
    @(q) model_manifold(q,Dclean,derivativeScale), ...
    q0,cfg.jacobianStep);

FIMclean = Jclean'*Jclean;

[~,Sclean,Vclean] = svd(Jclean,'econ');

singularValuesClean = diag(Sclean);

conditionNumberClean = ...
    singularValuesClean(1)^2 / ...
    max(singularValuesClean(end)^2,eps);

fprintf('Jacobian dimensions: %d x %d\n', ...
    size(Jclean,1),size(Jclean,2));

fprintf('Clean FIM condition number: %.8e\n', ...
    conditionNumberClean);

fprintf('\nClean singular values:\n');

for k = 1:numel(singularValuesClean)
    fprintf('Mode %d: %.8e\n',k,singularValuesClean(k));
end

vClean = Vclean(:,end);
vClean = vClean/norm(vClean);
vClean = cfg.initialGeodesicSpeed*vClean;

fprintf('\nClean sloppiest direction:\n');

for k = 1:numel(vClean)
    fprintf('%-15s : %+ .8e\n',paramNames{k},vClean(k));
end


%% 6. NOISY MODEL MANIFOLD

fprintf('\n============================================\n');
fprintf(' NOISY-DATA MODEL MANIFOLD\n');
fprintf('============================================\n');

rNoisy = model_manifold(q0,Dnoisy,derivativeScale);

Jnoisy = numerical_jacobian( ...
    @(q) model_manifold(q,Dnoisy,derivativeScale), ...
    q0,cfg.jacobianStep);

FIMnoisy = Jnoisy'*Jnoisy;

[~,Snoisy,Vnoisy] = svd(Jnoisy,'econ');

singularValuesNoisy = diag(Snoisy);

conditionNumberNoisy = ...
    singularValuesNoisy(1)^2 / ...
    max(singularValuesNoisy(end)^2,eps);

fprintf('Jacobian dimensions: %d x %d\n', ...
    size(Jnoisy,1),size(Jnoisy,2));

fprintf('Noisy FIM condition number: %.8e\n', ...
    conditionNumberNoisy);

fprintf('\nNoisy singular values:\n');

for k = 1:numel(singularValuesNoisy)
    fprintf('Mode %d: %.8e\n',k,singularValuesNoisy(k));
end

vNoisy = Vnoisy(:,end);
vNoisy = vNoisy/norm(vNoisy);
vNoisy = cfg.initialGeodesicSpeed*vNoisy;

fprintf('\nNoisy sloppiest direction:\n');

for k = 1:numel(vNoisy)
    fprintf('%-15s : %+ .8e\n',paramNames{k},vNoisy(k));
end


%% 7. COMPARE SLOPPY SPECTRA

fprintf('\n============================================\n');
fprintf(' CLEAN VS NOISY COMPARISON\n');
fprintf('============================================\n');

fprintf('\n%-15s %-18s %-18s\n', ...
    'Mode','Clean','Noisy');

for k = 1:numel(singularValuesClean)

    fprintf('%-15d %-18.8e %-18.8e\n', ...
        k,singularValuesClean(k),singularValuesNoisy(k));

end

fprintf('\nClean FIM condition number: %.8e\n',conditionNumberClean);
fprintf('Noisy FIM condition number: %.8e\n',conditionNumberNoisy);

figure('Color','w');

semilogy(1:numel(singularValuesClean), ...
    singularValuesClean,'o-','LineWidth',1.5);

hold on;

semilogy(1:numel(singularValuesNoisy), ...
    singularValuesNoisy,'s--','LineWidth',1.5);

xlabel('Singular-value index');
ylabel('Singular value');
title('Clean vs Noisy SG4 Model-Manifold Spectrum');

legend('Clean data','Noisy data','Location','best');
grid on;

exportgraphics(gcf, ...
    fullfile(cfg.outputDir, sprintf('clean_vs_noisy_sloppy_spectrum_%s.png', noiseTag)));


%% 8. CLEAN-DATA MBAM GEODESIC

fprintf('\n============================================\n');
fprintf(' CLEAN-DATA MBAM GEODESIC\n');
fprintf('============================================\n');

y0Clean = [q0(:);vClean(:)];

geoOptions = odeset( ...
    'RelTol',cfg.relTol, ...
    'AbsTol',cfg.absTol, ...
    'MaxStep',0.02);

geoFunctionClean = @(tau,y) mbam_geodesic_rhs( ...
    tau,y,Dclean,derivativeScale,cfg);

[tauClean,geoClean] = ode45( ...
    geoFunctionClean, ...
    [0,cfg.geodesicEndTime], ...
    y0Clean,geoOptions);

nParameters = numel(p0);

qPathClean = geoClean(:,1:nParameters);
vPathClean = geoClean(:,nParameters+1:end);

pPathClean = exp(qPathClean);

pEndClean = pPathClean(end,:);

fprintf('Clean geodesic points: %d\n',numel(tauClean));

fprintf('\nClean geodesic endpoint:\n');

for k = 1:nParameters
    fprintf('%-15s : %.8g\n',paramNames{k},pEndClean(k));
end


%% 9. NOISY-DATA MBAM GEODESIC

fprintf('\n============================================\n');
fprintf(' NOISY-DATA MBAM GEODESIC\n');
fprintf('============================================\n');

y0Noisy = [q0(:);vNoisy(:)];

geoFunctionNoisy = @(tau,y) mbam_geodesic_rhs( ...
    tau,y,Dnoisy,derivativeScale,cfg);

[tauNoisy,geoNoisy] = ode45( ...
    geoFunctionNoisy, ...
    [0,cfg.geodesicEndTime], ...
    y0Noisy,geoOptions);

qPathNoisy = geoNoisy(:,1:nParameters);
vPathNoisy = geoNoisy(:,nParameters+1:end);

pPathNoisy = exp(qPathNoisy);

pEndNoisy = pPathNoisy(end,:);

fprintf('Noisy geodesic points: %d\n',numel(tauNoisy));

fprintf('\nNoisy geodesic endpoint:\n');

for k = 1:nParameters
    fprintf('%-15s : %.8g\n',paramNames{k},pEndNoisy(k));
end


%% 10. COMPARE PARAMETER EVOLUTION

figure('Color','w','Position',[100 100 1200 800]);

for k = 1:nParameters

    subplot(4,2,k);

    semilogy(tauClean,pPathClean(:,k), ...
        'LineWidth',1.5);

    hold on;

    semilogy(tauNoisy,pPathNoisy(:,k), ...
        '--','LineWidth',1.5);

    yline(p0(k),':k');

    xlabel('Geodesic parameter');
    ylabel(paramNames{k});
    title(paramNames{k});

    grid on;

end

sgtitle('Clean vs Noisy MBAM Parameter Evolution');

exportgraphics(gcf, ...
    fullfile(cfg.outputDir, sprintf('clean_vs_noisy_parameter_evolution_%s.png', noiseTag)));


%% 11. ENDPOINT AND BOUNDARY DIAGNOSTICS

relativeChangeClean = abs(pEndClean-p0)./abs(p0);
relativeChangeNoisy = abs(pEndNoisy-p0)./abs(p0);

fprintf('\n============================================\n');
fprintf(' GEODESIC ENDPOINT COMPARISON\n');
fprintf('============================================\n');

fprintf('\n%-15s %-18s %-18s\n', ...
    'Parameter','Clean change %','Noisy change %');

for k = 1:nParameters

    fprintf('%-15s %-18.8f %-18.8f\n', ...
        paramNames{k}, ...
        100*relativeChangeClean(k), ...
        100*relativeChangeNoisy(k));

end

fprintf('\nBoundary assessment:\n');

for k = 1:nParameters

    fprintf('\n%s:\n',paramNames{k});

    if pEndClean(k)<1e-5 || pEndClean(k)>1e5
        fprintf('  Clean: extreme parameter value; investigate boundary.\n');
    else
        fprintf('  Clean: no demonstrated asymptotic boundary.\n');
    end

    if pEndNoisy(k)<1e-5 || pEndNoisy(k)>1e5
        fprintf('  Noisy: extreme parameter value; investigate boundary.\n');
    else
        fprintf('  Noisy: no demonstrated asymptotic boundary.\n');
    end

end

% Keep legacy variable names for the remaining plotting and saving sections.
% These refer to the noisy-data analysis.
pPath = pPathNoisy;
pEnd = pEndNoisy;
qPath = qPathNoisy;
vPath = vPathNoisy;
tau = tauNoisy;
singularValues = singularValuesNoisy;
FIM = FIMnoisy;
J = Jnoisy;

%% 12. HELD-OUT TEST TRAJECTORY

fprintf('\nSimulating held-out physical SG4 test trajectory...\n');

if isempty(selectedTestIdx)
    error('No test trajectories are available for this condition.');
end

testTrajectory = selectedTestIdx(1);

tTest = t;

Xclean = double(ds.X_clean(:,:,testTrajectory));
Xnoisy = double(ds.X_noisy(:,:,testTrajectory));
Yclean = double(ds.Y_clean(:,:,testTrajectory));
Ynoisy = double(ds.Y_noisy(:,:,testTrajectory));

UtestRaw = double(ds.U(:,:,testTrajectory));

% Canonical test input order.
if isempty(idxKPg)
    kPgTest = ones(size(tTest));
else
    kPgTest = UtestRaw(:,idxKPg);
end

UtestCanonical = [ ...
    UtestRaw(:,idxV), ...
    UtestRaw(:,idxTheta), ...
    UtestRaw(:,idxPm), ...
    UtestRaw(:,idxVf), ...
    kPgTest];

x0Test = Xclean(1,:)';

simOptions = odeset( ...
    'RelTol',1e-6, ...
    'AbsTol',1e-8, ...
    'MaxStep',0.02);

[tSim,Xsim] = ode15s( ...
    @(tt,xx) sg4_rhs_model( ...
        tt,xx,tTest,UtestCanonical,p0), ...
    tTest,x0Test,simOptions);

%% 13. COMPUTE ALGEBRAIC OUTPUTS

Ysim = sg4_output_model(Xsim,UtestCanonical,p0);

%% 14. RECONSTRUCTION METRICS

stateRMSE = sqrt(mean((Xsim-Xclean).^2,1));
outputRMSE = sqrt(mean((Ysim-Yclean).^2,1));

fprintf('\nTest reconstruction RMSE:\n');

stateNames = {'delta','omega','eq_prime','ed_prime'};
outputNames = {'id','iq','Pg','Qg'};

for k = 1:4
    fprintf('%-12s : %.8e\n',stateNames{k},stateRMSE(k));
end

for k = 1:4
    fprintf('%-12s : %.8e\n',outputNames{k},outputRMSE(k));
end

%% 15. PLOT RECONSTRUCTION

figure('Color','w','Position',[50 50 1400 950]);

allNoisy = [Xnoisy,Ynoisy];
allClean = [Xclean,Yclean];
allSim = [Xsim,Ysim];

allNames = {'delta','omega','eq prime','ed prime', ...
    'id','iq','Pg','Qg'};

for k = 1:8

    subplot(4,2,k);

    plot(tTest,allNoisy(:,k),'Color',[0.75 0.75 0.75]);
    hold on;

    plot(tTest,allClean(:,k),'k','LineWidth',1.2);
    plot(tSim,allSim(:,k),'r','LineWidth',1.4);

    title(allNames{k});
    xlabel('Time (s)');
    grid on;

    if k==1
        legend('Noisy measurement','Clean reference', ...
            'Physical SG4 reconstruction','Location','best');
    end

end

sgtitle('Repository-Based MBAM: SG4 Physical Reconstruction');

exportgraphics(gcf, ...
    fullfile(cfg.outputDir, sprintf('sg4_reconstruction_%s.png', noiseTag)), ...
    'Resolution',300);

%% 16. SAVE RESULTS

results = struct();

results.pInitial = p0;
results.pFit = pFit;
results.pGeodesic = pPath;
results.pEndpoint = pEnd;

results.qFit = qFit;
results.qPath = qPath;
results.velocityPath = vPath;

results.singularValues = singularValues;
results.FIM = FIM;
results.J = J;

results.geodesicTime = tau;

results.stateRMSE = stateRMSE;
results.outputRMSE = outputRMSE;

results.Xsim = Xsim;
results.Ysim = Ysim;

results.clean.pInitial = p0;
results.noisy.pInitial = p0;

results.clean.relativeParameterChange = relativeChangeClean;
results.noisy.relativeParameterChange = relativeChangeNoisy;
results.derivativeScale = derivativeScale;

results.clean.singularValues = singularValuesClean;
results.clean.FIM = FIMclean;
results.clean.J = Jclean;
results.clean.conditionNumber = conditionNumberClean;
results.clean.pPath = pPathClean;
results.clean.pEndpoint = pEndClean;
results.clean.geodesicTime = tauClean;

results.noisy.singularValues = singularValuesNoisy;
results.noisy.FIM = FIMnoisy;
results.noisy.J = Jnoisy;
results.noisy.conditionNumber = conditionNumberNoisy;
results.noisy.pPath = pPathNoisy;
results.noisy.pEndpoint = pEndNoisy;
results.noisy.geodesicTime = tauNoisy;

save(fullfile(cfg.outputDir, sprintf('mbam_sg4_results_%s.mat', noiseTag)), ...
    'results','cfg','-v7.3');

fprintf('\n============================================\n');
fprintf(' MBAM SG4 ANALYSIS FINISHED\n');
fprintf('============================================\n');
fprintf('Results directory: %s\n',cfg.outputDir);
fprintf('Saved results MAT file.\n');
fprintf('Saved parameter evolution and reconstruction plots.\n');

%% ================================================================
%% LOCAL FUNCTIONS
%% ================================================================

function residual = model_residual(q,D,scale,target)

    prediction = model_manifold(q,D,scale);
    residual = prediction-target;

end

function r = model_manifold(q,D,scale)

    p = exp(q(:))';

    f = sg4_vector_field(D.X,D.U,p);

    r = reshape((f./scale)',[],1);

end

function J = numerical_jacobian(fun,q,h)

    q = q(:);
    n = numel(q);

    r0 = fun(q);

    J = zeros(numel(r0),n);

    for k = 1:n

        dq = zeros(n,1);
        dq(k) = h;

        rp = fun(q+dq);
        rm = fun(q-dq);

        J(:,k) = (rp-rm)/(2*h);

    end

end

function Avv = directional_second_derivative(fun,q,v,h)

    v = v(:);

    if norm(v)<1e-12
        Avv = zeros(size(fun(q)));
        return;
    end

    step = h/norm(v);

    rp = fun(q+step*v);
    r0 = fun(q);
    rm = fun(q-step*v);

    Avv = (rp-2*r0+rm)/(step^2);

end

function dy = mbam_geodesic_rhs(~,y,D,scale,cfg)

    n = numel(y)/2;

    q = y(1:n);
    v = y(n+1:end);

    fun = @(z) model_manifold(z,D,scale);

    J = numerical_jacobian(fun,q,cfg.jacobianStep);

    Avv = directional_second_derivative( ...
        fun,q,v,cfg.directionalStep);

    [U,S,V] = svd(J,'econ');

    singularValues = diag(S);

    tolerance = max(singularValues)*1e-10;

    keep = singularValues>tolerance;

    acceleration = zeros(n,1);

    if any(keep)

        acceleration = -V(:,keep)* ...
            ((U(:,keep)'*Avv)./singularValues(keep));

    end

    dy = [v;acceleration];

end

function f = sg4_vector_field(X,U,p)

    H2 = p(1);
    D = p(2);
    Td = p(3);
    Tq = p(4);
    xd = p(5);
    xq = p(6);
    xdp = p(7);
    xqp = p(8);

    delta = X(:,1);
    omega = X(:,2);
    eqp = X(:,3);
    edp = X(:,4);

    V = U(:,1);
    theta = U(:,2);
    Pm = U(:,3);
    vf = U(:,4);
    kPg = U(:,5);

    dth = delta-theta;

    id = (eqp-V.*cos(dth))./xdp;
    iq = (V.*sin(dth)-edp)./xqp;

    Pg = V.*sin(dth).*id + V.*cos(dth).*iq;

    f = zeros(size(X));

    f(:,1) = 2*pi*50*(omega-1);

    f(:,2) = (Pm-kPg.*Pg-D.*(omega-1))./H2;

    f(:,3) = (-eqp-(xd-xdp).*id+vf)./Td;

    f(:,4) = (-edp+(xq-xqp).*iq)./Tq;

end

function dx = sg4_rhs_model(t,x,tData,UData,p)

    u = interp1(tData,UData,t,'linear','extrap');

    dx = sg4_vector_field(x(:)',u,p)';

end

function Y = sg4_output_model(X,UData,p)

    xdp = p(7);
    xqp = p(8);

    V = UData(:,1);
    theta = UData(:,2);

    delta = X(:,1);
    eqp = X(:,3);
    edp = X(:,4);

    dth = delta-theta;

    id = (eqp-V.*cos(dth))./xdp;
    iq = (V.*sin(dth)-edp)./xqp;

    Pg = V.*sin(dth).*id + V.*cos(dth).*iq;
    Qg = V.*cos(dth).*id - V.*sin(dth).*iq;

    Y = [id,iq,Pg,Qg];

end
