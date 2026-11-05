function [t_all, Y] = sg4_sensitivity_simulate(p, x0, Pm0, vf0)

% Simulate the SG4 model using the same voltage disturbance.
% Output columns: [Pg, Qg, id, iq]

theta0 = 0;

intervals = [
    0.0  0.3  1.0;
    0.3  0.5  0.8;
    0.5  1.0  1.0
];

t_all = [];
x_all = [];
x_current = x0;

options = odeset('RelTol',1e-8,'AbsTol',1e-10);

for k = 1:size(intervals,1)

    t_start = intervals(k,1);
    t_end = intervals(k,2);
    V_current = intervals(k,3);

    % Use the same 101 output points per interval
    t_segment = linspace(t_start,t_end,101);

    % Define constant inputs
    u.t = [t_start t_end];
    u.V = [V_current V_current];
    u.theta = [theta0 theta0];
    u.Pm = [Pm0 Pm0];
    u.vf = [vf0 vf0];

    % Integrate the SG4 model
    [t_segment,x_segment] = ode15s( ...
        @(t,x) sg4_rhs(t,x,p,u), ...
        t_segment,x_current,options);

    % Store trajectories
    if k == 1
        t_all = t_segment;
        x_all = x_segment;
    else
        t_all = [t_all; t_segment(2:end)];
        x_all = [x_all; x_segment(2:end,:)];
    end

    x_current = x_segment(end,:)';

end

% Calculate algebraic outputs
Y = zeros(length(t_all),4);

for k = 1:length(t_all)

    % Apply the voltage corresponding to this time
    if t_all(k) < 0.3
        V_current = 1.0;
    elseif t_all(k) < 0.5
        V_current = 0.8;
    else
        V_current = 1.0;
    end

    x = x_all(k,:)';

    y = sg4_outputs(x,V_current,theta0,p);

    Y(k,:) = [y.Pg, y.Qg, y.id, y.iq];

end

end