function dx = sg3_reduced_rhs(t, x, p, u)

% Reduced three-state SG model
% States: delta, omega, ed_prime

delta = x(1);
omega = x(2);
edp   = x(3);

% External inputs
V     = interp1(u.t, u.V, t, 'linear', 'extrap');
theta = interp1(u.t, u.theta, t, 'linear', 'extrap');
Pm    = interp1(u.t, u.Pm, t, 'linear', 'extrap');
vf    = interp1(u.t, u.vf, t, 'linear', 'extrap');

% Algebraic calculation of eq_prime (Paper equation A4)
eqp = (p.xd_prime/p.xd)*vf + ...
      ((p.xd-p.xd_prime)/p.xd)*V*cos(delta-theta);

% Stator currents
id = (eqp - V*cos(delta-theta))/p.xd_prime;
iq = (V*sin(delta-theta) - edp)/p.xq_prime;

% Generator active power
Pg = V*sin(delta-theta)*id + ...
     V*cos(delta-theta)*iq;

% Differential equations
ddelta = p.Omega_b*(omega-p.omega0);

domega = (Pm-Pg-p.D*(omega-p.omega0))/(2*p.H);

dedp = (-(p.xq/p.xq_prime)*edp + ...
        ((p.xq-p.xq_prime)/p.xq_prime)*V*sin(delta-theta)) ...
        /p.Tq0_prime;

dx = [ddelta; domega; dedp];

end