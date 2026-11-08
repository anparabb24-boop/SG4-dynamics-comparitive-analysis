function dx = sg4_rhs(t, x, p, u)
% SG4_RHS
% Right-hand side of the fourth-order synchronous generator model.
%
% State vector:
% x(1) = delta
% x(2) = omega
% x(3) = eq_prime
% x(4) = ed_prime
%
% Input signals (each interpolated linearly over u.t):
% u.t     time points of the current interval
% u.V     terminal voltage magnitude
% u.theta terminal voltage angle
% u.Pm    mechanical power
% u.vf    exciter (field) voltage
% u.kPg   (optional) electrical-power scaling factor, 1 = no change

% Extract states
delta = x(1);
omega = x(2);
eqp   = x(3);
edp   = x(4);

% Interpolate input signals at the current time
V     = interp1(u.t, u.V,     t, 'linear', 'extrap');
theta = interp1(u.t, u.theta, t, 'linear', 'extrap');
Pm    = interp1(u.t, u.Pm,    t, 'linear', 'extrap');
vf    = interp1(u.t, u.vf,    t, 'linear', 'extrap');

% Electrical power scaling factor (represents a sudden change in the
% electrical power drawn from the generator, e.g. partial load rejection)
if isfield(u,'kPg')
    kPg = interp1(u.t, u.kPg, t, 'linear', 'extrap');
else
    kPg = 1;
end

% Rotor angle difference
angle_diff = delta - theta;

% Algebraic stator currents
id = (eqp - V*cos(angle_diff))/p.xd_prime;
iq = (V*sin(angle_diff) - edp)/p.xq_prime;

% Generator active power
Pg = V*sin(angle_diff)*id + ...
     V*cos(angle_diff)*iq;

% Differential equations
ddelta = p.Omega_b*(omega - p.omega0);

domega = (Pm - kPg*Pg - p.D*(omega - p.omega0))/(2*p.H);

deqp = (-eqp - (p.xd - p.xd_prime)*id + vf) ...
       /p.Td0_prime;

dedp = (-edp + (p.xq - p.xq_prime)*iq) ...
       /p.Tq0_prime;

% Return state derivatives
dx = [ddelta; domega; deqp; dedp];

end