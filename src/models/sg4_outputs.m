function y = sg4_outputs(x, V, theta, p)
% SG4_OUTPUTS
% Calculate algebraic currents and generator powers.

delta = x(1);
eqp   = x(3);
edp   = x(4);

angle_diff = delta - theta;

% Stator currents
id = (eqp - V*cos(angle_diff))/p.xd_prime;

iq = (V*sin(angle_diff) - edp)/p.xq_prime;

% Generator powers
Pg = V*sin(angle_diff)*id + ...
     V*cos(angle_diff)*iq;

Qg = V*cos(angle_diff)*id - ...
     V*sin(angle_diff)*iq;

% Return outputs
y.id = id;
y.iq = iq;
y.Pg = Pg;
y.Qg = Qg;

end