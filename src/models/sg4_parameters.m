function p = sg4_parameters()
% SG4_PARAMETERS
% Parameters for the fourth-order synchronous generator model.
% Values taken from Appendix A of the reference paper.

% Generator ratings
p.Sn = 190;             % MVA
p.Vn = 15.75;           % kV
p.f  = 50;              % Hz

% Angular speed
p.Omega_b = 2*pi*p.f;   % rad/s
p.omega0  = 1.0;        % pu

% Mechanical parameters
p.H = 12.467/2;         % Inertia constant
p.D = 1.9;              % Damping coefficient

% Transient time constants
p.Td0_prime = 6.5;      % s
p.Tq0_prime = 0.2;      % s

% Synchronous reactances
p.xd = 0.655;
p.xq = 0.487;

% Transient reactances
p.xd_prime = 0.196;
p.xq_prime = 0.263;

end