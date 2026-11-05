function [Y, J, Gamma] = compute_geodesic_derivatives_2param(phi, p_nominal, u_data, t_span)
    n_params = 2;
    theta = exp(phi);
    p = p_nominal;
    p.D = theta(1);
    p.Td0_prime = theta(2);
    
    [t, x_traj] = ode15s(@(t,x) sg4_rhs(t, x, p, u_data), t_span, u_data.x0);
    n_time = length(t);
    Y = zeros(n_time, 2);
    for i = 1:n_time
        out = sg4_outputs(x_traj(i,:)', u_data.V(i), u_data.theta(i), p);
        Y(i, :) = [x_traj(i, 1), out.Pg];
    end
    Y_flat = Y(:);
    n_tot = length(Y_flat);
    
    eps_fd = 1e-5;
    J = zeros(n_tot, n_params);
    for j = 1:n_params
        phi_p = phi; phi_p(j) = phi_p(j) + eps_fd;
        phi_m = phi; phi_m(j) = phi_m(j) - eps_fd;
        J(:, j) = (eval_2p(phi_p, p_nominal, u_data, t_span, n_time) - ...
                   eval_2p(phi_m, p_nominal, u_data, t_span, n_time)) / (2 * eps_fd);
    end
    
    I = J' * J;
    I_inv = inv(I + 1e-10 * eye(2));
    
    % Christoffel symbols for 2 parameters
    d2Y = zeros(n_tot, 2, 2);
    for alpha = 1:2
        phi_p = phi; phi_p(alpha) = phi_p(alpha) + eps_fd;
        phi_m = phi; phi_m(alpha) = phi_m(alpha) - eps_fd;
        [~, J_p] = compute_geodesic_derivatives_2param(phi_p, p_nominal, u_data, t_span);
        [~, J_m] = compute_geodesic_derivatives_2param(phi_m, p_nominal, u_data, t_span);
        d2Y(:, :, alpha) = (J_p - J_m) / (2 * eps_fd);
    end
    
    Gamma = zeros(2, 2, 2);
    for mu = 1:2
        for alpha = 1:2
            for beta = 1:2
                term = 0;
                for nu = 1:2
                    term = term + I_inv(mu, nu) * (J(:, nu)' * d2Y(:, alpha, beta));
                end
                Gamma(mu, alpha, beta) = term;
            end
        end
    end
end