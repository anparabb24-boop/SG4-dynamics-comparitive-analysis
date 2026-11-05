function [Y, J, Gamma] = compute_geodesic_derivatives(phi, p_nominal, u_data, t_span)
    % COMPUTE_GEODESIC_DERIVATIVES (Optimized) Computes outputs, Jacobian, 
    % and Christoffel symbols for MBAM geodesic integration using parfor.
    
    n_params = length(phi);
    theta = exp(phi); 
    p = update_parameters(p_nominal, theta);
    
    % 1. Baseline simulation
    [t, x_traj] = ode15s(@(t,x) sg4_rhs(t, x, p, u_data), t_span, u_data.x0);
    n_time = length(t);
    n_outputs = 4; % [delta, omega, ed_prime, Pg]
    
    Y = zeros(n_time, n_outputs);
    for i = 1:n_time
        out = sg4_outputs(x_traj(i,:)', u_data.V(i), u_data.theta(i), p);
        Y(i, :) = [x_traj(i, 1), x_traj(i, 2), x_traj(i, 3), out.Pg];
    end
    Y_flat = Y(:);
    n_tot_out = length(Y_flat);
    
    % 2. Compute Jacobian J = dY/dphi using parallel central differences
    eps_fd = 1e-5;
    J = zeros(n_tot_out, n_params);
    
    Y_plus_cell = cell(n_params, 1);
    Y_minus_cell = cell(n_params, 1);
    
    parfor j = 1:n_params
        phi_p = phi; phi_p(j) = phi_p(j) + eps_fd;
        phi_m = phi; phi_m(j) = phi_m(j) - eps_fd;
        Y_plus_cell{j} = evaluate_flat_outputs(phi_p, p_nominal, u_data, t_span, n_time, n_outputs);
        Y_minus_cell{j} = evaluate_flat_outputs(phi_m, p_nominal, u_data, t_span, n_time, n_outputs);
    end
    
    for j = 1:n_params
        J(:, j) = (Y_plus_cell{j} - Y_minus_cell{j}) / (2 * eps_fd);
    end
    
    % 3. Fisher Information Matrix and Inverse
    I = J' * J;
    I_inv = inv(I + 1e-10 * eye(size(I))); 
    
    % 4. Compute second derivatives and Christoffel symbols
    d2Y = zeros(n_tot_out, n_params, n_params);
    J_plus_cell = cell(n_params, 1);
    J_minus_cell = cell(n_params, 1);
    
    parfor alpha = 1:n_params
        phi_p = phi; phi_p(alpha) = phi_p(alpha) + eps_fd;
        phi_m = phi; phi_m(alpha) = phi_m(alpha) - eps_fd;
        [~, J_plus_cell{alpha}] = compute_geodesic_derivatives_light(phi_p, p_nominal, u_data, t_span, n_time, n_outputs);
        [~, J_minus_cell{alpha}] = compute_geodesic_derivatives_light(phi_m, p_nominal, u_data, t_span, n_time, n_outputs);
    end
    
    for alpha = 1:n_params
        d2Y(:, :, alpha) = (J_plus_cell{alpha} - J_minus_cell{alpha}) / (2 * eps_fd);
    end
    
    % Assemble Christoffel symbols
    Gamma = zeros(n_params, n_params, n_params);
    for mu = 1:n_params
        for alpha = 1:n_params
            for beta = 1:n_params
                term = 0;
                for nu = 1:n_params
                    dot_prod = J(:, nu)' * d2Y(:, alpha, beta);
                    term = term + I_inv(mu, nu) * dot_prod;
                end
                Gamma(mu, alpha, beta) = term;
            end
        end
    end
end

function Y_flat = evaluate_flat_outputs(phi, p_nominal, u_data, t_span, n_time, n_outputs)
    try
        theta = exp(phi);
        p = update_parameters(p_nominal, theta);
        
        % Use strict tolerances and handle potential failures gracefully
        options = odeset('RelTol', 1e-4, 'AbsTol', 1e-5);
        [t, x_traj] = ode15s(@(t,x) sg4_rhs(t, x, p, u_data), t_span, u_data.x0, options);
        
        % If solver failed or returned fewer points than expected, throw error to catch block
        if length(t) < length(u_data.t)
            error('ODE solver truncated steps.');
        end
        
        Y = zeros(n_time, n_outputs);
        for i = 1:n_time
            out = sg4_outputs(x_traj(i,:)', u_data.V(i), u_data.theta(i), p);
            Y(i, :) = [x_traj(i, 1), x_traj(i, 2), x_traj(i, 3), out.Pg];
        end
        Y_flat = Y(:);
    catch
        % Fallback: return large numbers or NaNs so the optimizer/geodesic knows this step is invalid
        Y_flat = 1e6 * ones(n_time * n_outputs, 1);
    end
end

function [Y_flat, J] = compute_geodesic_derivatives_light(phi, p_nominal, u_data, t_span, n_time, n_outputs)
    theta = exp(phi);
    p = update_parameters(p_nominal, theta);
    [~, x_traj] = ode15s(@(t,x) sg4_rhs(t, x, p, u_data), t_span, u_data.x0);
    Y = zeros(n_time, n_outputs);
    for i = 1:n_time
        out = sg4_outputs(x_traj(i,:)', u_data.V(i), u_data.theta(i), p);
        Y(i, :) = [x_traj(i, 1), x_traj(i, 2), x_traj(i, 3), out.Pg];
    end
    Y_flat = Y(:);
    
    eps_fd = 1e-5;
    n_params = length(phi);
    J = zeros(length(Y_flat), n_params);
    
    % Serial lightweight loop for internal Jacobian evaluation
    for j = 1:n_params
        phi_p = phi; phi_p(j) = phi_p(j) + eps_fd;
        phi_m = phi; phi_m(j) = phi_m(j) - eps_fd;
        J(:, j) = (evaluate_flat_outputs(phi_p, p_nominal, u_data, t_span, n_time, n_outputs) - ...
                   evaluate_flat_outputs(phi_m, p_nominal, u_data, t_span, n_time, n_outputs)) / (2 * eps_fd);
    end
end

function p = update_parameters(p_base, theta)
    p = p_base;
    p.H = theta(1);
    p.D = theta(2);
    p.Td0_prime = theta(3);
    p.Tq0_prime = theta(4);
    p.xd = theta(5);
    p.xq = theta(6);
    p.xd_prime = theta(7);
    p.xq_prime = theta(8);
end