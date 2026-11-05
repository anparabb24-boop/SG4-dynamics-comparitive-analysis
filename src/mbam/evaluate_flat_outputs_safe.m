function Y_flat = evaluate_flat_outputs_safe(phi, p_nominal, u_data, t_span, n_time, n_outputs)
    try
        % Convert from log-parameters to physical parameters
        theta = exp(phi);
        p = update_parameters(p_nominal, theta);
        
        % Strict integration tolerances to handle stiffness safely
        options = odeset('RelTol', 1e-5, 'AbsTol', 1e-6, 'MaxStep', 0.05);
        [t, x_traj] = ode15s(@(t,x) sg4_rhs(t, x, p, u_data), t_span, u_data.x0, options);
        
        % If solver failed or truncated time steps, throw to catch
        if length(t) < length(u_data.t)
            error('ODE solver truncated or failed.');
        end
        
        Y = zeros(n_time, n_outputs);
        for i = 1:n_time
            out = sg4_outputs(x_traj(i,:)', u_data.V(i), u_data.theta(i), p);
            Y(i, :) = [x_traj(i, 1), x_traj(i, 2), x_traj(i, 3), out.Pg];
        end
        Y_flat = Y(:);
    catch
        % Fallback penalty vector for unstable parameter sets
        Y_flat = 1e6 * ones(n_time * n_outputs, 1);
    end
end

function p = update_parameters(p_base, theta)
    p = p_base;
    % Map vector components back to physical parameters
    if length(theta) == 2
        p.D = theta(1);
        p.Td0_prime = theta(2);
    else
        p.H = theta(1);
        p.D = theta(2);
        p.Td0_prime = theta(3);
        p.Tq0_prime = theta(4);
        p.xd = theta(5);
        p.xq = theta(6);
        p.xd_prime = theta(7);
        p.xq_prime = theta(8);
    end
end
