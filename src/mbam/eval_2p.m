function Y_flat = eval_2p(phi, p_nominal, u_data, t_span, n_time)
    theta = exp(phi);
    p = p_nominal;
    p.D = theta(1);
    p.Td0_prime = theta(2);
    [~, x_traj] = ode15s(@(t,x) sg4_rhs(t, x, p, u_data), t_span, u_data.x0);
    Y = zeros(n_time, 2);
    for i = 1:n_time
        out = sg4_outputs(x_traj(i,:)', u_data.V(i), u_data.theta(i), p);
        Y(i, :) = [x_traj(i, 1), out.Pg];
    end
    Y_flat = Y(:);
end