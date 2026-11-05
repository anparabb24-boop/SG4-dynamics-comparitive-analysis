function dydtau = geodesic_rhs_2param(tau, y, p_nominal, u_data, t_span)
    n_params = 2;
    phi = y(1:n_params);
    v = y(n_params+1:end);
    
    [~, ~, Gamma] = compute_geodesic_derivatives_2param(phi, p_nominal, u_data, t_span);
    
    dv = zeros(n_params, 1);
    for mu = 1:n_params
        term = 0;
        for alpha = 1:n_params
            for beta = 1:n_params
                term = term + Gamma(mu, alpha, beta) * v(alpha) * v(beta);
            end
        end
        dv(mu) = -term;
    end
    dydtau = [v; dv];
end