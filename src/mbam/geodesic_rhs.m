function dydtau = geodesic_rhs(tau, y, p_nominal, u_data, t_span)
    % GEODESIC_RHS First-order system representation of the second-order geodesic equation.
    % y = [phi; v] where phi = log(theta) and v = d(phi)/d(tau)
    
    n_params = length(y) / 2;
    phi = y(1:n_params);
    v = y(n_params+1:end);
    
    % Compute model outputs, Jacobian, and Christoffel symbols
    [~, ~, Gamma] = compute_geodesic_derivatives(phi, p_nominal, u_data, t_span);
    
    % Compute acceleration: dv^mu / dtau = - sum_{alpha,beta} Gamma^mu_{alpha,beta} v^alpha v^beta
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