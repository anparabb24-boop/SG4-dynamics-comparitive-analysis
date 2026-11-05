function [value, isterminal, direction] = geodesic_stop_event(~, y)
    phi = y(1:length(y)/2);
    
    % Stop if any log-parameter goes out of bounds (e.g., parameter < 1e-6 or > 1e6)
    % In log-space, this corresponds to abs(phi) > 14 (~6 orders of magnitude)
    bound_limit = 14;
    value = max(abs(phi)) - bound_limit; 
    
    isterminal = 1; % Halt integration
    direction = 1;  % Trigger when crossing from inside to outside
end