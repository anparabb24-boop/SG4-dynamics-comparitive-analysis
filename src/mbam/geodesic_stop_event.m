function [value, isterminal, direction] = geodesic_stop_event(~, y)
    phi = y(1:length(y)/2);
    value = max(abs(phi)) - 15; 
    isterminal = 1;
    direction = 1;
end
