function [a, tangent_point] = find_ellipse_a(points, h, k, b)
    % points: N x 2 matrix, each row is [x_i, y_i]
    % h, k: center of the ellipse
    % b: minor axis radius
    % a: major axis radius to be found (output)
    % tangent_point: the point where the ellipse is tangent to the discrete curve [x_tangent, y_tangent]
    tol = 1e-6;
    a = NaN;
    tangent_point = [];
    
    [N, ~] = size(points);
    valid_a = [];
    valid_points = []; % Store tangent points corresponding to valid a
    
    for i = 1:N
        xi = points(i, 1);
        yi = points(i, 2);
        
        % Check if (yi - k)^2 / b^2 < 1
        y_term = (yi - k)^2 / b^2;
        if y_term < 1-tol
            % Compute a_i for this point
            a_i = abs(xi - h) / sqrt(1 - y_term);
            
            % Check if all other points are outside the ellipse
            valid = true;
            for j = 1:N
                if j == i
                    continue;
                end
                xj = points(j, 1);
                yj = points(j, 2);
                D = (xj - h)^2 / (a_i^2) + (yj - k)^2 / (b^2);
                if D < 1-tol
                    valid = false;
                    break;
                end
            end
            
            if valid
                valid_a = [valid_a; a_i];
                valid_points = [valid_points; xi, yi]; % Store the tangent point
            end
        end
    end
    
    if ~isempty(valid_a)
        % Choose the smallest a (most restrictive ellipse)
        [a, idx] = min(valid_a);
        tangent_point = valid_points(idx, :);
    end
end
