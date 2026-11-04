function [Fx, Fy, Fz, nz_stats] = panel_integration_diag2(xingxian, Ma, M, nn, label)
%% Same as panel_integration_diag but without fprintf (caller handles output)
    gamma = 1.4;
    fzero_opts = optimset('Display', 'off', 'TolX', 1e-12);
    Fx = 0; Fy = 0; Fz = 0;
    nz_min = inf; nz_max = -inf; nz_sum = 0; nz_count = 0;
    for i = 1:M-1
        for j = 1:nn-1
            P1 = [xingxian(j,1,i), xingxian(j,2,i), xingxian(j,3,i)];
            P2 = [xingxian(j,1,i+1), xingxian(j,2,i+1), xingxian(j,3,i+1)];
            P3 = [xingxian(j+1,1,i+1), xingxian(j+1,2,i+1), xingxian(j+1,3,i+1)];
            P4 = [xingxian(j+1,1,i), xingxian(j+1,2,i), xingxian(j+1,3,i)];
            C = cross(P3 - P1, P2 - P4);
            C_mod = norm(C);
            if C_mod < 1e-14, continue; end
            Aw = C_mod / 2;
            nx = C(1) / C_mod;
            ny = C(2) / C_mod;
            nz = C(3) / C_mod;
            nz_min = min(nz_min, nz); nz_max = max(nz_max, nz);
            nz_sum = nz_sum + nz; nz_count = nz_count + 1;
            theta = asin(nx);
            p_ratio = local_pressure_ratio2(Ma, theta, gamma, fzero_opts);
            Fx = Fx + p_ratio * Aw * nx;
            Fy = Fy + p_ratio * Aw * ny;
            Fz = Fz + p_ratio * Aw * nz;
        end
    end
    nz_stats = struct('min', nz_min, 'max', nz_max, 'mean', nz_sum/nz_count);
    fprintf('%s: nz [%.4f, %.4f] mean=%.4f\n', label, nz_min, nz_max, nz_sum/nz_count);
end

function p_ratio = local_pressure_ratio2(Ma, theta, gamma, fzero_opts)
    if abs(theta) < 1e-10, p_ratio = 1.0; return; end
    K = Ma * theta;
    if K >= 0
        beta_guess = asin(1/Ma) + 0.05;
        if beta_guess <= theta, beta_guess = theta + 0.1; end
        try
            beta = fzero(@(b) shock_residual2(b, theta, Ma, gamma), beta_guess, fzero_opts);
            p_ratio = 1 + 2*gamma/(gamma+1) * (Ma^2 * sin(beta)^2 - 1);
        catch
            p_ratio = 1 + gamma * Ma^2 * theta^2;
        end
    else
        p_ratio = (1 + (gamma-1)/2 * K)^(2*gamma/(gamma-1));
    end
end

function res = shock_residual2(beta, theta, M, gamma)
    res = tan(theta) - 2/tan(beta) * (M^2 * sin(beta)^2 - 1) / (M^2 * (gamma + cos(2*beta)) + 2);
end