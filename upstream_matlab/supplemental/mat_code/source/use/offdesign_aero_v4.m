function [L_D, Force_x, Force_y, Force_z] = offdesign_aero_v4(wav_x, wav_y, wav_z, wav_up_x, wav_up_y, wav_up_z, Ma, AOA)
% offdesign_aero_v2 - Improved pressure model (2018) streamline-tracing approach.
% Traces streamlines along each spanwise station using Eq.(1): Vi = ni x Vinf x ni.
% Each panel's effective inclination is computed relative to the LOCAL flow
% direction (turned by upstream panels), not the freestream.
% Pressure model: oblique shock (compression) + PM (expansion) + Newtonian
% for panels where local Mach cannot sustain attached shock.

    %% rot_x(180): program frame -> real physics frame
    wav_z = -wav_z;
    wav_y = -wav_y;
    wav_up_z = -wav_up_z;
    wav_up_y = -wav_up_y;

    [M, nn] = size(wav_x);

   %% Build 3D point arrays in [nn, 3, M] format
    x_y_z_wav = zeros(nn, 3, M);
    x_y_z_up  = zeros(nn, 3, M);
    for i = 1:M
        x_y_z_wav(:, 1, i) = wav_x(i, :)';
        x_y_z_wav(:, 2, i) = wav_y(i, :)';
        x_y_z_wav(:, 3, i) = wav_z(i, :)';
        x_y_z_up(:, 1, i)  = wav_up_x(i, :)';
        x_y_z_up(:, 2, i)  = wav_up_y(i, :)';
        x_y_z_up(:, 3, i)  = wav_up_z(i, :)';
    end

    %% Rotate to off-design freestream-aligned frame (V_inf -> [1,0,0])
    Vector_V = [cosd(AOA), 0, sind(AOA)]';
    rat = rotate_matrix(Vector_V, [1, 0, 0]');

    x_y_z_wav_perm = permute(x_y_z_wav, [2, 1, 3]);
    x_y_z_wav_r = permute(pagemtimes(rat, x_y_z_wav_perm), [2, 1, 3]);
    x_y_z_up_perm = permute(x_y_z_up, [2, 1, 3]);
    x_y_z_up_r = permute(pagemtimes(rat, x_y_z_up_perm), [2, 1, 3]);

    %% Panel integration
    [Fx_l, Fy_l, Fz_l] = panel_integration(x_y_z_wav_r, Ma, M, nn, false);
    [Fx_u, Fy_u, Fz_u] = panel_integration(x_y_z_up_r, Ma, M, nn, true);

    %% Combine forces (same sign convention as generatewaverider)
    Force_x = Fx_u + Fx_l;
    Force_y = Fy_u + Fy_l;
    Force_z = Fz_l + Fz_u;

    L_D = Force_z / Force_x;
end

function [Fx, Fy, Fz] = panel_integration(xingxian, Ma, M, nn, is_upper)
% Panel integration with local inclination pressure model.
% xingxian: [nn, 3, M] array. is_upper: true for upper surface.
% The cross product gives opposite normals for upper vs lower.
%
% For nx/ny: flip upper surface to ensure consistent force direction.
% - Force_x and Force_y use direct summation (Fx_u + Fx_l, Fy_u + Fy_l)
% - So nx and ny must be flipped for upper surface so forces add correctly.
%
% For nz: do NOT flip. Force_z uses subtraction (Fz_l - Fz_u),
% so the '-' already handles the sign difference.
    gamma = 1.4;
    fzero_opts = optimset('Display', 'off', 'TolX', 1e-12);

    Fx = 0; Fy = 0; Fz = 0;

    for i = 1:M-1
        for j = 1:nn-1
            P1 = [xingxian(j,   1, i),   xingxian(j,   2, i),   xingxian(j,   3, i)  ];
            P2 = [xingxian(j,   1, i+1), xingxian(j,   2, i+1), xingxian(j,   3, i+1)];
            P3 = [xingxian(j+1, 1, i+1), xingxian(j+1, 2, i+1), xingxian(j+1, 3, i+1)];
            P4 = [xingxian(j+1, 1, i),   xingxian(j+1, 2, i),   xingxian(j+1, 3, i)  ];

            C = cross(P3 - P1, P2 - P4);
            C_mod = norm(C);

            if C_mod < 1e-14
                continue;
            end

            Aw = C_mod / 2;
            if is_upper
                nx = -C(1) / C_mod;
                ny = -C(2) / C_mod;
                nz = -C(3) / C_mod;
            else
                nx = C(1) / C_mod;
                ny = C(2) / C_mod;
                nz = C(3) / C_mod;
            end
            %% Pressure model: theta uses consistent nx for both surfaces
            theta = asin(nx);
            p_ratio = local_pressure_ratio(Ma, theta, gamma, fzero_opts);

            %% Force projection
            Fx = Fx + p_ratio * Aw * nx;
            Fy = Fy + p_ratio * Aw * ny;
            Fz = Fz + p_ratio * Aw * nz;
        end
    end
end

function p_ratio = local_pressure_ratio(Ma, theta, gamma, fzero_opts)
% Liu Wen (2019) unified formula:
%   K = Ma * theta >= 0 -> oblique shock (theta-beta-M)
%   K = Ma * theta < 0  -> PM hypersonic similarity

    if abs(theta) < 1e-10
        p_ratio = 1.0;
        return;
    end

    K = Ma * theta;

    if K >= 0
        beta_guess = asin(1/Ma) + 0.05;
        if beta_guess <= theta
            beta_guess = theta + 0.1;
        end
        try
            % beta = fzero(@(b) shock_residual(b, theta, Ma, gamma), ...
            %              beta_guess, fzero_opts);
            % p_ratio = 1 + 2*gamma/(gamma+1) * (Ma^2 * sin(beta)^2 - 1);
            [beta,~,~,~,pf_cone] = conical_flowfield(Ma,theta,gamma);
            p_ratio = pf_cone(end);
        catch
            % Correct modified Newtonian from forcecst.m (catch for detached shock)
            p02_over_p0 = (((gamma+1)^2*Ma^2/(4*gamma*Ma^2-2*(gamma-1)))^(gamma/(gamma-1))) * (1-gamma+2*gamma*Ma^2)/(gamma+1);
            p_ratio = 1 + (p02_over_p0 - 1) * sin(theta)^2;
        end
    else
        p_ratio = (1 + (gamma-1)/2 * K)^(2*gamma/(gamma-1));
    end
end

function res = shock_residual(beta, theta, M, gamma)
    res = tan(theta) - 2/tan(beta) * (M^2 * sin(beta)^2 - 1) / ...
          (M^2 * (gamma + cos(2*beta)) + 2);
end
