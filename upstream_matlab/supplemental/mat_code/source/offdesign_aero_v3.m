function [L_D, Force_x, Force_y, Force_z] = offdesign_aero_v3(wav_x, wav_y, wav_z, wav_up_x, wav_up_y, wav_up_z, Ma, AOA)
% offdesign_aero_v3 - Hybrid Liu Wen (2019) + standard oblique shock pressure model.
%
% Hybrid strategy:
%   |theta| > theta_threshold_deg  -> Liu Wen eq (4) improved tangent-cone (large inclination)
%   |theta| <= theta_threshold_deg -> standard 2D oblique shock (small inclination)
%
% Both branches use exact theta-beta-M (fzero) for shock angle.
% Expansion (K < 0): Prandtl-Meyer hypersonic similarity (same as both v2 and v3).
%
% Reference:
%   Liu W et al., "Design method of a new hypersonic waverider
%   configuration", Sci Sin Tech, 2019.
%
% Inputs:
%   wav_x, wav_y, wav_z       - lower surface [M x nn] (program frame)
%   wav_up_x, wav_up_y, wav_up_z - upper surface [M x nn] (program frame)
%   Ma                         - freestream Mach number
%   AOA                        - angle of attack [deg]
%   theta_threshold_deg        - inclination threshold [deg]:
%                                above -> Liu Wen eq (4), below -> standard oblique shock
%
% Geometry/normal vectors/force synthesis retained from v2 (confirmed correct).

    theta_threshold_deg = 5;
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
    [Fx_l, Fy_l, Fz_l] = panel_integration(x_y_z_wav_r, Ma, M, nn, false, theta_threshold_deg);
    [Fx_u, Fy_u, Fz_u] = panel_integration(x_y_z_up_r, Ma, M, nn, true,  theta_threshold_deg);

    %% Combine forces (same sign convention as v2 / generatewaverider)
    Force_x = Fx_u + Fx_l;
    Force_y = Fy_u + Fy_l;
    Force_z = Fz_l + Fz_u;

    L_D = Force_z / Force_x;
end

function [Fx, Fy, Fz] = panel_integration(xingxian, Ma, M, nn, is_upper, theta_threshold_deg)
% Panel integration with hybrid pressure model.
% xingxian: [nn, 3, M] array. is_upper: true for upper surface.
%
% Normal vector convention (same as v2, confirmed correct):
%   Upper surface: nx, ny, nz all flipped (negative sign)
%   Lower surface: nx, ny, nz as computed from cross product

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

            %% Pressure model: hybrid Liu Wen eq (4) / standard oblique shock
            theta = asin(nx);
            p_ratio = hybrid_pressure(Ma, theta, gamma, fzero_opts, theta_threshold_deg);

            %% Force projection
            Fx = Fx + p_ratio * Aw * nx;
            Fy = Fy + p_ratio * Aw * ny;
            Fz = Fz + p_ratio * Aw * nz;
        end
    end
end

function p_ratio = hybrid_pressure(Ma, theta, gamma, fzero_opts, theta_threshold_deg)
% Hybrid pressure model:
%   |theta| > threshold -> Liu Wen eq (4) improved tangent-cone
%   |theta| <= threshold -> standard 2D oblique shock
%
% Expansion (K < 0): PM hypersonic similarity (same for both branches).

    if abs(theta) < 1e-12
        p_ratio = 1.0;
        return;
    end

    K = Ma * theta;

    gm1 = gamma - 1;
    gp1 = gamma + 1;

    if K >= 0
        % --- Compression ---

        beta_guess = asin(1/Ma) + 0.05;
        if beta_guess <= theta
            beta_guess = theta + 0.1;
        end
        try
            beta = fzero(@(b) shock_residual(b, theta, Ma, gamma), ...
                         beta_guess, fzero_opts);
            sin_b = sin(beta);

            theta_deg = abs(theta) * 180/pi;

            if theta_deg > theta_threshold_deg
                % --- Large inclination: Liu Wen eq (4) ---
                cos_b = cos(beta);

                Term1 = 2*gamma/gp1 * Ma^2 * sin_b^2 - gm1/gp1;
                Term2_denom = 1 + gm1/2 * Ma^2 * sin_b^2;
                Term2 = 1 + gamma * Ma * (beta - theta)^2 * cos_b^2 / Term2_denom;
                p_ratio = 1 + Term1 / Term2;
            else
                % --- Small inclination: standard oblique shock (v2) ---
                p_ratio = 1 + 2*gamma/gp1 * (Ma^2 * sin_b^2 - 1);
            end

            % Sanity check
            if p_ratio < 1.0 || ~isfinite(p_ratio)
                p02 = ((gp1^2*Ma^2/(4*gamma*Ma^2 - 2*gm1))^(gamma/gm1)) * (1-gamma+2*gamma*Ma^2)/gp1;
                p_ratio = 1 + (p02 - 1) * sin(theta)^2;
            end
        catch
            % fzero failed (detached shock) -> Newtonian fallback
            p02 = ((gp1^2*Ma^2/(4*gamma*Ma^2 - 2*gm1))^(gamma/gm1)) * (1-gamma+2*gamma*Ma^2)/gp1;
            p_ratio = 1 + (p02 - 1) * sin(theta)^2;
        end

    else
        % --- Expansion: PM hypersonic similarity ---
        base = 1 + gm1/2 * K;
        if base > 0
            p_ratio = base^(2*gamma/gm1);
        else
            p_ratio = 1e-6;
        end
    end
end

function res = shock_residual(beta, theta, M, gamma)
% Exact theta-beta-M relation for oblique shock.
    res = tan(theta) - 2/tan(beta) * (M^2 * sin(beta)^2 - 1) / ...
          (M^2 * (gamma + cos(2*beta)) + 2);
end