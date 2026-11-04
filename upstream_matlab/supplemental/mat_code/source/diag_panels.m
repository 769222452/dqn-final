
%% Diagnostic: panel angle distribution at various AOAs
clc; clear;
addpath('D:\ZXG\Codex-project\Project_M15\.code');

load('wave_rider_optimization_final.mat');
[sorted_front, sort_idx] = sortrows(pareto_front, 1);
sorted_design = design_vars(sort_idx, :);
i = 45;

angle_shock_a1 = sorted_design(i,1); angle_shock_a2 = sorted_design(i,2);
angle_shock_b1 = sorted_design(i,3); angle_shock_b2 = sorted_design(i,4);
c1 = 1.0331; c2 = 1.0331;
d1 = sorted_design(i,5); AOA_prog = -4; a_xishu_1 = sorted_design(i,6); Ma = 15;

[wav_x, wav_y, wav_z] = generatewaverider(Ma, angle_shock_a1, angle_shock_a2, ...
    angle_shock_b1, angle_shock_b2, c1, c2, d1, AOA_prog, a_xishu_1);

% Upper surface
[M, nn] = size(wav_x);
V = [cosd(AOA_prog), 0, sind(AOA_prog)];
wav_up_x = zeros(M, nn); wav_up_y = zeros(M, nn); wav_up_z = zeros(M, nn);
x_design = max(wav_x(:));
for ii = 1:M
    x_le = wav_x(ii, 1); y_le = wav_y(ii, 1); z_le = wav_z(ii, 1);
    for jj = 1:nn
        wav_up_x(ii, jj) = x_le + (x_design - x_le) * (jj-1)/(nn-1);
        wav_up_y(ii, jj) = y_le + (wav_up_x(ii, jj) - x_le) * V(2) / V(1);
        wav_up_z(ii, jj) = z_le + (wav_up_x(ii, jj) - x_le) * V(3) / V(1);
    end
end

% Rotate to real frame
wav_z_r = -wav_z; wav_y_r = -wav_y;
wav_up_z_r = -wav_up_z; wav_up_y_r = -wav_up_y;

for AOA_real = [2, 4, 6, 8, 10]
    V_ref = [cosd(AOA_real), 0, sind(AOA_real)]';
    rat = rotate_matrix(V_ref, [1, 0, 0]');
    
    % Build array for lower surface
    xyz_low = zeros(nn, 3, M);
    for ii = 1:M
        xyz_low(:, 1, ii) = wav_x(ii, :)';
        xyz_low(:, 2, ii) = wav_y_r(ii, :)';
        xyz_low(:, 3, ii) = wav_z_r(ii, :)';
    end
    xyz_low_r = permute(pagemtimes(rat, permute(xyz_low, [2,1,3])), [2,1,3]);
    
    % Compute panel angles
    thetas = [];
    for ii = 1:M-1
        for jj = 1:nn-1
            P1 = [xyz_low_r(jj,1,ii),   xyz_low_r(jj,2,ii),   xyz_low_r(jj,3,ii)];
            P2 = [xyz_low_r(jj,1,ii+1),   xyz_low_r(jj,2,ii+1),   xyz_low_r(jj,3,ii+1)];
            P3 = [xyz_low_r(jj+1,1,ii+1), xyz_low_r(jj+1,2,ii+1), xyz_low_r(jj+1,3,ii+1)];
            P4 = [xyz_low_r(jj+1,1,ii),   xyz_low_r(jj+1,2,ii),   xyz_low_r(jj+1,3,ii)];
            C = cross(P3 - P1, P2 - P4);
            if norm(C) < 1e-14, continue; end
            ni = C / norm(C);
            theta = asin(ni(1));
            thetas = [thetas, theta];
        end
    end
    
    thetas_deg = rad2deg(thetas);
    fprintf('AOA=%d deg: theta range [%.1f, %.1f] deg, mean=%.1f deg, positive=%.1f%%, negative=%.1f%%\n', ...
        AOA_real, min(thetas_deg), max(thetas_deg), mean(thetas_deg), ...
        100*sum(thetas>0)/length(thetas), 100*sum(thetas<0)/length(thetas));
end
fprintf('\n');
