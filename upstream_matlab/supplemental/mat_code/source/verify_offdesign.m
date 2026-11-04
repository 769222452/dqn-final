%% Verify updated offdesign_aero.m with rot_x(180) + real AOA
clc; clear; format short;

load('wave_rider_optimization_final.mat');
[sorted_front, sort_idx] = sortrows(pareto_front, 1);
sorted_design = design_vars(sort_idx, :);

Ma = 15; i = 45;
angle_shock_a1 = sorted_design(i,1);
angle_shock_a2 = sorted_design(i,2);
angle_shock_b1 = sorted_design(i,3);
angle_shock_b2 = sorted_design(i,4);
c1 = 1.0331; c2 = 1.0331;
d1 = sorted_design(i,5);
AOA_prog = -4;
a_xishu_1 = sorted_design(i,6);

if isempty(gcp('nocreate'))
    parpool('local', 20);
end

%% Generate waverider
[wav_x, wav_y, wav_z, LD_gen, LW, H, Vol, Vol_eff, Time] = ...
    generatewaverider(Ma, angle_shock_a1, angle_shock_a2, ...
                      angle_shock_b1, angle_shock_b2, c1, c2, d1, AOA_prog, a_xishu_1);
fprintf('generatewaverider LD = %.6f\n', LD_gen);

%% Generate upper surface
[M, nn] = size(wav_x);
V_prog = [cosd(AOA_prog), 0, sind(AOA_prog)];
wav_up_x = zeros(M, nn);
wav_up_y = zeros(M, nn);
wav_up_z = zeros(M, nn);
x_design = max(wav_x(:));
for ii = 1:M
    x_le = wav_x(ii, 1);
    y_le = wav_y(ii, 1);
    z_le = wav_z(ii, 1);
    for jj = 1:nn
        wav_up_x(ii, jj) = x_le + (x_design - x_le) * (jj-1)/(nn-1);
        wav_up_y(ii, jj) = y_le + (wav_up_x(ii, jj) - x_le) * V_prog(2) / V_prog(1);
        wav_up_z(ii, jj) = z_le + (wav_up_x(ii, jj) - x_le) * V_prog(3) / V_prog(1);
    end
end

%% Test 1: Design point (real AOA = +4)
fprintf('\n=== Test 1: Design point ===\n');
AOA_real = 4;
[LD1, Fx1, Fy1, Fz1] = offdesign_aero(wav_x, wav_y, wav_z, wav_up_x, wav_up_y, wav_up_z, Ma, AOA_real);
fprintf('AOA = %d deg: LD = %.6f, Fx = %.6f, Fz = %.6f\n', AOA_real, LD1, Fx1, Fz1);
fprintf('  vs generatewaverider LD = %.6f (err=%.2f%%)\n', LD_gen, abs(LD1-LD_gen)/LD_gen*100);

%% Test 2: Off-design AOA sweep
fprintf('\n=== Test 2: AOA sweep (Ma=%d) ===\n', Ma);
for aoa = [0, 2, 4, 6, 8]
    [LD, ~, ~, ~] = offdesign_aero(wav_x, wav_y, wav_z, wav_up_x, wav_up_y, wav_up_z, Ma, aoa);
    fprintf('AOA = %d deg: LD = %.6f\n', aoa, LD);
end

%% Test 3: Off-design Ma sweep
fprintf('\n=== Test 3: Ma sweep (AOA=%d deg) ===\n', AOA_real);
for ma = [10, 12, 15, 18]
    [LD, ~, ~, ~] = offdesign_aero(wav_x, wav_y, wav_z, wav_up_x, wav_up_y, wav_up_z, ma, AOA_real);
    fprintf('Ma = %d: LD = %.6f\n', ma, LD);
end
