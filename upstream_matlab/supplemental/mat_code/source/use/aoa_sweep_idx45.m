%% AOA sweep for offdesign_aero: design point idx=45
%% AIAA Journal figure formatting (corrected: Mach-angle floor on lower surface)
clc; clear; format short;
if isempty(gcp('nocreate'))
    parpool('local',6);
end

load('wave_rider_optimization_final.mat');
[sorted_front, sort_idx] = sortrows(pareto_front, 1);
sorted_design = design_vars(sort_idx, :);

i = 45;
angle_shock_a1 = sorted_design(i,1);
angle_shock_a2 = sorted_design(i,2);
angle_shock_b1 = sorted_design(i,3);
angle_shock_b2 = sorted_design(i,4);
c1 = 1.0331; c2 = 1.0331;
d1 = sorted_design(i,5);
AOA_prog = -4;
a_xishu_1 = sorted_design(i,6);
Ma = 15;

fprintf('Generating waverider for design #45...\n');

[wav_x, wav_y, wav_z, LD_gen, LW, H, Vol, Vol_eff, Time] = ...
    generatewaverider(Ma, angle_shock_a1, angle_shock_a2, ...
                      angle_shock_b1, angle_shock_b2, c1, c2, d1, AOA_prog, a_xishu_1);
fprintf('LD_gen = %.4f, Vol_eff = %.6f\n', LD_gen, Vol_eff);

%% Upper surface: fixed at design AOA (-4 deg prog)
[M, nn] = size(wav_x);
V = [cosd(AOA_prog), 0, sind(AOA_prog)];
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
        wav_up_y(ii, jj) = y_le + (wav_up_x(ii, jj) - x_le) * V(2) / V(1);
        wav_up_z(ii, jj) = z_le + (wav_up_x(ii, jj) - x_le) * V(3) / V(1);
    end
end

%% AOA sweep (real physics frame)
AOA_list = 0:0.5:10;
n_aoa = length(AOA_list);
LD_arr = zeros(1, n_aoa);
Fx_arr = zeros(1, n_aoa);
Fz_arr = zeros(1, n_aoa);

fprintf('\nSweeping AOA (real physics frame, theta_min=%.2f deg)...\n', asind(1/Ma));
for k = 1:n_aoa
    AOA_real = AOA_list(k);
    [LD_arr(k), Fx_arr(k), ~, Fz_arr(k)] = offdesign_aero_v2(...
        wav_x, wav_y, wav_z, wav_up_x, wav_up_y, wav_up_z, Ma, AOA_real);
    fprintf('  AOA=%5.1f  LD=%.4f  Fx=%.4f  Fz=%.4f\n', ...
            AOA_real, LD_arr(k), Fx_arr(k), Fz_arr(k));
end

save('aoa_sweep_idx45.mat', 'AOA_list', 'LD_arr', 'Fx_arr', 'Fz_arr', ...
     'LD_gen', 'angle_shock_a1', 'angle_shock_a2', 'angle_shock_b1', ...
     'angle_shock_b2', 'd1', 'a_xishu_1');

%% AIAA Journal Figure
fig_width  = 3.5;
fig_height = 2.7;

figure('Units', 'inches', 'Position', [2, 2, fig_width, fig_height], ...
       'PaperPositionMode', 'auto', 'Color', 'w');
hold on; box on;

plot(AOA_list, LD_arr, 'k-', 'LineWidth', 1.2);
plot(AOA_list, LD_arr, 'ko', 'MarkerSize', 4, 'MarkerFaceColor', 'k');

[~, idx_design] = min(abs(AOA_list - 4));
plot(AOA_list(idx_design), LD_arr(idx_design), 'r^', ...
     'MarkerSize', 7, 'MarkerFaceColor', 'r', 'LineWidth', 1.0);

xlabel('\\alpha (deg)', 'FontName', 'Times New Roman', ...
       'FontSize', 10, 'FontWeight', 'bold');
ylabel('L/D', 'FontName', 'Times New Roman', ...
       'FontSize', 10, 'FontWeight', 'bold');

set(gca, 'FontName', 'Times New Roman', 'FontSize', 9, ...
         'LineWidth', 0.8);

grid on;
set(gca, 'GridLineStyle', ':');
set(gca, 'MinorGridLineStyle', 'none');

print('-depsc2', '-painters', 'aoa_sweep_idx45.eps');
print('-dpng', '-r300', 'aoa_sweep_idx45.png');

fprintf('\nFigures saved: aoa_sweep_idx45.eps, aoa_sweep_idx45.png\n');
fprintf('=== Done ===\n');
