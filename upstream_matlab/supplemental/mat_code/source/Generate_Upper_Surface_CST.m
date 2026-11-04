function [X_upper, Y_upper, Z_upper] = Generate_Upper_Surface_CST(leading_edge_points, cst_params, options, delta_h_sym)
% GENERATE_UPPER_SURFACE_CST_UPDATED
% 基于等Y截面方法的乘波体上表面生成
%
% 修改说明：
%   1. 类别函数：c(y) = [(y/(y_max-y_min)+1)^N1] * [(1-y/(y_max-y_min))^N2]
%      确保Y=0时c(y)≠0，Y=2时c(y)=0
%   2. 上表面Z坐标从前缘点Z往上加（而非从下表面Z往上加）
%
% 输入参数：
%   leading_edge_points: 前缘点坐标 [X_LE, Y_LE, Z_LE, Z_exit_lower]
%   cst_params: CST参数 [N1, N2, bernstein_degree, coeffs]
%   options: [nx, L]
%   delta_h_sym: Y=0处的高度差
%
% 输出参数：
%   X_upper, Y_upper, Z_upper: 上表面坐标
%   Z_exit: 出口处上表面Z值

fprintf('========================================\n');
fprintf('乘波体上表面生成（更新版）\n');
fprintf('========================================\n\n');

%% 参数初始化
if nargin < 4
    options = struct();
end
if ~isfield(options, 'nx'), options.nx = 50; end
if ~isfield(options, 'L'), options.L = 5.0; end

nx = options.nx;
L = options.L;

% 验证输入
n_points = length(leading_edge_points.X_LE);
if n_points == 0
    error('前缘点数量为空！');
end

fprintf('输入验证：\n');
fprintf('  - 前缘点数量: %d\n', n_points);
fprintf('  - Y_LE范围: [%.3f, %.3f] m\n', min(leading_edge_points.Y_LE), max(leading_edge_points.Y_LE));
fprintf('  - 对称面(Y=0)高度差: %.4f m\n', delta_h_sym);

% CST参数
if ~isfield(cst_params, 'N1'), cst_params.N1 = 0.5; end
if ~isfield(cst_params, 'N2'), cst_params.N2 = 1.0; end
if ~isfield(cst_params, 'bernstein_degree'), cst_params.bernstein_degree = 3; end
if ~isfield(cst_params, 'coeffs') || isempty(cst_params.coeffs)
    error('必须提供CST Bernstein系数！');
end

degree = cst_params.bernstein_degree;
coeffs = cst_params.coeffs;
if length(coeffs) < (degree + 1)
    coeffs = [coeffs, zeros(1, (degree + 1) - length(coeffs))];
end
coeffs = coeffs(1:(degree + 1));

fprintf('\nCST参数：\n');
fprintf('  - N1 = %.2f\n', cst_params.N1);
fprintf('  - N2 = %.2f\n', cst_params.N2);
fprintf('  - Bernstein阶数 = %d\n', degree);

%% ============================================
%% 步骤1：识别对称面(Y=0)和最宽处(Y=2)
%% ============================================
fprintf('\n步骤1：识别关键截面...\n');

[~, idx_sym] = min(abs(leading_edge_points.Y_LE));   % Y=0
[~, idx_max] = max(leading_edge_points.Y_LE);        % Y=2

Y_sym = leading_edge_points.Y_LE(idx_sym);
Y_max = leading_edge_points.Y_LE(idx_max);

fprintf('  - 对称面(Y=0)索引: %d, Y = %.6f m\n', idx_sym, Y_sym);
fprintf('  - 最宽处(Y=2)索引: %d, Y = %.6f m\n', idx_max, Y_max);

%% ============================================
%% 步骤2：计算CST函数（修正的类别函数）
%% ============================================
fprintf('\n步骤2：计算CST函数...\n');

Y_LE = leading_edge_points.Y_LE;
Y_min = min(Y_LE);
Y_max = max(Y_LE);

% 归一化参数
Y_range = Y_max - Y_min;
if Y_range == 0
    Y_range = 1e-6;
end

N1 = cst_params.N1;
N2 = cst_params.N2;

% 用户要求的类别函数：c(y) = [(y/(y_max-y_min)+1)^N1] * [(1-y/(y_max-y_min))^N2]
% 在 Y=0 时：c = (0+1)^N1 * (1-0)^N2 = 1 * 1 = 1
% 在 Y=2 时：c = (1+1)^N1 * (1-1)^N2 = 2^N1 * 0 = 0

Y_normalized = Y_LE / Y_range;  % y/(y_max-y_min)

% 计算类别函数
Class_y = (Y_normalized + 1).^N1 .* (1 - Y_normalized).^N2;

% Bernstein形状函数
Bernstein_y = zeros(n_points, 1);
for i = 1:n_points
    y = Y_normalized(i);
    S_y = 0;
    for j = 0:degree
        S_y = S_y + coeffs(j+1) * nchoosek(degree, j) * y.^j * (1-y).^(degree-j);
    end
    Bernstein_y(i) = S_y;
end

% CST函数
Z_CST_norm = Class_y .* Bernstein_y;

fprintf('  - Y归一化范围: [%.6f, %.6f]\n', min(Y_normalized), max(Y_normalized));
fprintf('  - 类别函数值范围: [%.6f, %.6f]\n', min(Class_y), max(Class_y));
fprintf('  - Bernstein值范围: [%.6f, %.6f]\n', min(Bernstein_y), max(Bernstein_y));
fprintf('  - CST归一化值范围: [%.6f, %.6f]\n', min(Z_CST_norm), max(Z_CST_norm));
fprintf('  - Y=0处CST值: %.6f (应≠0)\n', Z_CST_norm(idx_sym));
fprintf('  - Y=2处CST值: %.6f (应≈0)\n', Z_CST_norm(idx_max));
%% ============================================
%% 步骤3：计算缩放因子
%% ============================================
fprintf('\n步骤3：计算缩放因子...\n');

% 在Y=0处：高度增量 = delta_h_sym
% scale_factor = delta_h_sym / Z_CST_norm(idx_sym)
scale_factor = delta_h_sym / Z_CST_norm(idx_sym);

fprintf('  - 缩放因子: %.4f / %.6f = %.6f m\n', delta_h_sym, Z_CST_norm(idx_sym), scale_factor);

%% ============================================
%% 步骤4：计算出口处上表面Z值
%% ============================================
fprintf('\n步骤4：计算出口处上表面Z值...\n');

% 关键修改：从前缘点Z往上加（而非从下表面Z往上加）
Z_LE = leading_edge_points.Z_LE;
Z_exit_upper = zeros(n_points, 1);

for i = 1:n_points
    % 上表面Z = 前缘点Z + 缩放的CST增量
    Z_exit_upper(i) = Z_LE(i) + scale_factor * Z_CST_norm(i);
end

% 验证边界条件
delta_h_0_actual = Z_exit_upper(idx_sym) - Z_LE(idx_sym);
delta_h_2_actual = Z_exit_upper(idx_max) - Z_LE(idx_max);

fprintf('  - 前缘Z范围: [%.4f, %.4f] m\n', min(Z_LE), max(Z_LE));
fprintf('  - 出口上表面Z范围: [%.4f, %.4f] m\n', min(Z_exit_upper), max(Z_exit_upper));
fprintf('  - Y=0高度差: %.6f m (期望: %.4f m)\n', delta_h_0_actual, delta_h_sym);
fprintf('  - Y=2高度差: %.6f m (期望: 0.0 m)\n', delta_h_2_actual);

Z_exit = Z_exit_upper;

%% ============================================
%% 步骤5：冯卡门曲线生成上表面
%% ============================================
fprintf('\n步骤5：冯卡门曲线生成上表面...\n');

X_upper = zeros(n_points, nx);
Y_upper = zeros(n_points, nx);
Z_upper = zeros(n_points, nx);

X_LE = leading_edge_points.X_LE;

for i = 1:n_points
    Y_current = Y_LE(i);
    X_LE_current = X_LE(i);
    Z_LE_current = Z_LE(i);
    Z_exit_upper_current = Z_exit_upper(i);
    
    % 离散化X方向
    if X_LE_current < L
        X_stream = linspace(X_LE_current, L, nx);
    else
        X_stream = L * ones(1, nx);
    end
    
    % 关键修改：高度增量从前缘到出口的上表面
    H_local = Z_exit_upper_current - Z_LE_current;
    
    % 归一化流向位置
    if X_LE_current < L
        xi = (X_stream - X_LE_current) / (L - X_LE_current);
    else
        xi = zeros(1, nx);
    end
    
    xi(xi < 0) = 0; xi(xi > 1) = 1;
    
    % 冯卡门曲线
    theta = acos(1-2*xi);
    von_karman = (1/sqrt(pi)) * sqrt(theta-0.5*sin(2*theta));
    
    % 关键修改：Z = 前缘Z + 冯卡门增量（从前缘点Z往上加）
    Z_upper_stream = Z_LE_current + von_karman * H_local;
    
    X_upper(i, :) = X_stream;
    Y_upper(i, :) = Y_current;
    Z_upper(i, :) = Z_upper_stream;
end

% 验证
Z_upper_at_exit = Z_upper(:, end);
exit_error = max(abs(Z_upper_at_exit - Z_exit_upper));

fprintf('  - 上表面Z范围: [%.4f, %.4f] m\n', min(Z_upper(:)), max(Z_upper(:)));
fprintf('  - 出口验证误差: %.6f m\n', exit_error);

fprintf('\n========================================\n');
fprintf('生成完成！\n');
fprintf('========================================\n');

% 可视化
% if nargout == 0
Visualize_Updated(X_upper, Y_upper, Z_upper, leading_edge_points, Z_exit, ...
                 Z_CST_norm, scale_factor, idx_sym, idx_max, Y_normalized, Class_y, options);
% end

end

function Visualize_Updated(X_upper, Y_upper, Z_upper, leading_edge_points, Z_exit, ...
                          Z_CST_norm, scale_factor, idx_sym, idx_max, Y_normalized, Class_y, options)
    
    figure('Name', '上表面生成结果（更新版）', 'Position', [50, 50, 1600, 900], 'Color', 'w');
    
    subplot(2, 3, 1);
    surf(X_upper, Y_upper, Z_upper, 'EdgeColor', 'none', 'FaceColor', 'interp');
    axis equal
    hold on;
    plot3(leading_edge_points.X_LE, leading_edge_points.Y_LE, leading_edge_points.Z_LE, 'ro', 'MarkerSize', 5);
    plot3(options.L*ones(size(leading_edge_points.Y_LE)), leading_edge_points.Y_LE, Z_exit, 'bo', 'MarkerSize', 5);
    xlabel('X'); ylabel('Y'); zlabel('Z');
    title('3D视图');
    view(140, 30);
    grid on;
    hold off;
    
    subplot(2, 3, 2);
    plot(leading_edge_points.Y_LE, leading_edge_points.Z_LE, 'r-o', 'LineWidth', 2);
    axis equal
    hold on;
    plot(leading_edge_points.Y_LE, Z_exit, 'b-s', 'LineWidth', 2);
    plot(leading_edge_points.Y_LE(idx_sym), leading_edge_points.Z_LE(idx_sym), 'go', 'MarkerSize', 10);
    plot(leading_edge_points.Y_LE(idx_sym), Z_exit(idx_sym), 'go', 'MarkerSize', 10);
    xlabel('Y'); ylabel('Z');
    title('前缘和出口Z值');
    legend('前缘', '出口', 'Y=0', 'Location', 'best');
    grid on;
    hold off;
    
    subplot(2, 3, 3);
    plot(Y_normalized, Class_y, 'g-', 'LineWidth', 2);
    axis equal
    hold on;
    plot(Y_normalized(idx_sym), Class_y(idx_sym), 'ro', 'MarkerSize', 10);
    plot(Y_normalized(idx_max), Class_y(idx_max), 'yo', 'MarkerSize', 10);
    xlabel('归一化Y'); ylabel('类别函数');
    title('类别函数 c(y)');
    grid on;
    hold off;
    
    subplot(2, 3, 4);
    plot(leading_edge_points.Y_LE, Z_CST_norm, 'b-', 'LineWidth', 2);
    axis equal
    hold on;
    plot(leading_edge_points.Y_LE(idx_sym), Z_CST_norm(idx_sym), 'ro', 'MarkerSize', 10);
    plot(leading_edge_points.Y_LE(idx_max), Z_CST_norm(idx_max), 'yo', 'MarkerSize', 10);
    xlabel('Y'); ylabel('CST值');
    title('CST归一化值');
    grid on;
    hold off;
    
    subplot(2, 3, 5);
    plot(X_upper(idx_sym,:), Z_upper(idx_sym,:), 'r-', 'LineWidth', 2);
    axis equal
    hold on;
    plot(X_upper(idx_max,:), Z_upper(idx_max,:), 'b-', 'LineWidth', 2);
    xlabel('X'); ylabel('Z');
    title('Y=0和Y=2截面对比');
    legend('Y=0', 'Y=2', 'Location', 'best');
    grid on;
    hold off;
    
    subplot(2, 3, 6);
    xi_vk = linspace(0,1,100);
    axis equal
    % vk = (1/sqrt(pi))*acos(1-2*xi_vk) - 0.5*sin(2*acos(1-2*xi_vk));
    theta = acos(1-2*xi_vk);
    vk = (1/sqrt(pi))*sqrt(theta-sin(2*theta)/2);
    plot(xi_vk, vk, 'b-', 'LineWidth', 2);
    xlabel('\xi'); ylabel('冯卡门');
    title('冯卡门曲线');
    grid on;
end
