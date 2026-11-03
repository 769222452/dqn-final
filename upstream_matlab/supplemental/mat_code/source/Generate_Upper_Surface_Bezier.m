function [X_upper, Y_upper, Z_upper] = Generate_Upper_Surface_Bezier(leading_edge_points, bezier_params, options, delta_h_sym)
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
%   bezier_params: 贝塞尔曲线控制点参数
%       .control_points: 3×2矩阵，每行是[控制点Y坐标, 控制点Z坐标归一化值]
%           控制点1: [Y1, Z_norm1] - 通常Y1=0，Z_norm1=1.0（起点）
%           控制点2: [Y2, Z_norm2] - Y2可调，Z_norm2可调（中间控制点）
%           控制点3: [Y3, Z_norm3] - Y3=2，Z_norm3=0.0（终点）
%   options: [nx, L]
%   delta_h_sym: Y=0处的高度差
%
% 输出参数：
%   X_upper, Y_upper, Z_upper: 上表面坐标
%   Z_exit: 出口处上表面Z值

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

% 贝塞尔参数（默认值）
if ~isfield(bezier_params, 'control_points') || isempty(bezier_params.control_points)
    % 默认控制点：四点控制三次贝塞尔曲线
    bezier_params.control_points = [
        0.0, 1.0;    % P0: Y=0, Z_norm=1.0（起点高）
        0.3, 1.0;    % P1: Y=0.4, Z_norm=1.0（第一个中间控制点）
        0.8, 0.3;    % P2: Y=1.2, Z_norm=0.3（第二个中间控制点）
        1.0, 0.0     % P3: Y=2.0, Z_norm=0.0（终点低）
    ];
end
control_pts = bezier_params.control_points;

% 验证控制点
if size(control_pts, 1) ~= 4
    error('贝塞尔曲线需要4个控制点！');
end

%% ============================================
%% 步骤1：识别对称面(Y=0)和最宽处(Y=2)
%% ============================================

[~, idx_sym] = min(abs(leading_edge_points.Y_LE));   % Y=0
[~, idx_max] = max(leading_edge_points.Y_LE);        % Y=2

Y_sym = leading_edge_points.Y_LE(idx_sym);
Y_max = leading_edge_points.Y_LE(idx_max);

%% ============================================
%% 步骤2：计算贝塞尔曲线
%% ============================================

Y_LE = leading_edge_points.Y_LE;
Y_min = min(Y_LE);
Y_max_width = max(Y_LE);

% 提取控制点
P0 = control_pts(1, :);  % 起点
P1 = control_pts(2, :);  % 第一个中间控制点
P2 = control_pts(3, :);  % 第二个中间控制点
P3 = control_pts(4, :);  % 终点

% 计算贝塞尔曲线
Z_Bezier_norm = zeros(n_points, 1);

for i = 1:n_points
    Y_current = Y_LE(i);
    
    % 计算归一化参数 t (0 ≤ t ≤ 1)
    if Y_max_width - Y_min > 1e-10
        t = Y_current / (Y_max_width - Y_min);  % t从0到1
    else
        t = 0;
    end
    
    % 三次贝塞尔曲线公式
    % B(t) = (1-t)^3 * P0 + 3(1-t)^2*t * P1 + 3(1-t)*t^2 * P2 + t^3 * P3
    
    w0 = (1 - t)^3;                % P0的权重
    w1 = 3 * (1 - t)^2 * t;         % P1的权重
    w2 = 3 * (1 - t) * t^2;         % P2的权重
    w3 = t^3;                       % P3的权重
    
    % 计算Z值（只取Z坐标）
    Z_Bezier_norm(i) = w0 * P0(2) + w1 * P1(2) + w2 * P2(2) + w3 * P3(2);
end

%% ============================================
%% 步骤3：计算缩放因子
%% ============================================
scale_factor = delta_h_sym / Z_Bezier_norm(idx_sym);
%% ============================================
%% 步骤4：计算出口处上表面Z值
%% ============================================
Z_LE = leading_edge_points.Z_LE;
Z_exit_upper = zeros(n_points, 1);

for i = 1:n_points
    % 上表面Z = 前缘点Z + 缩放的CST增量
    Z_exit_upper(i) = Z_LE(i) + scale_factor * Z_Bezier_norm(i);
end

% 验证边界条件
% delta_h_0_actual = Z_exit_upper(idx_sym) - Z_LE(idx_sym);
% delta_h_2_actual = Z_exit_upper(idx_max) - Z_LE(idx_max);
Z_exit = Z_exit_upper;

%% ============================================
%% 步骤5：冯卡门曲线生成上表面
%% ============================================

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
% Z_upper_at_exit = Z_upper(:, end);
% exit_error = max(abs(Z_upper_at_exit - Z_exit_upper));

% 可视化
% if nargout == 0
% Visualize_Bezier(X_upper, Y_upper, Z_upper, leading_edge_points, Z_exit, ...
%                  Z_Bezier_norm, control_pts, idx_sym, idx_max, options);
% end

end

function Visualize_Bezier(X_upper, Y_upper, Z_upper, leading_edge_points, Z_exit, ...
                         Z_Bezier_norm, control_pts, idx_sym, idx_max, options)
    
        figure('Name', '上表面生成结果（四点贝塞尔曲线）', 'Position', [50, 50, 1800, 900], 'Color', 'w');
    
    % 子图1: 3D视图
    subplot(2, 3, 1);
    surf(X_upper, Y_upper, Z_upper, 'EdgeColor', 'none', 'FaceColor', 'interp');
    axis equal
    hold on;
    plot3(leading_edge_points.X_LE, leading_edge_points.Y_LE, leading_edge_points.Z_LE, 'ro', 'MarkerSize', 5);
    plot3(options.L*ones(size(leading_edge_points.Y_LE)), leading_edge_points.Y_LE, Z_exit, 'bo', 'MarkerSize', 5);
    % 标记4个控制点
    plot3(repmat(leading_edge_points.X_LE(idx_sym), 4, 1), control_pts(:,1), ...
          min(leading_edge_points.Z_LE) * ones(4, 1), 'gs', 'MarkerSize', 10, 'LineWidth', 2);
    xlabel('X'); ylabel('Y'); zlabel('Z');
    title('3D视图（四点贝塞尔）');
    view(140, 30);
    grid on;
    hold off;
    
    % 子图2: 贝塞尔曲线和4个控制点
    subplot(2, 3, 2);
    plot(leading_edge_points.Y_LE, Z_Bezier_norm, 'b-', 'LineWidth', 2);
    axis equal
    hold on;
    % 绘制4个控制点
    plot(control_pts(:,1), control_pts(:,2), 'go', 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', '控制点');
    % 绘制控制多边形（连接4个点）
    plot(control_pts(:,1), control_pts(:,2), 'g--', 'LineWidth', 1);
    % 标记特殊点
    plot(leading_edge_points.Y_LE(idx_sym), Z_Bezier_norm(idx_sym), 'ro', 'MarkerSize', 10);
    plot(leading_edge_points.Y_LE(idx_max), Z_Bezier_norm(idx_max), 'yo', 'MarkerSize', 10);
    xlabel('Y (m)'); ylabel('归一化Z值');
    title('四点贝塞尔曲线');
    legend(['贝塞尔曲线', '控制点', 'Y=0', 'Y=2'], 'Location', 'best');
    grid on;
    hold off;
    
    % 子图3: 前缘和出口Z值
    subplot(2, 3, 3);
    plot(leading_edge_points.Y_LE, leading_edge_points.Z_LE, 'r-o', 'LineWidth', 1.5);
    axis equal
    hold on;
    plot(leading_edge_points.Y_LE, Z_exit, 'b-s', 'LineWidth', 2);
    plot(leading_edge_points.Y_LE(idx_sym), leading_edge_points.Z_LE(idx_sym), 'go', 'MarkerSize', 10);
    plot(leading_edge_points.Y_LE(idx_sym), Z_exit(idx_sym), 'go', 'MarkerSize', 10);
    xlabel('Y'); ylabel('Z (m)');
    title('前缘和出口Z值');
    legend('前缘', '出口', 'Y=0', 'Location', 'best');
    grid on;
    hold off;
    
    % 子图4: 四点贝塞尔权重分布
    subplot(2, 3, 4);
    t_values = linspace(0, 1, 100);
    w0 = (1 - t_values).^3;                % P0权重
    w1 = 3 * (1 - t_values).^2 .* t_values;     % P1权重
    w2 = 3 * (1 - t_values) .* t_values.^2;     % P2权重
    w3 = t_values.^3;                                 % P3权重
    
    hold on;
    plot(t_values * 2, w0, 'r-', 'LineWidth', 2, 'DisplayName', 'P0权重');
    plot(t_values * 2, w1, 'g-', 'LineWidth', 2, 'DisplayName', 'P1权重');
    plot(t_values * 2, w2, 'b-', 'LineWidth', 2, 'DisplayName', 'P2权重');
    plot(t_values * 2, w3, 'm-', 'LineWidth', 2, 'DisplayName', 'P3权重');
    xlabel('Y (m)'); ylabel('权重');
    title('四点贝塞尔基函数权重');
    legend('Location', 'best');
    grid on;
    hold off;
    
    % 子图5: Y=0和Y=2截面对比
    subplot(2, 3, 5);
    plot(X_upper(idx_sym,:), Z_upper(idx_sym,:), 'r-', 'LineWidth', 2);
    axis equal
    hold on;
    plot(X_upper(idx_max,:), Z_upper(idx_max,:), 'b-', 'LineWidth', 2);
    xlabel('X (m)'); ylabel('Z (m)');
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
