function [y_fct, z_fct] = swc_shape_3(M, x_v_r_func, y_v_r_func, z_v_r_func, d_fct, x_design, dz_func, ddz_func)
    % 预分配数组
    x_end = zeros(1, M);
    y_end = zeros(1, M);
    z_end = zeros(1, M);
    y_fct = zeros(1, M);
    z_fct = zeros(1, M);
    v_end = zeros(1, M);
    
    d = d_fct;
    a1 = 0;
    
    % 创建残差函数用于求解方程组
    function F = equations(uv)
        u_val = uv(1);
        v_val = uv(2);
        F = zeros(2, 1);
        F(1) = z_v_r_func(u_val, v_val) - (a1 * y_v_r_func(u_val, v_val)^2 + d);
        F(2) = x_v_r_func(u_val, v_val) - x_design;
    end

    % 求解初始点
    options = optimoptions('fsolve', 'Display', 'off', 'Algorithm', 'levenberg-marquardt');
    uv0 = [0.5, 1.0]; % 初始猜测
    solution = fsolve(@equations, uv0, options);
    u_s = solution(1);
    v_s = solution(2);
    
    % 确保结果在合理范围内
    u_s = max(0, min(u_s, 1.5));
    v_s = max(0, min(v_s, pi/2));
    
    % 设置 v 端点
    v_end = linspace(v_s, pi/2, M);
    
    % 求解 u 值 (x = x_design)
    u_end = zeros(1, M);
    for i = 1:M
        v_val = v_end(i);
        
        % 创建方程求解 u
        eq = @(u) x_v_r_func(u, v_val) - x_design;
        
        % 使用 fzero 求解
        u_guess = 0.5; % 初始猜测
        try
            u_sol = fzero(eq, u_guess);
            u_end(i) = max(0, min(u_sol, 1.5)); % 约束范围
        catch
            % 使用线性插值作为回退
            if i > 1
                u_end(i) = u_end(i-1);
            else
                u_end(i) = u_s;
            end
        end
    end
    
    % 计算端点坐标
    for i = 1:M
        u_val = u_end(i);
        v_val = v_end(i);
        y_end(i) = y_v_r_func(u_val, v_val);
        z_end(i) = z_v_r_func(u_val, v_val);
    end
    
    % 设置 y_fct 曲线
    y_min = min(y_end);
    y_max = max(y_end);
    transition_point = 0.09; % 过渡点
    
    % 第一部分：从起点到过渡点
    idx1 = 1:floor(M*0.5);
    y_fct(idx1) = linspace(y_end(1), transition_point, length(idx1));
    det_y = abs(y_fct(1)-y_fct(2));
    % det_y = 0.01;

    % 第二部分：从过渡点到终点
    idx2 = (floor(M*0.5)+1):M;
    y_fct(idx2) = linspace(transition_point-det_y, y_end(end), length(idx2));
    
    % 计算 z_fct
    z_fct = a1 * y_fct.^2 + d;
    
    % 绘制结果
    % figure(2)
    % plot(y_end, z_end, 'g', 'LineWidth', 1.5);
    % hold on;
    % scatter(y_fct, z_fct, 30, 'b', 'filled');
    % hold on;
    
    % 计算并绘制曲率点
    afa = zeros(1, M);
    bta = zeros(1, M);
    for i = 1:M
        u_val = u_end(i);
        v_val = v_end(i);
        dz_val = dz_func(u_val, v_val);
        ddz_val = ddz_func(u_val, v_val);
        
        afa(i) = y_end(i) - (dz_val * (1 + dz_val^2)) / ddz_val;
        bta(i) = z_end(i) + (1 + dz_val^2) / ddz_val;
    end
    % scatter(afa, bta, 30, 'g', 'filled');
    % 
    % axis equal;
    % grid on;
    % legend('端点曲线', '前缘点', '曲率点', 'Location', 'best');
    % title('SWC 前缘形状');
    % xlabel('Y');
    % ylabel('Z');
    % hold off;
end
