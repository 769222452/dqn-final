function [pareto_front, design_vars] = surrogate_assisted_ga_wave_rider_6()
    % 1. 参数设置
    n_vars = 6;
    lb = [3, 3, 3, 3, 0.02, 0.9];
    ub = [8, 8, 8, 8, 0.18, 1.1];

    % 数据点参数
    min_valid_points = 50; % 初始采样点数
    ga_pop_size = 100; % GA 种群参数
    eval_per_gen = 20; % 代理模型每代新增点数
    max_attempts_per_gen = 50; % 为确保代理模型每代新增的点数而最多的
    
    % 收敛监测参数
    max_stall_generations = 15;  % 连续停滞代数阈值
    hv_threshold = 0.0008;      % 超体积相对变化阈值
    max_estimation_point = 5000; % 最大评估点数

    % % 用于对目标函数的归一/反归一化
    % global_norm = struct('min_vals', [0, 0], 'range_vals', [1, 1]);

    % 2. 初始采样（完全串行）
    fprintf('初始采样中...目标：%d个有效点\n', min_valid_points);
    X_train = [];
    Y_train = [];
    valid_count = 0;
    attempt_count = 0;
    
    while valid_count < min_valid_points
        new_X = lb + (ub - lb) .* lhsdesign(1, n_vars, 'criterion','maximin');
        [success, L_D, vol_frac] = evaluatePoint(new_X);
        attempt_count = attempt_count + 1;
        
        if success
            X_train = [X_train; new_X];
            Y_train = [Y_train; [L_D, vol_frac]];
            valid_count = valid_count + 1;
        end
        
        if mod(attempt_count, 10) == 0
            fprintf('已尝试 %d 点，有效点数: %d/%d (%.1f%%)\n', ...
                attempt_count, valid_count, min_valid_points, ...
                100*valid_count/attempt_count);
        end
    end
    fprintf('初始采样完成! 共尝试 %d 点，获得 %d 有效点\n', attempt_count, valid_count);
    
    % 3. 构建初始代理模型
    fprintf('构建代理模型...\n');
    n_samples = size(X_train,1);
    if n_samples <= 500
        fit_method = 'exact'; % 小数据量用精确解，精度高
    else
        fit_method = 'sr'; % 大数据量用稀疏近似，速度快
    end
    opts = {'KernelFunction','squaredexponential', 'FitMethod',fit_method, 'PredictMethod',fit_method};
    krig_LD = fitrgp(X_train, Y_train(:,1), opts{:});
    krig_vol = fitrgp(X_train, Y_train(:,2), opts{:});
    
    % 4. 初始化优化循环
    fprintf('开始优化循环...\n');
    total_time = 0;
    stall_count = 0;
    hv_history = [];
    gen = 0;
    
    % 获取初始Pareto前沿和超体积
    [~, true_pf] = extract_pareto_front(Y_train, false);
    % ref_point_fixed = [min(Y_train(:,1))-0.1, min(Y_train(:,2))-0.01];
    ref_point_fixed = [0,0];
    hv_current = calculateHypervolume(true_pf, ref_point_fixed);
    hv_history(1) = hv_current;
    fprintf('初始超体积: %.4f\n', hv_current);
    
    % 5. 主优化循环
    while stall_count < max_stall_generations
        gen = gen + 1;
        tic;
        fprintf('\n=== 第 %d 代 ===\n', gen);
        
        % NSGA-II优化
        surrogateFitness = @(x) [-predict(krig_LD, x), -predict(krig_vol, x)];
        % 设置遗传算法参数
        GA_opts = optimoptions('gamultiobj', ...
            'PopulationSize', ga_pop_size, ...
            'CrossoverFraction', 0.85, ...           % 交叉概率85%
            'CrossoverFcn', @crossoverintermediate, ... % 分散交叉
            'MutationFcn', {@mutationadaptfeasible, 0.07, 0.3}, ... % 7%个体变异，标准差0.3
            'SelectionFcn', {@selectiontournament, 3}, ... % 锦标赛选择（规模3）
            'FunctionTolerance', 1e-4, ...            % 函数值收敛容差
            'ParetoFraction', 0.7, ...                % Pareto解保留比例
            'Display', 'off', ...
            'UseParallel', false);
        [ga_x, current_pf] = gamultiobj(surrogateFitness, n_vars, [], [], [], [], lb, ub, GA_opts);
        current_pf = -current_pf;
        % 预测候选点的均值和标准差
        [mu_LD, sd_LD] = predict(krig_LD, ga_x);
        [mu_vol, sd_vol] = predict(krig_vol, ga_x);
        mu = [mu_LD, mu_vol];
        sigma = [sd_LD, sd_vol];
        % 混合pareto前缘
        % fused_pf = [true_pf;ga_pf];
        % [~, fused_pf] = extract_pareto_front(fused_pf, false);
        % 动态平衡开发与探索
        % kappa=0~1,0为纯开发模式（只关注高预测值区域）,1为强探索模式（偏向高不确定性区域）
        base_kappa = 0.5 + min(0.5, 0.015 * (gen - 1));
        if stall_count > 0
            kappa = min(1.0, base_kappa + 0.2 * stall_count);
        else
            kappa = base_kappa;
        end
        adjusted_sigma = sigma .* (1 + 1.5 * kappa);
        % 计算EHVI
        % ref_point_fixed = [min(Y_train(:,1))-0.1, min(Y_train(:,2))-0.01];
        ehvi_vals = calculateNHVI(mu, adjusted_sigma, current_pf, ref_point_fixed);
        
        % 按EHVI排序候选点
        [~, idx] = sort(ehvi_vals, 'descend');
        candidate_points = ga_x(idx, :);
        candidate_ehvi_vals = ehvi_vals(idx,:);
        current_ga_candidate_idex = size(candidate_points, 1);
        
        % 确保获得eval_per_gen个有效点
        new_points = zeros(eval_per_gen, n_vars);
        new_vals = zeros(eval_per_gen, 2);
        valid_count_gen = 0;
        attempt_count_gen = 0;
        candidate_idx = 1;
        max_perturb_attempts = 0;  % 单点最大扰动尝试次数
        
        while valid_count_gen < eval_per_gen && attempt_count_gen < current_ga_candidate_idex+max_attempts_per_gen
            if candidate_idx > size(candidate_points, 1)
                % === 高效升阻比约束点生成 ===
                % 策略1：利用已知高升阻比区域
                valid_ld_points = X_train(Y_train(:,1) >= 9.0, :);
                
                if ~isempty(valid_ld_points)
                    % 在已知高升阻比点附近采样
                    base_idx = randi(size(valid_ld_points, 1));
                    base_point = valid_ld_points(base_idx, :);
                    perturbation = (rand(1, n_vars) - 0.5) .* (ub - lb) * 0.1;
                    new_candidate = base_point + perturbation;
                    new_candidate = min(max(new_candidate, lb), ub);
                    
                    % 策略2：代理模型引导优化
                    [mu_ld, ~] = predict(krig_LD, new_candidate);

                    if mu_ld < 9.0
                        % 使用fmincon微调点位置
                        options = optimoptions('fmincon', 'Display', 'none', 'Algorithm', 'sqp');
                        [new_candidate, ~] = fmincon(@(x) abs(predict(krig_LD, x) - 10.0), ...
                            new_candidate, [], [], [], [], lb, ub, ...
                            @(x) ldConstraint(x, krig_LD), options);
                    end
                else
                    % 策略3：目标导向采样
                    target_ld = 10.0; % 目标升阻比
                    new_candidate = fmincon(@(x) abs(predict(krig_LD, x) - target_ld), ...
                        (lb + ub)/2, [], [], [], [], lb, ub, [], ...
                        optimoptions('fmincon', 'Display', 'none'));
                end
                
                % 添加到候选列表
                candidate_points = [candidate_points; new_candidate];
                [mu_ld, sd_ld] = predict(krig_LD, new_candidate);
                [mu_vol, sd_vol] = predict(krig_vol, new_candidate);
                ehvi_new = calculateNHVI([mu_ld, mu_vol], [sd_ld, sd_vol], current_pf, ref_point_fixed);
                candidate_ehvi_vals = [candidate_ehvi_vals; ehvi_new];
                
                % fprintf('  新增点: 预测L/D=%.3f\n', mu_ld);
            end
            
            base_point = candidate_points(candidate_idx, :);
            candidate_idx = candidate_idx + 1;
            % 尝试原始点
            current_point = base_point;
            [success, L_D, vol_frac] = evaluatePoint(current_point);
            attempt_count_gen = attempt_count_gen + 1;
            perturb_count = 0;

            % 原始点失败时添加扰动重试
            while ~success && perturb_count < max_perturb_attempts
                % 生成小扰动（设计变量范围的±1%）
                perturbation = (rand(1, n_vars) - 0.5) .* (ub - lb) * 0.1;
                current_point = base_point + perturbation;
                current_point = min(max(current_point, lb), ub);  % 约束边界
                % 计算扰动点EHVI
                [mu_ld, sd_ld] = predict(krig_LD, current_point);
                [mu_vol, sd_vol] = predict(krig_vol, current_point);
                ehvi_perturb = calculateNHVI([mu_ld, mu_vol], [sd_ld, sd_vol], current_pf, ref_point_fixed);                
                % 调用程序尝试扰动
                [success, L_D, vol_frac] = evaluatePoint(current_point);
                % attempt_count_gen = attempt_count_gen + 1;
                perturb_count = perturb_count + 1;
                if success
                    candidate_ehvi_vals(candidate_idx-1)=ehvi_perturb;
                end
            end

            % 成功则记录结果（可能是原始点或扰动点）
            if success
                valid_count_gen = valid_count_gen + 1;                
                new_points(valid_count_gen, :) = current_point;  % 记录实际评估点
                new_vals(valid_count_gen, :) = [L_D, vol_frac];  % 记录评估值
                ehvi_val = NaN;
                if (candidate_idx - 1) <= length(candidate_ehvi_vals)
                    ehvi_val = candidate_ehvi_vals(candidate_idx - 1);
                end
                % 显示结果（注明是否为扰动点）
                if perturb_count > 0
                    fprintf('  成功点 %d/%d (扰动%d次): L/D=%.3f, Vol=%.3f, NHVI=%.4f\n', ...
                        valid_count_gen, eval_per_gen, perturb_count, L_D, vol_frac, ...
                        ehvi_val);
                elseif (candidate_idx - 1) <=current_ga_candidate_idex
                    fprintf('  成功点 %d/%d (原始点): L/D=%.3f, Vol=%.3f, NHVI=%.4f\n', ...
                        valid_count_gen, eval_per_gen, L_D, vol_frac, ...
                        ehvi_val);
                else
                    fprintf('  成功点 %d/%d (新增点): L/D=%.3f, Vol=%.3f, NHVI=%.4f\n', ...
                        valid_count_gen, eval_per_gen, L_D, vol_frac, ...
                        ehvi_val);
                end
            % else
                % % 打印失败警告
                % fprintf('  放弃点 (尝试%d次): EHVI=%.4f\n', ...
                %     perturb_count + 1, candidate_ehvi_vals(candidate_idx - 1));
            end
        end
        
        % 裁剪到实际有效点
        new_points = new_points(1:valid_count_gen, :);
        new_vals = new_vals(1:valid_count_gen, :);
        
        % 更新代理模型和Pareto前沿
        if ~isempty(new_points)
            X_train = [X_train; new_points];
            Y_train = [Y_train; new_vals];
            
            fprintf('重新训练代理模型...\n');
            n_samples = size(X_train,1);
            if n_samples <= 500
                fit_method = 'exact'; % 小数据量用精确解，精度高
            else
                fit_method = 'sr'; % 大数据量用稀疏近似，速度快
            end
            opts = {'KernelFunction','squaredexponential', 'FitMethod',fit_method, 'PredictMethod',fit_method};
            krig_LD = fitrgp(X_train, Y_train(:,1), opts{:});
            krig_vol = fitrgp(X_train, Y_train(:,2), opts{:});
            
            % 更新Pareto前沿
            [~, true_pf] = extract_pareto_front(Y_train, false);
        else
            fprintf('警告：本代未获得新有效点！\n');
        end
        
        % 计算超体积并检查收敛
        hv_previous = hv_current;
        hv_current = calculateHypervolume(true_pf, ref_point_fixed);
        hv_history(gen+1) = hv_current;
        
        % 计算相对改进
        if gen == 1
            hv_improvement = (hv_current - hv_previous) / abs(hv_previous);
        else
            hv_improvement = (hv_current - hv_history(end-1)) / abs(hv_history(end-1));
        end
        
        % 检查是否停滞
        if gen > 1 && abs(hv_improvement) < hv_threshold
            stall_count = stall_count + 1;
            fprintf('停滞计数: %d/%d (改进: %.4f%%)\n', ...
                stall_count, max_stall_generations, hv_improvement*100);
        else
            stall_count = 0; % 重置计数器
        end
        
        % 显示进度
        gen_time = toc;
        total_time = total_time + gen_time;
        fprintf('超体积: %.4f (改进: %.4f%%)', hv_current, hv_improvement*100);
        fprintf('新增有效点: %d/%d, 总点数: %d\n', ...
            size(new_points,1), eval_per_gen, size(X_train,1));
        fprintf('本代用时: %.1f秒, 累计用时: %.1f分钟\n', gen_time, total_time/60);

        % 显示每一代的结果
        % figure(3)
        % scatter(ga_pf(:,1), ga_pf(:,2), 30, 'b', 'filled');
        % hold on;
        % scatter(Y_train(:,1), Y_train(:,2), 20, 'filled', 'MarkerFaceAlpha', 0.3);
        % hold on
        % scatter(true_pf(:,1), true_pf(:,2), 60, 'r', 'filled');
        % hold on
        
        % 每代保存结果
        save(sprintf('wave_rider_optimization_gen%d.mat', gen), ...
            'X_train', 'Y_train', 'true_pf', 'hv_history');
        
        % 安全停止条件
        if size(X_train, 1) >= max_estimation_point
            fprintf('达到最大评估点数(%d)，停止优化\n', max_estimation_point);
            break;
        end
    end
    
    % 6. 提取最终结果
    % front_idx = fastParetoFront(Y_train, 'max');
    % pareto_front = Y_train(front_idx, :);
    [front_idx, pareto_front] = extract_pareto_front(Y_train, false);
    design_vars = X_train(front_idx, :);
    
    % 7. 结果可视化
    [sorted_front, sort_idx] = sortrows(pareto_front, 1);
    sorted_design = design_vars(sort_idx, :);    
    figure;
    plot(sorted_front(:,1), sorted_front(:,2), 'LineWidth', 2);
    hold on;
    scatter(Y_train(:,1), Y_train(:,2), 20, 'filled', 'MarkerFaceAlpha', 0.3);
    hold on
    scatter(sorted_front(:,1), sorted_front(:,2), 60, 'r', 'filled');
    hold on
    xlabel('升阻比 (L/D)');
    ylabel('容积率 (Vol. Fraction)');
    title('乘波体设计Pareto前沿');
    legend('Pareto 前沿', '所有评估点', 'Pareto 最优解');
    grid on;
    hold off;
    
    % 绘制超体积进展
    figure;
    plot(0:gen, hv_history, 'bo-', 'LineWidth', 1.5);
    xlabel('代数');
    ylabel('超体积');
    title('优化过程超体积进展');
    grid on;
    
    % 8. 保存最终结果
    save('wave_rider_optimization_final.mat', 'pareto_front', 'design_vars', 'X_train', 'Y_train', 'hv_history');
    fprintf('\n优化完成! 总代数: %d, 总点数: %d, 总用时: %.1f分钟\n', ...
        gen, size(X_train,1), total_time/60);
end

% ... (保持所有辅助函数不变) ...
function [success, L_D, vol_frac] = evaluatePoint(x)
    success = false;
    L_D = NaN;
    vol_frac = NaN;
    Ma = 15; 
    angle_shock_a1 = x(1); 
    angle_shock_a2 = x(2); 
    angle_shock_b1 = x(3); 
    angle_shock_b2 = x(4);
    c1 = 1.0331;
    c2 = 1.0331;
    d1 = x(5);
    AOA = -4; 
    a_xishu_1 = x(6);
        
    try
        % 尝试执行计算
        [result.L_D, result.vol_frac] = waveRiderEval(Ma, angle_shock_a1, angle_shock_a2, angle_shock_b1, angle_shock_b2, c1, c2, d1, AOA, a_xishu_1);

        % 检查结果是否有效
        if ~isnan(result.L_D) && ~isnan(result.vol_frac)
            L_D = result.L_D;
            vol_frac = result.vol_frac;
            success = true;
        end
        
    catch ME
        % 如果是超时错误，特别处理
        if contains(ME.message,'Waverider_timeout')
            fprintf('  评估超时，已终止\n');
        else
            fprintf('  评估失败: %s\n', ME.message);
        end
        
    end
end

function hv = calculateHypervolume(front, ref_point)
    
    
    % 移除比参考点差的解
    valid_idx = front(:,1) > ref_point(1) & front(:,2) > ref_point(2);
    front = front(valid_idx, :);
    
    if isempty(front)
        hv = 0;
        return;
    end
    
    % 按第一个目标降序排列 (X轴：升阻比)
    [sorted_front, idx] = sortrows(front, 1, 'descend');
    n = size(sorted_front, 1);
    
    % 初始化超体积和Y方向最大值
    hv = 0;
    last_y = ref_point(2);
    
    % 累加矩形面积
    for i = 1:n
        x = sorted_front(i, 1);
        y = sorted_front(i, 2);
        
        if y > last_y
            hv = hv + (x - ref_point(1)) * (y - last_y);
            last_y = y;
        end
    end
end

function [pf_indices, pf_obj] = extract_pareto_front(obj, minimize)
    % 从目标函数值矩阵中提取Pareto前沿
    % 输入:
    %   obj      : N x M 矩阵，N个解，M个目标函数
    %   minimize : 布尔值，true=最小化目标（默认），false=最大化目标
    % 输出:
    %   pf_indices : Pareto前沿解的行索引
    %   pf_obj     : Pareto前沿的目标函数值

    if nargin < 2
        minimize = true; % 默认为最小化问题
    end

    [N, ~] = size(obj);
    dominated = false(N, 1); % 标记解是否被支配

    % 最小化与最大化的支配条件不同
    if minimize
        for i = 1:N
            if dominated(i), continue; end % 跳过已标记的解
            for j = 1:N
                if i == j, continue; end % 不自比
                % 检查支配：j是否所有目标 <= i 且 至少一个目标 < i
                if all(obj(j,:) <= obj(i,:)) && any(obj(j,:) < obj(i,:))
                    dominated(i) = true;
                    break; % 发现支配立即跳出
                end
            end
        end
    else % 最大化问题
        for i = 1:N
            if dominated(i), continue; end
            for j = 1:N
                if i == j, continue; end
                % 检查支配：j是否所有目标 >= i 且 至少一个目标 > i
                if all(obj(j,:) >= obj(i,:)) && any(obj(j,:) > obj(i,:))
                    dominated(i) = true;
                    break;
                end
            end
        end
    end

    pf_indices = find(~dominated); % 提取非支配解索引
    pf_obj = obj(pf_indices, :);   % 获取对应的目标值
end

% 辅助函数：升阻比约束
function [c, ceq] = ldConstraint(x, krig_model)
    % 约束升阻比在3.6-4.0范围内
    mu_ld = predict(krig_model, x);
    c = [mu_ld - 10.0;  % mu_ld ≤ 4.0
         8.0 - mu_ld]; % mu_ld ≥ 3.6
    ceq = [];
end

function nhvi = calculateNHVI(mu, sigma, pareto_front, ref_point)
    % 计算负超体积改进
    % 基于: "Swrm and Evolutionary Computation 99(2025):102190"
    % 核心改进：对被支配区域赋予负值，消除零梯度平台，提供搜索动力。
    
    % 预筛选：只计算有潜力的候选点
    if isempty(pareto_front)
        min_LD = inf;
        min_Vol = inf;
    else
        min_LD = min(pareto_front(:, 1));
        min_Vol = min(pareto_front(:, 2));
    end
    
    n = size(mu, 1);
    nhvi = zeros(n, 1);
    n_samples = 200; % 蒙特卡洛样本数
    
    % 预计算当前Pareto前沿的超体积作为基准
    hv_ref = calculateHypervolume(pareto_front, ref_point);
    
    parfor i = 1:n
        % 跳过明显劣质的候选点 (此预筛选逻辑在NHVI中可选，因为NHVI允许负值，
        % 但为了计算效率，保留此筛选可以跳过极差点的计算)
        if mu(i, 1) < min_LD && mu(i, 2) < min_Vol
            nhvi(i) = 0; 
            continue;
        end
        
        % 生成样本：分别对每个目标生成
        samples_ld = normrnd(mu(i, 1), sigma(i, 1), n_samples, 1);
        samples_vol = normrnd(mu(i, 2), sigma(i, 2), n_samples, 1);
        samples = [samples_ld, samples_vol];
        
        % 计算每个样本的超体积改进
        hv_improvement = zeros(n_samples, 1);
        
        for j = 1:n_samples
            sample = samples(j, :);
            
            % --- NHVI 核心逻辑修改开始 ---
            
            % 1. 检查样本是否被当前前沿支配
            % 用户代码逻辑：若 sample <= pf 且 sample < pf，则被支配
            dominated = false;
            dominating_indices = []; % 记录哪些Pareto点支配了该样本
            
            for k = 1:size(pareto_front, 1)
                pf_point = pareto_front(k, :);
                if all(sample <= pf_point) && any(sample < pf_point)
                    dominated = true;
                end
                % 同时检查该Pareto点是否支配样本 (用于构建新前沿)
                % 如果 pf >= sample (且不全等)，则 pf 支配 sample
                if all(pf_point >= sample) && any(pf_point > sample)
                    dominating_indices = [dominating_indices; k];
                end
            end
            
            if ~dominated
                % --- 情况 A: 样本未被支配 (标准 HVI 逻辑) ---
                % 新前沿 = 当前前沿 + 样本
                temp_front = [pareto_front; sample];
                
                % 移除被新样本支配的点 (如果新样本非常好，可能会支配旧点)
                dominated_idx = false(size(temp_front, 1), 1);
                for k = 1:size(temp_front, 1)
                    for l = 1:size(temp_front, 1)
                        if k ~= l && all(temp_front(l, :) >= temp_front(k, :)) && any(temp_front(l, :) > temp_front(k, :))
                            dominated_idx(k) = true;
                            break;
                        end
                    end
                end
                temp_front = temp_front(~dominated_idx, :);
                
                hv_new = calculateHypervolume(temp_front, ref_point);
                hv_improvement(j) = hv_new - hv_ref; % 结果为正
                
            else
                % --- 情况 B: 样本被支配 (NHVI 特有逻辑) ---
                % 根据论文 Eq.6 和 Eq.7: H(Pn ∩ {y'})
                % NHVI的定义是：新前沿 = (原前沿 - 支配sample的点) + sample
                
                keep_mask = true(size(pareto_front, 1), 1);
                if ~isempty(dominating_indices)
                    keep_mask(dominating_indices) = false;
                end
                
                temp_front = [pareto_front(keep_mask, :); sample];
                
                % 计算新的超体积
                hv_new = calculateHypervolume(temp_front, ref_point);
                
                % 改进值 = 新体积 - 原体积
                % 由于用较差的点替换了较好的点，hv_new < hv_ref，结果为负值
                hv_improvement(j) = hv_new - hv_ref; 
            end
            
            % --- NHVI 核心逻辑修改结束 ---
        end
        
        % NHVI 是改进的期望值 (包含正值和负值)
        nhvi(i) = mean(hv_improvement);
    end
end
