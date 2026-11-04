function [pareto_front, design_vars] = surrogate_assisted_ga_wave_rider_9()
    % 1. 参数设置
    n_vars = 5;
    lb = [4, 4, 4, 4, 0.96];
    ub = [8, 8, 8, 8, 1.04];
    min_valid_points = 50; % 初始采样点数
    ga_pop_size = 100; % GA 种群参数
    eval_per_gen = 20; % 代理模型每代新增点数
    max_attempts_per_gen = 50; % 为确保代理模型每代新增的点数而最多的扰动尝试
    
    max_stall_generations = 20; % 连续停滞代数阈值 
    hv_threshold = 0.001; % 超体积相对变化阈值     
    max_estimation_point = 5000; % 最大评估点数
    
    % 2. 初始采样
    fprintf('初始采样中...目标：%d个有效点\n', min_valid_points);
    X_all = [];  Y_feas = [];   % 全局池：存储所有尝试点及可行性(0/1)
    X_valid = []; Y_valid = []; % 有效池：仅存储成功点及目标值
    valid_count = 0; attempt_count = 0;
    
    while valid_count < min_valid_points
        new_X = lb + (ub - lb) .* lhsdesign(1, n_vars, 'criterion','maximin');
        [success, L_D, vol_frac] = evaluatePoint(new_X);
        attempt_count = attempt_count + 1;
        
        X_all = [X_all; new_X]; 
        Y_feas = [Y_feas; success];
        
        if success
            X_valid = [X_valid; new_X];
            Y_valid = [Y_valid; L_D, vol_frac];
            valid_count = valid_count + 1;
        end
        
        if mod(attempt_count, 10) == 0
            fprintf('已尝试 %d 点，有效点数: %d/%d (%.1f%%)\n', ...
                attempt_count, valid_count, min_valid_points, 100*valid_count/attempt_count);
        end
    end
    fprintf('初始采样完成! 共尝试 %d 点，获得 %d 有效点\n', attempt_count, valid_count);
    
    % 3. 构建初始代理模型
    opts_feas = getGPROpts(size(X_all, 1));
    krig_feas = fitrgp(X_all, Y_feas, opts_feas{:});
    
    opts_perf = getGPROpts(size(X_valid, 1));
    krig_LD = fitrgp(X_valid, Y_valid(:,1), opts_perf{:});
    krig_vol = fitrgp(X_valid, Y_valid(:,2), opts_perf{:});
    
    % 4. 初始化优化循环
    total_time = 0; 
    stall_count = 0; 
    gen = 0;
    [~, current_pf] = extract_pareto_front(Y_valid, false);
    ref_point_fixed = [0, 0];
    hv_current = calculateHypervolume(current_pf, ref_point_fixed);
    hv_history = hv_current;
    fprintf('初始超体积: %.4f\n', hv_current);
    
    % 5. 主优化循环
    while stall_count < max_stall_generations
        gen = gen + 1;
        tic;
        fprintf('\n=== 第 %d 代 ===\n', gen);
        
        % NSGA-II优化代理模型
        surrogateFitness = @(x) [-predict(krig_LD, x), -predict(krig_vol, x)];
        GA_opts = optimoptions('gamultiobj', ...
            'PopulationSize', ga_pop_size, ...
            'CrossoverFraction', 0.85, ...
            'CrossoverFcn', @crossoverscattered, ...
            'MutationFcn', {@mutationgaussian, 0.07, 0.3}, ...
            'SelectionFcn', {@selectiontournament, 3}, ...
            'FunctionTolerance', 1e-5, ...
            'ParetoFraction', 0.8, ...
            'Display', 'off', ...
            'UseParallel', false);
        [ga_x, ~] = gamultiobj(surrogateFitness, n_vars, [], [], [], [], lb, ub, GA_opts);
        
        % 预测均值与标准差
        [mu_LD, sd_LD] = predict(krig_LD, ga_x);
        [mu_vol, sd_vol] = predict(krig_vol, ga_x);
        mu = [mu_LD, mu_vol]; sigma = [sd_LD, sd_vol];
        
        % 动态平衡开发与探索 (kappa策略)
        base_kappa = 0.5 + min(0.5, 0.015 * (gen - 1));
        kappa = min(1.0, base_kappa + 0.2 * stall_count);
        adjusted_sigma = sigma .* (1 + 1.5 * kappa);
        
        % 计算NHVI并排序
        ehvi_vals = calculateNHVI(mu, adjusted_sigma, current_pf, ref_point_fixed);
        [~, idx] = sort(ehvi_vals, 'descend');
        sorted_ga_x = ga_x(idx, :);
        sorted_ehvi = ehvi_vals(idx);
        
        % 候选点池分层
        pos_idx = sorted_ehvi > 0;
        pool_positive = sorted_ga_x(pos_idx, :);
        pool_positive_ehvi = sorted_ehvi(pos_idx);
        pool_negative = sorted_ga_x(~pos_idx, :);
        pool_negative_ehvi = sorted_ehvi(~pos_idx);
        
        perturb_bases = pool_positive;
        if isempty(perturb_bases)
            perturb_bases = pool_negative; 
        end
        
        fprintf('候选点分层: 正NHVI %d, 负NHVI %d\n', size(pool_positive,1), size(pool_negative,1));
        
        % 评估候选点
        valid_count_gen = 0; 
        attempt_count_gen = 0;
        pos_pool_exhausted = isempty(pool_positive);
        perturb_mode_active = false; 
        perturb_base_idx = 1; 
        perturb_try_count = 0;
        
        gen_X_all = []; gen_Y_feas = []; 
        gen_X_valid = []; gen_Y_valid = [];
        
        max_total_attempts = size(sorted_ga_x, 1) + max_attempts_per_gen;
        
        while valid_count_gen < eval_per_gen && attempt_count_gen < max_total_attempts
            current_point = []; current_ehvi = NaN; skip_eval = false;
            
            % --- 优先级 1: 正池 ---
            if ~pos_pool_exhausted
                current_point = pool_positive(1, :);
                current_ehvi = pool_positive_ehvi(1);
                pool_positive(1,:) = []; pool_positive_ehvi(1) = [];
                if isempty(pool_positive)
                    pos_pool_exhausted = true; 
                    fprintf('  正池耗尽\n'); 
                end
                
            % --- 优先级 2: 扰动 ---
            elseif ~perturb_mode_active
                perturb_try_count = perturb_try_count + 1;
                if perturb_try_count <= max_attempts_per_gen
                    base_idx = mod(perturb_base_idx - 1, size(perturb_bases, 1)) + 1;
                    base_point = perturb_bases(base_idx, :);
                    perturb_base_idx = perturb_base_idx + 1;
                    noise = (ub - lb) .* randn(1, n_vars) .* 0.05;
                    current_point = max(lb, min(ub, base_point + noise));
                    
                    [m_ld, s_ld] = predict(krig_LD, current_point);
                    [m_vol, s_vol] = predict(krig_vol, current_point);
                    current_ehvi = calculateNHVI([m_ld, m_vol], [s_ld, s_vol], current_pf, ref_point_fixed);
                    if current_ehvi <= 0
                        skip_eval = true; 
                    end 
                else
                    perturb_mode_active = true; 
                    fprintf('  扰动达上限，进负池\n');
                end
                
            % --- 优先级 3: 负池 & 随机兜底 ---
            else
                if ~isempty(pool_negative)
                    current_point = pool_negative(1, :);
                    current_ehvi = pool_negative_ehvi(1);
                    pool_negative(1,:) = []; pool_negative_ehvi(1) = [];
                else
                    current_point = lb + (ub - lb) .* rand(1, n_vars);
                    [m_ld, s_ld] = predict(krig_LD, current_point);
                    [m_vol, s_vol] = predict(krig_vol, current_point);
                    current_ehvi = calculateNHVI([m_ld, m_vol], [s_ld, s_vol], current_pf, ref_point_fixed);
                end
            end
            
            % 执行真实评估
            if ~skip_eval && ~isempty(current_point)
                mu_feas_candidate = predict(krig_feas, current_point);
                if mu_feas_candidate < 0.4
                    continue; % 可行性过低，跳过
                end
                
                attempt_count_gen = attempt_count_gen + 1;
                [success, L_D, vol_frac] = evaluatePoint(current_point);
                
                gen_X_all = [gen_X_all; current_point];
                gen_Y_feas = [gen_Y_feas; success];
                
                if success
                    valid_count_gen = valid_count_gen + 1;
                    gen_X_valid = [gen_X_valid; current_point];
                    gen_Y_valid = [gen_Y_valid; L_D, vol_frac];
                    fprintf('  成功 %d/%d: L/D=%.3f, V=%.3f, NHVI=%.4f\n', valid_count_gen, eval_per_gen, L_D, vol_frac, current_ehvi);
                else
                    fprintf('  失败(预测可行=%.3f)\n', mu_feas_candidate);
                end
            end
            
            % 安全阀防死循环
            if pos_pool_exhausted && isempty(pool_negative) && perturb_try_count > max_attempts_per_gen + 10
                break;
            end
        end
        
        % 更新全局数据
        X_all = [X_all; gen_X_all]; Y_feas = [Y_feas; gen_Y_feas];
        X_valid = [X_valid; gen_X_valid]; Y_valid = [Y_valid; gen_Y_valid];
        
        % 重新训练代理模型
        opts_feas = getGPROpts(size(X_all, 1));
        krig_feas = fitrgp(X_all, Y_feas, opts_feas{:});
        
        if ~isempty(gen_X_valid)
            opts_perf = getGPROpts(size(X_valid, 1));
            krig_LD = fitrgp(X_valid, Y_valid(:,1), opts_perf{:});
            krig_vol = fitrgp(X_valid, Y_valid(:,2), opts_perf{:});
            [~, current_pf] = extract_pareto_front(Y_valid, false);
        else
            fprintf('警告：本代未获新有效点\n');
        end
        
        % 收敛监测
        hv_previous = hv_current;
        hv_current = calculateHypervolume(current_pf, ref_point_fixed);
        hv_history = [hv_history; hv_current];
        hv_improvement = (hv_current - hv_previous) / abs(hv_previous);
        
        if gen > 1 && abs(hv_improvement) < hv_threshold
            stall_count = stall_count + 1;
            fprintf('停滞 %d/%d (改进: %.4f%%)\n', stall_count, max_stall_generations, hv_improvement*100);
        else
            stall_count = 0;
        end
        
        gen_time = toc; total_time = total_time + gen_time;
        fprintf('HV: %.4f (+%.4f%%) | 新增有效: %d | 总有效: %d | 总点数: %d | 耗时: %.1fs (累计 %.1fmin)\n', ...
            hv_current, hv_improvement*100, size(gen_X_valid,1), size(X_valid,1), size(X_all,1), gen_time, total_time/60);
        
        % 每代保存
        save(sprintf('wave_rider_optimization_gen%d.mat', gen), 'X_valid', 'Y_valid', 'current_pf', 'hv_history');
        
        if size(X_all, 1) >= max_estimation_point
            fprintf('达到最大评估点数(%d)，停止\n', max_estimation_point);
            break;
        end
    end
    
    % 6. 提取最终结果
    [front_idx, pareto_front] = extract_pareto_front(Y_valid, false);
    design_vars = X_valid(front_idx, :);
    
    % 7. 可视化
    [sorted_front, sort_idx] = sortrows(pareto_front, 1);
    sorted_design = design_vars(sort_idx, :);    
    figure;
    scatter(Y_valid(:,1), Y_valid(:,2), 20, 'filled', 'MarkerFaceAlpha', 0.3); hold on;
    plot(sorted_front(:,1), sorted_front(:,2), 'LineWidth', 2); 
    scatter(sorted_front(:,1), sorted_front(:,2), 60, 'r', 'filled');
    xlabel('升阻比 (L/D)'); ylabel('容积率 (Vol. Fraction)');
    title('乘波体Pareto前沿'); legend('所有有效点', 'Pareto前沿', 'Pareto最优解'); grid on;
    
    figure; plot(0:length(hv_history)-1, hv_history, 'bo-', 'LineWidth', 1.5);
    xlabel('代数'); ylabel('超体积'); title('超体积进展'); grid on;
    
    % 8. 保存
    save('wave_rider_optimization_final.mat', 'pareto_front', 'design_vars', 'X_valid', 'Y_valid', 'hv_history');
    fprintf('\n优化完成! 代数: %d, 总点数: %d, 用时: %.1fmin\n', gen, size(X_all,1), total_time/60);
end

%% ==================== 辅助函数 ====================
% 提取重复的GPR参数设置
function opts = getGPROpts(n_samples)
    fit_method = 'exact';
    if n_samples > 500, fit_method = 'sr'; end
    opts = {'KernelFunction','matern52', 'FitMethod',fit_method, 'PredictMethod',fit_method};
end
% 评估函数
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
    d1 = 0.07;
    AOA = -4; 
    a_xishu_1 = x(5);
        
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
        
        % % 确保清理计时器
        % if exist('t', 'var') && isvalid(t)
        %     stop(t);
        %     delete(t);
        % end
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
    n_samples = 2000; % 蒙特卡洛样本数
    
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
