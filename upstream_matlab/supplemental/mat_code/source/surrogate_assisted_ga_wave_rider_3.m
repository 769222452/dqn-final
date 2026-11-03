function [pareto_front, design_vars] = surrogate_assisted_ga_wave_rider_3()
    % 1. 参数设置
    n_vars = 5;
    lb = [13, 0.1, 0.8, -0.1, 0.96];
    ub = [20, 0.9, 1.2, 0.1, 1.04];

    % 数据点参数
    min_valid_points = 300; % 初始采样点数
    ga_pop_size = 150; % GA 种群参数
    eval_per_gen = 30; % 代理模型每代新增点数
    max_attempts_per_gen = 50; % 为确保代理模型每代新增的点数而最多的
    
    % 收敛监测参数
    max_stall_generations = 20;  % 连续停滞代数阈值
    hv_threshold = 0.001;      % 超体积相对变化阈值
    max_estimation_point = 20000; % 最大评估点数

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
    opts = {'KernelFunction','squaredexponential', 'FitMethod','sr', 'PredictMethod','sr'};
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
    ref_point_fixed = [min(Y_train(:,1))-0.1, min(Y_train(:,2))-0.01];
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
            'CrossoverFraction', 0.9, ...           % 交叉概率85%
            'CrossoverFcn', @crossoverintermediate, ... % 分散交叉
            'MutationFcn', {@mutationadaptfeasible, 0.07}, ... % 7%个体变异，标准差0.3
            'SelectionFcn', {@selectiontournament, 3}, ... % 锦标赛选择（规模3）
            'FunctionTolerance', 1e-4, ...            % 函数值收敛容差
            'ParetoFraction', 0.8, ...                % Pareto解保留比例
            'Display', 'off', ...
            'UseParallel', false);
        [ga_x, ga_pf] = gamultiobj(surrogateFitness, n_vars, [], [], [], [], lb, ub, GA_opts);
        
        % 预测候选点的均值和标准差
        [mu_LD, sd_LD] = predict(krig_LD, ga_x);
        [mu_vol, sd_vol] = predict(krig_vol, ga_x);
        mu = [mu_LD, mu_vol];
        sigma = [sd_LD, sd_vol];
        % 混合pareto前缘
        fused_pf = [true_pf;ga_pf];
        [~, fused_pf] = extract_pareto_front(fused_pf, false);
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
        ref_point_fixed = [min(Y_train(:,1))-0.1, min(Y_train(:,2))-0.01];
        ehvi_vals = calculateEHVI(mu, adjusted_sigma, fused_pf, ref_point_fixed);
        
        % 按EHVI排序候选点
        [~, idx] = sort(ehvi_vals, 'descend');
        candidate_points = ga_x(idx, :);
        current_ga_candidate_idex = size(candidate_points, 1);
        
        % 确保获得eval_per_gen个有效点
        new_points = zeros(eval_per_gen, n_vars);
        new_vals = zeros(eval_per_gen, 2);
        valid_count_gen = 0;
        attempt_count_gen = 0;
        candidate_idx = 1;
        
        while valid_count_gen < eval_per_gen && attempt_count_gen < current_ga_candidate_idex+max_attempts_per_gen
            if candidate_idx > size(candidate_points, 1)
                new_candidate = lb + (ub - lb) .* rand(1, n_vars);
                candidate_points = [candidate_points; new_candidate];
                % 为新点计算EHVI (可选)
                [mu_ld, sd_ld] = predict(krig_LD, new_candidate);
                [mu_vol, sd_vol] = predict(krig_vol, new_candidate);
                ehvi_new = calculateEHVI([mu_ld, mu_vol], [sd_ld, sd_vol], fused_pf, ref_point_fixed);
                ehvi_vals = [ehvi_vals; ehvi_new];
            end
            
            current_point = candidate_points(candidate_idx, :);
            candidate_idx = candidate_idx + 1;
            
            [success, L_D, vol_frac] = evaluatePoint(current_point);
            attempt_count_gen = attempt_count_gen + 1;
            
            if success
                valid_count_gen = valid_count_gen + 1;
                new_points(valid_count_gen, :) = current_point;
                new_vals(valid_count_gen, :) = [L_D, vol_frac];
                
                ehvi_val = NaN;
                if (candidate_idx - 1) <= length(ehvi_vals)
                    ehvi_val = ehvi_vals(candidate_idx - 1);
                end
                
                fprintf('  成功点 %d/%d: L/D=%.3f, Vol=%.3f, EHVI=%.4f\n', ...
                    valid_count_gen, eval_per_gen, L_D, vol_frac, ehvi_val);
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
            opts = {'KernelFunction','squaredexponential', 'FitMethod','sr', 'PredictMethod','sr'};
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
    Ma = 8; 
    beta = x(1); 
    Ri = x(2); 
    c0 = x(3); 
    c1 = x(4); 
    AOA = 0; 
    a_xishu_1 = x(5);
    
    % 设置超时时间（3分钟）
    timeout = 180; 
    
    try
        % 创建共享变量来存储结果
        result = struct('L_D', NaN, 'vol_frac', NaN, 'completed', false);
        
        % 创建并启动计时器
        t = timer('TimerFcn', @(~,~) error('Timeout'), ...
                  'StartDelay', timeout);
        start(t);
        
        % 尝试执行计算
        [result.L_D, result.vol_frac] = waveRiderEval(Ma, beta, Ri, c0, c1, AOA, a_xishu_1);
        result.completed = true;
        
        % 停止并删除计时器
        stop(t);
        delete(t);
        
        % 检查结果是否有效
        if result.completed && ~isnan(result.L_D) && ~isnan(result.vol_frac)
            L_D = result.L_D;
            vol_frac = result.vol_frac;
            success = true;
        end
        
    catch ME
        % 如果是超时错误，特别处理
        if strcmp(ME.message, 'Timeout')
            fprintf('  评估超时（超过 %d 秒），已终止\n', timeout);
        else
            fprintf('  评估失败: %s\n', ME.message);
        end
        
        % 确保清理计时器
        if exist('t', 'var') && isvalid(t)
            stop(t);
            delete(t);
        end
    end
end

function ehvi = calculateEHVI(mu, sigma, pareto_front, ref_point)
    % 计算期望超体积改进 (EHVI)
    % mu: 候选点的预测均值 (n x 2) [L/D, Vol]
    % sigma: 预测标准差 (n x 2)
    % pareto_front: 当前Pareto前沿 (m x 2)
    % ref_point: 参考点 (1 x 2) [min_L/D, min_Vol]
    
    % 预筛选：只计算有潜力的候选点
    if isempty(pareto_front)
        min_LD = inf;
        min_Vol = inf;
    else
        min_LD = min(pareto_front(:, 1));
        min_Vol = min(pareto_front(:, 2));
    end
    
    n = size(mu, 1);
    ehvi = zeros(n, 1);
    n_samples = 200; % 蒙特卡洛样本数
    
    % 预计算参考点超体积
    hv_ref = calculateHypervolume(pareto_front, ref_point);
    
    for i = 1:n
        % 跳过明显劣质的候选点
        if mu(i, 1) < min_LD && mu(i, 2) < min_Vol
            ehvi(i) = 0;
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
            
            % 检查样本是否被当前前沿支配
            dominated = false;
            for k = 1:size(pareto_front, 1)
                if all(sample <= pareto_front(k, :)) && any(sample < pareto_front(k, :))
                    dominated = true;
                    break;
                end
            end
            
            if dominated
                hv_improvement(j) = 0;
            else
                % 创建临时前沿 = 当前前沿 + 新样本
                temp_front = [pareto_front; sample];
                
                % 移除被支配的点（包括新样本可能支配的点）
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
                
                % 计算超体积改进
                hv_new = calculateHypervolume(temp_front, ref_point);
                hv_improvement(j) = max(0, hv_new - hv_ref);
            end
        end
        
        % EHVI 是改进的期望值
        ehvi(i) = mean(hv_improvement);
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

function [Y_norm, min_vals, range_vals] = normalizeObjectives(Y)
    min_vals = min(Y);
    max_vals = max(Y);
    range_vals = max_vals - min_vals;
    
    % 防止除零错误（当所有值相同时）
    range_vals(range_vals == 0) = 1;
    
    % 线性归一化到[0,1]
    Y_norm = (Y - min_vals) ./ range_vals;
end

function Y_orig = denormalizeObjectives(Y_norm, min_vals, range_vals)
    Y_orig = Y_norm .* range_vals + min_vals;
end

