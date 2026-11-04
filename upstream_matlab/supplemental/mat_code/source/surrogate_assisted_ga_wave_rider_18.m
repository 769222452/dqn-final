function [pareto_front, design_vars] = surrogate_assisted_ga_wave_rider_18()
    %%%%%%%%%%%%%%%%Introduction%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 高斯过程回归预测升阻比与容积率，fitrgp
    % QDA 分类器预测可行性，fitcdiscr
    % NHVI * Sparsity^2 的混合加点准则
    % 使用WaveRiderUtils.m将辅助函数集成
    % Phase 1(Pareto前沿快速收敛阶段) + Phase 2（Pareto前沿插值）双重距离去重
    % 每一代内先phase 1(Pareto前沿快速收敛阶段)，再Phase 2（Pareto前沿插值）
    % 去重不使用设计空间，而利用目标空间
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Feasibility classifier as nonlinear constraint in GA for higher candidate feasibility rate
    % 1. 参数设置
    n_vars = 6;
    lb = [4, 4, 4, 4, 0.02, 0.96];
    ub = [15, 15, 15, 15, 0.18,1.04];
    min_valid_points = 50; % 初始采样点数
    ga_pop_size = 100; % GA 种群参数
    eval_per_gen = 20; % 代理模型每代新增点数
    max_attempts_per_gen = 50; % 为确保代理模型每代新增的点数而最多的扰动尝试
    
    max_stall_generations = 5; % 连续停滞代数阈值 
    hv_threshold = 0.008; % 超体积相对变化阈值     
    max_estimation_point = 5000; % 最大评估点数
    skip_count = 0; % 用于记录分类跳过的点数
    dup_skip_count = 0; % 用于记录距离去重跳过
    fitc_count = 1; % 用于记录分类器训练次数


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
    % fprintf('初始采样完成! 共尝试 %d 点，获得 %d 有效点\n', attempt_count, valid_count);
    % save(sprintf('wave_rider_optimization_initial_date.mat'), 'X_valid', 'Y_valid', 'X_all', 'Y_feas');
    % load wave_rider_optimization_initial_date.mat X_valid Y_valid X_all Y_feas
    % 3. 构建初始代理模型
    % QDA classifier for feasibility prediction
    feas_qda = fitcdiscr(X_all, Y_feas, ...
        'DiscrimType', 'quadratic');


    % 性能预测模型
    opts_perf = WaveRiderUtils.getGPROpts(size(X_valid, 1));
    krig_LD = fitrgp(X_valid, Y_valid(:,1), opts_perf{:});
    krig_vol = fitrgp(X_valid, Y_valid(:,2), opts_perf{:});
    
    % 4. 初始化优化循环
    total_time = 0; 
    stall_count = 0; 
    gen = 0;
    [~, current_pf] = WaveRiderUtils.extract_pareto_front(Y_valid, false);
    ref_point_fixed = [min(Y_valid(:,1))-0.1, min(Y_valid(:,2))-0.01];
    hv_current = WaveRiderUtils.calculateHypervolume(current_pf, ref_point_fixed);
    hv_history = hv_current;
    spacing_current = WaveRiderUtils.calculateSpacing(current_pf);
    spacing_history = spacing_current;
    fprintf('初始超体积: %.4f\n', hv_current);
    fprintf('初始Spacing指标: %.4f (越小越均匀)\n', spacing_current);
    
    % 5. 主优化循环
    while stall_count < max_stall_generations        
        gen = gen + 1;
        tic;
        fprintf('\n=== 第 %d 代 ===\n', gen);        
        fprintf('Phase 1——前缘快速收敛阶段\n');
        feas_threshold = min(0.5, 0.2 + 0.05 * (fitc_count-1)); % 动态调整可行性阈值：前期宽松(0.2)，后期严格(0.4)
        % NSGA-II优化代理模型
        surrogateFitness = @(x) [-predict(krig_LD, x), -predict(krig_vol, x)];
        GA_opts = optimoptions('gamultiobj', ...
            'PopulationSize', ga_pop_size, ...
            'CrossoverFraction', 0.85, ...
            'CrossoverFcn', @crossoverscattered, ...
            'MutationFcn', @mutationadaptfeasible, ...
            'SelectionFcn', {@selectiontournament, 3}, ...
            'FunctionTolerance', 1e-5, ...
            'ParetoFraction', 0.8, ...
            'Display', 'off', ...
            'UseParallel', false);
        % Feasibility nonlinear constraint using QDA model
        feasFunc = @(x) deal(feas_threshold - predict(feas_qda, x), []);
        [ga_x, ~] = gamultiobj(surrogateFitness, n_vars, [], [], [], [], lb, ub, feasFunc, GA_opts);
        
        % 预测均值与标准差
        [mu_LD, sd_LD] = predict(krig_LD, ga_x);
        [mu_vol, sd_vol] = predict(krig_vol, ga_x);
        mu = [mu_LD, mu_vol]; sigma = [sd_LD, sd_vol];         
        % 计算NHVI并排序
        ehvi_vals = WaveRiderUtils.calculateNHVI(mu, sigma, current_pf, ref_point_fixed);
        % ===== 计算稀疏度指标 =====
        sparsity_vals = WaveRiderUtils.calculateSparsity(mu, current_pf);
        % === 归一化 ===
        % NHVI: sigmoid归一化 → 正值映射到[0.5,1]，负值映射到[0,0.5]
        abs_max_ehvi = max(abs(ehvi_vals));
        if abs_max_ehvi > eps
            ehvi_norm = 0.5 + 0.5 * ehvi_vals / abs_max_ehvi;
        else
            ehvi_norm = 0.5 * ones(size(ehvi_vals));
        end        
        % Sparsity: 线性归一化到[0,1]
        sparsity_range = max(sparsity_vals) - min(sparsity_vals);
        if sparsity_range > eps
            sparsity_norm = (sparsity_vals - min(sparsity_vals)) / sparsity_range;
        else
            sparsity_norm = 0.5 * ones(size(sparsity_vals));
        end
        % === 组合指标排序 ===
        combined_score = ehvi_norm .* (sparsity_norm .^ 2);
        [~, idx] = sort(combined_score, 'descend');
        sorted_ga_x = ga_x(idx, :);
        sorted_ehvi = ehvi_vals(idx);  % 保留原始NHVI用于后续分层和日志        
        
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
        
        fprintf('分类器阈值（%.3f）, 候选点分层（正NHVI %d, 负NHVI %d）\n', feas_threshold, size(pool_positive,1), size(pool_negative,1));
        
        % 评估候选点
        valid_count_gen = 0; 
        attempt_count_gen = 0;
        pos_pool_exhausted = isempty(pool_positive);
        neg_pool_exhausted = isempty(pool_negative);
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
                    current_ehvi = WaveRiderUtils.calculateNHVI([m_ld, m_vol], [s_ld, s_vol], current_pf, ref_point_fixed);
                    if current_ehvi <= 0
                        current_sparse_val = WaveRiderUtils.calculateSparsity([m_ld, m_vol], current_pf);
                        % 归一化稀疏度（使用本代sparsity_vals的统计量）
                        sp_median = median(sparsity_vals);
                        if current_sparse_val > sp_median
                            skip_eval = false;  % 后期且在稀疏区域：允许探索
                        else
                            skip_eval = true;   % 前期或非稀疏区域：跳过
                        end
                    end 
                else
                    perturb_mode_active = true; 
                    fprintf('  扰动达上限，进负池\n');
                end
                
            % --- 优先级 3: 负池 & 随机兜底 ---
            else
                if ~neg_pool_exhausted
                    current_point = pool_negative(1, :);
                    current_ehvi = pool_negative_ehvi(1);
                    pool_negative(1,:) = []; pool_negative_ehvi(1) = [];
                    [m_ld, ~] = predict(krig_LD, current_point);
                    [m_vol, ~] = predict(krig_vol, current_point);
                    current_sparse_val = WaveRiderUtils.calculateSparsity([m_ld, m_vol], current_pf);
                    sp_median = median(sparsity_vals);
                    if current_sparse_val > sp_median
                        skip_eval = false;
                    else
                        skip_eval = true; 
                    end
                    if isempty(pool_negative)
                        neg_pool_exhausted = true;
                        fprintf('  负池耗尽，随机扰动\n'); 
                    end
                else                    
                    current_point = lb + (ub - lb) .* rand(1, n_vars);
                    [m_ld, s_ld] = predict(krig_LD, current_point);
                    [m_vol, s_vol] = predict(krig_vol, current_point);
                    current_ehvi = WaveRiderUtils.calculateNHVI([m_ld, m_vol], [s_ld, s_vol], current_pf, ref_point_fixed);
                end
            end
            
            % 执行真实评估
            if ~skip_eval && ~isempty(current_point)
                [~, feas_score] = predict(feas_qda, current_point);
                mu_feas_candidate = feas_score(:, 2);
                if mu_feas_candidate < feas_threshold
                    skip_count = skip_count+1;
                    continue; % 可行性过低，跳过
                end
                % 距离去重：检查与所有已评估点的相对距离(设计空间)
                all_evaluated = [X_all; gen_X_all];
                min_dist = min(pdist2(all_evaluated, current_point));
                design_range = ub - lb;
                rel_dist = min_dist / norm(design_range);
                if rel_dist < 0.01
                    dup_skip_count = dup_skip_count + 1;
                    continue;
                end
                attempt_count_gen = attempt_count_gen + 1;
                [success, L_D, vol_frac] = evaluatePoint(current_point);
                
                gen_X_all = [gen_X_all; current_point];
                gen_Y_feas = [gen_Y_feas; success];
                
                if success
                    new_y = [L_D, vol_frac];
                    % 利用目标空间去重
                    if isempty(Y_valid)
                        target_dist = inf;
                    else  
                        obj_min = min(Y_valid, [], 1); obj_max = max(Y_valid, [], 1);
                        obj_range = obj_max - obj_min;
                        obj_range(obj_range < eps) = 1;
                        Y_valid_norm = (Y_valid - obj_min) ./ obj_range;
                        new_y_norm = (new_y - obj_min) ./ obj_range;
                        target_dist = min(pdist2(Y_valid_norm, new_y_norm));
                    end
                    if target_dist < 0.01
                        dup_skip_count = dup_skip_count + 1;                        
                    else                        
                        valid_count_gen = valid_count_gen + 1;
                        gen_X_valid = [gen_X_valid; current_point];
                        gen_Y_valid = [gen_Y_valid; L_D, vol_frac];
                        fprintf('  成功 %d/%d: L/D=%.3f, V=%.3f, NHVI=%.4f\n', valid_count_gen, eval_per_gen, L_D, vol_frac, current_ehvi);
                    end                    
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
        if ~isempty(gen_X_valid)
            [~, current_pf] = WaveRiderUtils.extract_pareto_front(Y_valid, false);
            add_count = size(gen_X_valid,1);
        else
            add_count = 0;
            fprintf('警告：本代Phase 1未获新有效点\n');
        end        
        % 收敛监测
        hv_previous = hv_current;
        hv_current = WaveRiderUtils.calculateHypervolume(current_pf, ref_point_fixed);
        hv_improvement_1 = (hv_current - hv_previous) / abs(hv_previous);
        spacing_current = WaveRiderUtils.calculateSpacing(current_pf); 
        fprintf('HV: %.4f (+%.4f%%) | Spacing指标: %.4f (越小越均匀)\n', hv_current, hv_improvement_1*100, spacing_current);       
        % Phase 2        
        fprintf('Phase 2——Pareto前缘均布插值阶段\n');
        [front_idx, current_pf] = WaveRiderUtils.extract_pareto_front(Y_valid, false);
        pf_design = X_valid(front_idx, :);
        % 按升阻比排序前沿和设计变量（保证弧长计算与设计变量对应）
        [sorted_pf, sort_idx] = sortrows(current_pf, 1);
        sorted_design = pf_design(sort_idx, :);        
        % 识别前沿上的大间隙，返回需要插值的间隙信息
        n_interp_points = max(20, ceil(size(sorted_pf, 1) * 0.6));
        % n_interp_points = eval_per_gen;
        interp_info = WaveRiderUtils.getInterpGaps(sorted_pf, n_interp_points);
        if ~isempty(interp_info)
            fprintf('  前沿点数: %d, 插值间隙数: %d, 插值点数: %d\n', ...
                size(sorted_pf,1), size(unique(interp_info(:,1)),1), size(interp_info,1));
            % 设计变量空间插值并真实评估
            interp_success_count = 0;
            interp_X_all = []; interp_Y_feas = [];
            interp_X_valid = []; interp_Y_valid = [];
            for i = 1:size(interp_info, 1)
                seg = interp_info(i, 1);   % 间隙起始Pareto点的索引
                
                % 相邻 Pareto 点的设计变量
                A_design = sorted_design(seg, :);
                B_design = sorted_design(seg + 1, :);
                
                % 目标空间间隙中点
                target_mid = (sorted_pf(seg, :) + sorted_pf(seg + 1, :)) / 2;
                
                % 代理模型引导的线搜索: 在线段上找到预测最接近间隙中点的 t 值
                t_candidates = linspace(0, 1, 21);
                best_t = 0;
                best_dist = inf;
                for t = t_candidates
                    x_candidate = (1 - t) * A_design + t * B_design;
                    pred_target = [predict(krig_LD, x_candidate), predict(krig_vol, x_candidate)];
                    dist_to_mid = norm(pred_target - target_mid);
                    if dist_to_mid < best_dist
                        best_dist = dist_to_mid;
                        best_t = t;
                    end
                end
                
                x_interp = (1 - best_t) * A_design + best_t * B_design;
                x_interp = max(lb, min(ub, x_interp));  % 确保在边界内                
                % 可行性预筛选
                [~, feas_score] = predict(feas_qda, x_interp);
                feas_prob = feas_score(:, 2);
                if feas_prob < 0.1
                    skip_count = skip_count + 1;
                    fprintf('  插值点%d: 可行性低(%.2f)，跳过\n', i, feas_prob);
                    continue;
                end
                % 检查是否与已评估点过于接近（设计空间）
                min_dist = min(pdist2(X_all, x_interp));
                design_range = ub - lb;
                rel_dist = min_dist / norm(design_range);
                if rel_dist < 0.01
                    dup_skip_count = dup_skip_count + 1;
                    continue;
                end                    
                % 真实评估
                [success, L_D, vol_frac] = evaluatePoint(x_interp);
                interp_X_all = [interp_X_all; x_interp]; interp_Y_feas = [interp_Y_feas; success];
                X_all = [X_all; x_interp]; Y_feas = [Y_feas; success];                
                if success
                    new_y = [L_D, vol_frac];
                    % 利用目标空间去重
                    if isempty(Y_valid)
                        target_dist = inf;
                    else  
                        obj_min = min(Y_valid, [], 1); obj_max = max(Y_valid, [], 1);
                        obj_range = obj_max - obj_min;
                        obj_range(obj_range < eps) = 1;
                        Y_valid_norm = (Y_valid - obj_min) ./ obj_range;
                        new_y_norm = (new_y - obj_min) ./ obj_range;
                        target_dist = min(pdist2(Y_valid_norm, new_y_norm));
                    end
                    if target_dist<0.01
                        dup_skip_count = dup_skip_count + 1;
                        fprintf('  插值点%d成功但目标空间过近(%.2f)，不加入有效池\n', i, target_dist);
                    else          
                        % Check if dominated by current Pareto front
                        temp_front = [current_pf; new_y];
                        [~, new_front] = WaveRiderUtils.extract_pareto_front(temp_front, false);
                        new_y_is_pareto = any(all(abs(new_front - new_y) < 1e-10, 2));
                        if new_y_is_pareto
                            interp_X_valid = [interp_X_valid; x_interp];
                            interp_Y_valid = [interp_Y_valid; L_D, vol_frac];
                            X_valid = [X_valid; x_interp]; Y_valid = [Y_valid; L_D, vol_frac];
                            interp_success_count = interp_success_count + 1;
                            fprintf('  插值点%d成功: L/D=%.3f, Vol=%.3f (设计变量插值比例=%.2f)\n', ...
                                i, L_D, vol_frac, best_t);
                        else
                            dup_skip_count = dup_skip_count + 1;
                            fprintf('  插值点%d成功但被现有前缘支配，不加入有效池\n', i);
                        end
                    end
                else
                    fprintf('  插值点%d评估失败\n', i);
                end
            end
            fprintf('均布插值完成: 成功 %d/%d\n', interp_success_count, size(interp_info, 1));
            if interp_success_count > 0              
                add_count = add_count + size(interp_X_valid,1);
                [~, current_pf] = WaveRiderUtils.extract_pareto_front(Y_valid, false);
                hv_previous = hv_current;
                hv_current = WaveRiderUtils.calculateHypervolume(current_pf, ref_point_fixed);                
                hv_improvement_2 = (hv_current - hv_previous) / abs(hv_previous);
                spacing_current = WaveRiderUtils.calculateSpacing(current_pf);
                fprintf('HV: %.4f (+%.4f%%) | Spacing指标: %.4f (越小越均匀)\n', hv_current, hv_improvement_2*100, spacing_current);
            else
                fprintf('  均布插值本轮无成功点\n');
            end
            % feas_qda = fitcdiscr(X_all, Y_feas, ...
            %     'KernelFunction', 'squaredexponential', 'Sigma', 0.1, 'Standardize', true);
        end
        % Re-train feasibility QDA model
        if mod(gen, 2) == 0
            fitc_count = fitc_count + 1;
            fprintf('Retraining feasibility QDA model...\n');
            feas_qda = fitcdiscr(X_all, Y_feas, ...
                'DiscrimType', 'quadratic');
        end
        if ~isempty(gen_X_valid) || ~isempty(interp_X_valid)
            opts_perf = WaveRiderUtils.getGPROpts(size(X_valid, 1));
            krig_LD = fitrgp(X_valid, Y_valid(:,1), opts_perf{:});
            krig_vol = fitrgp(X_valid, Y_valid(:,2), opts_perf{:});            
        end
        hv_history = [hv_history; hv_current];
        spacing_history = [spacing_history; spacing_current];          
        hv_previous = hv_history(end-1);
        hv_improvement = (hv_current - hv_previous) / abs(hv_previous);  
        if gen > 1 && abs(hv_improvement) < hv_threshold
            stall_count = stall_count + 1;
            fprintf('停滞 %d/%d (改进: %.4f%%)\n', stall_count, max_stall_generations, hv_improvement*100);
        else
            stall_count = 0;
        end        
        gen_time = toc; total_time = total_time + gen_time;
        fprintf('HV: %.4f (+%.4f%%) | 分类跳过: %d | 距离跳过: %d | 新增有效: %d | 总有效: %d | 总点数: %d | 耗时: %.1fs (累计 %.1fmin)\n', ...
            hv_current, hv_improvement*100, skip_count, dup_skip_count, add_count, size(X_valid,1), size(X_all,1), gen_time, total_time/60);
        
        % 每代保存
        save(sprintf('wave_rider_optimization_gen%d.mat', gen), 'X_valid', 'Y_valid', 'current_pf', 'hv_history', 'spacing_history');
        
        if size(X_all, 1) >= max_estimation_point
            fprintf('达到最大评估点数(%d)，停止\n', max_estimation_point);
            break;
        end
    end
    
    % 6. 提取最终结果
    [front_idx, pareto_front] = WaveRiderUtils.extract_pareto_front(Y_valid, false);
    design_vars = X_valid(front_idx, :);
    
    % 7. 可视化
    % Pareto前沿图
    [sorted_front, sort_idx] = sortrows(pareto_front, 1);
    sorted_design = design_vars(sort_idx, :);    
    figure;
    scatter(Y_valid(:,1), Y_valid(:,2), 20, 'filled', 'MarkerFaceAlpha', 0.3); hold on;
    plot(sorted_front(:,1), sorted_front(:,2), 'LineWidth', 2); 
    scatter(sorted_front(:,1), sorted_front(:,2), 60, 'r', 'filled');
    xlabel('升阻比 (L/D)'); ylabel('容积率 (Vol. Fraction)');
    title('乘波体Pareto前沿'); legend('所有有效点', 'Pareto前沿', 'Pareto最优解'); grid on;
    
    % 超体积收敛曲线
    figure; plot(0:length(hv_history)-1, hv_history, 'bo-', 'LineWidth', 1.5);
    xlabel('代数'); ylabel('超体积'); title('超体积进展'); grid on;

    % 前缘空间收敛曲线
    figure; plot(0:length(spacing_history)-1, spacing_history, 'bo-', 'LineWidth', 1.5);
    xlabel('代数'); ylabel('前缘空间'); title('前缘空间进展'); grid on;
    
    % 均匀性诊断图
    figure;
    [sorted_final, ~] = sortrows(pareto_front, 1);
    pf_min_f = min(sorted_final, [], 1);
    pf_max_f = max(sorted_final, [], 1);
    pf_range_f = pf_max_f - pf_min_f;
    pf_range_f(pf_range_f < eps) = 1;
    pf_norm_f = (sorted_final - pf_min_f) / pf_range_f;
    gaps_final = zeros(size(pf_norm_f, 1)-1, 1);
    for i = 1:length(gaps_final)
        gaps_final(i) = norm(pf_norm_f(i+1,:) - pf_norm_f(i,:));
    end
    bar(gaps_final);
    xlabel('间隙编号'); ylabel('归一化弧长间隙');
    title('Pareto前沿间隙分布（越均匀越好）');
    yline(mean(gaps_final), 'r--', sprintf('均值=%.3f', mean(gaps_final)));
    
    % 设计变量分布图
    % figure;
    % for v = 1:n_vars
    %     subplot(2, 3, v);
    %     plot(sorted_design(:, v), sorted_front(:, 1), 'b.-');
    %     xlabel(['x_', num2str(v)]); ylabel('L/D');
    %     title(['设计变量', num2str(v), ' vs L/D']);
    %     grid on;
    % end

    % 8. 保存
    save('wave_rider_optimization_final.mat', 'pareto_front', 'design_vars', 'X_valid', 'Y_valid', 'hv_history', 'spacing_history');
    fprintf('\n优化完成! 代数: %d, 总点数: %d, 用时: %.1fmin\n', gen, size(X_all,1), total_time/60);
end

%% ==================== 辅助函数 ====================
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
    d1 = x(5);
    AOA = -5; 
    a_xishu_1 = x(6);
        
    try
        % 尝试执行计算
        [L_D, vol_frac] = waveRiderEval(Ma, angle_shock_a1, angle_shock_a2, angle_shock_b1, angle_shock_b2, c1, c2, d1, AOA, a_xishu_1);

        % 检查结果是否有效
        if ~isnan(L_D) && ~isnan(vol_frac)
            success = true;
        else
            L_D = NaN; vol_frac = NaN; % 确保失败时返回NaN
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

