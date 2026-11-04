classdef WaveRiderUtils
    %%%%%%%%%%%%%%%%乘波体优化辅助工具集%%%%%%%%%%%%%
    % 用法：WaveRiderUtils.方法名(参数)
    % 类似Python: module.function(args)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    methods(Static)

        function spacing_metric = calculateSpacing(pareto_front)
            % 计算Pareto前沿的均匀性指标
            % spacing = std(gaps) / mean(gaps)
            % 值越小 → 前沿越均匀；值=0 → 完美均匀            
            if size(pareto_front, 1) < 3
                spacing_metric = NaN;
                return;
            end            
            [sorted_pf, ~] = sortrows(pareto_front, 1);            
            % 归一化           
            pf_min = min(sorted_pf, [], 1);
            pf_max = max(sorted_pf, [], 1);
            pf_range = pf_max - pf_min;
            pf_range(pf_range < eps) = 1;
            pf_norm = (sorted_pf - pf_min) / pf_range;            
            % 弧长间隙
            gaps = zeros(size(pf_norm, 1) - 1, 1);
            for i = 1:length(gaps)
                gaps(i) = norm(pf_norm(i+1,:) - pf_norm(i,:));
            end            
            spacing_metric = std(gaps) / mean(gaps);
        end

        function interp_info = getInterpGaps(pareto_front, n_interp)
            % 识别Pareto前沿上的大间隙，返回需要插值的间隙信息
            % 输入：
            %   sorted_pf : M×2 按升阻比排序的Pareto前沿目标值
            %   n_interp  : 期望插值点数上限
            % 输出：
            %   interp_info : K×3 矩阵 [间隙索引, 插值比例, 间隙弧长]
            %                间隙索引: 相邻Pareto点的起始索引 (1~M-1)
            %                插值比例: 0~1，在设计变量空间中的插值位置
            %                间隙弧长: 该间隙在归一化目标空间中的弧长            
            if size(pareto_front, 1) < 2
                interp_info = [];
                return;
            end
            [sorted_pf, ~] = sortrows(pareto_front, 1);  
            % 归一化目标空间
            pf_min = min(sorted_pf, [], 1);
            pf_max = max(sorted_pf, [], 1);
            pf_range = pf_max - pf_min;
            pf_range(pf_range < eps) = 1;
            pf_norm = (sorted_pf - pf_min) ./ pf_range;            
            % 计算累积弧长
            n_pf = size(pf_norm, 1);
            arc_length = zeros(n_pf, 1);
            for i = 2:n_pf
                arc_length(i) = arc_length(i-1) + norm(pf_norm(i,:) - pf_norm(i-1,:));
            end
            total_length = arc_length(end);
            if total_length < eps
                interp_info = [];
                return;
            end            
            % 计算各间隙弧长
            gaps = diff(arc_length);
            avg_gap = mean(gaps);
            if avg_gap < eps
                interp_info = [];
                return;
            end            
            % 只在大间隙中插值（间隙 > 30%平均间隙才插值）
            min_gap_ratio = 1;            
            interp_info = [];
            gap_weight = [];            
            for seg = 1:length(gaps)
                if gaps(seg) < min_gap_ratio * avg_gap
                    continue;
                end                
                % 每个间隙的插值点数与间隙大小成正比
                % n_pts = max(1, round(gaps(seg) / avg_gap)*2);
                % fracs = linspace(0.25, 0.75, n_pts);
                % for f = fracs
                %     interp_info = [interp_info; seg, f, gaps(seg)];
                %     gap_weight = [gap_weight; gaps(seg)];
                % end
                interp_info = [interp_info; seg, gaps(seg)];
                gap_weight = [gap_weight; gaps(seg)];
            end            
            if isempty(interp_info)
                return;
            end            
            % 按间隙大小排序优先保留（大间隙中的点更有价值）
            [~, sort_idx] = sort(gap_weight, 'descend');
            n_keep = min(n_interp, size(interp_info, 1));
            % n_keep = min(20, size(interp_info, 1));
            interp_info = interp_info(sort_idx(1:n_keep), :);
        end

        function sparsity = calculateSparsity(mu, pareto_front)
            % 计算候选点在目标空间中的稀疏度指标
            % 稀疏度 = 候选点到最近Pareto前沿点的归一化欧氏距离
            %
            % 原理：
            %   - 距离大 → 该区域Pareto点稀疏 → 加入后均匀性改善大 → 应优先选择
            %   - 距离小 → 该区域Pareto点密集 → 加入后均匀性改善小 → 应降低优先级
            %
            % 输入：
            %   mu          : N×2 矩阵，候选点在目标空间的预测均值 [L/D, vol_frac]
            %   pareto_front: M×2 矩阵，当前Pareto前沿的目标值
            %
            % 输出：
            %   sparsity    : N×1 向量，每个候选点的稀疏度值            
            n_candidates = size(mu, 1);
            sparsity = zeros(n_candidates, 1);            
            if isempty(pareto_front)
                sparsity = ones(n_candidates, 1);
                return;
            end            
            % 归一化目标空间，使两个目标量级一致
            % L/D量级约3~6，vol_frac量级约0.1~0.5，差异约10倍
            pf_min = min(pareto_front, [], 1);
            pf_max = max(pareto_front, [], 1);
            pf_range = pf_max - pf_min;
            pf_range(pf_range < eps) = 1;            
            pf_norm = (pareto_front - pf_min) / pf_range;
            mu_norm = (mu - pf_min) / pf_range;            
            % 使用pdist2计算所有候选点到所有Pareto点的距离矩阵
            dist_matrix = pdist2(pf_norm, mu_norm);            
            % 稀疏度 = 到最近Pareto点的距离
            sparsity = min(dist_matrix, [], 1)';
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

        function opts = getGPROpts(n_samples)
            fit_method = 'exact';
            if n_samples > 500, fit_method = 'sr'; end
            opts = {'KernelFunction','matern52', ...
                    'FitMethod',fit_method, ...
                    'PredictMethod',fit_method, ...
                    'Standardize', true, ...       % 标准化输入，消除量级差异
                    'Sigma', 0.01};               % 固定小噪声，保证正定性
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
            % 预置parfor内随机种子，保证normrnd结果可重复

            % 预计算当前Pareto前沿的超体积作为基准
            hv_ref = WaveRiderUtils.calculateHypervolume(pareto_front, ref_point);            
            % parfor i = 1:n
            for i = 1:n
                % 跳过明显劣质的候选点 (此预筛选逻辑在NHVI中可选，因为NHVI允许负值，
                % 但为了计算效率，保留此筛选可以跳过极差点的计算)
                if mu(i, 1) < min_LD && mu(i, 2) < min_Vol
                    nhvi(i) = 0; 
                    continue;
                end                
                % rng(11); % 每次迭代独立种子，保证parfor内normrnd结果可重复
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
                        
                        hv_new = WaveRiderUtils.calculateHypervolume(temp_front, ref_point);
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
                        hv_new = WaveRiderUtils.calculateHypervolume(temp_front, ref_point);                        
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

    end
end

