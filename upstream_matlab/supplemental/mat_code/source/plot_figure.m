clc
clear
load wave_rider_optimization_gen130.mat
% 画Pareto前缘
% [sorted_front, sort_idx] = sortrows(pareto_front, 1);
% sorted_design = design_vars(sort_idx, :);   
% figure;
% plot(sorted_front(:,1), sorted_front(:,2), 'LineWidth', 2);
% hold on;
% scatter(Y_valid(:,1), Y_valid(:,2), 20, 'cyan', 'filled', 'MarkerFaceAlpha', 0.3);
% hold on
% scatter(sorted_front(:,1), sorted_front(:,2), 60, 'r', 'filled');
% hold on
% xlabel('升阻比 (L/D)');
% ylabel('容积率 (Vol. Fraction)');
% title('乘波体设计Pareto前沿');
% legend('Pareto 前沿', '所有评估点', 'Pareto 最优解');
% grid on;
% hold off;
% 画收敛曲线
figure;
plot(0:length(hv_history)-1, hv_history, 'bo-', 'LineWidth', 1.5);
xlabel('代数');
ylabel('超体积');
title('优化过程超体积进展');
grid on;