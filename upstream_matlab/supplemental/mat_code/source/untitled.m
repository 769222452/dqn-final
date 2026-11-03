clc
clear
format short
load wave_rider_optimization_final
[sorted_front, sort_idx] = sortrows(pareto_front, 1);
sorted_design = design_vars(sort_idx, :);   
if isempty(gcp('nocreate'))
    parpool('local',20);
end
Ma=15;
i = 45;
angle_shock_a1=sorted_design(i,1); 
angle_shock_a2=sorted_design(i,2); 
angle_shock_b1=sorted_design(i,3); 
angle_shock_b2=sorted_design(i,4); 
c1=1.0331; c2=1.0331; 
d1=sorted_design(i,5); 
AOA=-4; 
a_xishu_1=sorted_design(i,6);

[wav_x_new,wav_y_new,wav_z_new,LD,LW,H,Vol,Vol_eff,Time]=generatewaverider(Ma, angle_shock_a1, angle_shock_a2, angle_shock_b1, angle_shock_b2, c1, c2, d1, AOA, a_xishu_1);

fid = fopen('waverider_M15_sym.dat','wt');
for j = 1:length(wav_x_new(1,:))
    fprintf(fid,'ROW\n');
    for i = 1:length(wav_x_new(:,1))
        fprintf(fid,'%e\t',wav_x_new(i,j)*1000);
        fprintf(fid,'%e\t',wav_y_new(i,j)*1000);
        fprintf(fid,'%e\n',-wav_z_new(i,j)*1000);
    end  
end