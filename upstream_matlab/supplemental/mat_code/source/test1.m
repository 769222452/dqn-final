clc
clear
format short
if isempty(gcp('nocreate'))
    parpool('local',20);
end
Ma = 15; 
beta = 5.0910; 
Ri = 0.2693; 
c0 = 1.0976; 
c1 = 0.0217; 
d1 = 0.1;
AOA = -4; 
a_xishu_1 = 0.9791;
% [result.L_D, result.vol_frac] = waveRiderEval(Ma, beta, Ri, c0, c1, d1, AOA, a_xishu_1);
[wav_x_new,wav_y_new,wav_z_new,L_D,L_W,Xcp,Vol,Vol_eff,Time_LTOCs] = generatewaverider(Ma, beta, Ri, c0, c1, d1, AOA, a_xishu_1);