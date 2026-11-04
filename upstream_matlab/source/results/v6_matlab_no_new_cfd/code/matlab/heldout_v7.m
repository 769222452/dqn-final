function heldout_v7()
here = fileparts(mfilename('fullpath')); addpath(here); R = [fileparts(fileparts(here)) filesep];
T = readtable([R 'work\heldout_inputs.csv']); refg = struct('area_m2',6.931920786884801,'length_m',5,'moment_origin_m',[2.5 0 -0.9481139754931033]); n = height(T); out = nan(n,5);
for k = 1:n
    S = load(v7_project_path(T.mat_path{k})); g.lo = cat(3,S.wav_x,S.wav_y,S.wav_z); g.up = cat(3,S.wav_up_x,S.wav_up_y,S.wav_up_z);
    o = offdesign_aero_v7(g,struct('Mach',T.Mach(k),'alpha_rad',T.alpha_deg(k)*pi/180,'T_inf_K',T.T_K(k),'rho_inf_kgm3',T.rho(k)),refg,struct('mode','P2','friction','turbulent','Tw_K',1000,'base_pressure_ratio',0.5));
    out(k,:) = [o.CL_pressure o.CD_pressure o.CL_friction o.CD_friction o.CD_base];
end
writetable([T(:,{'branch','param','xi','Mach','alpha_deg','h_m'}) array2table(out,'VariableNames',{'CL_p','CD_p','CL_f','CD_f','CD_base05'})],[R 'tests\heldout_direct_matlab.csv']);
end
