function replay_cfd()
here = fileparts(mfilename('fullpath')); study = fileparts(fileparts(here));
project_root = fileparts(fileparts(study));
addpath(here); addpath(fullfile(project_root,'matlab','active'));
W = [fullfile(study,'work') filesep]; refg = struct('area_m2',6.931920786884801,'length_m',5,'moment_origin_m',[2.5 0 0]);
for pass = 1:2
    if pass==1, T = readtable([W 'cfd_replay_inputs.csv']); else, T = readtable([W 'cfd_sweep_inputs.csv']); end
    n = height(T); out = nan(n,6); cacheg = containers.Map();
    for k = 1:n
        mp = v7_project_path(T.mat{k}); if ~isKey(cacheg,mp), S = load(mp); g.lo = cat(3,S.wav_x,S.wav_y,S.wav_z); g.up = cat(3,S.wav_up_x,S.wav_up_y,S.wav_up_z); cacheg(mp) = g; end
        g = cacheg(mp); fl = struct('Mach',T.Mach(k),'alpha_rad',T.alpha_deg(k)*pi/180);
        for m = 1:3
            mdl = {'P0','P1','P2'}; md = mdl{m}; o = offdesign_aero_v7(g,fl,refg,struct('mode',md)); out(k,2*m-1) = o.CL; out(k,2*m) = o.CD_pressure;
        end
    end
    R = array2table(out,'VariableNames',{'CL_P0','CD_P0','CL_P1','CD_P1','CL_P2','CD_P2'});
    writetable([T(:,{'geom','Mach','alpha_deg'}) R],[W sprintf('cfd_replay_out_%d.csv',pass)]);
end
end

