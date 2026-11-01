function ablation_v7()
% Equal-lift CD of selected geometry nodes under frozen friction / base scenarios (component ablation). No parameter is tuned.
here = fileparts(mfilename('fullpath')); addpath(here);
R = [fileparts(fileparts(here)) filesep]; Mt = readtable([R 'GEOMETRY_METADATA.csv']); St = readtable([R 'work\sens_states.csv']);
refg = struct('area_m2',6.931920786884801,'length_m',5,'moment_origin_m',[2.5 0 -0.9481139754931033]);
sel = {'STRICT_xi0.0000','STRICT_xi0.5000','STRICT_xi1.0000','AREA_lam0.900','AREA_lam1.000','AREA_lam1.100'};
sc = {'pressure_only',struct('friction','none'); 'pressure+turb_Tw1000(nominal)',struct('friction','turbulent','Tw_K',1000); 'turb_Tw300',struct('friction','turbulent','Tw_K',300); 'turb_Tw1500',struct('friction','turbulent','Tw_K',1500);
      'transition_Retr5e6_Tw1000',struct('friction','transition','Tw_K',1000,'Re_tr',5e6); 'laminar_Tw1000',struct('friction','laminar','Tw_K',1000); 'nominal+base_pb0.5',struct('friction','turbulent','Tw_K',1000,'base_pressure_ratio',0.5); 'nominal+base_pb0.3',struct('friction','turbulent','Tw_K',1000,'base_pressure_ratio',0.3)};
rows = {};
for g = 1:numel(sel)
    r = Mt(strcmp(Mt.geometry_id,sel{g}),:); S = load(v7_project_path(r.mat_path{1})); geo.lo = cat(3,S.wav_x,S.wav_y,S.wav_z); geo.up = cat(3,S.wav_up_x,S.wav_up_y,S.wav_up_z);
    for s = 1:height(St), for k = 1:size(sc,1)
        op = struct('mode','P2'); f = fieldnames(sc{k,2}); for q = 1:numel(f), op.(f{q}) = sc{k,2}.(f{q}); end
        ev = @(a) offdesign_aero_v7(geo,struct('Mach',St.Mach(s),'alpha_rad',a*pi/180,'T_inf_K',St.T_K(s),'rho_inf_kgm3',St.rho(s)),refg,op);
        a0 = 4; a1 = 6; f0 = ev(a0).CL-St.CL_req(s); f1 = ev(a1).CL-St.CL_req(s);
        for it = 1:12, a2 = a1-f1*(a1-a0)/(f1-f0); a0 = a1; f0 = f1; a1 = a2; o = ev(a1); f1 = o.CL-St.CL_req(s); if abs(f1)<1e-10, break; end, end
        rows(end+1,:) = {sel{g},St.state{s},sc{k,1},a1,o.CD_total,o.CD_pressure,o.CD_friction,o.CD_base}; %#ok<AGROW>
    end, end
end
T = cell2table(rows,'VariableNames',{'geometry_id','state','scenario','alpha_iso_deg','CD_total','CD_pressure','CD_friction','CD_base'});
writetable(T,[R 'tests\ablation_equal_lift.csv']);
end
