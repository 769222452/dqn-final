function sens_v7()
% Independent control authority at the representative states: CL/CD/Cm derivatives wrt alpha and the geometry parameter, R_D, R_m, and the actual finite-change check.
here = fileparts(mfilename('fullpath')); addpath(here);
R = [fileparts(fileparts(here)) filesep]; Mt = readtable([R 'GEOMETRY_METADATA.csv']); St = readtable([R 'work\sens_states.csv']);
refg = struct('area_m2',6.931920786884801,'length_m',5,'moment_origin_m',[2.5 0 -0.9481139754931033]);
strict_nodes = [0 .125 .25 .375 .5 .625 .75 .875 1]; rows = {};
for br = {'STRICT_ORIGINAL','AREA_MORPH_EXPLORATORY'}
    B = Mt(strcmp(Mt.branch,br{1}),:); if strcmp(br{1},'STRICT_ORIGINAL'), B = B(ismember(round(B.value*1e4)/1e4,strict_nodes),:); end
    [~,ord] = sort(B.value); B = B(ord,:); n = height(B); G = cell(n,1);
    for k = 1:n, S = load(v7_project_path(B.mat_path{k})); G{k}.lo = cat(3,S.wav_x,S.wav_y,S.wav_z); G{k}.up = cat(3,S.wav_up_x,S.wav_up_y,S.wav_up_z); end
    for s = 1:height(St)
        ev = @(k,a,comp) evalg(G{k},St.Mach(s),a,St.T_K(s),St.rho(s),refg,comp);
        alp = nan(n,1); CD_iso = nan(n,1); CDp_iso = nan(n,1); CDf_iso = nan(n,1);
        for k = 1:n
            a0 = 4; a1 = 6; f0 = ev(k,a0,'CL')-St.CL_req(s); f1 = ev(k,a1,'CL')-St.CL_req(s);
            for it = 1:12, a2 = a1-f1*(a1-a0)/(f1-f0); a0 = a1; f0 = f1; a1 = a2; f1 = ev(k,a1,'CL')-St.CL_req(s); if abs(f1)<1e-10, break; end, end
            alp(k) = a1; [~,CD_iso(k),~,CDp_iso(k),CDf_iso(k)] = ev(k,a1,'all');
        end
        for k = 1:n
            da = 0.25;
            [CLa_p,CDa_p,Cma_p] = dalpha(G{k},St,s,alp(k),da,refg);
            kp = min(k+1,n); km = max(k-1,1); dp = (B.value(kp)-B.value(km)); 
            [cl_p,cd_p,cm_p] = at(G{kp},St,s,alp(k),refg); [cl_m,cd_m,cm_m] = at(G{km},St,s,alp(k),refg);
            CL_eta = (cl_p-cl_m)/dp; CD_eta = (cd_p-cd_m)/dp; Cm_eta = (cm_p-cm_m)/dp;
            RD = CD_eta-CDa_p(1)/CLa_p(1)*CL_eta; Rm = Cm_eta-Cma_p(1)/CLa_p(1)*CL_eta;
            RDact = (CD_iso(kp)-CD_iso(km))/dp;
            rows(end+1,:) = {br{1},St.state{s},B.geometry_id{k},B.value(k),alp(k),CD_iso(k),CDp_iso(k),CDf_iso(k),CLa_p(1),CDa_p(1),Cma_p(1),CL_eta,CD_eta,Cm_eta,RD,Rm,RDact,CLa_p(2),CDa_p(2)}; %#ok<AGROW>
        end
    end
end
T = cell2table(rows,'VariableNames',{'branch','state','geometry_id','param','alpha_iso_deg','CD_total_iso','CD_pressure_iso','CD_friction_iso','CL_alpha_total','CD_alpha_total','Cm_alpha_pressure','CL_eta','CD_eta','Cm_eta','R_D','R_m','R_D_actual_finite_change','CL_alpha_pressure','CD_alpha_pressure'});
writetable(T,[R 'tests\control_authority_sensitivity.csv']);
end
function varargout = evalg(g,M,a,T,rho,refg,comp)
o = offdesign_aero_v7(g,struct('Mach',M,'alpha_rad',a*pi/180,'T_inf_K',T,'rho_inf_kgm3',rho),refg,struct('mode','P2','friction','turbulent','Tw_K',1000));
if ~o.valid, error('invalid state'); end
if strcmp(comp,'CL'), varargout{1} = o.CL; else, varargout = {o.CL,o.CD_total,o.Cm,o.CD_pressure,o.CD_friction}; end
end
function [cl,cd,cm] = at(g,St,s,a,refg)
o = offdesign_aero_v7(g,struct('Mach',St.Mach(s),'alpha_rad',a*pi/180,'T_inf_K',St.T_K(s),'rho_inf_kgm3',St.rho(s)),refg,struct('mode','P2','friction','turbulent','Tw_K',1000)); cl = o.CL; cd = o.CD_total; cm = o.Cm;
end
function [CLa,CDa,Cma] = dalpha(g,St,s,a,da,refg)
[cp,dp_,mp] = at(g,St,s,a+da,refg); [cm_,dm_,mm_] = at(g,St,s,a-da,refg); d = 2*da*pi/180;
CLa = [(cp-cm_)/d 0]; CDa = [(dp_-dm_)/d 0]; Cma = (mp-mm_)/d;
o1 = offdesign_aero_v7(g,struct('Mach',St.Mach(s),'alpha_rad',(a+da)*pi/180),refg,struct('mode','P2')); o2 = offdesign_aero_v7(g,struct('Mach',St.Mach(s),'alpha_rad',(a-da)*pi/180),refg,struct('mode','P2'));
CLa(2) = (o1.CL-o2.CL)/d; CDa(2) = (o1.CD_total-o2.CD_total)/d;
end

