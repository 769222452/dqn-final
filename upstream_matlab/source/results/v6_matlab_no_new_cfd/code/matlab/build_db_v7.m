function build_db_v7(part, nparts)
% Small database: 13 Mach x 7 alpha x (9 geometry nodes of each branch) pressure (P2) + base-sensitivity terms, and nominal friction at 5 altitudes.
here = fileparts(mfilename('fullpath')); addpath(here);
R = [fileparts(fileparts(here)) filesep]; Mt = readtable([R 'GEOMETRY_METADATA.csv']); At = readtable([R 'work\atm_levels.csv']);
strict_nodes = [0 .125 .25 .375 .5 .625 .75 .875 1];
keep = strcmp(Mt.branch,'AREA_MORPH_EXPLORATORY') | (strcmp(Mt.branch,'STRICT_ORIGINAL') & ismember(round(Mt.value*1e4)/1e4, strict_nodes));
Mt = Mt(keep,:); Mt = Mt(mod((1:height(Mt))-1,nparts)==part,:);
Mach = linspace(3.0,8.5,13); Alpha = linspace(-2,12,7); nM = 13; nA = 7; nH = height(At);
refg = struct('area_m2',6.931920786884801,'length_m',5,'moment_origin_m',[2.5 0 -0.9481139754931033]);
for g = 1:height(Mt)
    S = load(v7_project_path(Mt.mat_path{g})); geo.lo = cat(3,S.wav_x,S.wav_y,S.wav_z); geo.up = cat(3,S.wav_up_x,S.wav_up_y,S.wav_up_z);
    CLp = nan(nM,nA); CDp = CLp; Cm = CLp; valid = false(nM,nA); det = CLp; CLb5 = CLp; CDb5 = CLp; CLb3 = CLp; CDb3 = CLp; CLf = nan(nM,nA,nH); CDf = CLf; Remax = CLp; Me_min = CLp;
    t0 = tic;
    for i = 1:nM, for j = 1:nA
        fl = struct('Mach',Mach(i),'alpha_rad',Alpha(j)*pi/180);
        o = offdesign_aero_v7(geo,fl,refg,struct('mode','P2','friction','none','base_pressure_ratio',0.5));
        valid(i,j) = o.valid; det(i,j) = o.detached_area_fraction; Cm(i,j) = o.Cm; CLp(i,j) = o.CL_pressure; CDp(i,j) = o.CD_pressure; CLb5(i,j) = o.CL_base; CDb5(i,j) = o.CD_base;
        o3 = offdesign_aero_v7(geo,fl,refg,struct('mode','P2','friction','none','base_pressure_ratio',0.3)); CLb3(i,j) = o3.CL_base; CDb3(i,j) = o3.CD_base;
        if o.valid, Me_min(i,j) = o.Me_min;
            for h = 1:nH
                fl.T_inf_K = At.T_K(h); fl.rho_inf_kgm3 = At.rho(h);
                of = offdesign_aero_v7(geo,fl,refg,struct('mode','P2','friction','turbulent','Tw_K',1000)); CLf(i,j,h) = of.CL_friction; CDf(i,j,h) = of.CD_friction; if h==1, Remax(i,j) = of.Re_x_max; end
            end
        end
    end, end
    branch = Mt.branch{g}; gid = Mt.geometry_id{g}; param = Mt.value(g); ghash = Mt.geometry_hash{g}; h_m = At.h_m'; wet = Mt.wetted_area(g); basearea = Mt.base_area(g); planf = Mt.planform_area(g);
    save([R 'db\g_' gid '.mat'],'branch','gid','param','ghash','Mach','Alpha','h_m','CLp','CDp','Cm','valid','det','CLb5','CDb5','CLb3','CDb3','CLf','CDf','Remax','Me_min','wet','basearea','planf','-v7');
    fprintf('%s done %.0f s valid %d/%d\n',gid,toc(t0),sum(valid(:)),numel(valid));
end
end

