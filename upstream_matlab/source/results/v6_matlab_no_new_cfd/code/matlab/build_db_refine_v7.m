function build_db_refine_v7(part, nparts)
% Pressure/base/moment refinement of the database grid (23 Mach x 15 alpha) after the 13x7 interpolation test showed ~1 % CD error between nodes.
here = fileparts(mfilename('fullpath')); addpath(here);
R = [fileparts(fileparts(here)) filesep]; Mt = readtable([R 'GEOMETRY_METADATA.csv']);
strict_nodes = [0 .125 .25 .375 .5 .625 .75 .875 1];
keep = strcmp(Mt.branch,'AREA_MORPH_EXPLORATORY') | (strcmp(Mt.branch,'STRICT_ORIGINAL') & ismember(round(Mt.value*1e4)/1e4, strict_nodes));
Mt = Mt(keep,:); Mt = Mt(mod((1:height(Mt))-1,nparts)==part,:);
Mach = linspace(3.0,8.5,23); Alpha = linspace(-2,12,15); nM = numel(Mach); nA = numel(Alpha);
refg = struct('area_m2',6.931920786884801,'length_m',5,'moment_origin_m',[2.5 0 -0.9481139754931033]);
for g = 1:height(Mt)
    S = load(v7_project_path(Mt.mat_path{g})); geo.lo = cat(3,S.wav_x,S.wav_y,S.wav_z); geo.up = cat(3,S.wav_up_x,S.wav_up_y,S.wav_up_z);
    CLp = nan(nM,nA); CDp = CLp; Cm = CLp; valid = false(nM,nA); CLb5 = CLp; CDb5 = CLp; CLb3 = CLp; CDb3 = CLp; t0 = tic;
    for i = 1:nM, for j = 1:nA
        fl = struct('Mach',Mach(i),'alpha_rad',Alpha(j)*pi/180);
        o = offdesign_aero_v7(geo,fl,refg,struct('mode','P2','base_pressure_ratio',0.5)); valid(i,j) = o.valid; CLp(i,j) = o.CL_pressure; CDp(i,j) = o.CD_pressure; Cm(i,j) = o.Cm; CLb5(i,j) = o.CL_base; CDb5(i,j) = o.CD_base;
        % base term is linear in (p_b/p_inf - 1): the 0.3 scenario follows exactly from the 0.5 one
        k = (0.3-1)/(0.5-1); CLb3(i,j) = k*o.CL_base; CDb3(i,j) = k*o.CD_base;
    end, end
    branch = Mt.branch{g}; gid = Mt.geometry_id{g}; param = Mt.value(g); ghash = Mt.geometry_hash{g};
    save([R 'db\p_' gid '.mat'],'branch','gid','param','ghash','Mach','Alpha','CLp','CDp','Cm','valid','CLb5','CDb5','CLb3','CDb3','-v7'); fprintf('%s done %.0f s\n',gid,toc(t0));
end
end

