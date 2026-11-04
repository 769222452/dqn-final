function aero_worker(exchange_dir, active_dir)
% Persistent hidden MATLAB aero worker for the 30-minute V2 pilot.
% Request = one geometry MAT x one flight Mach x a list of alpha (deg).
% Calls the active offdesign_aero_v4 unchanged; never writes outside exchange_dir.
restoredefaultpath;
addpath(active_dir, '-begin');
assert(strcmp(which('offdesign_aero_v4'),fullfile(active_dir,'offdesign_aero_v4.m')));
set(0,'DefaultFigureVisible','off');
write_json(fullfile(exchange_dir,'ready.json'),struct('status','READY','version',version, ...
    'resolved_aero',which('offdesign_aero_v4'),'pid',feature('getpid')));
while ~exist(fullfile(exchange_dir,'stop'),'file')
    requests=dir(fullfile(exchange_dir,'request_*.json'));
    for k=1:numel(requests)
        input_path=fullfile(exchange_dir,requests(k).name);
        output_path=strrep(input_path,'request_','response_');
        if exist(output_path,'file'); continue; end
        req=jsondecode(fileread(input_path));
        resp=struct('request_id',req.request_id,'status','PASS','message','','points',[]);
        t0=tic;
        try
            G=load(req.geometry_mat);
            pts=cell(1,numel(req.alpha_deg));
            for j=1:numel(req.alpha_deg)
                a=req.alpha_deg(j); t1=tic;
                p=struct('alpha_deg',a,'status','PASS','CL',NaN,'CD',NaN,'Cm',NaN, ...
                    'error_id','','error_message','','kernel_s',NaN,'model_version','', ...
                    'n_cone',NaN,'n_expansion',NaN,'n_freestream',NaN);
                try
                    o=offdesign_aero_v4(G.wav_x,G.wav_y,G.wav_z,G.wav_up_x,G.wav_up_y,G.wav_up_z, ...
                        req.M_flight,a*pi/180,req.reference,req.model_options);
                    p.CL=o.CL_pressure; p.CD=o.CD_pressure; p.Cm=o.Cm_pressure;
                    p.model_version=o.model_version;
                    p.n_cone=o.branch_counts.tangent_cone_compression;
                    p.n_expansion=o.branch_counts.hypersonic_expansion_similarity;
                    p.n_freestream=o.branch_counts.freestream;
                catch ME
                    p.status='FAIL'; p.error_id=ME.identifier; p.error_message=ME.message;
                end
                p.kernel_s=toc(t1);
                pts{j}=p;
            end
            resp.points=[pts{:}];
        catch ME
            resp.status='FAIL'; resp.message=[ME.identifier ': ' ME.message];
        end
        resp.elapsed_s=toc(t0);
        write_json(output_path,resp);
    end
    pause(0.02);
end
end

function write_json(path,value)
fid=fopen([path '.tmp'],'w','n','UTF-8'); assert(fid~=-1);
fprintf(fid,'%s',jsonencode(value)); fclose(fid);
movefile([path '.tmp'],path);
end
