function teacher_worker(exchange_dir, active_dir)
% One serial, hidden MATLAB process. No model changes, workers or graphics.
restoredefaultpath;
addpath(active_dir, '-begin');
assert(strcmp(which('build_geometry_core'),fullfile(active_dir,'build_geometry_core.m')));
assert(strcmp(which('offdesign_aero_v4'),fullfile(active_dir,'offdesign_aero_v4.m')));
set(0,'DefaultFigureVisible','off');
write_json(fullfile(exchange_dir,'ready.json'),struct('status','READY','version',version));
while ~exist(fullfile(exchange_dir,'stop'),'file')
    requests=dir(fullfile(exchange_dir,'request_*.json'));
    for k=1:numel(requests)
        input_path=fullfile(exchange_dir,requests(k).name);
        output_path=strrep(input_path,'request_','response_');
        if exist(output_path,'file'); continue; end
        req=jsondecode(fileread(input_path));
        out=struct('status','FAIL','operation',req.operation,'failure_type','');
        started=tic;
        try
            if strcmp(req.operation,'geometry')
                inputs10=req.inputs10;
                args=num2cell(inputs10);
                [wav_x,wav_y,wav_z,wav_up_x,wav_up_y,wav_up_z,diagnostics]=build_geometry_core(args{:});
                save(req.mat_output,'wav_x','wav_y','wav_z','wav_up_x','wav_up_y','wav_up_z','diagnostics','inputs10','-v7');
                out.diagnostics=diagnostics;
            elseif strcmp(req.operation,'aero')
                G=load(req.geometry_mat);
                out=offdesign_aero_v4(G.wav_x,G.wav_y,G.wav_z,G.wav_up_x,G.wav_up_y,G.wav_up_z,req.M_flight,req.alpha_rad,req.reference,req.model_options);
            else
                error('DirectTeacher:UnknownOperation','Unsupported operation');
            end
            out.status='PASS';
        catch ME
            out.status='FAIL';out.failure_type=ME.identifier;
            out.message=ME.message;out.stack=ME.stack;
        end
        out.elapsed_s=toc(started);
        out.resolved_geometry=which('build_geometry_core');
        out.resolved_aero=which('offdesign_aero_v4');
        write_json(output_path,out);
    end
    pause(0.05);
end
end

function write_json(path,value)
fid=fopen([path '.tmp'],'w','n','UTF-8'); assert(fid~=-1);
fprintf(fid,'%s',jsonencode(value)); fclose(fid);
movefile([path '.tmp'],path);
end
