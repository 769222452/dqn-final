function exp_worker(worker_dir,root)
% Single persistent MATLAB process; calls existing physics without modification.
restoredefaultpath;
addpath(fullfile(root,'matlab','active'));
addpath(fullfile(root,'tools','v6_wedge_geometry_screening'));
write_json(fullfile(worker_dir,'ready.json'),struct('pid',feature('getpid'),'core',which('build_geometry_core'),'matlab',version));
while ~isfile(fullfile(worker_dir,'stop'))
    files=dir(fullfile(worker_dir,'request_*.json'));
    for k=1:numel(files)
        request_path=fullfile(worker_dir,files(k).name);
        response_path=strrep(request_path,'request_','response_');
        if isfile(response_path), continue; end
        job=jsondecode(fileread(request_path)); t=tic;
        write_json(strrep(request_path,'request_','started_'),struct('pid',feature('getpid'),'operation',job.operation,'started',char(datetime('now'))));
        r=struct('status','PASS','operation',job.operation);
        try
            switch job.operation
                case 'generate'
                    status_path=fullfile(job.case_dir,'generation_status.json');
                    if ~isfile(status_path)
                        write_json(fullfile(job.case_dir,'started.json'),struct('status','STARTED','pid',feature('getpid'),'core_path',which('build_geometry_core')));
                        inputs10=job.inputs10(:)'; args=num2cell(inputs10);
                        try
                            [wav_x,wav_y,wav_z,wav_up_x,wav_up_y,wav_up_z,diagnostics]=build_geometry_core(args{:});
                            tmp=fullfile(job.case_dir,'native_geometry.tmp.mat');
                            save(tmp,'wav_x','wav_y','wav_z','wav_up_x','wav_up_y','wav_up_z','diagnostics','inputs10','-v7');
                            movefile(tmp,fullfile(job.case_dir,'native_geometry.mat'),'f');
                            st=struct('status','GENERATED','elapsed_s',toc(t),'diagnostics',diagnostics);
                        catch ME
                            st=struct('status','GENERATION_FAILED','elapsed_s',toc(t),'error_message',ME.message,'error_report',getReport(ME,'extended','hyperlinks','off'));
                        end
                        write_json(status_path,st);
                    end
                case 'evaluate'
                    if ~isfile(job.out_csv)
                        tmp=[job.out_csv '.tmp.csv'];
                        wedge_eval(job.query_csv,tmp,job.log_file);
                        movefile(tmp,job.out_csv,'f');
                    end
                otherwise
                    error('Unknown operation');
            end
        catch ME
            r.status='ERROR'; r.message=ME.message; r.report=getReport(ME,'extended','hyperlinks','off');
        end
        r.elapsed_s=toc(t); write_json(response_path,r);
    end
    pause(0.1);
end
end
function write_json(path,data)
fid=fopen([path '.tmp'],'w','n','UTF-8'); assert(fid~=-1);
fprintf(fid,'%s',jsonencode(data)); fclose(fid); movefile([path '.tmp'],path,'f');
end
