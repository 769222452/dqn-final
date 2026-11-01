function run_geometry_audit_batch(job_file, active_dir)
% Geometry only. This independent wrapper never invokes performance or aero.
restoredefaultpath;
addpath(active_dir);
assert(strcmp(which('build_geometry_core'),fullfile(active_dir,'build_geometry_core.m')));
jobs=jsondecode(fileread(job_file));
for k=1:numel(jobs)
    job=jobs(k);
    if ~exist(job.case_dir,'dir'); mkdir(job.case_dir); end
    status=struct('design_index',job.design_index,'status','STARTED','core_path',which('build_geometry_core'));
    write_json(fullfile(job.case_dir,'started.json'),status);
    started=tic;
    try
        args=num2cell(job.inputs10);
        [wav_x,wav_y,wav_z,wav_up_x,wav_up_y,wav_up_z,diagnostics]=build_geometry_core(args{:});
        inputs10=job.inputs10;
        save(fullfile(job.case_dir,'native_geometry.mat'),'wav_x','wav_y','wav_z','wav_up_x','wav_up_y','wav_up_z','diagnostics','inputs10','-v7');
        status.status='GENERATED';
        status.diagnostics=diagnostics;
    catch ME
        status.status='GENERATION_FAILED';
        status.error_identifier=ME.identifier;
        status.error_message=ME.message;
        status.error_report=getReport(ME,'extended','hyperlinks','off');
    end
    status.elapsed_s=toc(started);
    write_json(fullfile(job.case_dir,'generation_status.json'),status);
    fprintf('AUDIT design=%d status=%s seconds=%.3f\n',job.design_index,status.status,status.elapsed_s);
end
end
function write_json(path,data)
fid=fopen([path '.tmp'],'w','n','UTF-8');
assert(fid~=-1);
fprintf(fid,'%s',jsonencode(data)); fclose(fid);
movefile([path '.tmp'],path);
end
