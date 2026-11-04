function make_family(job_file, active_dir, out_dir)
% Generate the controlled family with the UNMODIFIED active geometry core (upstream_matlab).
% Resumable: a job whose status.json says GENERATED is skipped. One folder per job:
%   <out_dir>/<job_id>/native_geometry.mat  (wav_*, wav_up_*, diagnostics, inputs10; same format as the snapshot)
%   <out_dir>/<job_id>/status.json
restoredefaultpath;
addpath(active_dir);
assert(strcmp(which('build_geometry_core'), fullfile(active_dir, 'build_geometry_core.m')), 'wrong geometry core on path');
set(0, 'DefaultFigureVisible', 'off');
jobs = jsondecode(fileread(job_file));
for k = 1:numel(jobs)
    if iscell(jobs)
        job = jobs{k};
    else
        job = jobs(k);
    end
    d = fullfile(out_dir, job.job_id);
    sfile = fullfile(d, 'status.json');
    if isfile(sfile)
        s = jsondecode(fileread(sfile));
        if strcmp(s.status, 'GENERATED'), continue; end
    end
    if ~isfolder(d), mkdir(d); end
    r = struct('job_id', job.job_id, 'inputs10', job.inputs10, 'core_path', which('build_geometry_core'));
    t = tic;
    try
        args = num2cell(job.inputs10);
        [wav_x, wav_y, wav_z, wav_up_x, wav_up_y, wav_up_z, diagnostics] = build_geometry_core(args{:});
        inputs10 = job.inputs10;
        save(fullfile(d, 'native_geometry.mat'), 'wav_x', 'wav_y', 'wav_z', 'wav_up_x', 'wav_up_y', 'wav_up_z', ...
             'diagnostics', 'inputs10', '-v7');
        r.status = 'GENERATED';
        r.nonfinite = sum(~isfinite([wav_x(:); wav_y(:); wav_z(:); wav_up_x(:); wav_up_y(:); wav_up_z(:)]));
    catch ME
        r.status = 'GENERATION_FAILED';
        r.error_identifier = ME.identifier;
        r.error_message = ME.message;
    end
    r.elapsed_s = toc(t);
    fid = fopen(sfile, 'w'); fprintf(fid, '%s', jsonencode(r)); fclose(fid);
    fprintf('FAMILY %s %s %.1fs\n', job.job_id, r.status, r.elapsed_s);
end
end
