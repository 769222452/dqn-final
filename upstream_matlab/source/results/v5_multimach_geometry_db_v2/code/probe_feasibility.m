function probe_feasibility(job_file, active_dir, out_file)
% Independent geometry-only feasibility probe for the multimach geometry DB V2.
% Calls the confirmed active build_geometry_core and records status/timing only.
% No native geometry is stored and no aerodynamic or performance code is loaded.
restoredefaultpath;
addpath(active_dir);
assert(strcmp(which('build_geometry_core'), fullfile(active_dir, 'build_geometry_core.m')));
jobs = jsondecode(fileread(job_file));
results = cell(numel(jobs), 1);
for k = 1:numel(jobs)
    job = jobs(k);
    args = num2cell(job.inputs10);
    r = struct('probe_id', job.probe_id, 'inputs10', job.inputs10, ...
               'core_path', which('build_geometry_core'));
    started = tic;
    try
        [wav_x, wav_y, wav_z, wav_up_x, wav_up_y, wav_up_z, diagnostics] = build_geometry_core(args{:});
        r.status = 'GENERATED';
        r.lower_shape = size(wav_x);
        r.upper_shape = size(wav_up_x);
        r.length = max(wav_x(:));
        r.half_span = max(wav_y(:));
        r.lower_z_range = [min(wav_z(:)), max(wav_z(:))];
        r.upper_z_range = [min(wav_up_z(:)), max(wav_up_z(:))];
        r.nonfinite_count = sum(~isfinite([wav_x(:); wav_y(:); wav_z(:); ...
                                           wav_up_x(:); wav_up_y(:); wav_up_z(:)]));
        r.diagnostics = diagnostics;
    catch ME
        r.status = 'GENERATION_FAILED';
        r.error_identifier = ME.identifier;
        r.error_message = ME.message;
        r.error_stack_top = '';
        if ~isempty(ME.stack)
            r.error_stack_top = sprintf('%s:%d', ME.stack(1).name, ME.stack(1).line);
        end
    end
    r.elapsed_s = toc(started);
    results{k} = r;
    fprintf('PROBE %s status=%s seconds=%.2f\n', job.probe_id, r.status, r.elapsed_s);
    write_json(out_file, results);
end
end

function write_json(path, data)
fid = fopen([path '.tmp'], 'w', 'n', 'UTF-8');
assert(fid ~= -1);
fprintf(fid, '%s', jsonencode(data));
fclose(fid);
movefile([path '.tmp'], path);
end
