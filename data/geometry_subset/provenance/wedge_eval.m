function wedge_eval(query_csv, out_csv, log_file)
% WEDGE_EVAL  Batch driver around the EXISTING offdesign_aero_v7 (mode P2 = strip shock-expansion with real state transfer,
% friction none, no base term => CL/CD of the pressure-only tangent-wedge/shock-expansion model).  No model code is changed.
%   query_csv columns: geometry_id, mat_path, Mach, alpha_deg      (one row = one fixed-geometry evaluation)
%   Reference (from configs/mission_1200s.yaml and v7 database builder): area 6.931920786884801 m^2, length 5 m.
%   Identical queries are never recomputed within one call (geometry is loaded once and results are cached by key).
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));        % project root (tools/v6_wedge_geometry_screening/wedge_eval.m)
v7dir = fullfile(root, 'results', 'v6_matlab_no_new_cfd', 'code', 'matlab');
addpath(v7dir);
T = readtable(query_csv, 'Delimiter', ',', 'TextType', 'char');   % explicit delimiter: ids/paths contain '.'
refg = struct('area_m2', 6.931920786884801, 'length_m', 5, 'moment_origin_m', [2.5 0 -0.9481139754931033]);
opts = struct('mode', 'P2', 'friction', 'none', 'base_pressure_ratio', NaN);
n = height(T);
CL = nan(n,1); CD = nan(n,1); Cm = nan(n,1); valid = false(n,1); det = nan(n,1); vac = nan(n,1); Memin = nan(n,1); Memax = nan(n,1);
reason = repmat({''}, n, 1); ghash = repmat({''}, n, 1); err = repmat({''}, n, 1);
ids = unique(T.geometry_id, 'stable');
fid = fopen(log_file, 'a'); t00 = tic;
for g = 1:numel(ids)
    idx = find(strcmp(T.geometry_id, ids{g}));
    try
        mp = T.mat_path{idx(1)}; if isempty(regexp(mp, '^[A-Za-z]:|^[\\/]', 'once')), mp = fullfile(root, mp); end   % project-relative paths
        S = load(mp);
        geo.lo = cat(3, S.wav_x, S.wav_y, S.wav_z); geo.up = cat(3, S.wav_up_x, S.wav_up_y, S.wav_up_z);
    catch ME
        for k = idx', err{k} = ['load_failed:' ME.message]; end
        fprintf(fid, '%s load failed\n', ids{g}); continue
    end
    cache = containers.Map('KeyType', 'char', 'ValueType', 'double');
    for k = idx'
        key = sprintf('%.12g|%.12g', T.Mach(k), T.alpha_deg(k));
        if isKey(cache, key)
            j = cache(key);
            CL(k) = CL(j); CD(k) = CD(j); Cm(k) = Cm(j); valid(k) = valid(j); det(k) = det(j); vac(k) = vac(j);
            Memin(k) = Memin(j); Memax(k) = Memax(j); reason{k} = reason{j}; ghash{k} = ghash{j}; continue
        end
        try
            o = offdesign_aero_v7(geo, struct('Mach', T.Mach(k), 'alpha_rad', T.alpha_deg(k)*pi/180), refg, opts);
            CL(k) = o.CL_pressure; CD(k) = o.CD_pressure; Cm(k) = o.Cm; valid(k) = o.valid; det(k) = o.detached_area_fraction;
            vac(k) = o.vacuum_area_fraction; Memin(k) = o.Me_min; Memax(k) = o.Me_max; ghash{k} = o.geometry_hash;
            reason{k} = strjoin(o.invalid_reasons, ';');
        catch ME
            err{k} = ['eval_failed:' ME.message];
        end
        cache(key) = k;
    end
    fprintf(fid, '%s %d queries %.1f s (total %.0f s)\n', ids{g}, numel(idx), toc(t00), toc(t00));
end
fclose(fid);
R = table(T.geometry_id, T.Mach, T.alpha_deg, CL, CD, Cm, valid, det, vac, Memin, Memax, reason, ghash, err, ...
    'VariableNames', {'geometry_id','M_flight','alpha_deg','CL','CD_pressure','Cm_pressure','model_valid','detached_area_fraction','vacuum_area_fraction','Me_min','Me_max','invalid_reasons','v7_coordinate_hash','error'});
writetable(R, out_csv);
end
