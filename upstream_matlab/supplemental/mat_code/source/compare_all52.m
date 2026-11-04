function compare_all52
% Compare offdesign_aero vs generatewaverider LD for all 52 Pareto points
% Design conditions: Ma=15, AOA_prog=-4 deg
% Coded by Codex, 2026-07-31

    clc; clear; format short;
    warning('off', 'all');
    
    %% Load optimization data
    load('wave_rider_optimization_final.mat');
    if isempty(gcp('nocreate'))
        parpool('local',6);
    end
    %% Sort Pareto front
    [~, sort_idx] = sortrows(pareto_front, 1);
    sorted_design = design_vars(sort_idx, :);
    
    %% Design conditions
    Ma_des = 15;
    AOA_prog = -4;
    AOA_real = 4;
    c1 = 1.0331; c2 = 1.0331;
    
    %% Pre-allocate
    n_pts = 52;
    LD_gen_all  = zeros(n_pts, 1);
    LD_off_all  = zeros(n_pts, 1);
    err_all     = zeros(n_pts, 1);
    valid_idx   = true(n_pts, 1);
    
    fprintf('=== offdesign_aero vs generatewaverider LD Comparison ===\n');
    fprintf('Design Ma = %d, AOA(prog) = %d deg (real AOA = +%d deg)\n', ...
        Ma_des, AOA_prog, AOA_real);
    fprintf('C1 = %.4f, C2 = %.4f\n\n', c1, c2);
    
    t_total = tic;
    
    %% Main loop
    for k = 1:n_pts
        fprintf('[%3d/52] params: a1=%.4f a2=%.4f b1=%.4f b2=%.4f d1=%.4f ax=%.4f\n', ...
            k, sorted_design(k,1), sorted_design(k,2), sorted_design(k,3), ...
            sorted_design(k,4), sorted_design(k,5), sorted_design(k,6));
        
        %% Step 1: generatewaverider with robust error handling
        [wav_x, wav_y, wav_z, LD_gen] = generatewaverider(...
            Ma_des, sorted_design(k,1), sorted_design(k,2), ...
            sorted_design(k,3), sorted_design(k,4), c1, c2, ...
            sorted_design(k,5), AOA_prog, sorted_design(k,6));
        LD_gen_all(k) = LD_gen;
        if ~isreal(LD_gen) || isnan(LD_gen) || isempty(LD_gen)
            valid_idx(k) = false;
            LD_gen_all(k) = NaN;
            LD_off_all(k) = NaN;
            err_all(k) = NaN;
            if ~isempty(LD_gen)
                fprintf('  [FAIL] generatewaverider returned LD=%.4f\n', LD_gen);
            else
                fprintf('  [FAIL] generatewaverider failed or returned empty\n');
            end
            continue;
        end
        
        %% Step 2: Generate upper surface
        M  = size(wav_x, 1);
        nn = size(wav_x, 2);
        
        V_prog = [cosd(AOA_prog), 0, sind(AOA_prog)];
        x_design_max = max(wav_x(:));
        
        wav_up_x = zeros(M, nn);
        wav_up_y = zeros(M, nn);
        wav_up_z = zeros(M, nn);
        
        for ii = 1:M
            x_le = wav_x(ii, 1);
            y_le = wav_y(ii, 1);
            z_le = wav_z(ii, 1);
            for jj = 1:nn
                wav_up_x(ii, jj) = x_le + (x_design_max - x_le) * (jj-1)/(nn-1);
                wav_up_y(ii, jj) = y_le + (wav_up_x(ii, jj) - x_le) * V_prog(2) / V_prog(1);
                wav_up_z(ii, jj) = z_le + (wav_up_x(ii, jj) - x_le) * V_prog(3) / V_prog(1);
            end
        end
        
        %% Step 3: offdesign_aero evaluation
        [LD_off, ~, ~, ~] = offdesign_aero_v4(...
            wav_x, wav_y, wav_z, wav_up_x, wav_up_y, wav_up_z, ...
            Ma_des, AOA_real);
        
        if ~isfinite(LD_off)
            valid_idx(k) = false;
            LD_off_all(k) = NaN;
            err_all(k) = NaN;
            fprintf('  [WARN] offdesign_aero returned LD=%.4f (non-finite)\n', LD_off);
            continue;
        end
        LD_off_all(k) = LD_off;
        
        %% Step 4: Compute error
        if LD_gen ~= 0
            err_all(k) = (LD_off - LD_gen) / LD_gen * 100;
        else
            err_all(k) = 0;
        end
        
        fprintf('  LD_gen=%.6f, LD_off=%.6f, err=%.4f%%\n', ...
            LD_gen, LD_off, err_all(k));
    end
    
    %% Print results table
    fprintf('\n\n=== COMPARISON RESULTS TABLE ===\n');
    fprintf('%-4s %8s %8s %8s %8s %8s %8s %10s %10s %8s %6s\n', ...
        'Idx', 'a1', 'a2', 'b1', 'b2', 'd1', 'ax1', 'LD_gen', 'LD_off', 'Err_%', 'S');
    fprintf('%s\n', repmat('-', 90));
    
    n_valid  = 0;
    n_warn   = 0;
    n_fail   = 0;
    
    for k = 1:n_pts
        if ~valid_idx(k)
            fprintf('%4d %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %10s %10s %8s %6s\n', ...
                k, sorted_design(k,1), sorted_design(k,2), sorted_design(k,3), ...
                sorted_design(k,4), sorted_design(k,5), sorted_design(k,6), ...
                num2str(LD_gen_all(k), '%.4f'), num2str(LD_off_all(k), '%.4f'), 'N/A', 'F');
            n_fail = n_fail + 1;
        else
            if abs(err_all(k)) < 5
                status = 'OK';
                n_valid = n_valid + 1;
            else
                status = 'W';
                n_warn = n_warn + 1;
            end
            fprintf('%4d %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %10.4f %10.4f %7.2f%% %6s\n', ...
                k, sorted_design(k,1), sorted_design(k,2), sorted_design(k,3), ...
                sorted_design(k,4), sorted_design(k,5), sorted_design(k,6), ...
                LD_gen_all(k), LD_off_all(k), err_all(k), status);
        end
    end
    
    %% Summary statistics
    fprintf('\n%s\n', repmat('=', 90));
    fprintf('Summary:\n');
    fprintf('  Total points:     52\n');
    fprintf('  OK  (err<5%%):     %2d\n', n_valid);
    fprintf('  WARN (err>=5%%):   %2d\n', n_warn);
    fprintf('  FAIL (no result):  %2d\n', n_fail);
    
    if n_valid + n_warn > 0
        valid_errs = err_all(valid_idx & abs(err_all) < 1e10);
        if ~isempty(valid_errs)
            fprintf('  Mean |Err|:       %.2f%%\n', mean(abs(valid_errs)));
            fprintf('  Max |Err|:       %.2f%%\n', max(abs(valid_errs)));
            fprintf('  Min Err:         %.2f%%\n', min(valid_errs));
            fprintf('  Max Err:         %.2f%%\n', max(valid_errs));
        end
    end
    
    %% Write results to file
    fid = fopen(fullfile(pwd, 'compare_results52.txt'), 'wt');
    fprintf(fid, 'offdesign_aero vs generatewaverider LD Comparison\n');
    fprintf(fid, 'Design Ma=%d, AOA(prog)=%d deg, C1=%.4f, C2=%.4f\n\n', ...
        Ma_des, AOA_prog, c1, c2);
    fprintf(fid, '%-4s %8s %8s %8s %8s %8s %8s %10s %10s %8s %6s\n', ...
        'Idx', 'a1', 'a2', 'b1', 'b2', 'd1', 'ax1', 'LD_gen', 'LD_off', 'Err_%', 'S');
    for k = 1:n_pts
        if ~valid_idx(k)
            fprintf(fid, '%4d %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %10s %10s %8s %6s\n', ...
                k, sorted_design(k,1), sorted_design(k,2), sorted_design(k,3), ...
                sorted_design(k,4), sorted_design(k,5), sorted_design(k,6), ...
                num2str(LD_gen_all(k), '%.4f'), num2str(LD_off_all(k), '%.4f'), 'N/A', 'F');
        else
            if abs(err_all(k)) < 5
                status = 'OK';
            else
                status = 'W';
            end
            fprintf(fid, '%4d %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %10.4f %10.4f %7.2f%% %6s\n', ...
                k, sorted_design(k,1), sorted_design(k,2), sorted_design(k,3), ...
                sorted_design(k,4), sorted_design(k,5), sorted_design(k,6), ...
                LD_gen_all(k), LD_off_all(k), err_all(k), status);
        end
    end
    fclose(fid);
    
    fprintf('\nResults saved to: compare_results52.txt\n');
    fprintf('Total time: %.1f seconds\n', toc(t_total));
end
