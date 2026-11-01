function v5_batch(input_path, output_path)
% Fixed-path, no graphics/workers, complete JSON status on every failure.
input=jsondecode(fileread(input_path));
out=struct('status','FAIL','failure_type','','message','');
t=tic;
try
    set(0,'DefaultFigureVisible','off');
    if strcmp(input.operation,'geometry')
        p=input.parameters;
        [wav_x,wav_y,wav_z,wav_up_x,wav_up_y,wav_up_z,diagnostics]=build_geometry_core(p(1),p(2),p(3),p(4),p(5),p(6),p(7),p(8),p(9),p(10));
        save(input.mat_output,'wav_x','wav_y','wav_z','wav_up_x','wav_up_y','wav_up_z','diagnostics','-v7');
        out.diagnostics=diagnostics;
    elseif strcmp(input.operation,'aero')
        G=load(input.geometry_mat);
        out=offdesign_aero_v4(G.wav_x,G.wav_y,G.wav_z,G.wav_up_x,G.wav_up_y,G.wav_up_z,input.M_flight,input.alpha_rad,input.reference,input.model_options);
    elseif strcmp(input.operation,'components')
        if isfield(input,'gamma_global_junk');global specific_heat_ratio;specific_heat_ratio=input.gamma_global_junk;end
        [beta,~,~,mf,pf]=conical_flowfield(input.M_flight,input.theta_rad,input.gamma_gas);
        out.beta_rad=beta;out.M_cone=mf(end);out.pressure_ratio=pf(end);
    else
        error('V5:InputInvalid','Unknown operation');
    end
    out.status='PASS';
catch ME
    out.status='FAIL';out.failure_type=ME.identifier;out.message=ME.message;
    out.stack=ME.stack;
end
out.elapsed_s=toc(t);
out.resolved_geometry=which('build_geometry_core');
out.resolved_aero=which('offdesign_aero_v4');
out.resolved_cone=which('comptue_conicalfield');
active_files=dir(fullfile(fileparts(mfilename('fullpath')),'*.m'));
out.resolved_dependencies=struct();
for k=1:numel(active_files)
    [~,name]=fileparts(active_files(k).name);
    out.resolved_dependencies.(name)=which(name);
end
out.matlab_version=version;
out.toolboxes=ver;
fid=fopen(output_path,'w');fprintf(fid,'%s',jsonencode(out));fclose(fid);
end
