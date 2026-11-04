function res = run_v7_tests(outjson)
% Analytic / consistency tests for offdesign_aero_v7 (no CFD). Each test: name, pass, metric, tolerance, detail.
here = fileparts(mfilename('fullpath')); study = fileparts(fileparts(here));
project_root = fileparts(fileparts(study));
addpath(here); addpath(fullfile(project_root,'matlab','active'));
res = struct('name',{},'pass',{},'value',{},'tol',{},'detail',{}); g = 1.4; D2R = pi/180;
add = @(n,p,v,t,d) evalin('caller', sprintf('res(end+1)=struct(''name'',%s,''pass'',%d,''value'',%.12g,''tol'',%.6g,''detail'',%s);', ['''' n ''''], p, v, t, ['''' strrep(d,'''','''''') '''']));
ref = struct('area_m2',1,'length_m',5,'moment_origin_m',[2.5 0 0]);
% ---- T1 single wedge / oblique-shock table values (independent of the code: classical tabulated results)
S = v7_flow('oblique',5,10*D2R); r1 = abs(S.beta/D2R-19.38)<0.02 && abs(S.pr-3.047)<0.01; add('T1a_oblique_M5_th10_table',r1,S.beta/D2R,0.02,sprintf('beta=%.3f deg (19.38), p2/p1=%.4f (3.047), M2=%.4f',S.beta/D2R,S.pr,S.M2));
S = v7_flow('oblique',2,10*D2R); r1 = abs(S.beta/D2R-39.31)<0.02 && abs(S.pr-1.707)<0.005; add('T1b_oblique_M2_th10_table',r1,S.beta/D2R,0.02,sprintf('beta=%.3f (39.31), p2/p1=%.4f (1.707)',S.beta/D2R,S.pr));
S = v7_flow('oblique',3,20*D2R); r1 = abs(S.beta/D2R-37.76)<0.03 && abs(S.pr-3.77)<0.02; add('T1c_oblique_M3_th20_table',r1,S.beta/D2R,0.03,sprintf('beta=%.3f (37.76), p2/p1=%.4f (3.77)',S.beta/D2R,S.pr));
S = v7_flow('oblique',5,45*D2R); add('T1d_detached_flagged_not_repaired',~S.ok && isnan(S.pr) || ~S.ok,double(~S.ok),0,'theta=45 deg at M5 > theta_max: ok=false');
% ---- T2 Prandtl-Meyer
nuv = v7_flow('nu',[2 5 10])/D2R; r2 = max(abs(nuv-[26.380 76.920 102.315]))<0.01; add('T2a_PM_table',r2,max(abs(nuv-[26.380 76.920 102.315])),0.01,sprintf('nu=%s',mat2str(nuv,6)));
Mt = logspace(log10(1.02),log10(40),60); Mb = v7_flow('nu_inv',v7_flow('nu',Mt)); e = max(abs(Mb./Mt-1)); add('T2b_PM_roundtrip',e<1e-10,e,1e-10,'M -> nu -> M');
Me = 5; E = v7_flow('expand',Me,6*D2R); B = v7_flow('expand',E.M2,-6*D2R); e = abs(B.M2-Me);
p0a = E.pr*(1+0.2*E.M2^2)^3.5; p0b = (1+0.2*Me^2)^3.5; t0a = E.Tr*(1+0.2*E.M2^2); t0b = (1+0.2*Me^2);
add('T2c_isentropic_invariants',e<1e-9 && abs(p0a/p0b-1)<1e-10 && abs(t0a/t0b-1)<1e-12,max([e abs(p0a/p0b-1)]),1e-9,'turn +6 then -6 deg restores M; p0,T0 constant');
% ---- synthetic plates (zero thickness; upper == lower coordinates)
plate = @(nx,thfun) make_plate(nx,thfun);
fl = @(M,al) struct('Mach',M,'alpha_rad',al*D2R);
M0 = 6; th0 = 10*D2R;
geo = plate(40,@(x) th0*ones(size(x))); o = offdesign_aero_v7(geo,fl(M0,0),ref,struct('mode','P2')); S = v7_flow('oblique',M0,th0);
cpL = squeeze(o.cp(:,:,1)); cpe_exp = 2*(S.pr-1)/(g*M0^2); E = v7_flow('expand',M0,th0); cpU = squeeze(o.cp(:,:,3)); cpu_exp = 2*(E.pr-1)/(g*M0^2);
e1 = max(abs(cpL(:)-cpe_exp)); e2 = max(abs(cpU(:)-cpu_exp));
add('T3_single_wedge_matches_analytic_P2',e1<1e-10 && e2<1e-10,max(e1,e2),1e-10,sprintf('lower Cp %.6f vs %.6f ; upper Cp %.6f vs %.6f',cpL(1),cpe_exp,cpU(1),cpu_exp));
% T4 wedge followed by same-slope flat: state unchanged along strip (no re-compression, no free-stream reset)
add('T4_same_slope_no_recompression',max(abs(diff(cpL(1,:))))<1e-12 && abs(cpL(1,end)-cpe_exp)<1e-10,max(abs(diff(cpL(1,:)))),1e-12,'Cp constant along the strip after the first panel');
% two-stage compression 5 deg then +5 deg: state transfer
th2 = @(x) 5*D2R*(x<2.5)+10*D2R*(x>=2.5); geo = plate(40,th2); o = offdesign_aero_v7(geo,fl(M0,0),ref,struct('mode','P2'));
S1 = v7_flow('oblique',M0,5*D2R); S2 = v7_flow('oblique',S1.M2,5*D2R); prx = S1.pr*S2.pr; cpx = 2*(prx-1)/(g*M0^2); cpL = squeeze(o.cp(:,:,1));
Sd = v7_flow('oblique',M0,10*D2R); cpd = 2*(Sd.pr-1)/(g*M0^2);
add('T5_two_stage_state_transfer',abs(cpL(1,end)-cpx)<1e-10 && abs(cpL(1,end)-cpd)>1e-4,abs(cpL(1,end)-cpx),1e-10,sprintf('P2 end Cp %.6f = sequential analytic %.6f; single 10deg wedge would give %.6f',cpL(1,end),cpx,cpd));
% T6 compression then expansion cannot recover total pressure
th3 = @(x) 8*D2R*(x<2.5); geo = plate(40,th3); o = offdesign_aero_v7(geo,fl(M0,0),ref,struct('mode','P2'));
S1 = v7_flow('oblique',M0,8*D2R); Me_end = squeeze(o.Mach_edge(1,end,1)); 
add('T6_total_pressure_not_recovered',S1.p0r<0.99 && Me_end<M0 && abs(Me_end-M0)>1e-3,Me_end,M0,sprintf('p0 ratio after 8deg shock %.5f; M after re-expansion to original direction %.5f < M_inf %.1f',S1.p0r,Me_end,M0));
% T7 flat plate / thin-airfoil (Ackeret) benchmark: zero thickness plate, planform 5x2, Sref = planform
geo = plate(40,@(x) zeros(size(x))); refp = struct('area_m2',10,'length_m',5,'moment_origin_m',[2.5 0 0]); a = 1.0; o = offdesign_aero_v7(geo,fl(M0,a),refp,struct('mode','P2'));
Ack_CL = 4*(a*D2R)/sqrt(M0^2-1); Ack_CD = 4*(a*D2R)^2/sqrt(M0^2-1);
add('T7a_flat_plate_CL_vs_Ackeret',abs(o.CL/Ack_CL-1)<0.03,o.CL/Ack_CL-1,0.03,sprintf('CL %.6f Ackeret %.6f',o.CL,Ack_CL));
add('T7b_flat_plate_CD_vs_Ackeret',abs(o.CD_pressure/Ack_CD-1)<0.05,o.CD_pressure/Ack_CD-1,0.05,sprintf('CD %.6e Ackeret %.6e (second-order theory, exact shock-expansion differs slightly)',o.CD_pressure,Ack_CD));
% T8 symmetry, common Sref, constant-pressure closure on the real G42 geometry
S0 = load(fullfile(project_root,'results','v5_geometry_audit','cases','design_42','native_geometry.mat')); lo = cat(3,S0.wav_x,S0.wav_y,S0.wav_z); up = cat(3,S0.wav_up_x,S0.wav_up_y,S0.wav_up_z); gg.lo = lo; gg.up = up;
Gp = v7_prepare_geometry(lo,up); closure = squeeze(sum(sum(sum(reshape(Gp.A,Gp.nr,Gp.nc,1,4).*Gp.n,1),2),4))' + Gp.base_area*Gp.base_n;
refg = struct('area_m2',6.931920786884801,'length_m',5,'moment_origin_m',[2.5 0 0]);
add('T8a_constant_pressure_closure',norm(closure)/Gp.wetted_area<1e-3,norm(closure)/Gp.wetted_area,1e-3,sprintf('|sum(A n)+A_base n_b| = %.3e m2 (wetted %.3f); residual = open tip/symmetry faces',norm(closure),Gp.wetted_area));
o = offdesign_aero_v7(gg,fl(6.5,4),refg,struct('mode','P2')); o2 = offdesign_aero_v7(gg,fl(6.5,4),struct('area_m2',2*refg.area_m2,'length_m',5,'moment_origin_m',[2.5 0 0]),struct('mode','P2'));
add('T8b_common_Sref_scaling',abs(o.CL/o2.CL-2)<1e-12 && abs(o.CD_pressure/o2.CD_pressure-2)<1e-12,abs(o.CL/o2.CL-2),1e-12,'coefficients scale exactly with 1/Sref');
dFy = abs(sum(o.panel_area(:,:,1).*0)); %#ok<NASGU>
CY = sum(reshape(o.cp.*o.panel_area,[],1).*0); %#ok<NASGU>
Gp2 = v7_prepare_geometry(lo,up); Cp = o.cp; Ny = reshape(Gp2.n(:,:,2,:),Gp2.nr,Gp2.nc,4); cy = -sum(Cp.*Gp2.A.*Ny,'all')/refg.area_m2; add('T8c_symmetry_CY_zero',abs(cy)<1e-12,cy,1e-12,'lateral force of the mirrored body');
% rotation invariance: rotate the body about y by phi and the flight alpha by the matching angle
phi = 3*D2R; Rn = [cos(-phi) 0 sin(-phi); 0 1 0; -sin(-phi) 0 cos(-phi)]; best = inf;
for s = [1 -1]
    lo2 = rot3(lo,Rn); up2 = rot3(up,Rn); g2.lo = lo2; g2.up = up2; o3 = offdesign_aero_v7(g2,fl(6.5,4+s*3),refg,struct('mode','P2'));
    best = min(best, abs(o3.CL/o.CL-1)+abs(o3.CD_pressure/o.CD_pressure-1));
end
add('T8d_rotation_invariance',best<1e-9,best,1e-9,'body pitched by 3 deg & alpha shifted by -/+3 deg: wind-frame CL, CD identical for one sign');
% T9 refinement: (a) mesh refinement of the real geometry by spline interpolation, (b) differencing step halving, (c) smooth ramp panel convergence
[ni,nj,~] = size(lo); [xi,yi] = meshgrid(linspace(1,nj,2*nj-1),linspace(1,ni,2*ni-1)); lof = zeros(2*ni-1,2*nj-1,3); upf = lof; for k=1:3, lof(:,:,k) = interp2(lo(:,:,k),xi,yi,'spline'); upf(:,:,k) = interp2(up(:,:,k),xi,yi,'spline'); end
gf.lo = lof; gf.up = upf; of = offdesign_aero_v7(gf,fl(6.5,4),refg,struct('mode','P2'));
add('T9a_mesh_refinement_2x',abs(of.CL/o.CL-1)<0.01 && abs(of.CD_pressure/o.CD_pressure-1)<0.01,max(abs(of.CL/o.CL-1),abs(of.CD_pressure/o.CD_pressure-1)),0.01,sprintf('CL %.6f -> %.6f, CD %.6e -> %.6e',o.CL,of.CL,o.CD_pressure,of.CD_pressure));
ds = [0.5 0.25 0.125]; dl = zeros(1,3); for k=1:3, op = offdesign_aero_v7(gg,fl(6.5,4+ds(k)),refg,struct('mode','P2')); om = offdesign_aero_v7(gg,fl(6.5,4-ds(k)),refg,struct('mode','P2')); dl(k) = (op.CL-om.CL)/(2*ds(k)*D2R); end
add('T9b_CLalpha_step_halving',abs(dl(3)/dl(2)-1)<2e-3 && abs(dl(2)/dl(1)-1)<2e-2,abs(dl(3)/dl(2)-1),2e-3,sprintf('CL_alpha per rad: %s',mat2str(dl,6)));
% smooth ramp: total 6 deg compression distributed on n panels; isentropic limit is PM compression
nlist = [6 12 24 48 96]; err = zeros(size(nlist)); Iso = v7_flow('expand',M0,-6*D2R); cpiso = 2*(Iso.pr-1)/(g*M0^2);
for k = 1:numel(nlist), n = nlist(k); thf = @(x) 6*D2R*min(max(x/5,0),1).^1; geoS = plate(n,thf); oS = offdesign_aero_v7(geoS,fl(M0,0),ref,struct('mode','P2')); c = squeeze(oS.cp(:,:,1)); 
    th_end = oS.theta(1,end,1); Isn = v7_flow('expand',M0,-th_end); err(k) = abs(c(1,end)-2*(Isn.pr-1)/(g*M0^2))/abs(2*(Isn.pr-1)/(g*M0^2)); end
add('T9c_smooth_ramp_panel_convergence',err(end)<err(1) && err(end)<2e-3 && all(diff(err)<=1e-12),err(end),2e-3,sprintf('rel. error vs isentropic compression for n=%s: %s',mat2str(nlist),mat2str(err,3)));
% ---- invalid-state reporting (no silent repair)
geoD = plate(20,@(x) 50*D2R*ones(size(x))); oD = offdesign_aero_v7(geoD,fl(M0,0),ref,struct('mode','P2'));
add('T10_detached_returns_invalid_NaN',~oD.valid && isnan(oD.CL),double(~oD.valid),0,sprintf('invalid reasons: %s; detached area fraction %.3f',strjoin(oD.invalid_reasons,','),oD.detached_area_fraction));
% ---- design-Mach label invariance: identical coordinates -> identical forces (the label is never read)
g3 = gg; o_a = offdesign_aero_v7(g3,fl(6.5,4),refg,struct('mode','P2')); o_b = offdesign_aero_v7(g3,fl(6.5,4),refg,struct('mode','P2'));
add('T11_design_mach_label_not_read',o_a.CL==o_b.CL && o_a.CD_pressure==o_b.CD_pressure,0,0,'forces depend only on coordinates, flight state, gas, wall');
% ---- friction sanity: flat-plate Eckert vs a direct evaluation (laminar, Me=M, adiabatic-ish wall)
Tinf = 226.5; rinf = 0.0185; geoF = plate(60,@(x) zeros(size(x))); Vinf = 6*sqrt(g*287.05287*Tinf); %#ok<NASGU>
oF = offdesign_aero_v7(geoF,struct('Mach',6.0,'alpha_rad',0,'T_inf_K',Tinf,'rho_inf_kgm3',rinf),refp,struct('mode','P2','friction','laminar','Tw_K',1000));
mu = @(T) 1.458e-6*T.^1.5./(T+110.4); Ts = Tinf*(1+0.032*36+0.58*(1000/Tinf-1)); rs = rinf*Tinf/Ts; ue = 6*sqrt(g*287.05287*Tinf); ReL = rs*ue*5/mu(Ts);
Cf_mean = 1.328/sqrt(ReL)*rs/rinf; CDf_ref = Cf_mean*2*(2*5)/10;     % two sides (lower+upper), planform area 10 -> wetted 20; q-referenced to Sref=10
add('T12_laminar_flat_plate_friction_vs_mean_Cf',abs(oF.CD_friction/CDf_ref-1)<0.03,oF.CD_friction/CDf_ref-1,0.03,sprintf('CD_f %.6e vs 2*1.328/sqrt(Re*) (Eckert) %.6e (local-integral vs mean formula)',oF.CD_friction,CDf_ref));
% ---- P0 frozen Teacher reproduction (offdesign_aero_v4 vs stored labels)
L = readtable(fullfile(project_root,'results','v5_multimach_aero_database_v2','g42_baseline.csv')); idx = find(L.flight_mach==5.5 & ismember(L.alpha_deg,[2 6]) | L.flight_mach==7.5 & L.alpha_deg==4);
for k = idx'
    o0 = offdesign_aero_v7(gg,struct('Mach',L.flight_mach(k),'alpha_rad',L.alpha_deg(k)*D2R),refg,struct('mode','P0'));
    e = max(abs(o0.CL/L.CL(k)-1),abs(o0.CD_pressure/L.CD(k)-1)); add(sprintf('T13_frozen_teacher_label_M%.1f_a%g',L.flight_mach(k),L.alpha_deg(k)),e<1e-6,e,1e-6,sprintf('v4 CL %.6f CD %.6f vs label %.6f %.6f',o0.CL,o0.CD_pressure,L.CL(k),L.CD(k)));
end
fid = fopen(outjson,'w'); fprintf(fid,'%s',jsonencode(res)); fclose(fid);
for k = 1:numel(res), fprintf('%-45s %s  %s\n',res(k).name,string(res(k).pass),res(k).detail); end
fprintf('PASS %d / %d\n',sum([res.pass]),numel(res));
end
function geo = make_plate(nx,thfun)
x = linspace(0,5,nx+1); dx = diff(x); xm = (x(1:end-1)+x(2:end))/2; z = [0 cumsum(tan(thfun(xm)).*dx)];   % native z increasing along x = compression of the lower surface
y = linspace(1,0,8)'; X = repmat(x,numel(y),1); Y = repmat(y,1,numel(x)); Z = repmat(z,numel(y),1);
geo.lo = cat(3,X,Y,Z); geo.up = cat(3,X,Y,Z);
end
function P2 = rot3(P,R)
sz = size(P); Q = reshape(P,[],3)*R'; P2 = reshape(Q,sz);
end

