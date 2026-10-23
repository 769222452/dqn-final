function out = offdesign_aero_v7(geometry, flight, reference, options)
% OFFDESIGN_AERO_V7  isolated engineering aero module (NO CFD).  Pressure + friction + base components kept separate.
%   geometry : struct lo, up (native 50x50x3 grids; body = diag(1,-1,-1)) [+ id]
%   flight   : Mach, alpha_rad, [T_inf_K, rho_inf_kgm3 -> needed only for friction]
%   reference: area_m2, length_m, [moment_origin_m]   (common for all geometries; never rescaled with the shape)
%   options  : mode 'P0' (frozen Teacher offdesign_aero_v4) | 'P1' (tangent wedge + exact Prandtl-Meyer from the free stream)
%              | 'P2' (strip shock-expansion with REAL state transfer; default)
%              friction 'none'|'turbulent'|'laminar'|'transition' ; Tw_K ; Re_tr ; base_pressure_ratio (p_b/p_inf, NaN = none)
% Invalid states (no attached weak shock, subsonic post-shock, vacuum, strip fold-back) set valid=false and coefficients NaN.
% No silent fallback, no Mach clamping, no abs() repairs.
VERSION = 'offdesign_aero_v7-1.0';
if nargin<4, options = struct(); end
opt = struct('mode','P2','friction','none','Tw_K',1000,'Re_tr',5e6,'base_pressure_ratio',NaN);
fn = fieldnames(options); for k=1:numel(fn), opt.(fn{k}) = options.(fn{k}); end
M = flight.Mach; al = flight.alpha_rad; g = 1.4;
if ~(isreal(M) && isfinite(M) && M>1 && isfinite(al) && abs(al)<pi/2), error('v7:input','invalid flight condition'); end
Sref = reference.area_m2; Lref = reference.length_m; org = [0 0 0]; if isfield(reference,'moment_origin_m'), org = reshape(reference.moment_origin_m,1,3); end
out = struct('version',VERSION,'mode',opt.mode,'valid',true,'invalid_reasons',{{}});
ed = [cos(al) 0 sin(al)]; el = [-sin(al) 0 cos(al)];
if strcmp(opt.mode,'P0')
    ro = struct('area_m2',Sref,'length_m',Lref,'moment_origin_m',org);
    t = offdesign_aero_v4(geometry.lo(:,:,1),geometry.lo(:,:,2),geometry.lo(:,:,3),geometry.up(:,:,1),geometry.up(:,:,2),geometry.up(:,:,3),M,al,ro,struct());
    out.CL_pressure = t.CL_pressure; out.CD_pressure = t.CD_pressure; out.CD_friction = 0; out.CD_base = 0; out.CL_friction = 0; out.CL_base = 0;
    out.CL = t.CL_pressure; out.CD_total = t.CD_pressure; out.Cm = t.Cm_pressure; out.source = 'frozen Teacher offdesign_aero_v4 (tangent cone, Cp=0 base, no friction)';
    out.detached_area_fraction = 0; return
end
G = v7_prepare_geometry(geometry.lo, geometry.up);
if G.foldback, out.valid = false; out.invalid_reasons{end+1} = 'strip_foldback'; end
nr = G.nr; nc = G.nc; N = G.n; A = G.A;
inc = -(N(:,:,1,:)*ed(1) + N(:,:,2,:)*ed(2) + N(:,:,3,:)*ed(3)); inc = squeeze(inc);   % [nr nc 4]
if any(abs(inc(:))>1+1e-12), error('v7:numeric','incidence outside [-1,1]'); end
th = asin(inc);
pr = ones(nr,nc,4); Tr = ones(nr,nc,4); rr = ones(nr,nc,4); Me = M*ones(nr,nc,4); p0r = ones(nr,nc,4); bad = false(nr,nc,4); vac = false(nr,nc,4);
switch opt.mode
    case 'P1'     % every panel referred to the free stream
        for b = 1:4
            t = th(:,:,b); c = t>1e-12; e = t<-1e-12;
            if any(c(:)), S = v7_flow('oblique', M*ones(sum(c(:)),1), t(c)); q = pr(:,:,b); q(c) = S.pr; pr(:,:,b) = q;
                q = Tr(:,:,b); q(c) = S.Tr; Tr(:,:,b) = q; q = rr(:,:,b); q(c) = S.rhor; rr(:,:,b) = q; q = Me(:,:,b); q(c) = S.M2; Me(:,:,b) = q; q = p0r(:,:,b); q(c) = S.p0r; p0r(:,:,b) = q;
                q = bad(:,:,b); q(c) = ~S.ok; bad(:,:,b) = q; end
            if any(e(:)), S = v7_flow('expand', M*ones(sum(e(:)),1), -t(e)); q = pr(:,:,b); q(e) = S.pr; pr(:,:,b) = q;
                q = Tr(:,:,b); q(e) = S.Tr; Tr(:,:,b) = q; q = rr(:,:,b); q(e) = S.rhor; rr(:,:,b) = q; q = Me(:,:,b); q(e) = S.M2; Me(:,:,b) = q;
                q = vac(:,:,b); q(e) = S.vac; vac(:,:,b) = q; end
        end
    case 'P2'     % strips along the grid lines, state carried from the previous panel (turn measured relative to the LOCAL flow direction = previous wall)
        for b = 1:4
            Ml = M*ones(nr,1); pl = ones(nr,1); Tl = ones(nr,1); rl = ones(nr,1); p0l = ones(nr,1); prev = zeros(nr,1); dead = false(nr,1); vc = false(nr,1);
            for j = 1:nc
                d = th(:,j,b)-prev; live = ~dead & ~vc;
                c = live & d>1e-12; e = live & d<-1e-12;
                k_pr = ones(nr,1); k_T = ones(nr,1); k_r = ones(nr,1); M2 = Ml; k_p0 = ones(nr,1);
                if any(c), S = v7_flow('oblique', Ml(c), d(c)); k_pr(c) = S.pr; k_T(c) = S.Tr; k_r(c) = S.rhor; M2(c) = S.M2; k_p0(c) = S.p0r;
                    ci = find(c); dead(ci(~S.ok)) = true; end
                if any(e), S = v7_flow('expand', Ml(e), -d(e)); k_pr(e) = S.pr; k_T(e) = S.Tr; k_r(e) = S.rhor; M2(e) = S.M2;
                    ei = find(e); vc(ei(S.vac)) = true; end
                pl = pl.*k_pr; Tl = Tl.*k_T; rl = rl.*k_r; p0l = p0l.*k_p0; Ml = M2; prev = th(:,j,b);   % d==0: state unchanged (never reset to the free stream)
                pr(:,j,b) = pl; Tr(:,j,b) = Tl; rr(:,j,b) = rl; Me(:,j,b) = Ml; p0r(:,j,b) = p0l; bad(:,j,b) = dead; vac(:,j,b) = vc;
            end
        end
    otherwise, error('v7:option','unknown mode');
end
bad = bad | ~isfinite(pr) | ~isfinite(Me);
Aall = sum(A,'all'); out.detached_area_fraction = sum(A(bad),'all')/Aall; out.vacuum_area_fraction = sum(A(vac),'all')/Aall;
if any(bad(:)), out.valid = false; out.invalid_reasons{end+1} = 'no_attached_weak_shock_or_subsonic'; end
if any(vac(:)), out.valid = false; out.invalid_reasons{end+1} = 'prandtl_meyer_vacuum'; end
out.total_pressure_ratio_min = min(p0r(~bad),[],'all'); out.Me_min = min(Me(~bad & ~vac),[],'all'); out.Me_max = max(Me(~bad & ~vac),[],'all');
Cp = 2*(pr-1)/(g*M^2);
dF = -reshape(Cp.*A,nr,nc,1,4).*N/Sref; CFp = reshape(sum(sum(sum(dF,1),2),4),1,3);           % pressure force coefficient vector (1x3)
Rr = G.R; CM = reshape(sum(sum(sum(cross(Rr-reshape(org,1,1,3,1), dF, 3),1),2),4),1,3)/Lref;
CFf = [0 0 0]; CFb = [0 0 0];
if ~strcmp(opt.friction,'none')
    if ~(isfield(flight,'T_inf_K') && isfield(flight,'rho_inf_kgm3')), error('v7:input','friction needs T_inf_K and rho_inf_kgm3'); end
    Tinf = flight.T_inf_K; rinf = flight.rho_inf_kgm3; a_inf = sqrt(g*287.05287*Tinf); Vinf = M*a_inf; qinf = 0.5*rinf*Vinf^2;
    Te = Tr*Tinf; re_ = rr*rinf; ae = sqrt(g*287.05287*Te); ue = Me.*ae; mu = @(T) 1.458e-6*T.^1.5./(T+110.4);
    Ts = Te.*(1+0.032*Me.^2+0.58*(opt.Tw_K./Te-1));                 % Eckert reference temperature
    rs = re_.*Te./Ts;                                               % rho* (isobaric)
    x = max(G.s,1e-9); Rex = rs.*ue.*x./mu(Ts);
    avg = @(p) ((max(G.s2,1e-9).^(1-p)-max(G.s1,1e-9).^(1-p))./((1-p)*max(G.s2-G.s1,1e-12)))./x.^(-p);   % exact cell average of a power-law Cf(x) (removes the 1/sqrt(x) midpoint error)
    cfl = 0.664./sqrt(Rex).*avg(0.5); cft = 0.0592./Rex.^0.2.*avg(0.2); cfs = zeros(size(Rex));   % incompressible-form local Cf at the reference state, cell-averaged
    switch opt.friction
        case 'laminar', cfs = cfl;
        case 'turbulent', cfs = cft;
        case 'transition', Ree = re_.*ue.*x./mu(Te); cfs = where3(Ree<opt.Re_tr, cfl, cft);
        otherwise, error('v7:option','unknown friction');
    end
    cfloc = cfs.*rs./re_;                                           % Cf referred to edge dynamic pressure
    tau = cfloc.*0.5.*re_.*ue.^2;                                   % wall shear (Pa)
    tau(bad|vac) = NaN;
    nd = N(:,:,1,:)*ed(1)+N(:,:,2,:)*ed(2)+N(:,:,3,:)*ed(3); nd = squeeze(nd);
    T3 = zeros(nr,nc,3,4);
    for k = 1:3, T3(:,:,k,:) = reshape(ed(k)-nd.*squeeze(N(:,:,k,:)),nr,nc,1,4); end
    tm = sqrt(sum(T3.^2,3)); out.friction_min_tangent = min(tm,[],'all'); T3 = T3./tm;     % surface flow direction = free-stream projection on the panel
    dFf = reshape(tau.*A,nr,nc,1,4).*T3/(qinf*Sref); CFf = reshape(sum(sum(sum(dFf,1),2),4),1,3);
    out.Re_x_max = max(Rex(~bad),[],'all'); out.Cf_local_mean = mean(cfloc(~bad),'all');
end
if ~isnan(opt.base_pressure_ratio)
    Cpb = 2*(opt.base_pressure_ratio-1)/(g*M^2); CFb = -Cpb*G.base_area*G.base_n/Sref;
end
out.CL_pressure = CFp*el'; out.CD_pressure = CFp*ed'; out.CL_friction = CFf*el'; out.CD_friction = CFf*ed'; out.CL_base = CFb*el'; out.CD_base = CFb*ed';
out.CL = out.CL_pressure+out.CL_friction+out.CL_base; out.CD_total = out.CD_pressure+out.CD_friction+out.CD_base; out.Cm = CM(2); out.Cm_base_excluded = true;
out.base_area_m2 = G.base_area; out.wetted_area_m2 = G.wetted_area; out.geometry_hash = G.hash;
out.cp = Cp; out.theta = th; out.Mach_edge = Me; out.pr = pr; out.panel_area = A;
if ~out.valid, for f = {'CL_pressure','CD_pressure','CL_friction','CD_friction','CL_base','CD_base','CL','CD_total','Cm'}, out.(f{1}) = NaN; end, end
out.source = ['strip/wedge engineering model ' opt.mode ' + friction ' opt.friction ' + base ' num2str(opt.base_pressure_ratio)];
end
function z = where3(c,a,b), z = b; z(c) = a(c); end
