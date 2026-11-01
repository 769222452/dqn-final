function varargout = v7_flow(op, varargin)
% Exact perfect-gas relations used by offdesign_aero_v7 (vectorised, no silent clamping).
%   beta_max, theta_of_beta, [beta,ok]=beta_weak(M,theta), S=oblique(M,theta), nu(M), [M,vac]=nu_inv(v), S=expand(M,dtheta), p0_normal(Mn)
g = 1.4;
switch op
    case 'beta_max'
        M = varargin{1}; s2 = ((g+1)*M.^2/4 - 1 + sqrt((g+1)*(1+(g-1)/2*M.^2+(g+1)*M.^4/16)))./(g*M.^2);
        varargout{1} = asin(sqrt(min(max(s2,0),1)));
    case 'theta_of_beta'
        M = varargin{1}; b = varargin{2};
        varargout{1} = atan(2./tan(b).*(M.^2.*sin(b).^2-1)./(M.^2.*(g+cos(2*b))+2));
    case 'beta_weak'
        [M,th] = deal(varargin{1}, varargin{2}); [M,th] = v7_bc(M,th);
        lo = asin(1./M)+1e-13; hi = v7_flow('beta_max',M);
        ok = M>1 & th>=0 & th <= v7_flow('theta_of_beta',M,hi);
        for it=1:90
            mid = (lo+hi)/2; f = v7_flow('theta_of_beta',M,mid)-th;
            hi = where(f>0,mid,hi); lo = where(f>0,lo,mid);
        end
        b = (lo+hi)/2; b(~ok) = NaN; varargout{1} = b; varargout{2} = ok;
    case 'oblique'      % weak attached oblique shock; NaN fields + ok=false when no attached weak solution
        [M,th] = deal(varargin{1}, varargin{2}); [M,th] = v7_bc(M,th);
        [b,ok] = v7_flow('beta_weak',M,th);
        mn1 = M.*sin(b); mn2 = sqrt((1+(g-1)/2*mn1.^2)./(g*mn1.^2-(g-1)/2));
        S.beta = b; S.ok = ok; S.pr = 1+2*g/(g+1)*(mn1.^2-1);
        S.rhor = (g+1)*mn1.^2./((g-1)*mn1.^2+2); S.Tr = S.pr./S.rhor; S.M2 = mn2./sin(b-th);
        S.p0r = v7_flow('p0_normal',mn1);
        S.ok = ok & isfinite(S.M2) & S.M2>1;      % subsonic downstream state is declared invalid, not repaired
        varargout{1} = S;
    case 'p0_normal'
        mn = varargin{1};
        varargout{1} = ((g+1)*mn.^2./((g-1)*mn.^2+2)).^(g/(g-1)).*((g+1)./(2*g*mn.^2-(g-1))).^(1/(g-1));
    case 'nu'
        M = varargin{1}; k = sqrt((g+1)/(g-1));
        varargout{1} = k*atan(sqrt((M.^2-1)/k^2)) - atan(sqrt(M.^2-1));
    case 'nu_inv'       % bisection in log10(M); returns Inf (vacuum) at/above nu_max
        v = varargin{1}; numax = pi/2*(sqrt((g+1)/(g-1))-1); vac = v >= numax*(1-1e-12);
        lo = zeros(size(v)); hi = 4*ones(size(v));
        for it=1:100
            mid = (lo+hi)/2; f = v7_flow('nu',10.^mid)-v;
            hi = where(f>0,mid,hi); lo = where(f>0,lo,mid);
        end
        M = 10.^((lo+hi)/2); M(vac) = Inf; M(v<0) = NaN; varargout{1} = M; varargout{2} = vac;
    case 'expand'       % isentropic Prandtl-Meyer turn (dtheta>0 expansion); exact nu inversion
        [M,dth] = deal(varargin{1}, varargin{2}); [M,dth] = v7_bc(M,dth);
        [M2,vac] = v7_flow('nu_inv', v7_flow('nu',M)+dth);
        t = (1+(g-1)/2*M.^2)./(1+(g-1)/2*M2.^2); t(vac) = 0;
        S.M2 = M2; S.vac = vac; S.Tr = t; S.pr = t.^(g/(g-1)); S.rhor = t.^(1/(g-1)); S.p0r = ones(size(M)); S.ok = isfinite(M2) | vac;
        varargout{1} = S;
    otherwise, error('v7_flow:op','unknown op %s',op);
end
end
function [a,b] = v7_bc(a,b)
if isscalar(a), a = a*ones(size(b)); end
if isscalar(b), b = b*ones(size(a)); end
end
function z = where(c,a,b)
z = b; z(c) = a(c);
end
