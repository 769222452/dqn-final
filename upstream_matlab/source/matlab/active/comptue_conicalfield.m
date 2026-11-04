function [index,beta,angles,vt,vr,info]=comptue_conicalfield(M,theta,gamma_gas)
% V5 bounded Taylor-Maccoll weak attached cone. ALL angles radians.
% Velocity normalized by sqrt(2*h0); explicit gamma in every RHS.
if ~isscalar(M)||~isscalar(theta)||~isscalar(gamma_gas)||~isreal([M,theta,gamma_gas])||any(~isfinite([M,theta,gamma_gas]))||M<=1||theta<=0||theta>=pi/2||gamma_gas<=1
    error('V5:InputInvalid','Cone needs M>1, 0<theta<pi/2 and gamma>1');
end
started=tic;mu=asin(1/M);grid=linspace(mu+1e-7,pi/2-1e-4,48);
% At exactly beta=Mach angle the shock vanishes and cone angle tends to 0.
% Use this analytic endpoint; imposing a finite beta excess wrongly excludes
% small WALL inclinations. This is not a floor on wall half-angle.
last_b=mu;last_f=-theta;lo=NaN;hi=NaN;valid_count=0;
for k=1:numel(grid)
    [wall,~,~,ok]=wall_solution(grid(k));
    if ok
        valid_count=valid_count+1;f=wall-theta;
        if isfinite(last_f)&&last_f<0&&f>=0;lo=last_b;hi=grid(k);break;end
        last_b=grid(k);last_f=f;
    end
end
if ~isfinite(lo)
    if valid_count>=4;error('V5:NoAttachedSolution','No weak attached cone solution in bounded shock scan');
    else;error('V5:RootBracketFailure','Cone shock scan failed');end
end
opts=optimset('Display','off','TolX',eps,'MaxIter',70,'MaxFunEvals',100);
[beta,residual,flag,fz]=fzero(@wall_residual,[lo hi],opts);
if flag<=0||abs(residual)>2e-7;error('V5:RootConvergenceFailure','Cone wall residual did not converge');end
[wall,angles,y,ok]=wall_solution(beta);
if ~ok||abs(wall-theta)>2e-7;error('V5:ODEIntegrationFailure','Cone final integration invalid');end
vr=y(:,1);vt=y(:,2);index=numel(angles)-1;
if nargout>5
    info=struct('beta_minus_mu_rad',beta-mu,'fzero_residual_rad',residual,'final_wall_residual_rad',wall-theta, ...
        'fzero_iterations',fz.iterations,'fzero_funcCount',fz.funcCount,'bracket',[lo,hi], ...
        'bracket_lo_is_mach_angle',lo==mu,'scan_valid_count',valid_count,'elapsed_s',toc(started));
end
    function f=wall_residual(b)
        if b==mu;f=-theta;return;end
        [a,~,~,good]=wall_solution(b);
        if ~good;error('V5:ODEIntegrationFailure','Cone bracket integration failed');end
        f=a-theta;
    end
    function [wall,t,y,ok]=wall_solution(b)
        if toc(started)>8;error('V5:NumericalTimeout','Cone component exceeded 8 s');end
        Mn=M*sin(b);rr=(gamma_gas+1)*Mn^2/(2+(gamma_gas-1)*Mn^2);
        vinf=sqrt((gamma_gas-1)*M^2/(2+(gamma_gas-1)*M^2));y0=[vinf*cos(b);-vinf*sin(b)/rr];
        % Behind a near-Mach-angle shock vt^2-a^2 starts at -margin, so the
        % Taylor-Maccoll RHS is 0/0-like across a layer of width ~layer. The
        % default ode45 first step jumps over it and the shooting wall angle
        % becomes discontinuous in b. Start inside the layer; the equations,
        % tolerances and acceptance tests are unchanged.
        margin=(gamma_gas-1)/2*(1-sum(y0.^2))-y0(2)^2;
        layer=max(margin,0)/max(2*vinf^2*sin(b)*cos(b),1e-3);
        h0=min(1e-4,max(0.05*layer,64*eps(b)));
        optsode=odeset('RelTol',2e-10,'AbsTol',1e-12,'MaxStep',.01,'InitialStep',h0,'Events',@wall_event,'OutputFcn',@time_guard);
        [t,y,te,ye,ie]=ode45(@(a,u)rhs(a,u,gamma_gas),[b,1e-7],y0,optsode);
        ok=~isempty(te)&&ie(end)==1&&isreal(y)&&all(isfinite(y(:)));
        if ok;wall=te(end);t(end)=wall;y(end,:)=ye(end,:);else;wall=NaN;end
    end
    function [value,isterminal,direction]=wall_event(~,y)
        a2=(gamma_gas-1)/2*(1-sum(y.^2));value=[y(2);a2-y(2)^2-1e-10];isterminal=[1;1];direction=[1;-1];
    end
    function stop=time_guard(~,~,~)
        stop=0;if toc(started)>8;error('V5:NumericalTimeout','Cone component exceeded 8 s');end
    end
end
function dy=rhs(a,y,gamma_gas)
a2=(gamma_gas-1)/2*(1-sum(y.^2));den=y(2)^2-a2;
if ~isreal(y)||any(~isfinite(y))||a2<=0||abs(den)<1e-15
    error('V5:NumericalDomain','Taylor-Maccoll sonic or energy singularity');
end
dy=[y(2);(a2*(2*y(1)+y(2)/tan(a))-y(2)^2*y(1))/den];
end
