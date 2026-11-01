function out=offdesign_aero_v4(wav_x,wav_y,wav_z,wav_up_x,wav_up_y,wav_up_z,M,alpha,reference,options)
% V5 repaired selected v4 path. INPUT ANGLES RADIANS, pressure coefficients.
% Body x downstream,z up: native program -> body diag(1,-1,-1).
% Full mirrored surfaces; trailing base explicit Cp=0 approximation.
% Native QUAD area-vector integration preserved, not plotting resampling.
if ~isreal([M,alpha])||any(~isfinite([M,alpha]))||M<=1||abs(alpha)>=pi/2
    error('V5:InputInvalid','Invalid fixed-surface flight condition');
end
if reference.area_m2<=0||reference.length_m<=0;error('V5:InputInvalid','Invalid reference');end
g=1.4;ed=[cos(alpha),0,sin(alpha)];el=[-sin(alpha),0,cos(alpha)];
N=[];A=[];R=[];labels={};
for k=1:2
    if k==1;X=wav_x;Y=-wav_y;Z=-wav_z;sgn=-1;lab='lower';
    else;X=wav_up_x;Y=-wav_up_y;Z=-wav_up_z;sgn=1;lab='upper';end
    if any(~isfinite([X(:);Y(:);Z(:)]))||~isreal([X(:);Y(:);Z(:)])
        error('V5:InputInvalid','Invalid fixed surfaces');
    end
    if isfield(options,'mesh_stride')&&options.mesh_stride>1
        r=unique([1:options.mesh_stride:size(X,1),size(X,1)]);c=unique([1:options.mesh_stride:size(X,2),size(X,2)]);
        X=X(r,c);Y=Y(r,c);Z=Z(r,c);
    end
    for i=1:size(X,1)-1
        for j=1:size(X,2)-1
            p1=[X(i,j),Y(i,j),Z(i,j)];p2=[X(i+1,j),Y(i+1,j),Z(i+1,j)];
            p3=[X(i+1,j+1),Y(i+1,j+1),Z(i+1,j+1)];p4=[X(i,j+1),Y(i,j+1),Z(i,j+1)];
            v=sgn*cross(p3-p1,p2-p4);mag=norm(v);
            if mag<=1e-14;error('V5:GeometryDomain','Degenerate integration quad');end
            n=v/mag;centre=(p1+p2+p3+p4)/4;
            for mirror=[1,-1]
                N(end+1,:)=n.*[1,mirror,1];A(end+1,1)=mag/2;R(end+1,:)=centre.*[1,mirror,1];labels{end+1}=lab;
            end
        end
    end
end
incidence=-N*ed';
if any(abs(incidence)>1+1e-12);error('V5:NumericalDomain','Unit normal incidence outside [-1,1]');end
% No Mach-angle floor on WALL inclination. Zero incidence = freestream.
theta=asin(incidence);pr=ones(size(theta));branch=zeros(size(theta));
comp=theta>1e-9;expansion=theta< -1e-9;
% Bounded lookup of the exact same local tangent-cone component; pchip.
% Numerical lookup is explicit, validation compares 33 vs65 nodes separately.
lookup_n=33;if isfield(options,'cone_lookup_nodes');lookup_n=options.cone_lookup_nodes;end
if any(comp)
    knots=linspace(0,max(theta(comp)),lookup_n);values=ones(size(knots));
    for k=2:lookup_n
        [~,~,~,~,pc]=conical_flowfield(M,knots(k),g);values(k)=pc(end);
    end
    pr(comp)=interp1(knots,values,theta(comp),'pchip');branch(comp)=1;
end
K=M*theta(expansion);base=1+(g-1)/2*K;
if any(base<=0);error('V5:OutOfDomain','Hypersonic expansion similarity outside positive-base domain');end
pr(expansion)=base.^(2*g/(g-1));branch(expansion)=2;
if ~isreal(pr)||any(~isfinite(pr))||any(pr<=0);error('V5:NumericalDomain','Invalid pressure ratio');end
Cp=2*(pr-1)/(g*M^2);dF=-(Cp.*A).*N/reference.area_m2;CF=sum(dF,1);
origin=[0,0,0];if isfield(reference,'moment_origin_m');origin=reshape(reference.moment_origin_m,1,3);end
CM=sum(cross(R-origin,dF,2),1)/reference.length_m;
out=struct('status','PASS','CL_pressure',dot(CF,el),'CD_pressure',dot(CF,ed),'CY_pressure',CF(2), ...
    'pressure_moment_body',CM,'Cm_pressure',CM(2),'coefficient_force_body',CF, ...
    'pressure_ratio_min',min(pr),'pressure_ratio_max',max(pr),'Cp_min',min(Cp),'Cp_max',max(Cp), ...
    'branch_counts',struct('tangent_cone_compression',sum(branch==1),'hypersonic_expansion_similarity',sum(branch==2),'freestream',sum(branch==0),'modified_newtonian_fallback',0), ...
    'reference',reference,'model_version','teacher-offdesign-v4-repaired-v5.1','gamma_gas',g, ...
    'panel_count_full_body',numel(A),'cone_lookup_nodes',lookup_n,'base_assumption','Cp=0 on omitted trailing base', ...
    'missing_physics',{{'viscous_drag','bluntness','base_pressure','heating','validated_trim'}}, ...
    'moment_scope','pressure moment about declared origin only; NOT validated trim', ...
    'coordinate_convention','x downstream,z up; e_drag=[cos(alpha),0,sin(alpha)]; e_lift=[-sin(alpha),0,cos(alpha)]', ...
    'coefficient_convention','CF=-sum(Cp*A*n_out)/Sref; Cp=2*(p/pinf-1)/(gamma*M^2)');
end
