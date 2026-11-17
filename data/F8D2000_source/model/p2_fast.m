function out=p2_fast(lo,up,Ms,alphas)
% Same P2 panel/strip equations as the frozen original implementation.
% v7_flow_fast changes numerical root iteration only; validated against v7_flow.
% Batch independent flight points; mirror x/z forces analytically; cache exact upper geometry.
% Pressure only. Common reference area/origin, no friction or base pressure.
persistent cached_up cached_M cached_alpha upper_result;
Ms=Ms(:)';alphas=alphas(:)';assert(numel(Ms)==numel(alphas));
if isempty(cached_up)||~isequal(up,cached_up)||~isequal(Ms,cached_M)||~isequal(alphas,cached_alpha)
 upper_result=skin(up,1,Ms,alphas);cached_up=up;cached_M=Ms;cached_alpha=alphas;
end
lower_result=skin(lo,-1,Ms,alphas);u=upper_result;l=lower_result;
Fx=2*(u.Fx+l.Fx);Fz=2*(u.Fz+l.Fz);Cm=2*(u.Cm+l.Cm);
valid=u.valid&l.valid;
CL=-Fx.*sin(alphas)+Fz.*cos(alphas);CD=Fx.*cos(alphas)+Fz.*sin(alphas);
CL(~valid)=NaN;CD(~valid)=NaN;Cm(~valid)=NaN;
out=struct('CL',CL(:),'CD_pressure',CD(:),'Cm_pressure',Cm(:),'model_valid',valid(:),...
 'detached_area_fraction',((u.bad_area+l.bad_area)/(u.area+l.area))',...
 'vacuum_area_fraction',((u.vac_area+l.vac_area)/(u.area+l.area))');
end

function o=skin(native,sgn,Ms,alpha)
P=native.*reshape([1 -1 -1],1,1,3);
p1=P(1:end-1,1:end-1,:);p2=P(2:end,1:end-1,:);p3=P(2:end,2:end,:);p4=P(1:end-1,2:end,:);
v=sgn*cross(p3-p1,p2-p4,3);mag=sqrt(sum(v.^2,3));assert(all(mag(:)>1e-14));
normal=v./mag;area=mag/2;centre=(p1+p2+p3+p4)/4;
nr=size(area,1);nc=size(area,2);nf=numel(Ms);Sref=6.931920786884801;Lref=5;origin=[2.5 0 -0.9481139754931033];
mid=(native(1:end-1,:,:)+native(2:end,:,:))/2;foldback=any(diff(mid(:,:,1),1,2)<-1e-12,'all');
Ml=repmat(Ms,nr,1);pl=ones(nr,nf);prev=zeros(nr,nf);dead=false(nr,nf);vc=dead;
Fx=zeros(1,nf);Fz=Fx;Cm=Fx;badarea=Fx;vacarea=Fx;
for j=1:nc
 th=asin(-normal(:,j,1)*cos(alpha)-normal(:,j,3)*sin(alpha));
 d=th-prev;live=~dead&~vc;c=live&d>1e-12;e=live&d<-1e-12;
 kpr=ones(nr,nf);M2=Ml;
 if any(c,'all')
  s=v7_flow_fast('oblique',Ml(c),d(c));kpr(c)=s.pr;M2(c)=s.M2;ix=find(c);dead(ix(~s.ok))=true;
 end
 if any(e,'all')
  s=v7_flow_fast('expand',Ml(e),-d(e));kpr(e)=s.pr;M2(e)=s.M2;ix=find(e);vc(ix(s.vac))=true;
 end
 pl=pl.*kpr;Ml=M2;prev=th;
 bad=dead|~isfinite(pl)|~isfinite(Ml);
 Cp=2*(pl-1)./(1.4*Ms.^2);
 dFx=-Cp.*(area(:,j).*normal(:,j,1))/Sref;dFz=-Cp.*(area(:,j).*normal(:,j,3))/Sref;
 Fx=Fx+sum(dFx,1);Fz=Fz+sum(dFz,1);
 Cm=Cm+sum((centre(:,j,3)-origin(3)).*dFx-(centre(:,j,1)-origin(1)).*dFz,1)/Lref;
 badarea=badarea+sum(bad.*area(:,j),1);vacarea=vacarea+sum(vc.*area(:,j),1);
end
o=struct('Fx',Fx,'Fz',Fz,'Cm',Cm,'valid',~any(dead|vc|~isfinite(pl)|~isfinite(Ml),1)&~foldback,...
 'bad_area',badarea,'vac_area',vacarea,'area',sum(area,'all'));
end
