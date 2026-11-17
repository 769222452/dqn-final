function varargout=v7_flow_fast(op,varargin)
% Identical perfect-gas equations/branches to frozen v7_flow.m.
% Only numerical root iteration is accelerated. Original bounds preserved.
g=1.4;
switch op
 case 'beta_max'
  M=varargin{1};s2=((g+1)*M.^2/4-1+sqrt((g+1)*(1+(g-1)/2*M.^2+(g+1)*M.^4/16)))./(g*M.^2);
  varargout{1}=asin(sqrt(min(max(s2,0),1)));
 case 'theta_of_beta'
  M=varargin{1};b=varargin{2};varargout{1}=atan(2./tan(b).*(M.^2.*sin(b).^2-1)./(M.^2.*(g+cos(2*b))+2));
 case 'beta_weak'
  [M,th]=bc(varargin{1},varargin{2});lo=asin(1./M)+1e-13;hi=v7_flow_fast('beta_max',M);
  maxth=v7_flow_fast('theta_of_beta',M,hi);ok=M>1&th>=0&th<=maxth;
  b=lo+(hi-lo).*min(max(th./maxth,0),1);active=ok;
  for it=1:60
   if ~any(active(:)),break;end
   ix=find(active);m=M(ix);x=b(ix);A=m.^2;sn=sin(x);ct=1./tan(x);
   N=2*ct.*(A.*sn.^2-1);D=A.*(g+cos(2*x))+2;
   f=atan(N./D)-th(ix);
   dn=2*(-(A.*sn.^2-1)./sn.^2+ct.*A.*sin(2*x));dd=-2*A.*sin(2*x);
   df=(dn.*D-N.*dd)./(D.^2+N.^2);
   done=abs(f)<2e-15 | (hi(ix)-lo(ix))<4e-15;
   active(ix(done))=false;ix=ix(~done);f=f(~done);df=df(~done);x=x(~done);
   above=f>0;hi(ix(above))=x(above);lo(ix(~above))=x(~above);
   next=x-f./df;bad=~isfinite(next)|next<=lo(ix)|next>=hi(ix);
   next(bad)=(lo(ix(bad))+hi(ix(bad)))/2;b(ix)=next;
  end
  if any(active(:))
   ix=find(active);m=M(ix);l=lo(ix);h=hi(ix);q=th(ix);
   for it=1:90,mid=(l+h)/2;f=v7_flow_fast('theta_of_beta',m,mid)-q;h(f>0)=mid(f>0);l(f<=0)=mid(f<=0);end
   b(ix)=(l+h)/2;
  end
  b(~ok)=NaN;varargout{1}=b;varargout{2}=ok;
 case 'oblique'
  [M,th]=bc(varargin{1},varargin{2});[b,ok]=v7_flow_fast('beta_weak',M,th);
  mn1=M.*sin(b);mn2=sqrt((1+(g-1)/2*mn1.^2)./(g*mn1.^2-(g-1)/2));
  S.beta=b;S.ok=ok;S.pr=1+2*g/(g+1)*(mn1.^2-1);
  S.rhor=(g+1)*mn1.^2./((g-1)*mn1.^2+2);S.Tr=S.pr./S.rhor;S.M2=mn2./sin(b-th);
  S.p0r=v7_flow_fast('p0_normal',mn1);S.ok=ok&isfinite(S.M2)&S.M2>1;varargout{1}=S;
 case 'p0_normal'
  mn=varargin{1};varargout{1}=((g+1)*mn.^2./((g-1)*mn.^2+2)).^(g/(g-1)).*((g+1)./(2*g*mn.^2-(g-1))).^(1/(g-1));
 case 'nu'
  M=varargin{1};k=sqrt((g+1)/(g-1));varargout{1}=k*atan(sqrt((M.^2-1)/k^2))-atan(sqrt(M.^2-1));
 case 'nu_inv'
  v=varargin{1};numax=pi/2*(sqrt((g+1)/(g-1))-1);vac=v>=numax*(1-1e-12);
  lo=zeros(size(v));hi=4*ones(size(v));q=log10(min(max(5./max(numax-v,1e-20),1),1e4));
  capped=v>=v7_flow_fast('nu',1e4);active=~vac&v>0&~capped;
  q(v<=0)=0;q(capped)=4;
  for it=1:60
   if ~any(active(:)),break;end
   ix=find(active);x=q(ix);m=10.^x;f=v7_flow_fast('nu',m)-v(ix);
   df=log(10)*sqrt(m.^2-1)./(1+(g-1)/2*m.^2);
   done=abs(f)<2e-15 | (hi(ix)-lo(ix))<4e-15;
   active(ix(done))=false;ix=ix(~done);f=f(~done);df=df(~done);x=x(~done);
   above=f>0;hi(ix(above))=x(above);lo(ix(~above))=x(~above);
   next=x-f./df;bad=~isfinite(next)|next<=lo(ix)|next>=hi(ix);
   next(bad)=(lo(ix(bad))+hi(ix(bad)))/2;q(ix)=next;
  end
  if any(active(:))
   ix=find(active);l=lo(ix);h=hi(ix);target=v(ix);
   for it=1:100,mid=(l+h)/2;f=v7_flow_fast('nu',10.^mid)-target;h(f>0)=mid(f>0);l(f<=0)=mid(f<=0);end
   q(ix)=(l+h)/2;
  end
  M=10.^q;M(vac)=Inf;M(v<0)=NaN;varargout{1}=M;varargout{2}=vac;
 case 'expand'
  [M,dth]=bc(varargin{1},varargin{2});[M2,vac]=v7_flow_fast('nu_inv',v7_flow_fast('nu',M)+dth);
  t=(1+(g-1)/2*M.^2)./(1+(g-1)/2*M2.^2);t(vac)=0;
  S.M2=M2;S.vac=vac;S.Tr=t;S.pr=t.^(g/(g-1));S.rhor=t.^(1/(g-1));S.p0r=ones(size(M));S.ok=isfinite(M2)|vac;varargout{1}=S;
 otherwise,error('v7_flow_fast:op','unknown op');
end
end
function [a,b]=bc(a,b)
if isscalar(a),a=a*ones(size(b));end
if isscalar(b),b=b*ones(size(a));end
end
