function [beta,velocity_x,velocity_y,mf_cone,pf_cone]=conical_flowfield(M,theta,gamma_gas)
% V5 radians end-to-end. Teacher Taylor-Maccoll pressure approach, bounded.
[~,beta,angles,vt,vr]=comptue_conicalfield(M,theta,gamma_gas);
v2=vr.^2+vt.^2;
if any(v2>=1)||any(v2<0);error('V5:NumericalDomain','Invalid cone energy');end
mf_cone=sqrt(2*v2./((gamma_gas-1)*(1-v2)));
Mn=M*sin(beta);pr_shock=1+2*gamma_gas/(gamma_gas+1)*(Mn^2-1);
p0ratio=pr_shock*(1+(gamma_gas-1)/2*mf_cone(1)^2)^(gamma_gas/(gamma_gas-1));
pf_cone=p0ratio./(1+(gamma_gas-1)/2*mf_cone.^2).^(gamma_gas/(gamma_gas-1));
velocity_x=vr.*cos(angles)-vt.*sin(angles);velocity_y=vr.*sin(angles)+vt.*cos(angles);
if ~isreal(pf_cone)||any(~isfinite(pf_cone))||any(pf_cone<=0)
    error('V5:NumericalDomain','Invalid cone pressure');
end
end
