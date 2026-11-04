function [Sb] = calculate_H(ru,rv,nu,nv,v_D)
%UNTITLED4 此处提供此函数的摘要
%   此处提供详细说明
% syms u v
% ru=[xu,yu,zu]/norm([xu,yu,zu]);
% rv=[xv,yv,zv]/norm([xv,yv,zv]);
n=cross(ru,rv)/norm(cross(ru,rv));
% v_D = [n(2)^2+n(3)^2,-n(1)*n(2),-n(1)*n(3)];
% v_D=v_D/norm(v_D);
l=cross(n,v_D)/norm(cross(n,v_D));
ab=[ru',rv']\v_D';
du=ab(1);
dv=ab(2);
ab=[ru',rv']\l';
dul=ab(1);
dvl=ab(2);
% du=dot(ru/norm(ru),v_D)/norm(ru);
% dv=dot(rv/norm(rv),v_D)/norm(rv);
% dul=dot(ru/norm(ru),l)/norm(ru);
% dvl=dot(rv/norm(rv),l)/norm(rv);
% nu=[diff(n(1),u),diff(n(2),u),diff(n(3),u)];
% nv=[diff(n(1),v),diff(n(2),v),diff(n(3),v)];
E=dot(ru,ru);
F=dot(ru,rv);
G=dot(rv,rv);
L=-dot(ru,nu);
M=-dot(ru,nv);
N=-dot(rv,nv);
H=(L*G-2*M*F+N*E)/(E*G-F^2);
Sa=(L*du^2+2*M*du*dv+N*dv^2)/(E*du^2+2*F*du*dv+G*dv^2);
Sb1=H-Sa;
Sb=(L*dul^2+2*M*dul*dvl+N*dvl^2)/(E*dul^2+2*F*dul*dvl+G*dvl^2);
end