function [x1,y1,p1,r1,v1,a1,v1_x,v1_y] = qiujie_shock(x_shock,r_shock,h_center_body,p0,a0,v0,r0,shock_angle)
g=1.4;
xishu=1;
g=1.4;xishu=1;
m0 = v0./sqrt(g*p0./r0);
for i=1
    x(1,i)=x_shock(i); y(1,i)=r_shock(i)+h_center_body;
end
for i=1
    bojiao(1,i)=shock_angle(1,i);%%%������
    p(1,i)=(2*g/(g+1)*(m0(i)*sin(bojiao(1,i)))^2-(g-1)/(g+1))*p0(i);%%%%�����ϵ�ѹ��
    r(1,i)=(g+1)*(m0(i)*sin(bojiao(1,i)))^2/(2+(g-1)*(m0(i)*sin(bojiao(1,i)))^2)*r0(i);%%%%�����ϵ��ܶ�
    pianzhuanjiao=bojiao(1,i)-atan(2*tan(bojiao(1,i))*(1/m0(i)^2/(sin(bojiao(1,i)))^2+(g-1)/2)/(g+1));%%%%�����ϵ�����ƫת��
    a(1,i)=pianzhuanjiao+a0(i);%%%%�����ϵ�����ƫת��,ʵ����Ϊ�ٶ�V��X��ļн�
    v(1,i)=sin(bojiao(1,i))/sin(bojiao(1,i)-pianzhuanjiao)*(2/(g+1)/(m0(i)*sin(bojiao(1,i)))^2+(g-1)/(g+1))*v0(i);%%%%�����ϵ��ٶ�
end
for i = 1
    x1 = x(1,i); y1 = y(1,i)-h_center_body;
    p1 = p(1,i); r1 = r(1,i); v1 = v(1,i); a1 = a(1,i);
    v1_x=v1(1,i)*cos(a1(1,i));v1_y=v1(1,i)*sin(a1(1,i));
end
end
