%%%%%%%%%%%%Ë«¶þ´Î±´Èû¶ûÇúÃæ%%%%%%%%%%%
clc
clear
N = [1,-2,1;-2,2,0;1,0,0];
p00=[0,-1,0]; p01=[0,0,1]; p02=[0,1,0];
p10=[2,-1.3,0]; p11=[2,0,2]; p12=[2,1.3,0];
p20=[4,-1.8,0]; p21=[4,0,3.5]; p22=[4,1.8,0];
P_X=[p00(1),p01(1),p02(1);p10(1),p11(1),p12(1);p20(1),p21(1),p22(1)];
P_Y=[p00(2),p01(2),p02(2);p10(2),p11(2),p12(2);p20(2),p21(2),p22(2)];
P_Z=[p00(3),p01(3),p02(3);p10(3),p11(3),p12(3);p20(3),p21(3),p22(3)];
x_ctrl = P_X;
y_ctrl = P_Y;
z_ctrl = P_Z;
u = linspace(0,1,100); v = linspace(0,1,100);
for i = 1:100
    for j = 1:100
        x(j,i) = [u(i)^2,u(i),1]*N*P_X*N'*[v(j)^2,v(j),1]';
        y(j,i) = [u(i)^2,u(i),1]*N*P_Y*N'*[v(j)^2,v(j),1]';
        z(j,i) = [u(i)^2,u(i),1]*N*P_Z*N'*[v(j)^2,v(j),1]';
    end
end
mesh(x,y,z,'edgecolor','k','facecolor','none')
axis equal
hold on
mesh(x_ctrl,y_ctrl,z_ctrl,'Facecolor','none','Edgecolor','r','LineWidth',1.5,'linestyle','--','Marker','*','Markerfacecolor','r','Markersize',8)
axis equal
hold on
