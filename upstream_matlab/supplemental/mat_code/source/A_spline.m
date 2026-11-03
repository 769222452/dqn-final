%%%%%%%%%%%%%%%%%三点控制贝塞尔曲线%%%%%%%%%%%%%
clc
clear
N = [1,-2,1; -2,2,0; 1,0,0];
p00 = [0,0]; p01 = [0,0.15]; p02 = [1,0.2];
P_X = [p00(1),p01(1),p02(1)];
P_Y = [p00(2),p01(2),p02(2)];
x_ctrl = P_X;
y_ctrl = P_Y;
u = linspace(0,1,10);
for i=1:10
    x(i)=[u(i)^2,u(i),1]*N*P_X';
    y(i)=[u(i)^2,u(i),1]*N*P_Y';
end
plot(x,y);
axis equal
hold on
plot(x_ctrl,y_ctrl,'r--','linewidth',1.5,'marker','*'); 
hold on