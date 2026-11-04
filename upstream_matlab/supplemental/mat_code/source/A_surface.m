%%%%%%%%%%%%二阶贝塞尔曲线与直纹面的耦合%%%%%%%%%%%%%
clc
clear
P = [0 -1 0; 
     0 0 1;
     0 1 0];
Q = [2.5 -2 0; 
     2.5 0 2;
     2.5 2 0];
u = linspace(0,1,100); v = linspace(0,1,100);
x_ctrl = [P(1,1),P(2,1),P(3,1);Q(1,1),Q(2,1),Q(3,1)];
y_ctrl = [P(1,2),P(2,2),P(3,2);Q(1,2),Q(2,2),Q(3,2)];
z_ctrl = [P(1,3),P(2,3),P(3,3);Q(1,3),Q(2,3),Q(3,3)];
for i = 1:100
    for j = 1:100
        x(j,i) = (1-u(i))*((1-v(j))^2*P(1,1)+2*v(j)*(1-v(j))*P(2,1)+v(j)^2*P(3,1))+u(i)*((1-v(j))^2*Q(1,1)+2*v(j)*(1-v(j))*Q(2,1)+v(j)^2*Q(3,1));
        y(j,i) = (1-u(i))*((1-v(j))^2*P(1,2)+2*v(j)*(1-v(j))*P(2,2)+v(j)^2*P(3,2))+u(i)*((1-v(j))^2*Q(1,2)+2*v(j)*(1-v(j))*Q(2,2)+v(j)^2*Q(3,2));
        z(j,i) = (1-u(i))*((1-v(j))^2*P(1,3)+2*v(j)*(1-v(j))*P(2,3)+v(j)^2*P(3,3))+u(i)*((1-v(j))^2*Q(1,3)+2*v(j)*(1-v(j))*Q(2,3)+v(j)^2*Q(3,3));
    end
end
mesh(x,y,z,'edgecolor','k','facecolor','none')
axis equal
hold on
mesh(x_ctrl,y_ctrl,z_ctrl,'Facecolor','none','Edgecolor','r','LineWidth',1.5,'linestyle','--','Marker','*','Markerfacecolor','r','Markersize',8)
axis equal
hold on
