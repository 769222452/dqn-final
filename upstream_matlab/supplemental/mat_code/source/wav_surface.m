clc
clear
load x_y_z_xingxian
[a,b,c] = swc;
Y_shock = linspace(x_y_z_xingxian (1,2,1),x_y_z_xingxian (1,2,40),20);
X_shock = 1.3*ones(size(Y_shock));
Z_shock = -(a*Y_shock.^2+b*Y_shock+c);
%% 输入激波曲面参数
X_wav(1,1:20)= x_y_z_xingxian (1,1,1);
Y_wav(1,1:20)= x_y_z_xingxian (1,2,1);
Z_wav(1,1:20)= x_y_z_xingxian (1,3,1);
for i = 2:40
    X_wav(i,1:20) = linspace(x_y_z_xingxian(1,1,i),x_y_z_xingxian(1,1,1),20);
    for j = 1:20
        Y_wav(i,j) = interp1(x_y_z_xingxian(:,1,i),x_y_z_xingxian(:,2,i),X_wav(i,j));
        Z_wav(i,j) = interp1(x_y_z_xingxian(:,1,i),x_y_z_xingxian(:,3,i),X_wav(i,j));
    end
end
X_wav(41,1:20)= x_y_z_xingxian (1,1,40);
Y_wav(41,1:20)= x_y_z_xingxian (1,2,40);
Z_wav(41,1:20)= x_y_z_xingxian (1,3,40);
Z_wav = -Z_wav;
X_up = X_wav;
Y_up = Y_wav;
Z_up = -0.09*ones(size(Z_wav));
% colormap([0,0,0]);
% set(gca,'xtick',[],'xticklabel',[],'xcolor','w')
% set(gca,'ytick',[],'yticklabel',[],'ycolor','w')
% set(gca,'ztick',[],'zticklabel',[],'zcolor','w')
figure;
set(gcf,'position',[0,0,1000,1000])
set(0,'defaultfigurecolor','w')
angle={[0 90],[-45,35],[0,0],[-90,0]};
for i = 1
subplot(2,2,i); 
mesh(X_up,Y_up,Z_up,'LineWidth',0.3,'EdgeColor','black');
axis tight
axis equal
hold on
mesh(X_wav,Y_wav,Z_wav,'LineWidth',0.3,'EdgeColor','black');
axis tight
axis equal
hold on
hidden off
set(gca,'xtick',[],'xticklabel',[],'xcolor','w')
set(gca,'ytick',[],'yticklabel',[],'ycolor','w')
set(gca,'ztick',[],'zticklabel',[],'zcolor','w')
view(angle{i});
end
for i = 2
subplot(2,2,i); 
mesh(X_up,Y_up,Z_up,'LineWidth',0.3,'EdgeColor','black');
axis tight
axis equal
hold on
mesh(X_wav,Y_wav,Z_wav,'LineWidth',0.3,'EdgeColor','black');
axis tight
axis equal
hold on
hidden off
set(gca,'xtick',[],'xticklabel',[],'xcolor','w')
set(gca,'ytick',[],'yticklabel',[],'ycolor','w')
set(gca,'ztick',[],'zticklabel',[],'zcolor','w')
view(angle{i});
end
for i = 3
subplot(2,2,i); 
mesh(X_up,Y_up,Z_up,'LineWidth',0.3,'EdgeColor','black');
axis tight
axis equal
hold on
mesh(X_wav,Y_wav,Z_wav,'LineWidth',0.3,'EdgeColor','black');
axis tight
axis equal
hold on
hidden off
set(gca,'xtick',[],'xticklabel',[],'xcolor','w')
set(gca,'ytick',[],'yticklabel',[],'ycolor','w')
set(gca,'ztick',[],'zticklabel',[],'zcolor','w')
view(angle{i});
end
for i = 4
subplot(2,2,i); 
mesh(X_up,Y_up,Z_up,'LineWidth',0.3,'EdgeColor','black');
axis tight
axis equal
hold on
mesh(X_wav,Y_wav,Z_wav,'LineWidth',0.3,'EdgeColor','black');
axis tight
axis equal
hold on
hidden off
plot3(X_shock,Y_shock,Z_shock,'k--','LineWidth',0.3)
axis tight
axis equal
hold on
set(gca,'xtick',[],'xticklabel',[],'xcolor','w')
set(gca,'ytick',[],'yticklabel',[],'ycolor','w')
set(gca,'ztick',[],'zticklabel',[],'zcolor','w')
view(angle{i});
end
% subplot(2,2,1);
% set(gca,'position',[0,0.4,0.4,0.4]);
subplot(2,2,2);
set(gca,'position',[0.47,0.4,0.5,0.5]);
% set(gca,'xtick',[],'xticklabel',[],'xcolor','w')
% set(gca,'ytick',[],'yticklabel',[],'ycolor','w')
% set(gca,'ztick',[],'zticklabel',[],'zcolor','w')
% box on
