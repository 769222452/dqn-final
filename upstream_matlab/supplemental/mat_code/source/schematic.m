
afa = 0:2*pi/99:2*pi;U = linspace(0,1,100);
%% 第一道激波shock_1
A_1 = 0; D_1 = 2; a_1 = 0.4; b_1 = 0.35; 
for i = 1:100
    for j = 1:100
        x_shock1(i,j) = (1-U(i))*A_1+U(i)*D_1;
        y_shock1(i,j) = U(i)*a_1*D_1*cos(afa(j));
        z_shock1(i,j) = U(i)*b_1*D_1*sin(afa(j));
    end
end
%% 第二道激波shock_2
A_2 = 1; D_2 = 2; a_2 = 0.4; b_2 = 0.35;
for i = 1:100
    for j = 1:100
        x_shock2(i,j) = (1-U(i))*A_2+U(i)*D_2;
        y_shock2(i,j) = U(i)*a_2*D_2*cos(afa(j));
        z_shock2(i,j) = U(i)*b_2*D_2*sin(afa(j));
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
set(gcf,'position',[0,0,1000,1000])
set(0,'defaultfigurecolor','w')
angle=[41,42];
% map = [1,0,0;0,1,0];
h1 = surf(x_shock1,y_shock1,z_shock1);
colormap([])
shading interp
alpha(0.1);
axis tight
axis equal 
hold on
h2 = surf(x_shock2,y_shock2,z_shock2);
shading interp
colormap(map)
alpha(0.2)
axis tight
axis equal 
hold on
set(gca,'xtick',[],'xticklabel',[],'xcolor','w')
set(gca,'ytick',[],'yticklabel',[],'ycolor','w')
set(gca,'ztick',[],'zticklabel',[],'zcolor','w')
view(angle);