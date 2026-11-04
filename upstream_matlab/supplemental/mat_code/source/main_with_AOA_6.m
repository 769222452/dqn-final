% 通过旋转激波来处理，椭圆锥激波
% 乘波体分为两个部分，前者保证激波，后者保证体积+重构各流面内的网格
clc
clear
M = 50;
N_l = 20;
format short
tic
%%%%%%%%%%%%%%%%%%%%%%%%来流条件%%%%%%%%%%%%%%%%%%%%%%%%%%
global p0
global r0
global m0
global v0
global a0
% 定义用于指定各流面内的沿程压力分布
global a_p
global b_p
global c_p
global d_p
global x_p
p0=1880;r0=0.02929783;m0=6;v0=m0*sqrt(1.4*p0/r0); a0=0;
AOA=4; Vector_V=[cosd(AOA) 0 sind(AOA)]'; % AOA为攻角，AOS为侧滑角，Vector_V表示来流方向
rat=rotate_matrix(Vector_V,[1 0 0]'); % rat为旋转矩阵，rat*a=b, inv（rat）*b=a；
a_p=0; b_p=0; c_p=-0.1; % 压力梯度的控制方程为y=a_p*(x-x_p)^3+b_p*(x-x_p)^2+c_p*(x-x_p)+d_p
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%激波形状%%%%%%%%%%%%%%%%%%%%%%%%%
%%绕Z轴旋转
x_design = 4; angle_shock_a=16; angle_shock_b=16;
b1=0.2; a1=b1/(tand(angle_shock_b)/tand(angle_shock_a));
u = linspace(0,1,100); v = linspace(0,pi/2,100);
for i = 1:100
    for j = 1:100
        x(j,i)=u(i)*x_design;
        y(j,i)=(u(i)*x_design*tand(angle_shock_a)+a1)*cos(v(j));
        z(j,i)=(u(i)*x_design*tand(angle_shock_b)+b1)*sin(v(j));        
        x_y_z_xuanzhuan=rat*[x(j,i),y(j,i),z(j,i)]';
        x_xuanzhuan(j,i)=x_y_z_xuanzhuan(1); y_xuanzhuan(j,i)=x_y_z_xuanzhuan(2); z_xuanzhuan(j,i)=x_y_z_xuanzhuan(3);
    end
end
figure(1)
mesh(x,y,z,'edgecolor','k','facecolor','none') % 旋转前激波形状
axis equal
hold on
% mesh(x_xuanzhuan,y_xuanzhuan,z_xuanzhuan,'edgecolor','k','facecolor','none') % 旋转后激波方向
% axis equal
% hold on
fid = fopen('SWC.shape-UG.dat','wt');
for i = 1:100
    fprintf(fid,'ROW\n');
    for j = 1:100
        fprintf(fid,'%e\t',x(i,j)*1000);
        fprintf(fid,'%e\t',y(i,j)*1000);
        fprintf(fid,'%e\n',z(i,j)*1000);
    end  
end
fid = fopen('SWC.shapae.dat','wt');
fprintf(fid,'Title=”SWC mesh”\n'); 
fprintf(fid,'Variables=“X”,“Y”,“Z”\n'); 
fprintf(fid,'Zone I=100,J=100,f=point\n'); 
for i = 1:100
    for j = 1:100
        fprintf(fid,'%e\t',x(i,j));
        fprintf(fid,'%e\t',y(i,j));
        fprintf(fid,'%e\n',-z(i,j));
    end  
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 此处表示的是旋转后的激波方程及曲率方向等，计算时使用的都是旋转后
% 旋转前x_v, y_v, z_v
% 旋转后x_v_r, y_v_r, z_v_r, F
syms x_v y_v z_v x_v_r y_v_r z_v_r F u v real
x_v=u*x_design;
y_v=(u*x_design*tand(angle_shock_a)+a1)*cos(v);
z_v=(u*x_design*tand(angle_shock_b)+b1)*sin(v);
% 旋转前曲率方程
xu_bf = diff(x_v,u); yu_bf = diff(y_v,u); zu_bf = diff(z_v,u);
xv_bf = diff(x_v,v); yv_bf = diff(y_v,v); zv_bf = diff(z_v,v);
if xv_bf == 0
   dz_bf = zv_bf/yv_bf;
   ddz_bf = diff(dz_bf,v)/yv_bf;
else
   dz_bf = (zu_bf+(-xu_bf/xv_bf)*zv_bf)/(yu_bf+(-xu_bf/xv_bf)*yv_bf);
   ddz_bf = (diff(dz_bf,u)+(-xu_bf/xv_bf)*diff(dz_bf,v))/(yu_bf+(-xu_bf/xv_bf)*yv_bf);
end
% 旋转后方程
x_y_z_v = rat*[x_v,y_v,z_v]';
x_v_r = x_y_z_v(1); y_v_r = x_y_z_v(2); z_v_r = x_y_z_v(3); 
xu = diff(x_v_r,u); yu = diff(y_v_r,u); zu = diff(z_v_r,u);
xv = diff(x_v_r,v); yv = diff(y_v_r,v); zv = diff(z_v_r,v);
% F = cross([xu,yu,zu],[xv,yv,zv]);
F = cross([xv,yv,zv],[xu,yu,zu]);
Fx = F(1);
Fy = F(2);
Fz = F(3);
if xv == 0
   dz = zv/yv;
   ddz = diff(dz,v)/yv;
else
   dz = (zu+(-xu/xv)*zv)/(yu+(-xu/xv)*yv);
   ddz = (diff(dz,u)+(-xu/xv)*diff(dz,v))/(yu+(-xu/xv)*yv);
end
syms u v real
v=0;
eq1=eval(x_v-x_design-0.1);
u_s=vpasolve(eq1); u_s=eval(u_s); u_end=u_s;
%%%%%%%%%%%%%%%%%%%%%%%%%前缘形状%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 定义前缘形状时使用的是旋转前，定义好后旋转再计算
d_fct = b1;
% [x_end,y_end,z_end,y_fct,z_fct,v_end]=swc_shape(M,x_v,y_v,z_v,d_fct,x_design,dz_bf,ddz_bf);
[x_end_1,y_end_1,z_end_1,y_fct_1,z_fct_1,v_end_1]=swc_shape_2(M,x_v,y_v,z_v,d_fct,x_design,dz_bf,ddz_bf);
x_y_z_xuanzhuan=[];
for i = 1:M
% for i=30
    syms u v real
    eq1=eval(z_v-z_fct_1(i));
    eq2=eval(y_v-y_fct_1(i));
    [u_s,v_s]=vpasolve(eq1,eq2,[u,v],[0,Inf;0,pi/2]);
    u_s = eval(u_s); v_s = eval(v_s);
    u_fct(i) = u_s; v_fct(i)=v_s;
    u = u_s; v = v_s;
    x_fct_1(i)=eval(x_v); 
    deta_x(i)=((x_design+0.1-x_fct_1(i))/(N_l-1))*cosd(AOA);
    x_fct(i)=eval(x_v_r);  y_fct(i)=eval(y_v_r);  z_fct(i)=eval(z_v_r);  
end
figure(1)
plot3(x_fct_1,y_fct_1,z_fct_1,'g.');
hold on
Time_discrete = toc;
tic
%%%%%%%%%%%%%%%%%%%%%%%%%离散激波曲面%%%%%%%%%%%%%%%%%%%%%
for i = 1:M
    x_y_z_shock(1,1,i) = x_fct(i);
    x_y_z_shock(1,2,i) = y_fct(i);
    x_y_z_shock(1,3,i) = z_fct(i);
    u = u_fct(i); v = v_fct(i);
    shock_angle(i,1) = pi/2-acos(abs(eval(Fx))/(sqrt((eval(Fx))^2+(eval(Fy))^2+(eval(Fz))^2)));
    beta_index(1,1,i) = included_angle([eval(Fy),eval(Fz)]);   
    x_r_shock(1,1,i) = x_y_z_shock(1,1,i);
    x_r_shock(1,2,i) = sqrt((x_y_z_shock(1,2,i)-x_y_z_shock(1,2,i)).^2+(x_y_z_shock(1,3,i)-x_y_z_shock(1,3,i)).^2);
    U_index(i,1) = u_fct(i); V_index(i,1) = v_fct(i);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%求解激波后参数%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for j = 1
    for i = 1:M 
        x_s = x_y_z_shock(j,1,i); 
        y_s = x_y_z_shock(j,2,i);
        z_s = x_y_z_shock(j,3,i);
        u = U_index(i,j); v = V_index(i,j);
        bta1 = eval(y_s-(dz.*(1+dz.^2))./ddz);
        bta2 = eval(z_s+(1+dz.^2)./ddz);
        x_y_center_of_shock(j,1,i) = x_y_z_shock(j,1,i);
        x_y_center_of_shock(j,2,i) = bta1;
        x_y_center_of_shock(j,3,i) = bta2;
        h_center_body(i,j) = sqrt((x_y_z_shock(j,3,i)-bta2).^2+(x_y_z_shock(j,2,i)-bta1).^2);
        x_r_shock(j,3,i)=sqrt((x_y_z_shock(j,3,i)-bta2).^2+(x_y_z_shock(j,2,i)-bta1).^2);
        p_bf(j,i) = p0;
        a_bf(j,i) = a0;
        v_bf(j,i) = m0*sqrt(1.4*p0/r0);
        r_bf(j,i) = r0;
        h_real = h_center_body(i,j)-x_r_shock(j,2,i);
        h_r_inter(j,j,i)=h_real;
        h_inter(j,j,i)=h_center_body(i,j);
        [x1,y1,p1,r1,v1,a1,v1_x,v1_r] = qiujie_shock(x_r_shock(j,1,i),x_r_shock(j,2,i),h_real,p_bf(j,i),a_bf(j,i),v_bf(j,i),r_bf(j,i),shock_angle(i,j));
        x_l(j,j,i) = x1;
        y_l(j,j,i) = y1;
        p_l(j,j,i) = p1;
        v_l(j,j,i) = v1;
        r_l(j,j,i) = r1;
        a_l(j,j,i) = a1;
        v_x(j,j,i) = v1_x;
        v_y(j,j,i) = v1_r*cos(beta_index(j,j,i));
        v_z(j,j,i) = v1_r*sin(beta_index(j,j,i));
        x_inter(j,j,i) = x_y_z_shock(j,1,i);
        y_inter(j,j,i) = x_y_z_shock(j,2,i);
        z_inter(j,j,i) = x_y_z_shock(j,3,i);
    end 
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x_index = 0*x_l; h1=x_index; h2=x_index;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%求内点参数%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for i = 1:M
for i=35
    times = i
    x_distance_1 = x_r_shock(1,1,i);
    j=2;
    % while x_distance_1<x_design+0.1
    % while U_index(i,j-1)<u_end
    for j=2:N_l+1
        u = U_index(i,j-1); v = V_index(i,j-1);
        vector_D = 0.02*[eval(Fz^2+Fy^2),eval(-Fx*Fy),eval(-Fx*Fz)];
        delta_x = deta_x(i);
        delta_y = delta_x*vector_D(2)/vector_D(1);
        x_y_z_shock(j,1,i) = x_y_z_shock(j-1,1,i)+delta_x;
        x_y_z_shock(j,2,i) = x_y_z_shock(j-1,2,i)+delta_y;        
        syms u v real
        eq1=x_v_r-x_y_z_shock(j,1,i);
        eq2=y_v_r-x_y_z_shock(j,2,i);
        [u_s,v_s]=vpasolve(eq1,eq2,[u,v],[0 Inf; 0 pi]);
        U_index(i,j)=u_s; V_index(i,j)=v_s;
        u = U_index(i,j); v = V_index(i,j);
        x_y_z_shock(j,3,i) = eval(z_v_r);        
        shock_angle(i,j) = pi/2-acos(abs(eval(Fx))/(sqrt((eval(Fx))^2+(eval(Fy))^2+(eval(Fz))^2)));
        beta_index(j,j,i) = included_angle([eval(Fy),eval(Fz)]);
        % plot3(x_y_z_shock(j-1:j,1,i),x_y_z_shock(j-1:j,2,i),x_y_z_shock(j-1:j,3,i),'r');
        % hidden off
        % hold on
        x_r_shock(j,1,i) = x_y_z_shock(j,1,i);
        x_r_shock(j,2,i) = sqrt((x_y_z_shock(j,3,i)-x_y_z_shock(j-1,3,i)).^2+(x_y_z_shock(j,2,i)-x_y_z_shock(j-1,2,i)).^2)+ x_r_shock(j-1,2,i);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        u = U_index(i,j); v = V_index(i,j);
        x_s = x_y_z_shock(j,1,i); 
        y_s = x_y_z_shock(j,2,i);
        z_s = x_y_z_shock(j,3,i);
        bta1 = eval(y_s-(dz.*(1+dz.^2))./ddz);
        bta2 = eval(z_s+(1+dz.^2)./ddz);
        x_y_center_of_shock(j,1,i) = x_y_z_shock(j,1,i);
        x_y_center_of_shock(j,2,i) = bta1;
        x_y_center_of_shock(j,3,i) = bta2;
        h_center_body(i,j) = sqrt((x_y_z_shock(j,3,i)-bta2).^2+(x_y_z_shock(j,2,i)-bta1).^2);
        x_r_shock(j,3,i)=sqrt((x_y_z_shock(j,3,i)-bta2).^2+(x_y_z_shock(j,2,i)-bta1).^2);
        p_bf(j,i) = p0;
        a_bf(j,i) = a0;
        v_bf(j,i) = m0*sqrt(1.4*p0/r0);
        r_bf(j,i) = r0;
        h_real = h_center_body(i,j)-x_r_shock(j,2,i);
        h_r_inter(j,j,i)=h_real;
        h_inter(j,j,i)=h_center_body(i,j);
        [x1,y1,p1,r1,v1,a1,v1_x,v1_r] = qiujie_shock(x_r_shock(j,1,i),x_r_shock(j,2,i),h_real,p_bf(j,i),a_bf(j,i),v_bf(j,i),r_bf(j,i),shock_angle(i,j));
        x_l(j,j,i) = x1; y_l(j,j,i) = y1; p_l(j,j,i) = p1;
        v_l(j,j,i) = v1; r_l(j,j,i) = r1; a_l(j,j,i) = a1;
        v_x(j,j,i) = v1_x;
        v_y(j,j,i) = v1_r*cos(beta_index(j,j,i));
        v_z(j,j,i) = v1_r*sin(beta_index(j,j,i));
        x_inter(j,j,i) = x_y_z_shock(j,1,i);
        y_inter(j,j,i) = x_y_z_shock(j,2,i);
        z_inter(j,j,i) = x_y_z_shock(j,3,i);
        for n = j-1:-1:1
                x1=[];y1=[];p1=[];r1=[];v1=[];a1=[];x2=[];y2=[];p2=[];r2=[];v2=[];a2=[];           
                h1(j-1,n,i) = interp1(x_r_shock(1:j,1,i),x_r_shock(1:j,3,i),x_l(j-1,n,i));
                h2(j-1,n,i) = interp1(x_r_shock(1:j,1,i),x_r_shock(1:j,2,i),x_l(j-1,n,i));
                h_real=h1(j-1,n,i)-h2(j-1,n,i);
                h_r_inter(j-1,n,i)=h_real;
                h_inter(j-1,n,i)=h1(j-1,n,i);
                x1 = x_l(j-1,n,i); y1 = y_l(j-1,n,i)+h_real; p1 = p_l(j-1,n,i); r1 = r_l(j-1,n,i); v1 = v_l(j-1,n,i); a1 = a_l(j-1,n,i);
                x2 = x_l(j,n+1,i); y2 = y_l(j,n+1,i)+h_real; p2 = p_l(j,n+1,i); r2 = r_l(j,n+1,i); v2 = v_l(j,n+1,i); a2 = a_l(j,n+1,i);
                [x3,y3,p3,r3,v3,a3,det3,v3_x,v3_r] = qiujie_inter(x1,y1,p1,r1,v1,a1,x2,y2,p2,r2,v2,a2);
                x_l(j,n,i) = x3; y_l(j,n,i) = y3-h_real; p_l(j,n,i) = p3; a_l(j,n,i) = a3; v_l(j,n,i) = v3; r_l(j,n,i) = r3; % x_index(j,n,i)=det3; x_inter(j,n,i) = x_l(j,n,i); v_x(j,n,i) = v3_x;
                x3=[];y3=[];p3=[];r3=[];v3=[];v3=[];a3=[];det3=[];v3_x=[];v3_r=[];y4=[];z4=[];theta4=[];     
        end
       x_distance=x_l(j,1,i); x_distance_1=x_inter(j,j,i);
       if isnan(x_distance)==1 || x_distance<x_inter(j-1,1,i)
           disp('error');
           break;
       end       
       % j=j+1;
       if abs(U_index(i,j-1)-u_end)<0.00001
          break;
       end         
    end
    index(i)=j; % 用于表示每张流面内激波决定段的位置
    for j=1:index(i)-1
        for n=j+1:index(i)
            x_l(j,n,i)=x_l(j,n-1,i); y_l(j,n,i)=y_l(j,n-1,i); p_l(j,n,i)=p_l(j,n-1,i);
            r_l(j,n,i)=r_l(j,n-1,i); v_l(j,n,i)=v_l(j,n-1,i); a_l(j,n,i)=a_l(j,n-1,i);
        end
    end
    P_h1=polyfit(x_r_shock(1:index(i),1,i),x_r_shock(1:index(i),3,i),4);
    P_h2=polyfit(x_r_shock(1:index(i),1,i),x_r_shock(1:index(i),2,i),4); 
    disp('激波决定段求解完成');
    % 给定壁面压力，反求等熵压缩段
    d_p = p_l(index(i),1,i); x_p = x_l(index(i),1,i);
    for n=1:index(i)
        x_l_2(1,n,i)=x_l(index(i),n,i); y_l_2(1,n,i)=y_l(index(i),n,i); p_l_2(1,n,i)=p_l(index(i),n,i);  
        v_l_2(1,n,i)=v_l(index(i),n,i); r_l_2(1,n,i)=r_l(index(i),n,i); a_l_2(1,n,i)=a_l(index(i),n,i);
    end
    for j=2:index(i)
        for n=1:index(i)-j+1
            x1=[];y1=[];p1=[];r1=[];v1=[];a1=[];
            x2=[];y2=[];p2=[];r2=[];v2=[];a2=[];
            x3=[];y3=[];p3=[];r3=[];v3=[];a3=[];
            h1_2(j-1,n,i)=polyval(P_h1,x_l_2(j-1,n,i));
            h2_2(j-1,n,i)=polyval(P_h2,x_l_2(j-1,n,i));
            h_real=h1_2(j-1,n,i)-h2_2(j-1,n,i);
            if n==1         
                x1 = x_l_2(j-1,n,i); y1 = y_l_2(j-1,n,i)+h_real; p1 = p_l_2(j-1,n,i); r1 = r_l_2(j-1,n,i); v1 = v_l_2(j-1,n,i); a1 = a_l_2(j-1,n,i);
                x2 = x_l_2(j-1,n+1,i); y2 = y_l_2(j-1,n+1,i)+h_real; p2 = p_l_2(j-1,n+1,i); r2 = r_l_2(j-1,n+1,i); v2 = v_l_2(j-1,n+1,i); a2 = a_l_2(j-1,n+1,i);
                [x3,y3,p3,r3,v3,a3]=wall(x1,y1,p1,r1,v1,a1,x2,y2,p2,r2,v2,a2);
                x_l_2(j,n,i)=x3; y_l_2(j,n,i)=y3-h_real; p_l_2(j,n,i)=p3; a_l_2(j,n,i)=a3; v_l_2(j,n,i)=v3; r_l_2(j,n,i)=r3;
            else
                x1 = x_l_2(j-1,n+1,i); y1 = y_l_2(j-1,n+1,i)+h_real; p1 = p_l_2(j-1,n+1,i); r1 = r_l_2(j-1,n+1,i); v1 = v_l_2(j-1,n+1,i); a1 = a_l_2(j-1,n+1,i);
                x2 = x_l_2(j,n-1,i); y2 = y_l_2(j,n-1,i)+h_real; p2 = p_l_2(j,n-1,i); r2 = r_l_2(j,n-1,i); v2 = v_l_2(j,n-1,i); a2 = a_l_2(j,n-1,i);
                [x3,y3,p3,r3,v3,a3]=inter_point(x1,y1,p1,r1,v1,a1,x2,y2,p2,r2,v2,a2);
                x_l_2(j,n,i)=x3; y_l_2(j,n,i)=y3-h_real; p_l_2(j,n,i)=p3; a_l_2(j,n,i)=a3; v_l_2(j,n,i)=v3; r_l_2(j,n,i)=r3;               
            end
        end
        for n=index(i)-j+2:index(i)
            x_l_2(j,n,i)=x_l_2(j,n-1,i); y_l_2(j,n,i)=y_l_2(j,n-1,i); p_l_2(j,n,i)=p_l_2(j,n-1,i); 
            a_l_2(j,n,i)=a_l_2(j,n-1,i); v_l_2(j,n,i)=v_l_2(j,n-1,i); r_l_2(j,n,i)=r_l_2(j,n-1,i);
        end
    end
    disp('压力决定段求解完成');
    % 重构网格
    x_2=[]; y_2=[]; p_2=[]; r_2=[]; v_2=[]; a_2=[];
    x_2=[x_l(:,:,i);x_l_2(2:end,:,i)]; y_2=[y_l(:,:,i);y_l_2(2:end,:,i)]; p_2=[p_l(:,:,i);p_l_2(2:end,:,i)];
    r_2=[r_l(:,:,i);r_l_2(2:end,:,i)]; v_2=[v_l(:,:,i);v_l_2(2:end,:,i)]; a_2=[a_l(:,:,i);a_l_2(2:end,:,i)];
    % 定义设计截面点数，重构网格时得从激波点出发
    % figure(3)
    % plot(x_2(:,1),y_2(:,1),'g')
    % axis equal 
    % hold on
    % plot(x_2(:,end),y_2(:,end),'g')
    % hold on
   for n=1:index(i)
        x_l_3(1,n,i)=x_2(n,end); y_l_3(1,n,i)=y_2(n,end); p_l_3(1,n,i)=p_2(n,end);
        r_l_3(1,n,i)=r_2(n,end); v_l_3(1,n,i)=v_2(n,end); a_l_3(1,n,i)=a_2(n,end);
        v_x_3(1,n,i)=v_x(n,n,i); v_y_3(1,n,i)=v_y(n,n,i); v_z_3(1,n,i)=v_z(n,n,i);
        x_inter_3(1,n,i)=x_inter(n,n,i); y_inter_3(1,n,i)=y_inter(n,n,i); z_inter_3(1,n,i)=z_inter(n,n,i);
        beta_index_3(1,n,i)=beta_index(n,n,i);        
    end
    for j=2:index(i)
        for n=j:index(i)
            if n==j
                for k=1:length(x_2(:,1))-1
                    if n==index(i)
                        f12=atan((y_l_3(j-1,n,i)-y_l_3(j-1,n-1,i))/(x_l_3(j-1,n,i)-x_l_3(j-1,n-1,i)))-pi/2;
                    else
                        f12=atan((y_l_3(j-1,n+1,i)-y_l_3(j-1,n-1,i))/(x_l_3(j-1,n+1,i)-x_l_3(j-1,n-1,i)))-pi/2;
                    end                    
                    f03=atan((y_2(k+1,1)-y_2(k,1))/(x_2(k+1,1)-x_2(k,1)));
                    x1=x_l_3(j-1,n,i); y1=y_l_3(j-1,n,i);
                    x3=x_2(k,1); y3=y_2(k,1);
                    [x0,y0]=intercept(x1,y1,f12,x3,y3,f03);
                    if x0>=x_2(k,1) && x0<=x_2(k+1,1)
                        x_l_3(j,n,i)=x0; y_l_3(j,n,i)=y0;                         
                        break
                    end
                end
                x0=[]; y0=[]; x1=[]; y1=[]; x3=[]; y3=[]; f03=[]; f13=[];
                p_l_3(j,n,i)=interp1(x_2(:,1),p_2(:,1),x_l_3(j,n,i));
                r_l_3(j,n,i)=interp1(x_2(:,1),r_2(:,1),x_l_3(j,n,i));
                v_l_3(j,n,i)=interp1(x_2(:,1),v_2(:,1),x_l_3(j,n,i));
                a_l_3(j,n,i)=interp1(x_2(:,1),a_2(:,1),x_l_3(j,n,i));
                det_s=sqrt((y_l_3(j,n,i)-y_l_3(j-1,n,i))^2+(x_l_3(j,n,i)-x_l_3(j-1,n,i))^2);
                x_inter_3(j,n,i)=x_l_3(j,n,i);
                if n==index(i)
                    [y4,z4,theta4]=dirction_bd(x_inter_3(j-1,n-1,i),y_inter_3(j-1,n-1,i),z_inter_3(j-1,n-1,i),x_inter_3(j-1,n,i),y_inter_3(j-1,n,i),z_inter_3(j-1,n,i),v_x_3(j-1,n,i),v_y_3(j-1,n,i),v_z_3(j-1,n,i),x_inter_3(j,n,i));
                else
                    [y4,z4,theta4]=dirction_cd(x_inter_3(j-1,n-1,i),y_inter_3(j-1,n-1,i),z_inter_3(j-1,n-1,i),x_inter_3(j-1,n+1,i),y_inter_3(j-1,n+1,i),z_inter_3(j-1,n+1,i),x_inter_3(j-1,n,i),y_inter_3(j-1,n,i),z_inter_3(j-1,n,i),v_x_3(j-1,n,i),v_y_3(j-1,n,i),v_z_3(j-1,n,i),x_inter_3(j,n,i));
                end                  
                y_inter_3(j,n,i)=y4; z_inter_3(j,n,i)=z4; beta_index_3(j,n,i)=theta4;
                v_x_3(j,n,i)=v_l_3(j,n,i)*cos(a_l_3(j,n,i));
                v_y_3(j,n,i)=v_l_3(j,n,i)*sin(a_l_3(j,n,i))*cos(beta_index_3(j,n,i));
                v_z_3(j,n,i)=v_l_3(j,n,i)*sin(a_l_3(j,n,i))*sin(beta_index_3(j,n,i));
                y4=[]; z4=[]; theta4=[];
            elseif n==index(i)
                alpha=atan((y_l_3(j-1,n,i)-y_l_3(j-1,n-1,i))/(x_l_3(j-1,n,i)-x_l_3(j-1,n-1,i)))-pi/2;
                x_l_3(j,n,i)=x_l_3(j-1,n,i)+det_s*cos(alpha);
                y_l_3(j,n,i)=y_l_3(j-1,n,i)+det_s*sin(alpha);
                p_l_3(j,n,i)=interpt_2D(x_2,y_2,p_2,x_l_3(j,n,i),y_l_3(j,n,i));
                r_l_3(j,n,i)=interpt_2D(x_2,y_2,r_2,x_l_3(j,n,i),y_l_3(j,n,i));
                v_l_3(j,n,i)=interpt_2D(x_2,y_2,v_2,x_l_3(j,n,i),y_l_3(j,n,i));
                a_l_3(j,n,i)=interpt_2D(x_2,y_2,a_2,x_l_3(j,n,i),y_l_3(j,n,i));
                x_inter_3(j,n,i)=x_l_3(j,n,i);
                [y4,z4,theta4]=dirction_bd(x_inter_3(j-1,n-1,i),y_inter_3(j-1,n-1,i),z_inter_3(j-1,n-1,i),x_inter_3(j-1,n,i),y_inter_3(j-1,n,i),z_inter_3(j-1,n,i),v_x_3(j-1,n,i),v_y_3(j-1,n,i),v_z_3(j-1,n,i),x_inter_3(j,n,i));
                y_inter_3(j,n,i)=y4; z_inter_3(j,n,i)=z4; beta_index_3(j,n,i)=theta4;
                v_x_3(j,n,i)=v_l_3(j,n,i)*cos(a_l_3(j,n,i));
                v_y_3(j,n,i)=v_l_3(j,n,i)*sin(a_l_3(j,n,i))*cos(beta_index_3(j,n,i));
                v_z_3(j,n,i)=v_l_3(j,n,i)*sin(a_l_3(j,n,i))*sin(beta_index_3(j,n,i));                
                y4=[]; z4=[]; theta4=[];
            else
                alpha=atan((y_l_3(j-1,n+1,i)-y_l_3(j-1,n-1,i))/(x_l_3(j-1,n+1,i)-x_l_3(j-1,n-1,i)))-pi/2;
                x_l_3(j,n,i)=x_l_3(j-1,n,i)+det_s*cos(alpha);
                y_l_3(j,n,i)=y_l_3(j-1,n,i)+det_s*sin(alpha);
                p_l_3(j,n,i)=interpt_2D(x_2,y_2,p_2,x_l_3(j,n,i),y_l_3(j,n,i));
                r_l_3(j,n,i)=interpt_2D(x_2,y_2,r_2,x_l_3(j,n,i),y_l_3(j,n,i));
                v_l_3(j,n,i)=interpt_2D(x_2,y_2,v_2,x_l_3(j,n,i),y_l_3(j,n,i));
                a_l_3(j,n,i)=interpt_2D(x_2,y_2,a_2,x_l_3(j,n,i),y_l_3(j,n,i));
                x_inter_3(j,n,i)=x_l_3(j,n,i);
                [y4,z4,theta4]=dirction_cd(x_inter_3(j-1,n-1,i),y_inter_3(j-1,n-1,i),z_inter_3(j-1,n-1,i),x_inter_3(j-1,n+1,i),y_inter_3(j-1,n+1,i),z_inter_3(j-1,n+1,i),x_inter_3(j-1,n,i),y_inter_3(j-1,n,i),z_inter_3(j-1,n,i),v_x_3(j-1,n,i),v_y_3(j-1,n,i),v_z_3(j-1,n,i),x_inter_3(j,n,i));  
                y_inter_3(j,n,i)=y4; z_inter_3(j,n,i)=z4; beta_index_3(j,n,i)=theta4;
                v_x_3(j,n,i)=v_l_3(j,n,i)*cos(a_l_3(j,n,i));
                v_y_3(j,n,i)=v_l_3(j,n,i)*sin(a_l_3(j,n,i))*cos(beta_index_3(j,n,i));
                v_z_3(j,n,i)=v_l_3(j,n,i)*sin(a_l_3(j,n,i))*sin(beta_index_3(j,n,i));  
                y4=[]; z4=[]; theta4=[];
            end
            % plot(x_l_3(j-1,n-1:n,i),y_l_3(j-1,n-1:n,i),'b')
            % axis equal 
            % hold on
            % plot(x_l_3(j-1:j,n,i),y_l_3(j-1:j,n,i),'b')
            % hold on
        end
        for n=j-1:-1:1
            x_l_3(j,n,i)=x_l_3(j,n+1,i); y_l_3(j,n,i)=y_l_3(j,n+1,i); p_l_3(j,n,i)=p_l_3(j,n+1,i);
            r_l_3(j,n,i)=r_l_3(j,n+1,i); v_l_3(j,n,i)=v_l_3(j,n+1,i); a_l_3(j,n,i)=a_l_3(j,n+1,i);
            x_inter_3(j,n,i)=x_inter_3(j,n+1,i); y_inter_3(j,n,i)=y_inter_3(j,n+1,i); z_inter_3(j,n,i)=z_inter_3(j,n+1,i);
            v_x_3(j,n,i)=v_x_3(j,n+1,i); v_y_3(j,n,i)=v_y_3(j,n+1,i); v_z_3(j,n,i)=v_z_3(j,n+1,i);
        end
    end
    figure(4)
    plot3(x_inter_3(:,1,i),y_inter_3(:,1,i),z_inter_3(:,1,i))
    axis equal
    hold on
end
for i = 1:M
% for i=15
    x_y_z_xingxian(:,1,i) = x_inter_3(:,1,i);
    x_y_z_xingxian(:,2,i) = y_inter_3(:,1,i);
    x_y_z_xingxian(:,3,i) = z_inter_3(:,1,i);
    p_wav_1=p_l_3(:,1,i);
    vx_wav_1=v_x_3(:,1,i); vy_wav_1=v_y_3(:,1,i); vz_wav_1=v_z_3(:,1,i);
    x_y_z_xingxian_r=inv(rat)*[x_y_z_xingxian(:,1,i),x_y_z_xingxian(:,2,i),x_y_z_xingxian(:,3,i)]';
    coordinate_x=x_y_z_xingxian_r(1,1:index(i));
    coordinate_y=x_y_z_xingxian_r(2,1:index(i));
    coordinate_z=x_y_z_xingxian_r(3,1:index(i));
    wav_x(i,:)=linspace(coordinate_x(1),x_design,50);
    for j=1:50
        wav_y(i,j)=interp1(coordinate_x,coordinate_y,wav_x(i,j));
        wav_z(i,j)=interp1(coordinate_x,coordinate_z,wav_x(i,j));
        p_wav(i,j)=interp1(coordinate_x,p_wav_1,wav_x(i,j));
        vx_wav(i,j)=interp1(coordinate_x,vx_wav_1,wav_x(i,j));
        vy_wav(i,j)=interp1(coordinate_x,vy_wav_1,wav_x(i,j));
        vz_wav(i,j)=interp1(coordinate_x,vz_wav_1,wav_x(i,j));
    end
    coordinate_x=[]; coordinate_y=[];coordinate_z=[]; p_wav_1=[]; vx_wav_1=[]; vy_wav_1=[]; vz_wav_1=[];
end
% 生成上壁面
for i=1:M
    wav_up_x(i,:)=linspace(x_fct_1(i),x_design,50);
    for j=1:50
        wav_up_y(i,j)=(wav_up_x(i,j)-wav_up_x(i,1))*Vector_V(2)/Vector_V(1)+y_fct_1(i);
        wav_up_z(i,j)=(wav_up_x(i,j)-wav_up_x(i,1))*Vector_V(3)/Vector_V(1)+z_fct_1(i);
    end
end
Time_LTOCs = toc;
figure(1)
mesh(wav_x,wav_y,wav_z,'edgecolor','g','facecolor','none') %
axis equal
hold on
mesh(wav_up_x,wav_up_y,wav_up_z,'edgecolor','b','facecolor','none')
hold on
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid = fopen('waverider_AOA_-4.dat','wt');
for j = 1:50
    fprintf(fid,'ROW\n');
    for i = 1:50
        fprintf(fid,'%e\t',wav_x(i,j)*1000);
        fprintf(fid,'%e\t',wav_y(i,j)*1000);
        fprintf(fid,'%e\n',-wav_z(i,j)*1000);
    end  
end
fid = fopen('waverider_AOA_-4_up.dat','wt');
for j = 1:50
    fprintf(fid,'ROW\n');
    for i = 1:50
        fprintf(fid,'%e\t',wav_up_x(i,j)*1000);
        fprintf(fid,'%e\t',wav_up_y(i,j)*1000);
        fprintf(fid,'%e\n',-wav_up_z(i,j)*1000);
    end  
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid = fopen('wav_surface_1.dat','wt');
fprintf(fid,'TITLE="COMBINE_FIELD"\n');
fprintf(fid,'VARIABLES=x,y,z,p,vx,vy,vz\n');
[data,np,nodes,Nelayer]= shujuzhenhe_2D( wav_x,wav_y,wav_z,p_wav,vx_wav,vy_wav,vz_wav,p0,length(wav_x(:,1)),length(wav_x(1,:)));       
fprintf(fid,strcat('ZONE\t','N=',num2str(nodes),',E=',num2str(Nelayer),',DATAPACKING=POINT,ZONETYPE=FEQuadrilateral\n'));            
fprintf(fid,'%f %f %f %f %f %f %f\n',data');
fprintf(fid,' %d %d %d %d %d %d %d\n',np');