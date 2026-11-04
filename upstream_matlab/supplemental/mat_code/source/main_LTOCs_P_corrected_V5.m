clc
clear
M = 50;
% N = 60;
format short
tic
%%%%%%%%%%%%%%%%%%%%%%%%来流条件%%%%%%%%%%%%%%%%%%%%%%%%%%
global p0
global r0
global m0
global v0
global a0
p0=1880;r0=0.02929783;m0=10;v0=m0*sqrt(1.4*p0/r0); a0=0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%激波形状%%%%%%%%%%%%%%%%%%%%%%%%%
%%绕Z轴旋转
a_z = 0;
A = [cosd(a_z),sind(a_z),0,0;-sind(a_z),cosd(a_z),0,0;0,0,1,0;0,0,0,1];
x_design = 4;
N1 = [1,-2,1;-2,2,0;1,0,0];
N2 = [-1,3,-3,1;3,-6,3,0;-3,3,0,0;1,0,0,0];
p00=[0,-1,0]; p01=[0,-0.5,0.5]; p02=[0,0.5,0.5]; p03=[0,1,0];
p10=[2,-1.4,0]; p11=[2,-0.7,1.2]; p12=[2,0.7,1.2]; p13=[2,1.4,0];
p20=[4,-1.8,0]; p21=[4,-0.9,2]; p22=[4,0.9,2]; p23=[4,1.8,0];
% p00=[0,-1,0]; p01=[0,-0.5,0.6]; p02=[0,0.5,0.6]; p03=[0,1,0];
% p10=[2,-1.4,0]; p11=[2,-0.8,1.1]; p12=[2,0.8,1.1]; p13=[2,1.4,0];
% p20=[4,-1.8,0]; p21=[4,-1.4,1.6]; p22=[4,1.4,1.6]; p23=[4,1.8,0];
P_X=[p00(1),p01(1),p02(1),p03(1);p10(1),p11(1),p12(1),p13(1);p20(1),p21(1),p22(1),p23(1)];
P_Y=[p00(2),p01(2),p02(2),p03(2);p10(2),p11(2),p12(2),p13(2);p20(2),p21(2),p22(2),p23(2)];
P_Z=[p00(3),p01(3),p02(3),p03(3);p10(3),p11(3),p12(3),p13(3);p20(3),p21(3),p22(3),p23(3)];
x_ctrl = P_X;
y_ctrl = P_Y;
z_ctrl = P_Z;
u = linspace(0,1,100); v = linspace(0,1,100);
for i = 1:100
    for j = 1:100
        x(j,i) = [u(i)^2,u(i),1]*N1*P_X*N2'*[v(j)^3,v(j)^2,v(j),1]';
        y(j,i) = [u(i)^2,u(i),1]*N1*P_Y*N2'*[v(j)^3,v(j)^2,v(j),1]';
        z(j,i) = [u(i)^2,u(i),1]*N1*P_Z*N2'*[v(j)^3,v(j)^2,v(j),1]';
    end
    x_y_z_xuanzhua(1:100,1:3,i) = [x(1:100,i),y(1:100,i),z(1:100,i)];
    x_y_z_xuanzhua(1:100,4,i) = 1;
    x_y_z_xuanzhua(1:100,:,i) =  x_y_z_xuanzhua(1:100,:,i) *A;
    x(1:100,i) = x_y_z_xuanzhua(:,1,i);
    y(1:100,i) = x_y_z_xuanzhua(:,2,i);
    z(1:100,i) = x_y_z_xuanzhua(:,3,i);    
end
mesh(x,y,z,'edgecolor','k','facecolor','none')
axis equal
hold on
mesh(x_ctrl,y_ctrl,z_ctrl,'Facecolor','none','Edgecolor','r','LineWidth',1.5,'linestyle','--','Marker','*','Markerfacecolor','r','Markersize',8)
axis equal
hold on
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
syms x_v y_v z_v x_v_r y_v_r z_v_r F u v real
x_v = [u^2,u,1]*N1*P_X*N2'*[v^3,v^2,v,1]';
y_v = [u^2,u,1]*N1*P_Y*N2'*[v^3,v^2,v,1]';
z_v = [u^2,u,1]*N1*P_Z*N2'*[v^3,v^2,v,1]';
x_y_z_v = [x_v, y_v, z_v, 1] * A;
x_v_r = x_y_z_v(1); y_v_r = x_y_z_v(2); z_v_r = x_y_z_v(3);
xu = diff(x_v_r,u); yu = diff(y_v_r,u); zu = diff(z_v_r,u);
xv = diff(x_v_r,v); yv = diff(y_v_r,v); zv = diff(z_v_r,v);
F = cross([xu,yu,zu],[xv,yv,zv]);
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
%%%%%%%%%%%%%%%%%%%%%%%%%前缘形状%%%%%%%%%%%%%%%%%%%%%%%%%%%
d_fct = 0.6; Ma_1 = 6; Ma_2 = 10; 
[x_end,y_end,z_end,y_fct,z_fct,v_end]=swc_shape(M,x_v_r,y_v_r,z_v_r,d_fct,x_design,dz,ddz);
for i = 1:M
%     v_fct(i) = v_end(i);
%     v = v_end(i);
    syms u v real
    eq1=eval(z_v_r-z_fct(i));
    eq2=eval(y_v_r-y_fct(i));
    [u_s,v_s]=vpasolve(eq1,eq2,[u,v],[0,Inf;0,1]);          
    u_s = eval(u_s); v_s = eval(v_s);
    u_fct(i) = u_s; v_fct(i)=v_s;
    u = u_s; v = v_s;
    x_fct(i)=eval(x_v_r);
    Ma_fct(i)=Ma_1+(Ma_2-Ma_1)*(i-1)/(M-1);
end

figure(1)
plot3(x_end,y_end,z_end,'g');
hold on
plot3(x_fct,y_fct,z_fct,'b.');
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
%         v_bf(j,i) = m0*sqrt(1.4*p0/r0);
        v_bf(j,i) = Ma_fct(i)*sqrt(1.4*p0/r0);
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
for i = 1:M
    times = i
    x_distance = x_r_shock(1,1,i);
    j=2;
    while x_distance<x_design+0.1
        u = U_index(i,j-1); v = V_index(i,j-1);
        vector_D = 0.02*[eval(Fz^2+Fy^2),eval(-Fx*Fy),eval(-Fx*Fz)];
        delta_x = 0.08;
        delta_y = delta_x*vector_D(2)/vector_D(1);
        x_y_z_shock(j,1,i) = x_y_z_shock(j-1,1,i)+delta_x;
        x_y_z_shock(j,2,i) = x_y_z_shock(j-1,2,i)+delta_y;        
        syms u v real
        eq1=x_v_r-x_y_z_shock(j,1,i);
        eq2=y_v_r-x_y_z_shock(j,2,i);
%         [u_s,v_s]=solve(eq1,eq2,'u','v');
%         u_s = eval(u_s); v_s = eval(v_s);
%         u=u_s; v=v_s;
%         distance = eval(sqrt((y_v_r-x_y_z_shock(j-1,2,i)).^2+(z_v_r-x_y_z_shock(j-1,3,i)).^2));
%         [m,index] = min(distance);
%         U_index(i,j) = u_s(index); V_index(i,j) = v_s(index);
        [u_s,v_s]=vpasolve(eq1,eq2,[u,v],[0 inf; 0 1]);
        U_index(i,j)=u_s; V_index(i,j)=v_s;
        u = U_index(i,j); v = V_index(i,j);
        x_y_z_shock(j,3,i) = eval(z_v_r);        
        shock_angle(i,j) = pi/2-acos(abs(eval(Fx))/(sqrt((eval(Fx))^2+(eval(Fy))^2+(eval(Fz))^2)));
        beta_index(j,j,i) = included_angle([eval(Fy),eval(Fz)]);
        plot3(x_y_z_shock(j-1:j,1,i),x_y_z_shock(j-1:j,2,i),x_y_z_shock(j-1:j,3,i),'r');
        hidden off
        hold on
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
%         v_bf(j,i) = m0*sqrt(1.4*p0/r0);
        v_bf(j,i) = Ma_fct(i)*sqrt(1.4*p0/r0);
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
                x_l(j,n,i) = x3; y_l(j,n,i) = y3-h_real; p_l(j,n,i) = p3; a_l(j,n,i) = a3; v_l(j,n,i) = v3; r_l(j,n,i) = r3; x_index(j,n,i)=det3; x_inter(j,n,i) = x_l(j,n,i); v_x(j,n,i) = v3_x;
                [y4,z4,theta4]=dirction(x_inter(j-1,n,i),y_inter(j-1,n,i),z_inter(j-1,n,i),v_x(j-1,n,i),v_y(j-1,n,i),v_z(j-1,n,i),x_inter(j,n+1,i),y_inter(j,n+1,i),z_inter(j,n+1,i),v_x(j,n+1,i),v_y(j,n+1,i),v_z(j,n+1,i),x_inter(j,n,i),x_index(j,n,i));
                y_inter(j,n,i)=y4; z_inter(j,n,i)=z4;
                beta_index(j,n,i)=theta4;
                v_y(j,n,i) = v3_r*cos(beta_index(j,n,i));
                v_z(j,n,i) = v3_r*sin(beta_index(j,n,i));
                y_l_index(j,n,i)=sqrt((y_inter(j,n,i)-y_inter(j-1,n,i))^2+(z_inter(j,n,i)-z_inter(j-1,n,i))^2)-(y_l(j,n,i)-y_l(j-1,n,i));
                x3=[];y3=[];p3=[];r3=[];v3=[];v3=[];a3=[];det3=[];v3_x=[];v3_r=[];y4=[];z4=[];theta4=[];     
        end
       x_distance=x_inter(j,1,i);
       if isnan(x_distance)==1 || x_distance<x_inter(j-1,1,i)
           disp('error');
           break;
       end       
       j=j+1;
    end
end
for i = 1:M
    x_y_z_xingxian(:,1,i) = x_inter(:,1,i);
    x_y_z_xingxian(:,2,i) = y_inter(:,1,i);
    x_y_z_xingxian(:,3,i) = z_inter(:,1,i);
    p_wav(i,:)=p_l(:,1,i);
    index(i)= find(x_inter(:,1,i),1,'last');
    plot3(x_y_z_xingxian(1:index(i),1,i),x_y_z_xingxian(1:index(i),2,i),x_y_z_xingxian(1:index(i),3,i),'k');
    hold on
end
Time_LTOCs = toc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
save p_l p_l
save a_l a_l
save x_l x_l
save y_l y_l
save v_l v_l
save r_l r_l
save x_inter x_inter
save y_inter y_inter
save z_inter z_inter
save x_y_z_xingxian x_y_z_xingxian
save h_inter h_inter
save h_r_inter h_r_inter
save v_x v_x
save v_y v_y
save v_z v_z
save p_wav p_wav
% [Force_x,Force_y,Force_z]=force_1(x_y_z_xingxian,p_wav,x_2,M,index);
% surface_out(x_y_z_xingxian,x_2);