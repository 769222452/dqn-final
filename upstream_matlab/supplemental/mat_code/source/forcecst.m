function [L,D,lift_to_drag]=forcecst(x,y,z,p,x1,y1,z1,Area_up,Area_base,Area_st,n1,n2)
%来流条件
global gama;% 比热比
global R;% 气体动力学常数
global m0;%来流马赫数
global p0;%来流压力
global t0;%来流静温
global AOA;%攻角


%判断如果用切锥法计算，物面角和攻角的和或差为给定的半锥角是否合理，角度小于马赫角只能用修正牛顿公式进行计算，大于马赫角到25度用切锥法计算，25度到最大偏转角用修正牛顿公式计算
mach_angle=asin(1/m0)*180/pi;%计算来流马赫数对应的马赫角，马赫角是最小可能存在的激波角，二维圆锥激波的激波角至少要大于马赫角，而给定的圆锥流场的初始半锥角肯定小于圆锥流场的激波角，所以如果给定的半锥角都小于马赫角的话，那么就不会存在圆锥流场激波，就要减小来流马赫数或增大圆锥流场的初始半锥角
%由斜激波关系式确定最大的流动偏转角
epsilon =0:0.01:pi/2;%假定激波角为从0到90度,MATLAB中的正余弦等相应函数都以弧度为单位
a1max=zeros(length(epsilon),1);%给定变量的大小
for ii=1:length(epsilon)%在不同来流马赫数条件下，当激波角从0到90度变化时求解出相应的流动偏转角,此时偏转角只有正值才是对的，即在给定的来流马赫数条件下，只有大于某个激波角才能求解出流动转折角
a1max(ii)=atan(((m0^2)*(sin(epsilon(ii))^2)-1)/(((m0^2)*((gama+1)/2-(sin(epsilon(ii))^2))+1)*tan(epsilon(ii))));
end
a1max1=max(a1max)*180/pi;%通过比较法取出在该来流马赫数条件下的最大的流动偏转角


%乘波体下表面升阻比计算，直接用每根流线上的压力分布进行面元积分计算
L1=0;
D1=0;
% % %如果攻角为0度，直接用乘波体下表面流线上的压力进行积分计算此时迎风面的升阻比
% if AOA==0
% for i=1:n1-1%n1为每一根流线上的点数，也就是x,y,z,p的行数，行是流线上的点数
%     for j=1:n2-1%n2为吻切面的数目即流线数目，也就是x,y,z,p的列数，列是流线        
%         S2=[(x(i+1,j+1))-x(i,j) (y(i+1,j+1))-y(i,j) (z(i+1,j+1))-z(i,j) ];%坐标矩阵正对角线的坐标相减
%         S1=[(x(i,j+1))-x(i+1,j) (y(i,j+1))-y(i+1,j) (z(i,j+1))-z(i+1,j) ];%坐标矩阵逆对角线的坐标相减
%         n=cross(S2,S1);%返回BC,AD向量的叉积，生成一个与AD、BC都垂直的向量
%         p_st_equal=(p(i,j)+p(i+1,j)+p(i,j+1)+p(i+1,j+1))/4;%4个点压力的平均值
%         dL=0.5*p_st_equal*dot(n,[0 0 1]);%0.5乘以压力平均值和垂直Z方向法向量[0 0 1]的点积
%         dD=0.5*p_st_equal*dot(n,[1 0 0]);%0.5乘以压力平均值和垂直X方向法向量[1 0 0]的点积，0.5为半个乘波体
%         L1=L1+dL;
%         D1=D1+dD;
%     end
% end

% %当AOA>0时,迎风面流动状态改变，用改进牛顿法/切锥法估算迎风面压力计算升阻比
%  else
  theta0=ones(n1,n2);  
  pf_cone0=ones(n1,n2);
  for i=1:(n1-1)
    for j=1:n2
   theta0(i,j)=atan(abs(z(i+1,j)-z(i,j))/abs(x(i+1,j)-x(i,j)))*180/pi+AOA;
   theta0(end,:)=theta0(end-1,:);%最后一行的角度直接等于前一行，便于压力计算
           if isnan(theta0(i,j))%如果没有楔角，则压力为0
              pf_cone0(i,j)=0;
           else        
                if theta0(i,j)<mach_angle||theta0(i,j)<a1max1||theta0(i,j)>25%角度过小或过大无法用切锥法则用修正的牛顿公式进行计算
                 p02=(((gama+1)^2*m0^2)/(4*gama*m0^2-2*(gama-1)))^(gama/(gama-1))*((1-gama+2*gama*m0^2)/(gama+1))*p0;%正激波后的滞止压力
                 r0=p0/(R*t0);%来流密度
                 D0=sqrt(p0*gama/r0);%来流音速
                 v0=D0*m0;%来流速度
                 cpmax=(p02-p0)/(0.5*r0*v0^2);%最大压力系数
                 cp=cpmax*sin(theta0(i,j)*pi/180)^2;%该点的压力系数
                 pf_cone0(i,j)=((0.5*r0*v0^2)*cp+p0)/p0;%该点的压力比值            
              else
              [~,~,~,~, pf_cone]=conical_flowfield(m0,theta0(i,j),p0,t0);%圆锥流场的外流场求解,输入来流马赫数m0、初始化半锥角theta、远场压力p0
               %远场温度t0；输出圆锥流场的球面角deltaf_cone，deltaf_cone(1)为圆锥流场激波角，deltaf_cone(end)为初始化半锥角，有量纲的X方向和Y方向速度及对应的马赫数mf_cone与压力pf_cone
                pf_cone0(i,j)=pf_cone(end);%取壁面压力
               end
           end
    end
 end  
 pf_cone0(end,:)=pf_cone0(end-1,:);%最后一行的压力直接等于前一行  
 pf_cone0(:,end)=pf_cone0(:,end-1);%最后一列是一个点，压力取前一列
 theta0(:,end)=theta0(:,end-1);%最后一列的角度也取前一列的角度
 for i=1:n1%n1为吻切面的数目即流线数目，也就是x,y,z,p的行数，行是流线
    for j=1:n2%n2为每一根流线上的点数，也就是x,y,z,p的列数，列是流线上的点数,下表面的x,y,z坐标都是准确的，适合用面元法计算
          dL1=pf_cone0(i,j)*cos(theta0(i,j)*pi/180);
          dD1=pf_cone0(i,j)*sin(theta0(i,j)*pi/180);
          if isnan(dD1)||isnan(dL1)
          else
            L1=L1+dL1;
            D1=D1+dD1;
          end
    end
 end   
% end







%CST上表面压力分布估算，背风面压力估算，角度小时用切锥/改进牛顿法计算，角度大时用激波膨胀波理论计算
%传输进来的x1,y1,z1每一行都是是横截面曲线，最后一列是最外围轮廓线，需要从最外围轮廓线生成相同Y值的竖直截面x2,y2,z2，便于计算对应的锥角，才能沿竖直截面采用切锥法/改进牛顿法或激波膨胀波法估算压力分布
x2=ones(n1,n2);
y2=ones(n1,n2);
z2=ones(n1,n2);
x2(1,:)=x1(:,end);%将x1外轮廓线上的x值赋给竖直截面的第一行
y2(1,:)=y1(:,end);%将y1外轮廓线上的y值赋给竖直截面的第一行
z2(1,:)=z1(:,end);%将z1外轮廓线上的z值赋给竖直截面的第一行
 for j=1:n2%竖直截面要求每一列都是从外轮廓线点出发的同一个Y值的截面
 y2(:,j)=y2(1,j);
 end
%每一列的X和Z值则从x1和z1的n行n列即对称线位置往下赋值
for i=1:n1
        x2(i:end,i)=x1(i:end,i);%x2对称点以下赋给x1对称点下每一列的
        x2(1:i,i)=x2(1,i);%2对称点以上赋给x2第一行每一列的值
        z2(i:end,i)=z1(i:end,i);
        z2(1:i,i)=z2(1,i);
end

%计算从起始点到终点的斜率角度，然后每一点都相当于来流条件下角度为斜率角度的切锥点的压力
theta=ones(n1,n2);
beta=ones(n1,n2);
K=ones(n1,n2);
pf_cone1=ones(n1,n2);
for i=1:(n1-1)
    for j=1:n2
   theta(i,j)=atan(abs(z2(i+1,j)-z2(i,j))/abs(x2(i+1,j)-x2(i,j)))*180/pi-AOA;
   theta(end,:)=theta(end-1,:);%最后一行的角度直接等于前一行，便于压力计算
           if isnan(theta(i,j))%如果没有物面角，就是来流压力
                pf_cone1(i,j)=p0/p0;
           else
              %膨胀波计算，参考《高超声速钝前缘乘波构型优化设计研究》
%                beta(i,j)=(gama+1)/(gama-1)*(1+(1+((2*(gama+3))/((gama+1)^2*m0^2*theta(i,j)^2)))^0.5)*theta(i,j);%计算出物面与来流夹角theta(i,j)对应的激波角
%                K(i,j)=m0*theta(i,j);%高超声速相似参数
%                if K(i,j)>0||K(i,j)==0
%                   pf_cone1(i,j)=1+(2*gama/(gama+1)*m0^2*sin(beta(i,j)*pi/180)^2-(gama+1)/(gama-1))*(1+(gama*m0*(beta(i,j)-theta(i,j))^2*cos(beta(i,j)*pi/180)^2)/(1+((gama-1)/2)*m0^2*sin(beta(i,j)*pi/180)^2))^(-1);
%                else%如果K<0
%                 pf_cone1(i,j)=real((1+(gama-1)/2*K(i,j))^((2*gama)/(gama-1)));%在计算指数函数时涉及到了负数开方运算，会出现复数，此时计算结果不准确，采用改进牛顿法计算
%                  p02=(((gama+1)^2*m0^2)/(4*gama*m0^2-2*(gama-1)))^(gama/(gama-1))*((1-gama+2*gama*m0^2)/(gama+1))*p0;%正激波后的滞止压力
%                  r0=p0/(R*t0);%来流密度
%                  D0=sqrt(p0*gama/r0);%来流音速
%                  v0=D0*m0;%来流速度
%                  cpmax=(p02-p0)/(0.5*r0*v0^2);%最大压力系数
%                  cp=cpmax*sin(theta(i,j)*pi/180)^2;%该点的压力系数
%                  pf_cone1(i,j)=((0.5*r0*v0^2)*cp+p0)/p0;%该点的压力比值 
%                end               
                 %切锥和牛顿法计算
               if theta(i,j)<mach_angle||theta(i,j)<a1max1||theta(i,j)>25%角度过小或过大无法用切锥法则用修正的牛顿公式进行计算
                 p02=(((gama+1)^2*m0^2)/(4*gama*m0^2-2*(gama-1)))^(gama/(gama-1))*((1-gama+2*gama*m0^2)/(gama+1))*p0;%正激波后的滞止压力
                 r0=p0/(R*t0);%来流密度
                 D0=sqrt(p0*gama/r0);%来流音速
                 v0=D0*m0;%来流速度
                 cpmax=(p02-p0)/(0.5*r0*v0^2);%最大压力系数
                 cp=cpmax*sin(theta(i,j)*pi/180)^2;%该点的压力系数
                 pf_cone1(i,j)=((0.5*r0*v0^2)*cp+p0)/p0;%该点的压力比值            
               else
               [~,~,~,~, pf_cone]=conical_flowfield(m0,theta(i,j),p0,t0);%圆锥流场的外流场求解,输入来流马赫数m0、初始化半锥角theta、远场压力p0
                %远场温度t0；输出圆锥流场的球面角deltaf_cone，deltaf_cone(1)为圆锥流场激波角，deltaf_cone(end)为初始化半锥角，有量纲的X方向和Y方向速度及对应的马赫数mf_cone与压力pf_cone
                pf_cone1(i,j)=pf_cone(end);%取壁面压力
               end
           end
    end
end 
pf_cone1(end,:)=pf_cone1(end-1,:);%最后一行的压力直接等于前一行

% %考虑边界层理论对计算出来的压力进行修正
% %参考温度Tck计算
% Tbm=1000;%壁面温度自己给定
% Tck=t0*(1.28+0.023*m0^2+0.58*(Tbm/t0-1));
% %绝热壁温Tad计算
% pr=0.72;%系数
% Tad=t0*(1+(pr)^0.5*(gama-1)/2*m0^2);
% %来流雷诺数
% miu0=1.716*10^(-5);%sutherland公式中t0为273.15K时的气体粘度系数
% miu=miu0*(t0/273.11)^1.5*((273.11+110.56)/(t0+110.56));%sutherland公式计算的气体粘度系数
% Rel=r0*v0*abs(x(1,1)/10^3)/(miu);%计算雷诺数，粘度
% %边界层位移厚度的计算
% Cw=(Tck/t0)^(0.5)*((t0+110.4)/(Tck+110.4));
% pisai=(m0)^3/(Rel)^0.5*Cw^0.5;
% epsl=0.425*(gama-1)*(Tbm/Tad+0.35)*pisai;
% for i=1:n1-1
%     for j=1:n2-1
%  ddeta=(abs(x2(1,1))+x2(i,j))*epsl/(m0*(pf_cone1(i,j)^0.5))*(1+epsl/(2*pf_cone1(i,j))*)
%         
%     end
% end
  
%CST上表面升阻比计算，直接用每根流线上的压力分布进行面元积分分解计算
L2=0;
D2=0;
for i=1:n1-1%n1为吻切面的数目即流线数目，也就是x2,y2,z2, pf_cone1的行数，行是流线
    for j=1:n2-1%n2为每一根流线上的点数，也就是x2,y2,z2, pf_cone1的列数，列是流线上的点数
          dD2=pf_cone1(i,j)*sin(theta(i,j)*pi/180);
          dL2=-pf_cone1(i,j)*cos(theta(i,j)*pi/180);
          if isnan(dD2)||isnan(dL2)
          else
            L2=L2+dL2;
            D2=D2+dD2;
          end
    end
end

%层流摩阻估算，适用于高度40km以上，引自《高超声速气动外形设计方法综述》，形状因子namada和修正系数Kd不准
miu0=1.716*10^(-5);%sutherland公式中t0为273.15K时的气体粘度系数
%参考温度Tck计算
Tbm=1000;%壁面温度自己给定
Tck=t0*(1.28+0.023*m0^2+0.58*(Tbm/t0-1));
miu=miu0*(Tck/273.11)^1.5*((273.11+110.56)/(Tck+110.56));%sutherland公式计算的气体粘度系数
namada=2.6*(1+0.2*AOA+0.01*AOA^2);%形状因子
Sf=(Area_up+Area_st)/(10^6);%迎风面和背风面面积和
Sref=(Area_base)/(10^6);%参考面积为出口截面面积


%迎风面摩阻总值法
Rel=r0*v0*abs(x(1,1))/10^3/miu;
Cxfp=1.328/Rel^0.5;%平板摩阻系数
Cxf=Cxfp*(1+0.03*m0^2)^(-1/3)*namada*Sf/Sref;%修正后的摩阻系数
D31=0.5*r0*v0^2*Area_up/(10^6)*Cxf;
D32=0.5*r0*v0^2*Area_st/(10^6)*Cxf;
D3=D31+D32;

% %迎风面摩阻,平均法
% D31=0;
% D32=0;
% for i=1:n1-1
%     for j=1:n2-1
%          Rel(i,j)=r0*v0*((abs(x(1,1))+0.01+x(i,j))/10^3)/(miu);%计算雷诺数，粘度
%          Cxfp(i,j)=1.328/(Rel(i,j))^0.5;%平板摩阻系数
%          Cxf(i,j)=Cxfp(i,j)*(1+0.03*m0^2)^(-1/3)*namada*Sf/Sref;%修正后的摩阻系数
%          AD=[(x(i+1,j+1))-x(i,j) (y(i+1,j+1))-y(i,j) (z(i+1,j+1))-z(i,j) ];
%          BC=[(x(i,j+1))-x(i+1,j) (y(i,j+1))-y(i+1,j) (z(i,j+1))-z(i+1,j) ];
%          n_up=cross(BC,AD);
%          dArea_up=0.5*norm(n_up)/(10^6);
%          dD31=0.5*r0*v0^2*dArea_up*Cxf(i,j);
%           if isnan(dD31)
%           else
%             D31=D31+dD31;
%           end
%     end
% end
% %背风面摩阻
% for i=1:n1-1
%     for j=1:n2-1
%          Rel(i,j)=r0*v0*((abs(x1(1,1))+0.01+x1(i,j))/10^3)/(miu);%计算雷诺数，粘度
%          Cxfp(i,j)=1.328/(Rel(i,j))^0.5;%平板摩阻系数
%          Cxf(i,j)=Cxfp(i,j)*(1+0.03*m0^2)^(-1/3)*namada*Sf/Sref;%修正后的摩阻系数
%          AD=[(x1(i+1,j+1))-x1(i,j) (y1(i+1,j+1))-y1(i,j) (z1(i+1,j+1))-z1(i,j) ];
%          BC=[(x1(i,j+1))-x1(i+1,j) (y1(i,j+1))-y1(i+1,j) (z1(i,j+1))-z1(i+1,j) ];
%          n_up=cross(BC,AD);
%          dArea_up=0.5*norm(n_up)/(10^6);
%          dD3=0.5*r0*v0^2*dArea_up*Cxf(i,j);
%           if isnan(dD3)
%           else
%             D32=D32+dD3;
%           end
%     end
% end
% D3=D31+D32;

Kd=1-0.015*AOA;%形状修正系数
Cxd=1.43/m0^2*(Area_base/10^6)/Sref*Kd;%修正后的底阻系数
D4=Cxd*0.5*r0*v0^2*(Area_base/10^6);%底阻





L=L1+L2;
D=D1+D2+D3+D4;
lift_to_drag=L/D

