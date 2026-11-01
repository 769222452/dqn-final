function [angle1_shock,velocity1_x,velocity1_y,mf_cone,pf_cone]=conical_flowfield(m0,delta0,gamma)
%% 输入参数
%来流马赫数
%初始化半锥角
%远场压力
%远场温度
%%%注意：
%1、远场压力和温度取1还是实际值不影响无量纲化压力、无量纲化温度和马赫数计算结果
%2、远场压力和温度取1还是实际值影响x和y方向速度的计算结果，各角度对应的速度为等比放大，且各角度放大比例相等，因此远场压力和温度取1还是实际值不影响流线追踪结果
%3、结论1和2经过验证，结果在同文件夹中的"压力温度取值影响验证.xls"中
mach_preshock = m0;
angle0_halfcone = delta0;
pressure_inlet =1;
temperature_inlet =1;
specific_heat_ratio=gamma;
gas_constant = 287;
%% 求解T-M方程
[index,angle1_shock,angle_span,velocitynormal,velocityradiul]=comptue_conicalfield(mach_preshock,angle0_halfcone, specific_heat_ratio);%输入来流马赫数mach_preshock、初始化半锥角angle0_halfcone
%输出激波角angle1_shock、球面坐标系的球面角angle_span和对应球面角上的垂直于球半径的球面角速度velocitynormal，球半径方向速度velocityradiul

%球坐标系下的无量纲化速度分量转换为无量纲的二维速度分量，参考左克罗129页公式16.73-16.76
mach = sqrt(velocitynormal.*velocitynormal+velocityradiul.*velocityradiul);%由无量纲的球面角速度velocitynormal和球半径方向速度velocityradiul计算对应的无量纲二维马赫数
velocity_x = velocityradiul.*cos(angle_span)- velocitynormal.*sin(angle_span);%由无量纲的球面角速度velocitynormal和球半径方向速度velocityradiul计算对应的无量纲二维Vx
velocity_y = velocityradiul.*sin(angle_span)+ velocitynormal.*cos(angle_span);%由无量纲的球面角速度velocitynormal和球半径方向速度velocityradiul计算对应的无量纲二维Vy
angle_flowturning = atan(velocity_y./velocity_x);%计算二维速度间的夹角即流动角



%将在求解T-M方程的过程中通过自由流临界速度无量纲化的马赫数、速度等量重新还原成有量纲的量
mach1 = sqrt(2*mach.*mach./(specific_heat_ratio+1-(specific_heat_ratio-1).*mach.*mach));%左克罗气体动力学上册 EQ3.189，无量纲马赫数与有量纲马赫数的关系式
%通过斜激波关系式中波前波后参数的计算关系式计算需要把velocitynormal、velocityradiul、velocity_x、velocity_y等速度值转换为计算有量纲的量所需的波后参数
temp = mach_preshock*sind(angle1_shock);%sind是“角度制”的三角函数
mach_aftshock = sqrt(((specific_heat_ratio+1)^2*temp^2*mach_preshock^2-4*(temp^2-1)*(specific_heat_ratio*temp^2+1))/...
    (2*specific_heat_ratio*temp^2-(specific_heat_ratio-1))/((specific_heat_ratio-1)*temp^2+2));%斜激波前后马赫数关系式 参考PPT
pressure_aftshock  = pressure_inlet*(2*specific_heat_ratio*temp^2-(specific_heat_ratio-1))/(specific_heat_ratio+1);%左克罗气体动力学下册 EQ16.66或者page286 EQ(7.49)斜激波前后压力关系式
temperature_aftshock = temperature_inlet*(2*specific_heat_ratio*temp^2-(specific_heat_ratio-1))*((specific_heat_ratio-1)*temp^2+2)/((specific_heat_ratio+1)^2*temp^2);%page286 EQ(7.51)的斜激波形式 斜激波前后温度关系式
totalpressure_aftshock = (1+(specific_heat_ratio-1)*mach_aftshock^2/2.)^(specific_heat_ratio/(specific_heat_ratio-1))*pressure_aftshock ;%总静压关系式
totaltemperature_aftshock = (1+(specific_heat_ratio-1)*mach_aftshock^2/2.)*temperature_aftshock;%总静温关系式
% "pressure_aftshock 、temperature_aftshock" see book of M.J.左克罗 "gas dynamics " rudin  page272 EQ(7.49)、(7.51)并改为斜激波形式
% "totalpressure_aftshock 、totaltemperature_aftshock" see book of M.J.左克罗 "gas dynamics " rudin  page107 EQ（3.76）、（3.69）
%球坐标系计算出来的velocitynormal、velocityradiul、velocity_x、velocity_y等无量纲的速度量，通过Vmax'这个临界声速转换为有量纲的量
velocity1_radiul= sqrt(2*specific_heat_ratio*gas_constant *totaltemperature_aftshock/(specific_heat_ratio+1)).*velocityradiul;
velocity1_normal = sqrt(2*specific_heat_ratio*gas_constant *totaltemperature_aftshock/(specific_heat_ratio+1)).*velocitynormal;
velocity1_x = sqrt(2*specific_heat_ratio*gas_constant *totaltemperature_aftshock/(specific_heat_ratio+1)).*velocity_x;
velocity1_y = sqrt(2*specific_heat_ratio*gas_constant *totaltemperature_aftshock/(specific_heat_ratio+1)).*velocity_y;
% "velocity1_radiul、velocity1_normal" is dimensional according the limited velocity（Vmax'）
% specified see book of M.J.左克罗 "gas dynamics "   page 124-125,EQ（3.187）

totalpressure_inlet = (1+(specific_heat_ratio-1)*mach_preshock^2/2.)^(specific_heat_ratio/(specific_heat_ratio-1))*pressure_inlet;%来流总压计算参考M.J.左克罗 "gas dynamics " page124，EQ3.182 
pressure1 = zeros(index+1,1);%给定锥激波波后流场压力、总压、总压恢复系数、温度等参数的矩阵大小
totalpressure1 = zeros(index+1,1);
totalpressure1_recovery = zeros(index+1,1);
temperature1 = zeros(index+1,1);
%计算锥激波波后流场的各项参数的大小
for i=1:index+1
    pressure1(i) = totalpressure_aftshock./(1+(specific_heat_ratio-1).*mach1(i)^2./2.)^(specific_heat_ratio./(specific_heat_ratio-1))/pressure_inlet;%总静温关系式see book of M.J.左克罗 "gas dynamics " rudin  page107 EQ（3.76）
    totalpressure1(i) = (1+(specific_heat_ratio-1).*mach1(i)^2./2.)^(specific_heat_ratio./(specific_heat_ratio-1)).*pressure1(i);%"totalpressure_aftshocksee book of M.J.左克罗 "gas dynamics " rudin  page107 EQ（3.76）
    totalpressure1_recovery(i) = totalpressure1(i)./totalpressure_inlet;%总压恢复系数
    temperature1(i) = totaltemperature_aftshock./(1+(specific_heat_ratio-1).*mach1(i)^2/2.)/temperature_inlet;%总静温关系式see book of M.J.左克罗 "gas dynamics " rudin  page107 EQ（3.69）
end
deltaf_cone=angle_span;%本函数返回锥激波流场的球面角
mf_cone=mach1;%本函数返回锥激波流场的马赫数
pf_cone=pressure1;%本函数返回锥激波流场的压力
