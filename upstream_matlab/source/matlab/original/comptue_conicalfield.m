function [index,angle1_shock,angle_span,velocitynormal,velocityradiul]=comptue_conicalfield(mach_preshock,angle0_halfcone, specific_heat_ratio)
%已知圆锥的半锥角、来流马赫数，求相应的激波角
format long 
esp1 = 1.0; %控制变量

angle0_halfcone = angle0_halfcone.*pi/180;%初始化半锥角进行角度和弧度转换
angle1_shock =angle0_halfcone+(asin(1./mach_preshock));%给定迭代求解时激波角的初值，(asin(1./mach_preshock)为自由流马赫角，参考左克罗气体动力学下册128- EQ16.64

while abs(esp1)>1e-5 %给定初始激波角angle1_shock，通过数值积分求解T-M方程得到新的计算出来的激波角angle1_shock，两者间的误差小于0.00001即可得到对应圆锥半角的激波角,该误差值可调，能够得到更精准的解算结果
machnormal_preshock = mach_preshock*sin(angle1_shock);
beta = atan(tan(angle1_shock)*2./(specific_heat_ratio+1)*(1/machnormal_preshock^2+(specific_heat_ratio-1)/2.));
%参考左克罗气体动力学下册129- EQ16.67
temp=sqrt((specific_heat_ratio+1)*mach_preshock^2./(2+(specific_heat_ratio-1)*mach_preshock^2));
%参考左克罗气体动力学下册129- EQ3.189
mach_aftshock = temp*(sin(angle1_shock)/sin(beta))*(2./((specific_heat_ratio+1)*machnormal_preshock^2)+(specific_heat_ratio-1)/(specific_heat_ratio+1));
%参考左克罗气体动力学下册129- EQ16.68
velocity0_radiul = mach_aftshock*cos(beta);%给定假定的激波角时求得的第一个U向分速度
%参考左克罗气体动力学下册129- EQ16.70
velocity0_normal = -mach_aftshock*sin(beta);%给定假定的激波角时求得的第一个V向分速度
y0 = [velocity0_radiul velocity0_normal];
%参考左克罗气体动力学下册129- EQ16.71

anglespan_halfcone = [angle1_shock:-0.001:angle0_halfcone];%确定激波面和锥面之间数值积分算法的步长为0.001，即从半锥角到激波角分成以0.001为步长的N个球面角
[angle,y] = ode45(@weifen,anglespan_halfcone,y0);%根据给定的每一步假定的球面角及初始的UV向分速度求解出紧靠下游的激波角和UV向分速度，4阶龙格库塔法对微分方程组进行数值求解
velocity_radiul = y(:,1);%每一步假定的球面角对应的U向分速度
velocity_normal = y(:,2);%每一步假定的球面角对应的V向分速度

for j = 2:numel(velocity_radiul)%初始加上迭代的所有点，找到上一步假定的球面角对应的V向分速度为0的点，对应的球面角即为此时的圆锥半角，但该圆锥半角
    %与输入给定的圆锥半角存在差值，需要迭代求解才能得出对应于输入给定圆锥半角的激波角
    if velocity_normal(end)>0%判断出V向速度为0的位置
        if velocity_normal(j)>0&&velocity_normal(j-1)<0
            index=j;
            break
        end
        else
        index=numel(velocity_radiul);
    end
end
        velocitynormal_chazhi1 = velocity_normal(1:index);%将velocity_normal大于0的速度值都赋给 velocitynormal_chazhi1
        anglehalfcone_chazhi1 = angle(1:index);%将velocity_normal大于0时对应的angle激波角值赋给anglehalfcone_chazhi1

        angle_velocity0 = interp1(velocitynormal_chazhi1,anglehalfcone_chazhi1,0,'spline'); %cubic方法，插值求解出V向分速度为0时的准确球面角值，
        %其即为此时的圆锥半角 angle_velocity0 
        esp1 = (angle0_halfcone-angle_velocity0)/10.;%计算出来的V向分速度为0时的圆锥半角angle_velocity0 和真正的输入的圆锥半角angle0_halfcon的差值除以10作为计算精度的标记值
        angle1_shock = angle1_shock+esp1;%通过第一次计算出来的圆锥半角的差值esp1来修正假定的激波角 angle1_shock
end 
        velocityradiul_chazhi1 = velocity_radiul(1:index);%将迭代求解出来的V向速度为0时对应的U向的速度值velocity_radiul(1:index)赋值给velocityradiul_chazhi1
        velocityradiul0 = interp1(velocitynormal_chazhi1,velocityradiul_chazhi1,0,'spline');%插值法准确找出V向分速度为0时的U向分速度
 %---------------------------output--------------------------------------%      
        angle1_shock = angle1_shock*180./pi;%迭代求解出的激波角
        angle_span = [anglehalfcone_chazhi1
                      angle_velocity0];%,球面角扇，从壁面到激波角的分出了N份
        velocitynormal =[velocitynormal_chazhi1
                         0];%对应球面角上的V向分速度
        velocityradiul=[velocityradiul_chazhi1
                        velocityradiul0];%对应球面角上的U向分速度
                                                         
%------------------------subfuction of EQ T-M----------------------------%
function dy=weifen(angle_halfcone,y) 
global specific_heat_ratio
dy = zeros(2,1);%定义dy的大小为两行一列

dy(1) = y(2);%U向分速度与球面角的微分值等于传入的初始V向分速度
%see book of M.J.左克罗 "gas dynamics "  page 128,EQ（16.60）

temp  = (specific_heat_ratio+1)/2.-(specific_heat_ratio-1)/2.*(y(1)^2+y(2)^2);
% sound speed is un_dimensional according the critical sound speed(a*)
%see book of M.J.左克罗 "gas dynamics "   page 128,EQ（16.62）与128,EQ（16.63）

dy(2) = -y(1)+(temp*(y(1)+y(2)/tan(angle_halfcone)))/(y(2)^2-temp);%计算出V向分速度与球面角的微分值
%see book of M.J.左克罗 "gas dynamics "   page 128,EQ（16.61）
