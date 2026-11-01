function [ x3,y3,p3,r3,v3,a3 ] = qiujie_end_point_improved( x1,y1,p1,r1,v1,a1,x2,y2,p2,r2,v2,a2,local_a_xishu )
global g
global xishu
global a_xishu
g=1.4;xishu=1; a_xishu=local_a_xishu;
D1=sqrt(g*p1/r1);M1=v1/D1;w1=asin(1/M1);%%%1点处的马赫数，马赫角miu
D2=sqrt(g*p2/r2);M2=v2/D2;w2=asin(1/M2);%%%2点处的马赫数，马赫角miu
a3=a_xishu*a1;
%%%%%%求3点处坐标%%%%%%
[x3,y3]=intercept(x1,y1,(a1+a3)/2,x2,y2,pi/2); %%%%%过1处的流线c0和过2处的截至坐标系
%%%%%%过3点出的特征线c-与12点连线的交点4%%%%%%
[x4,y4,p4,r4,v4,a4]=calculate_3point_parameter_prediction(x1,y1,p1,r1,v1,a1,x2,y2,p2,r2,v2,a2,x3,y3); 
D4=sqrt(g*p4/r4);M4=v4/D4;w4=asin(1/M4);
QM=sqrt((M4)^2-1)/r4/(v4)^2;
SM=xishu*sin(a4)/y4/M4/cos(a4-w4);
TM=-SM*(x3-x4)+QM*p4-a4;
%%%%%%3点就是要求的新点，此时的结果是预估步的结果
R0=r1*v1;A0=D1*D1;T01=R0*v1+p1;T02=p1-A0*r1;
p3=(TM+a3)/QM;
v3=(T01-p3)/R0;
r3=(p3-T02)/A0;
difference_p=1;difference_r=1;difference_v=1;difference_a=1;
% calculate the parameter at the interior point in the correction method
%%%%%%%此时是进行校正步结果，及为最终的结果
n_step=0;
while abs(difference_p)>0.1 || abs(difference_r)>0.0001 || abs(difference_v)>0.0001
    n_step=n_step+1;
    [x4,y4,p4,r4,v4,a4]=calculate_3point_parameter_check(x1,y1,p1,r1,v1,a1,x2,y2,p2,r2,v2,a2,x3,y3,p3,r3,v3,a3,x4,y4,p4,r4,v4,a4);  % calculate the parameter at point 3 in the correction interpolation method
    pM=(p4+p3)/2;aM=(a4+a3)/2;vM=(v4+v3)/2;rM=(r4+r3)/2;yM=(y4+y3)/2;
    DM=sqrt(g*pM/rM);MM=vM/DM;wM=asin(1/MM);
    QM=sqrt((MM)^2-1)/rM/(vM)^2;
    SM=xishu*sin(aM)/yM/MM/cos(aM-wM);
    TM=-SM*(x3-x4)+QM*p4-a4;
    p0=(p1+p3)/2; v0=(v1+v3)/2; r0=(r1+r3)/2; D0=sqrt((g*p0)/r0);
    R0=r0*v0; A0=D0*D0; T01=R0*v1+p1; T02=p1-A0*r1;
    p31=(TM+a3)/QM;
    v31=(T01-p31)/R0;
    r31=(p31-T02)/A0;
    difference_p=p31-p3; difference_r=r31-r3; difference_v=v31-v3;
    p3=p31; r3=r31; v3=v31;
    if n_step>=20
        x3=NAN; y3=NAN;
        break
    end
end
end

