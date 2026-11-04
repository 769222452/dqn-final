function [ x3,y3,p3,r3,v3,a3 ] = inter_point( x1,y1,p1,r1,v1,a1,x2,y2,p2,r2,v2,a2 )
global specific_heat_ratio
global xishu
specific_heat_ratio=1.4;xishu=1;
D1=sqrt(specific_heat_ratio*p1/r1);M1=v1/D1;w1=asin(1/M1);%%%1点出的马赫数，马赫角miu
D2=sqrt(specific_heat_ratio*p2/r2);M2=v2/D2;w2=asin(1/M2);%%%2点出的马赫数，马赫角miu
Qp=sqrt((M2)^2-1)/r2/(v2)^2;
Sp=xishu*sin(a2)/y2/M2/cos(a2+w2);
QM=sqrt((M1)^2-1)/r1/(v1)^2;
SM=xishu*sin(a1)/y1/M1/cos(a1-w1);
[x3,y3]=intercept(x1,y1,a1-w1,x2,y2,a2+w2);%%%%%过1处的流线c-和过2处的特征线c+的交点3
Tp=-Sp*(x3-x2)+Qp*p2+a2;
TM=-SM*(x3-x1)+QM*p1-a1;
%%%%%%过3点出的特征线c0与12点连线的交点4
[x4,y4,p4,r4,v4,a4]=calculate_3point_parameter_prediction_2(x1,y1,p1,r1,v1,a1,x2,y2,p2,r2,v2,a2,x3,y3);  % calculate the parameter at point 3 in the prediction interpolation method
D4=sqrt(specific_heat_ratio*p4/r4);M4=v4/D4;w4=asin(1/M4);
R0=r4*v4;A0=D4*D4;T01=R0*v4+p4;T02=p4-A0*r4;
%%%%%%3点就是要求的新点，此时的结果是预估步的结果
p3=(Tp+TM)/(Qp+QM);
a3=Tp-Qp*p3;
v3=(T01-p3)/R0;
r3=(p3-T02)/A0;
difference_x=1;difference_y=1;difference_p=1;difference_r=1;difference_v=1;difference_a=1;
% calculate the parameter at the interior point in the correction method
%%%%%%%此时是进行校正步结果，及为最终的结果
while difference_x>0.0001 || difference_y>0.0001 || difference_p>0.1 || difference_r>0.0001 || difference_v>0.0001 || difference_a>0.0001
    pp=(p2+p3)/2;ap=(a2+a3)/2;vp=(v2+v3)/2;rp=(r2+r3)/2;yp=(y2+y3)/2;
    pM=(p1+p3)/2;aM=(a1+a3)/2;vM=(v1+v3)/2;rM=(r1+r3)/2;yM=(y1+y3)/2;
    Dp=sqrt(specific_heat_ratio*pp/rp);Mp=vp/Dp;wp=asin(1/Mp);Qp=sqrt((Mp)^2-1)/rp/(vp)^2;Sp=xishu*sin(ap)/yp/Mp/cos(ap+wp);
    DM=sqrt(specific_heat_ratio*pM/rM);MM=vM/DM;wM=asin(1/MM);QM=sqrt((MM)^2-1)/rM/(vM)^2;SM=xishu*sin(aM)/yM/MM/cos(aM-wM);
    [x31,y31]=intercept(x1,y1,aM-wM,x2,y2,ap+wp);
    Tp=-Sp*(x31-x2)+Qp*p2+a2;
    TM=-SM*(x31-x1)+QM*p1-a1;
    [x4,y4,p4,r4,v4,a4]=calculate_3point_parameter_check_2(x1,y1,p1,r1,v1,a1,x2,y2,p2,r2,v2,a2,x31,y31,p3,r3,v3,a3,x4,y4,p4,r4,v4,a4);  % calculate the parameter at point 3 in the correction interpolation method    
    p0=(p4+p3)/2;a0=(a4+a3)/2;v0=(v4+v3)/2;r0=(r4+r3)/2;
    D0=sqrt(specific_heat_ratio*p0/r0);R0=r0*v0;A0=D0*D0;T01=R0*v1+p1;T02=p1-A0*r1;    
    p31=(Tp+TM)/(Qp+QM);
    a31=Tp-Qp*p31;
    v31=(T01-p31)/R0;
    r31=(p31-T02)/A0;
    difference_x=abs(x31-x3);difference_y=abs(y31-y3);difference_p=abs(p31-p3);difference_r=abs(r31-r3);difference_v=abs(v31-v3);difference_a=abs(a31-a3);
    x3=x31;y3=y31;p3=p31;r3=r31;v3=v31;a3=a31;
end
end
