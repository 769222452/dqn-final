function [ x4,y4,p4,r4,v4,a4 ] = wall_boundary( x1,y1,p1,r1,v1,a1,x2,y2,p2,r2,v2,a2,a_wall )
global specific_heat_ratio
global xishu
D2=sqrt((specific_heat_ratio*p2)/r2);
M2=v2/D2;
w2 =asin(1/M2);
QM=sqrt(M2*M2-1)/(r2*v2*v2);
SM=xishu*sin(a2)/(y2*M2*cos(a2-w2));
[x4,y4]=intercept(x1,y1,a_wall,x2,y2,a2-w2);
TM=-SM*(x4-x2)+QM*p2-a2;
a4=a_wall;
D1=sqrt((specific_heat_ratio*p1)/r1);
R0=r1*v1;A0=D1*D1;T01=R0*v1+p1;T02=p1-A0*r1;
p4=(TM+a4)/QM;
v4=(T01-p4)/R0;
r4=(p4-T02)/A0;
difference1=1;
while abs(difference1)>0.1
    pM=(p2+p4)/2;
    aM=(a2+a4)/2;
    vM=(v2+v4)/2;
    rM=(r2+r4)/2;
    yM=(y2+y4)/2;
    DM=sqrt(specific_heat_ratio*pM/rM);
    MM=vM/DM;
    wM=asin(1/MM);
    QM=sqrt(MM*MM-1)/(rM*vM*vM);
    SM=xishu*sin(aM)/(yM*MM*cos(aM-wM));
    [x4,y4]=intercept(x1,y1,a_wall,x2,y2,aM-wM);
    TM=-SM*(x4-x2)+QM*p2-a2;
    p0=(p1+p4)/2;
    v0=(v1+v4)/2;
    r0=(r1+r4)/2; 
    D0=sqrt((specific_heat_ratio*p0)/r0);
    R0=r0*v0;A0=D0*D0;T01=R0*v1+p1;T02=p1-A0*r1;
    a41=a_wall;
    p41=(TM+a41)/QM;
    v41=(T01-p41)/R0;
    r41=(p41-T02)/A0;
    difference1=p41-p4;
    p4=p41;
    v4=v41;
    r4=r41;
    a4=a41;
end
end

