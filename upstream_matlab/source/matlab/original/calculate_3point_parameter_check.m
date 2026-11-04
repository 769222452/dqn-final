function [  x4,y4,p4,r4,v4,a4 ] = calculate_3point_parameter_check( x1,y1,p1,r1,v1,a1,x2,y2,p2,r2,v2,a2,x3,y3,p3,r3,v3,a3,x4,y4,p4,r4,v4,a4 )
global g
difference_BM=1;f12=atan((y2-y1)/(x2-x1));
D1=sqrt(g*p1/r1);M1=v1/D1;B1=a1-asin(1/M1);
D2=sqrt(g*p2/r2);M2=v2/D2;B2=a2-asin(1/M2);
pM=(p3+p4)/2;rM=(r3+r4)/2;vM=(v3+v4)/2;aM=(a3+a4)/2;
DM=sqrt(g*pM/rM);MM=vM/DM;BM=aM-asin(1/MM);
n_step=0;
while difference_BM>0.0001
    n_step=n_step+1;
    [x4,y4]=intercept(x1,y1,f12,x3,y3,BM);
    B41=B2+(y4-y2)*(B1-B2)/(y1-y2);
    difference_BM=abs(B41-BM);
    BM=B41; 
    if n_step>=30
        x4=NAN; y4=NAN;
        break
    end
end
p4=p2+(y4-y2)*(p1-p2)/(y1-y2);
v4=v2+(y4-y2)*(v1-v2)/(y1-y2);
r4=r2+(y4-y2)*(r1-r2)/(y1-y2);
a4=a2+(y4-y2)*(a1-a2)/(y1-y2);
end

