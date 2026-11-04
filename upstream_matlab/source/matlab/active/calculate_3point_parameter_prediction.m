function [ x4,y4,p4,r4,v4,a4 ] = calculate_3point_parameter_prediction( x1,y1,p1,r1,v1,a1,x2,y2,p2,r2,v2,a2,x3,y3 )
g=1.4; % Explicit fixed ideal-gas geometry assumption
difference_B4=1;f12=atan((y2-y1)/(x2-x1));
D1=sqrt(g*p1/r1);M1=v1/D1;B1=a1-asin(1/M1);
D2=sqrt(g*p2/r2);M2=v2/D2;B2=a2-asin(1/M2);
B4=(B1+B2)/2; 
n_step=0;
while difference_B4>0.0001
    n_step=n_step+1;
    [x4,y4]=intercept(x1,y1,f12,x3,y3,B4);
    B41=B2+(y4-y2)*(B1-B2)/(y1-y2);
    difference_B4=abs(B41-B4);
    B4=B41;
    if n_step>=30
        error('V5:GeometryNonconverged','Characteristic iteration cap');
    end
end
p4=p2+(y4-y2)*(p1-p2)/(y1-y2);
v4=v2+(y4-y2)*(v1-v2)/(y1-y2);
r4=r2+(y4-y2)*(r1-r2)/(y1-y2);
a4=a2+(y4-y2)*(a1-a2)/(y1-y2);
end

