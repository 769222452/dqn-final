function fun=panel_parameters(p1,p2,p3)
% return 4 parameters a*x+b*y+c*z+d=0 [a,b,c,d] = >parameters
A=p2-p1;
B=p3-p1;

cr=cross(A,B);
a=cr(1);
b=cr(2);
c=cr(3);
d=-(a*p1(1)+b*p1(2)+c*p1(3));
fun=[a,b,c,d];