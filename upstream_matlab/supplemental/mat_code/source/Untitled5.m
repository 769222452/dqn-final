clc
clear
a_1 = 1.2; b_1 = 1.456;
a_2 = 4.44; b_2 = 3.64;
x_1 = 0; x_2 = 7;
syms x1 y1 x2 y2 x3 y3 a b c 
[a,b,c]=solve('y1 = a*x1^3+b*x1^2+c*x1+0.76','y2 = 3*a*x2^2+2*b*x2+c','y3 = 3*a*x3^2+2*b*x3+c','a','b','c');
x1 = 7; y1 = 1;
x2 = 0; y3 = 0.05428;
x3 = 7; y2 = 0.01428;
a = eval(a); b = eval(b); c = eval(c);
syms x_v A_1 B_1
u = (x_v-x_1)/(x_2-x_1);
A_1 = a_1*(1-u)+a_2*u;
B_1 = b_1*(1-u)+b_2*u;
A_B = A_1/B_1;
% A_B = B_1/A_1;
x_v = linspace(x_1,x_2,20);
x_v = 0.25*7;
% a_b = eval(A_1/B_1);
% b = eval(B_1);
% plot(x_v,b);
% axis equal
% hold on
A_B = a*x_v.^3+b*x_v.^2+c*x_v+0.76;
% A_B = (0.24*x_v/7+0.76);
% A_B = eval(A_B);
plot(x_v,A_B);
hold on