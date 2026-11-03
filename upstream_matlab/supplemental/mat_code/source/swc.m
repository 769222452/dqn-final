function [a,b,c]=swc()
clc
clear
M = 40;
A = 0.05;
%%%% 设计截面唇口参数
%%%%% 设计截面内激波形状
syms a b c x0 y0 x1 y1 x2 y2
y0 = a*x0^2+b*x0+c;
y1 = a*x1^2+b*x1+c;
y2 = a*x2^2+b*x2+c;
[a,b,c] = solve('y0 = a*x0^2+b*x0+c;','y1 = a*x1^2+b*x1+c;','y2 = a*x2^2+b*x2+c;','a','b','c');
x0 = -0.5;y0 = 0;
x1 = 0.5;y1 = 0;
x2 = 0;y2 = 0.37;
a = eval(a);b = eval(b);c = eval(c);
end