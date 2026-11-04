
function [x3,y3,p3,r3,v3,a3,det3,v3_x,v3_y] = qiujie_inter(x1,y1,p1,r1,v1,a1,x2,y2,p2,r2,v2,a2)

global g
global xishu
g=1.4;xishu=1; 
[x3,y3,p3,r3,v3,a3,det3]=reverse_shock(x1,y1,p1,r1,v1,a1,x2,y2,p2,r2,v2,a2);       
v3_x=v3*cos(a3);v3_y=v3*sin(a3);
end













