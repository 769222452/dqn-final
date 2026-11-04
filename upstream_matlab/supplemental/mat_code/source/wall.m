function [ x4,y4,p4,r4,v4,a4 ] = wall( x1,y1,p1,r1,v1,a1,x2,y2,p2,r2,v2,a2 )
% 给定壁面压力
global specific_heat_ratio
global xishu
specific_heat_ratio=1.4;xishu=1; 
a_wall_1=a1;
[ x4,y4,p4,r4,v4,a4 ] = wall_boundary( x1,y1,p1,r1,v1,a1,x2,y2,p2,r2,v2,a2,a_wall_1 );
difference1=p4-pressure_function(x4);
a_wall_2=0.99*a_wall_1;
while abs(difference1)>0.001
    [ x4,y4,p4,r4,v4,a4 ] = wall_boundary( x1,y1,p1,r1,v1,a1,x2,y2,p2,r2,v2,a2,a_wall_2 );
    difference2=p4-pressure_function(x4);
    a_wall_3=a_wall_2-difference2*(a_wall_2-a_wall_1)/(difference2-difference1);
    a_wall_1=a_wall_2;
    a_wall_2=a_wall_3;
    difference1=difference2;
end
end

