function [V] = volume_wumianti(point1,point2,point3,point4,point5)
    V1 = volume_simianti(point1,point2,point3,point4);
    V2 = volume_simianti(point1,point2,point5,point4);
    V = V1+V2;
end