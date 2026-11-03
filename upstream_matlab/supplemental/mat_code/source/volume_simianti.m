function [V] = volume_simianti(point1,point2,point3,point4)
    A=point2-point1;
    B=point3-point1;
    C=point4-point1;
    S = cross(A,B);
    V = abs(dot(S,C)/6);
end