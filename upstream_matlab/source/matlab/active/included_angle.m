function [afa]=included_angle(V2)
    V1=[1,0];
    V1_M = sqrt(V1(1)^2+V1(2)^2);
    V2_M = sqrt(V2(1)^2+V2(2)^2);
    beta = acos((V1(1)*V2(1)+V1(2)*V2(2))/V1_M/V2_M);
    if (V1(1)*V2(2)-V2(1)*V1(2))>=0
        afa = beta;
    else
        afa = 2*pi-beta;
    end
end