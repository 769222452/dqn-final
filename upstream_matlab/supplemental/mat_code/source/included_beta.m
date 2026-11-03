function [beta]=included_beta(v1,v2)
    v1_m = sqrt(v1(1)^2+v1(2)^2+v1(3)^2);
    v2_m = sqrt(v2(1)^2+v2(2)^2+v2(3)^2);
    beta = acos((v1(1)*v2(1)+v1(2)*v2(2)+v1(3)*v2(3))/v1_m/v2_m);
end