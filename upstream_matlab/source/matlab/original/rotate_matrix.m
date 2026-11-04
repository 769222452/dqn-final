function [rat]=rotate_matrix(a,b) % the rotate matrix for 'rat*a=b'
    b=b/norm(b); a=a/norm(a); % 将输入的向量单位化
    rotate_angle=acos(dot(a,b)); % 求出两向量之间的夹角
    rotate_axis=cross(a,b); % 求出旋转轴，即两向量构成平面的法向量
    rotate_axis=rotate_axis/norm(rotate_axis); % 旋转轴向量单位化
    matrix=zeros(3,3);
    matrix(1,2)=-rotate_axis(3);
    matrix(1,3)=rotate_axis(2);
    matrix(2,1)=rotate_axis(3);
    matrix(2,3)=-rotate_axis(1);
    matrix(3,1)=-rotate_axis(2);
    matrix(3,2)=rotate_axis(1);
    rat=eye(3)+(1-cos(rotate_angle))*matrix*matrix+sin(rotate_angle)*matrix; % 应用罗德里格公式求出旋转矩阵
    if isnan(rat)==1
        rat=eye(3);
    end
end