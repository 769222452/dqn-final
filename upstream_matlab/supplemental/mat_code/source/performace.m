clc
clear
load x_y_z_up x_y_z_up
load x_y_z_wav x_y_z_wav
load p_wav_up
load p_wav
AOA=-4; Vector_V=[cosd(AOA) 0 sind(AOA)]'; % AOA为攻角，AOS为侧滑角，Vector_V表示来流方向
rat=rotate_matrix(Vector_V,[1 0 0]'); % rat为旋转矩阵，rat*a=b, inv（rat）*b=a；
M=length(x_y_z_wav(1,1,:));
for i=1:M
    % x_y_z_wav(:,3,i)=-x_y_z_wav(:,3,i);
    % x_y_z_up(:,3,i)=-x_y_z_up(:,3,i);
    x_y_z_xingxian_1=rat*[x_y_z_wav(:,1,i),x_y_z_wav(:,2,i),x_y_z_wav(:,3,i)]';
    x_y_z_xingxian_2=rat*[x_y_z_up(:,1,i),x_y_z_up(:,2,i),x_y_z_up(:,3,i)]';
    x_y_z_wav_new(:,1,i)=x_y_z_xingxian_1(1,:);
    x_y_z_wav_new(:,2,i)=x_y_z_xingxian_1(2,:);
    x_y_z_wav_new(:,3,i)=x_y_z_xingxian_1(3,:);
    x_y_z_up_new(:,1,i)=x_y_z_xingxian_2(1,:);
    x_y_z_up_new(:,2,i)=x_y_z_xingxian_2(2,:);
    x_y_z_up_new(:,3,i)=x_y_z_xingxian_2(3,:);   
    wav_index(i)=length(x_y_z_wav(:,1,1));
    x_y_z_xingxian_1=[]; x_y_z_xingxian_2=[];
end
% 计算各方向的受力
[Force_x_upper,Force_y_upper,Force_z_upper,Sw_upper,Sb_upper,Sp_upper,My_upper]=force_upper(x_y_z_up_new,p_wav_up,M);
[Force_x_lower,Force_y_lower,Force_z_lower,Sw_lower,Sb_lower,Sp_lower,My_lower]=force_lower(x_y_z_wav_new,p_wav,M);
Force_x = Force_x_upper+Force_x_lower;
Force_y = Force_y_upper+Force_y_lower;
Force_z = Force_z_upper-Force_z_lower;
My = My_lower-My_upper;
% L_D=L/D;
L_D=Force_z/Force_x;
Sw = Sw_upper+Sw_lower;
Sp = 2*Sp_lower;
Sb = abs(Sb_upper)+abs(Sb_lower);
V=2*volume(x_y_z_up,x_y_z_wav);
eta = V^(2/3)/Sp;
Xcp=My*Force_z/(Force_z^2+Force_x^2)/x_y_z_wav_new(1,1,1);
% Xcp=-My_lower*Force_z_lower/(Force_z_lower^2+Force_x_lower^2)/x_y_z_wav_new(1,1,1);