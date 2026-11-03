function [data_xingxian_new]=shujuzhenghe(x_y_z_xingxian,D,index)
M = 200;
for i = 1:length(x_y_z_xingxian(1,1,:));
    data_xingxian_new(:,1,i) = linspace(x_y_z_xingxian(1,1,i),D,M);
    for j = 1:M
        data_xingxian_new(j,2,i) = interp1(x_y_z_xingxian(1:index(i),1,i),x_y_z_xingxian(1:index(i),2,i),data_xingxian_new(j,1,i));
        data_xingxian_new(j,3,i) = interp1(x_y_z_xingxian(1:index(i),1,i),x_y_z_xingxian(1:index(i),3,i),data_xingxian_new(j,1,i));
    end
end
end