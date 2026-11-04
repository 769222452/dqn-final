function surface_out(x_y_z_xingxian,D,index)
data_xingxian_new = shujuzhenghe(x_y_z_xingxian,D,index);
fid = fopen('waverider_coarse.dat','wt');
for i = 1:length(x_y_z_xingxian(1,1,:))
    fprintf(fid,'ROW\n');
    fprintf(fid,'%f %f %f \n',x_y_z_xingxian(:,1:3,i)'*1000);
end
fid = fopen('waverider_fine.dat','wt');
for i = 1:length(x_y_z_xingxian(1,1,:));
    fprintf(fid,'ROW\n');
    fprintf(fid,'%f %f %f \n',data_xingxian_new(:,1:3,i)'*1000);
end
end
