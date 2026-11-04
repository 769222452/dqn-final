clc
clear
load x_y_z_xingxian x_y_z_xingxian
load num_index num_index
load x_y_z_xingxian2 x_y_z_xingxian2
load num_index2 num_index2
n1 =  find(num_index2,1,'first'); n2 = find(num_index2,1,'last');
fid = fopen('waverider_1.dat','wt');
for i = 1:length(x_y_z_xingxian(1,1,:))
    fprintf(fid,'ROW\n');
    fprintf(fid,'%f %f %f \n',x_y_z_xingxian(1:num_index(i),1:3,i)'*1000);
end
fid = fopen('waverider_2.dat','wt');
for i = n1:n2
    fprintf(fid,'ROW\n');
    fprintf(fid,'%f %f %f \n',x_y_z_xingxian2(1:num_index2(i),1:3,i)'*1000);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [data_xingxian]=shujuzhenghe(x_y_z_xingxian,x_y_z_xingxian2,num_index,num_index2);
% fid = fopen('waverider_conical.dat','wt');
% for i = 1:M
%     fprintf(fid,'ROW\n');
%     fprintf(fid,'%f %f %f \n',data_xingxian(:,1:3,i)'*1000);
% end