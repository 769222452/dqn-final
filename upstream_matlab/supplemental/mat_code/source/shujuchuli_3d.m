clc
clear
load p_l p_l
load a_l a_l
load x_l x_l
load y_l y_l
load v_l v_l
load r_l r_l
load x_inter x_inter
load y_inter y_inter
load z_inter z_inter
load x_y_z_xingxian x_y_z_xingxian
load v_x v_x
load v_y v_y
load v_z v_z
for i = 1:length(x_inter(1,1,:))
index(i)= find(x_y_z_xingxian(:,1,i),1,'last');
end
fid = fopen('flow_block2.dat','wt');
fprintf(fid,'TITLE="COMBINE_FIELD"\n');
fprintf(fid,'VARIABLES=x,y,z,p,h,h_r,vx,vy,vz\n');

% for i=2:length(x_inter(1,1,:))-1
% for j=1:130
% k = x_y_z_xingxian(j,1,i)- x_y_z_xingxian(1,1,1);
% 
% if k >=0
%     count(i) = j;
%     break 
% end
% end
% end


for k=1:length(x_inter(1,1,:))
    k
    for i=1:index(k)
        for j=1:length(x_inter(1,:,k))
            if i<j
                x_inter(i,j,k) = x_inter(i,i,k);
                y_inter(i,j,k) = y_inter(i,i,k);
                z_inter(i,j,k) = z_inter(i,i,k);                                
                p_l(i,j,k) = p_l(i,i,k);
                v_x(i,j,k) = v_x(i,i,k);
                v_y(i,j,k) = v_y(i,i,k);
                v_z(i,j,k) = v_z(i,i,k);
            end
        end
    end
    for i=index(k)+1:length(x_inter(:,1,k))
        x_inter(i,:,k) = x_inter(i-1,:,k);
        y_inter(i,:,k) = y_inter(i-1,:,k);
        z_inter(i,:,k) = z_inter(i-1,:,k);                                
        p_l(i,:,k) = p_l(i-1,:,k);
        v_x(i,:,k) = v_x(i-1,:,k);
        v_y(i,:,k) = v_y(i-1,:,k);
        v_z(i,:,k) = v_z(i-1,:,k);
    end   
end

    [data,np,nodes,Nele]=shujuzhenhe_3D(x_inter,y_inter,z_inter,p_l,v_x,v_x,v_x,v_y,v_z,length(x_inter(:,1,k)),length(x_inter(1,:,k)),length(x_inter(1,1,:)));    
    fprintf(fid,strcat('ZONE\t','N=',num2str(nodes),',E=',num2str(Nele),',DATAPACKING=POINT,ZONETYPE=FEBrick\n'));
    fprintf(fid,'%f %f %f %f %f %f %f\n',data');
    fprintf(fid,' %d %d %d %d %d %d %d %d\n',np');