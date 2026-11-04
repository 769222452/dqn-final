x_wav_end=x_fct_1(end); nn=100;
for i = 1:M
% for i=15
    x_y_z_xingxian(:,1,i) = x_inter_3(:,1,i);
    x_y_z_xingxian(:,2,i) = y_inter_3(:,1,i);
    x_y_z_xingxian(:,3,i) = z_inter_3(:,1,i);
    p_wav_1=p_l_3(:,1,i);
    vx_wav_1=v_x_3(:,1,i); vy_wav_1=v_y_3(:,1,i); vz_wav_1=v_z_3(:,1,i);
    x_y_z_xingxian_r=inv(rat)*[x_y_z_xingxian(:,1,i),x_y_z_xingxian(:,2,i),x_y_z_xingxian(:,3,i)]';
    coordinate_x=x_y_z_xingxian_r(1,:);
    coordinate_y=x_y_z_xingxian_r(2,:);
    coordinate_z=x_y_z_xingxian_r(3,:);
    wav_x(i,:)=linspace(coordinate_x(1),x_wav_end,nn);
    for j=1:nn
        wav_y(i,j)=interp1(coordinate_x,coordinate_y,wav_x(i,j));
        wav_z(i,j)=interp1(coordinate_x,coordinate_z,wav_x(i,j));
        p_wav(i,j)=interp1(coordinate_x,p_wav_1,wav_x(i,j));
        vx_wav(i,j)=interp1(coordinate_x,vx_wav_1,wav_x(i,j));
        vy_wav(i,j)=interp1(coordinate_x,vy_wav_1,wav_x(i,j));
        vz_wav(i,j)=interp1(coordinate_x,vz_wav_1,wav_x(i,j));
        x_y_z_wav(j,1,i)=wav_x(i,j); x_y_z_wav(j,2,i)=wav_y(i,j); x_y_z_wav(j,3,i)=wav_z(i,j);
    end
    coordinate_x=[]; coordinate_y=[];coordinate_z=[]; p_wav_1=[]; vx_wav_1=[]; vy_wav_1=[]; vz_wav_1=[];
end
wav_x(M,:)=wav_x(M,1); wav_y(M,:)=wav_y(M,1); wav_z(M,:)=wav_z(M,1);
p_wav(M,:)=p_wav(M,1); vx_wav(M,:)=vx_wav(M,1); vy_wav(M,:)=vy_wav(M,1); vz_wav(M,:)=vz_wav(M,1);
x_y_z_wav(:,1,M)=wav_x(M,1); x_y_z_wav(:,2,M)=wav_y(M,1); x_y_z_wav(:,3,M)=wav_z(M,1);
% 生成上壁面
for i=1:M
    wav_up_x(i,:)=linspace(x_fct_1(i),x_wav_end,nn);
    for j=1:nn
        wav_up_y(i,j)=(wav_up_x(i,j)-wav_up_x(i,1))*Vector_V(2)/Vector_V(1)+y_fct_1(i);
        wav_up_z(i,j)=(wav_up_x(i,j)-wav_up_x(i,1))*Vector_V(3)/Vector_V(1)+z_fct_1(i);
        p_wav_up(i,j)=p0;
        x_y_z_up(j,1,i)=wav_up_x(i,j); x_y_z_up(j,2,i)=wav_up_y(i,j); x_y_z_up(j,3,i)=wav_up_z(i,j);
    end
end
Time_LTOCs = toc;
figure(1)
mesh(wav_x,wav_y,wav_z,'edgecolor','g','facecolor','none') %
axis equal
hold on
mesh(wav_up_x,wav_up_y,wav_up_z,'edgecolor','b','facecolor','none')
hold on
% 计算体积
V=volume(x_y_z_up,x_y_z_wav);
save x_y_z_up x_y_z_up
save x_y_z_wav x_y_z_wav
save p_wav_up p_wav_up
save p_wav p_wav
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid = fopen('waverider_AOA_-4.dat','wt');
for j = 1:nn
    fprintf(fid,'ROW\n');
    for i = 1:M
        fprintf(fid,'%e\t',wav_x(i,j)*1000);
        fprintf(fid,'%e\t',wav_y(i,j)*1000);
        fprintf(fid,'%e\n',-wav_z(i,j)*1000);
    end  
end
fid = fopen('waverider_AOA_-4_up.dat','wt');
for j = 1:nn
    fprintf(fid,'ROW\n');
    for i = 1:M
        fprintf(fid,'%e\t',wav_up_x(i,j)*1000);
        fprintf(fid,'%e\t',wav_up_y(i,j)*1000);
        fprintf(fid,'%e\n',-wav_up_z(i,j)*1000);
    end  
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid = fopen('wav_surface_1.dat','wt');
fprintf(fid,'TITLE="COMBINE_FIELD"\n');
fprintf(fid,'VARIABLES=x,y,z,p,vx,vy,vz\n');
[data,np,nodes,Nelayer]= shujuzhenhe_2D( wav_x,wav_y,wav_z,p_wav,vx_wav,vy_wav,vz_wav,p0,length(wav_x(:,1)),length(wav_x(1,:)));       
fprintf(fid,strcat('ZONE\t','N=',num2str(nodes),',E=',num2str(Nelayer),',DATAPACKING=POINT,ZONETYPE=FEQuadrilateral\n'));            
fprintf(fid,'%f %f %f %f %f %f %f\n',data');
fprintf(fid,' %d %d %d %d %d %d %d\n',np');