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
x_y_z_xingxian = real(x_y_z_xingxian);
fid = fopen('wav_surface.dat','wt');
fprintf(fid,'TITLE="COMBINE_FIELD"\n');
fprintf(fid,'VARIABLES=x,y,z,p,vx,vy,vz\n');
for i = 1:length(x_inter(1,1,:))
    wav_x(i,:) = x_inter(:,1,i)';
    wav_y(i,:) = y_inter(:,1,i)';
    wav_z(i,:) = z_inter(:,1,i)';
    p_wav(i,:) = p_l(:,1,i)';
    vx_wav(i,:) = v_x(:,1,i)';
    vy_wav(i,:) = v_y(:,1,i)';
    vz_wav(i,:) = v_z(:,1,i)';
end
for i = 1:length(x_inter(1,1,:))
index(i)= find(x_y_z_xingxian(:,1,i),1,'last');
end
n = 50;
    for i = 1:length(x_inter(1,1,:))
        wav_x_new(i,:) = linspace(wav_x(i,1),wav_x(1,1),n);
        for j = 1:n
            wav_y_new(i,j) = interp1(wav_x(i,1:index(i)),wav_y(i,1:index(i)),wav_x_new(i,j));
            wav_z_new(i,j) = interp1(wav_x(i,1:index(i)),wav_z(i,1:index(i)),wav_x_new(i,j));
             p_wav_new(i,j) = interp1(wav_x(i,1:index(i)),p_wav(i,1:index(i)),wav_x_new(i,j));
             vx_wav_new(i,j) = interp1(wav_x(i,1:index(i)),vx_wav(i,1:index(i)),wav_x_new(i,j));
             vy_wav_new(i,j) = interp1(wav_x(i,1:index(i)),vy_wav(i,1:index(i)),wav_x_new(i,j));
             vz_wav_new(i,j) = interp1(wav_x(i,1:index(i)),vz_wav(i,1:index(i)),wav_x_new(i,j));
        end
    end
p1 = 1880;
[data,np,nodes,Nelayer]= shujuzhenhe_2D( wav_x,wav_y,wav_z,p_wav,vx_wav,vy_wav,vz_wav,p1,length(wav_x(:,1)),length(wav_x(1,:)));       
fprintf(fid,strcat('ZONE\t','N=',num2str(nodes),',E=',num2str(Nelayer),',DATAPACKING=POINT,ZONETYPE=FEQuadrilateral\n'));            
fprintf(fid,'%f %f %f %f %f %f %f\n',data');
fprintf(fid,' %d %d %d %d %d %d %d\n',np');
fid = fopen('wav_surface_1.dat','wt');
fprintf(fid,'TITLE="COMBINE_FIELD"\n');
fprintf(fid,'VARIABLES=x,y,z,p,vx,vy,vz\n');
[data,np,nodes,Nelayer]= shujuzhenhe_2D( wav_x_new,wav_y_new,wav_z_new,p_wav_new,vx_wav_new,vy_wav_new,vz_wav_new,p1,length(wav_x_new(:,1)),length(wav_x_new(1,:)));       
fprintf(fid,strcat('ZONE\t','N=',num2str(nodes),',E=',num2str(Nelayer),',DATAPACKING=POINT,ZONETYPE=FEQuadrilateral\n'));            
fprintf(fid,'%f %f %f %f %f %f %f\n',data');
fprintf(fid,' %d %d %d %d %d %d %d\n',np');
    for i = 1:length(x_inter(1,1,:))
        wav_x_new(i,:) = linspace(wav_x(i,1),7,n);
        for j = 1:n
            wav_y_new(i,j) = interp1(wav_x(i,1:index(i)),wav_y(i,1:index(i)),wav_x_new(i,j));
            wav_z_new(i,j) = wav_z(i,1);
             p_wav_new(i,j) = 1;
             vx_wav_new(i,j) = interp1(wav_x(i,1:index(i)),vx_wav(i,1:index(i)),wav_x_new(i,j));
             vy_wav_new(i,j) = interp1(wav_x(i,1:index(i)),vy_wav(i,1:index(i)),wav_x_new(i,j));
             vz_wav_new(i,j) = interp1(wav_x(i,1:index(i)),vz_wav(i,1:index(i)),wav_x_new(i,j));
        end
    end
fid = fopen('wav_surface_2.dat','wt');
fprintf(fid,'TITLE="COMBINE_FIELD"\n');
fprintf(fid,'VARIABLES=x,y,z,p,vx,vy,vz\n');
[data,np,nodes,Nelayer]= shujuzhenhe_2D( wav_x_new,wav_y_new,wav_z_new,p_wav_new,vx_wav_new,vy_wav_new,vz_wav_new,p1,length(wav_x_new(:,1)),length(wav_x_new(1,:)));       
fprintf(fid,strcat('ZONE\t','N=',num2str(nodes),',E=',num2str(Nelayer),',DATAPACKING=POINT,ZONETYPE=FEQuadrilateral\n'));            
fprintf(fid,'%f %f %f %f %f %f %f\n',data');
fprintf(fid,' %d %d %d %d %d %d %d\n',np');