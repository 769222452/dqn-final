clc
clear
load x_y_z_up x_y_z_up
load x_y_z_wav x_y_z_wav
M=length(x_y_z_wav(1,1,:)); N=length(x_y_z_wav(:,1,1));
for i = 1:M
    for j = 1:N
        wav_x(i,j) = x_y_z_wav(j,1,i);
        wav_y(i,j) = x_y_z_wav(j,2,i);
        wav_z(i,j) = -x_y_z_wav(j,3,i);
        wav_up_x(i,j) = x_y_z_up(j,1,i);
        wav_up_y(i,j) = x_y_z_up(j,2,i);
        wav_up_z(i,j) = -x_y_z_up(j,3,i);
    end
end
wav_x=flipud(wav_x); wav_y=flipud(wav_y); wav_z=flipud(wav_z);
wav_up_x=flipud(wav_up_x); wav_up_y=flipud(wav_up_y); wav_up_z=flipud(wav_up_z);
figure(1)
mesh(wav_x,wav_y,wav_z,'edgecolor','k','facecolor','none') 
axis equal
hold on
mesh(wav_up_x,wav_up_y,wav_up_z,'edgecolor','k','facecolor','none')
hold on
plot3(wav_x(:,1),wav_y(:,1),wav_z(:,1),'r')
hold on
% N_2D=20; D_2D=linspace(0.1,1,N_2D);
N_3D=50; D_3D=linspace(0.3,1,N_3D);
fid = fopen('volume_eff.dat','wt');
for i=1:N_3D
% for i=N_3D
    x_slice=[]; y_up_slice=[]; z_up_slice=[]; 
    index=[]; y_slice=[]; z_slice=[]; 
    x_slice=D_3D(i)*(wav_x(1,end)-wav_x(1,1));
    index=find(wav_x(:,1)>=x_slice,1);
    if isempty(index)==1
        index=M;
    end    
    for n=1:index-1
        y_up_slice(n)=interp1(wav_up_x(n,:),wav_up_y(n,:),x_slice);
        z_up_slice(n)=interp1(wav_up_x(n,:),wav_up_z(n,:),x_slice);
        y_slice(n)=interp1(wav_x(n,:),wav_y(n,:),x_slice);
        z_slice(n)=interp1(wav_x(n,:),wav_z(n,:),x_slice);
    end
    y_up_slice(index)=interp1(wav_up_x(:,1),wav_up_y(:,1),x_slice);
    z_up_slice(index)=interp1(wav_up_x(:,1),wav_up_z(:,1),x_slice);
    y_slice(index)=y_up_slice(index); z_slice(index)=z_up_slice(index);
    figure(2)
    plot(y_up_slice,z_up_slice,'k','LineWidth',1.5)
    axis equal 
    hold on
    plot(y_slice,z_slice,'k','LineWidth',1.5)
    hold on
    y_center=(y_slice(1)+y_up_slice(1))/2; z_center=(z_slice(1)+z_up_slice(1))/2;    
    h=y_center; k=z_center; b=z_up_slice(1)-z_center;
    points=[y_up_slice;z_up_slice]'; 
    [a1, tangent_point1] = find_ellipse_a(points, h, k, b);
    points=[]; points=[y_slice;z_slice]';
    [a2, tangent_point2] = find_ellipse_a(points, h, k, b);
    a=min(a1,a2);
    theta=linspace(-pi/2,pi/2,50);
    y=h+a*cos(theta);
    z=k+b*sin(theta);
    plot(y,z,'r--','LineWidth',1.5)
    hold on 
    plot(tangent_point2(1),tangent_point2(2),'k.','LineWidth',1.5)
    hold on
    a_slice(i)=a; b_slice(i)=b;
    S_slice(i)=pi*a*b;
    fprintf(fid,'ROW\n');
    for j=1:50
        fprintf(fid,'%e\t',x_slice*1000);
        fprintf(fid,'%e\t',y(j)*1000);
        fprintf(fid,'%e\n',z(j)*1000);
    end
end
fclose(fid);
V_eff=0;
for i=1:N_3D-1
    V_eff=V_eff+(S_slice(i)+S_slice(i+1))*(D_3D(i+1)-D_3D(i))*(wav_x(1,end)-wav_x(1,1))/2;
end
fid = fopen('semi-axis-length-Case7.dat','wt');
fprintf(fid,'TITLE = "Translation of CGNS file VCSW-TC1-entire-1st-03000.cgns"\n');
fprintf(fid,'Variables="<i>D</i><sub>3D</sub>"\n');
fprintf(fid,'"<i>a</i><sub>ep</sub>"\n');
fprintf(fid,'"<i>b</i><sub>ep</sub>"\n');
fprintf(fid,'"<i>S</i><sub>ep</sub>"\n');
fprintf(fid,'DATASETAUXDATA Common.DensityVar="6"\n');
fprintf(fid,'DATASETAUXDATA Common.MachNumberVar="7"\n');
fprintf(fid,'DATASETAUXDATA Common.PressureVar="4"\n');
fprintf(fid,'DATASETAUXDATA Common.TemperatureVar="11"\n');
fprintf(fid,'DATASETAUXDATA Common.UVar="7"\n');
fprintf(fid,'DATASETAUXDATA Common.VectorVarsAreVelocity="TRUE"\n');
fprintf(fid,'DATASETAUXDATA Common.VVar="8"\n');
fprintf(fid,'DATASETAUXDATA Common.WVar="9"\n');
fprintf(fid,'ZONE T="Extracted Points"\n');
fprintf(fid,' STRANDID=0, SOLUTIONTIME=0\n');
fprintf(fid,' I=%e, J=1, K=1, ZONETYPE=Ordered\n',length(a_slice));
fprintf(fid,' DATAPACKING=POINT\n');
fprintf(fid,' DT=(SINGLE SINGLE )\n');
for i=1:length(a_slice)
    fprintf(fid,'%e\t',D_3D(i));
    fprintf(fid,'%e\t',a_slice(i)/1.701);
    fprintf(fid,'%e\t',b_slice(i)/1.701);
    fprintf(fid,'%e\n',S_slice(i)/1.1102);
end
