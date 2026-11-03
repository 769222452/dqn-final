function [V_eff] = vol_eff_sim(wav_x,wav_y,wav_z,wav_up_x,wav_up_y,wav_up_z)
    % 用于估算有效容积
    wav_x=flipud(wav_x); wav_y=flipud(wav_y); wav_z=flipud(wav_z); wav_z = -wav_z;
    wav_up_x=flipud(wav_up_x); wav_up_y=flipud(wav_up_y); wav_up_z=flipud(wav_up_z); wav_up_z=-wav_up_z;
    M = length(wav_x(:,1)); N=length(wav_x(1,:));
    % figure(1)
    % mesh(wav_x,wav_y,wav_z,'edgecolor','g','facecolor','none') 
    % axis equal
    % hold on
    % mesh(wav_up_x,wav_up_y,wav_up_z,'edgecolor','b','facecolor','none')
    % hold on
    % plot3(wav_x(:,1),wav_y(:,1),wav_z(:,1),'r')
    % hold on
    N_3D=50; D_3D=linspace(0.3,1,N_3D);
    for i=1:N_3D
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
        % figure(2)
        % plot(y_up_slice,z_up_slice,'b','LineWidth',1.5)
        % axis equal 
        % hold on
        % plot(y_slice,z_slice,'b','LineWidth',1.5)
        % hold on
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
        % plot(y,z,'r--','LineWidth',1.5)
        % hold on 
        % plot(tangent_point2(1),tangent_point2(2),'k.','LineWidth',1.5)
        % hold on
        a_slice(i)=a; b_slice(i)=b;
        S_slice(i)=pi*a*b;
    end
    V_eff=0;
    for i=1:N_3D-1
        V_eff=V_eff+(S_slice(i)+S_slice(i+1))*(D_3D(i+1)-D_3D(i))*(wav_x(1,end)-wav_x(1,1))/2;
    end
end