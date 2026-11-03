function [Fx,Fy,Fz]=force(x_y_z_xingxian,p_wav,D,M)
    n = 50;
    for i = 1:M
        xingxian(1:n,1,i) = linspace(x_y_z_xingxian(1,1,i),D,n);
        for j = 1:n
            xingxian(j,2,i) = interp1(x_y_z_xingxian(:,1,i),x_y_z_xingxian(:,2,i),xingxian(j,1,i));
            xingxian(j,3,i) = interp1(x_y_z_xingxian(:,1,i),x_y_z_xingxian(:,3,i),xingxian(j,1,i));
            P(i,j) = interp1(x_y_z_xingxian(:,1,i),p_wav(i,:),xingxian(j,1,i));
        end
    end
    for i = 1:M
        for j = 1:n
            x(i,j) = xingxian(j,1,i);
            y(i,j) = xingxian(j,2,i);
            z(i,j) = xingxian(j,3,i);
        end
    end
    surf(x,y,z,P);
    axis equal
    hold on
    Fx = 0; Fy = 0; Fz = 0; F = 0;
    for i = 1
        for j = 1:n-1
            point1 = [xingxian(j,1,i),xingxian(j,2,i),xingxian(j,3,i)];
            point2 = [xingxian(j,1,i+1),xingxian(j,2,i+1),xingxian(j,3,i+1)];
            point3 = [xingxian(j+1,1,i+1),xingxian(j+1,2,i+1),xingxian(j+1,3,i+1)];
            A = point2-point1;
            B = point3-point1;
            Cx = A(2)*B(3)-B(2)*A(3);
            Cy = A(3)*B(1)-B(3)*A(1);
            Cz = A(1)*B(2)-A(2)*B(1);
            C = [Cx,Cy,Cz];
            Module_C = sqrt(Cx^2+Cy^2+Cz^2);
            Aw = Module_C/2;
            P_avg = (P(i,j)+P(i+1,j)+P(i+1,j+1))/3;
            F = F+P_avg*Aw;
            Fx = Fx+P_avg*Aw*Cx/Module_C;
            Fy = Fy+P_avg*Aw*Cy/Module_C;
            Fz = Fz+P_avg*Aw*Cz/Module_C;
        end
    end
    for i = 2:M-2
        for j = 1:n-1
            %%%%第一个三角形%%%%%
            point1 = [xingxian(j,1,i),xingxian(j,2,i),xingxian(j,3,i)];
            point2 = [xingxian(j,1,i+1),xingxian(j,2,i+1),xingxian(j,3,i+1)];
            point3 = [xingxian(j+1,1,i+1),xingxian(j+1,2,i+1),xingxian(j+1,3,i+1)];
            A = point2-point1;
            B = point3-point1;
            Cx = A(2)*B(3)-B(2)*A(3);
            Cy = A(3)*B(1)-B(3)*A(1);
            Cz = A(1)*B(2)-A(2)*B(1);
            C = [Cx,Cy,Cz];
            Module_C = sqrt(Cx^2+Cy^2+Cz^2);
            Aw = Module_C/2;
            P_avg = (P(i,j)+P(i+1,j)+P(i+1,j+1))/3;
            F = F+P_avg*Aw;
            Fx = Fx+P_avg*Aw*Cx/Module_C;
            Fy = Fy+P_avg*Aw*Cy/Module_C;
            Fz = Fz+P_avg*Aw*Cz/Module_C;
            %%%%%第二个三角形%%%%%
            point1 = [xingxian(j,1,i),xingxian(j,2,i),xingxian(j,3,i)];
            point2 = [xingxian(j+1,1,i+1),xingxian(j+1,2,i+1),xingxian(j+1,3,i+1)];
            point3 = [xingxian(j+1,1,i),xingxian(j+1,2,i),xingxian(j+1,3,i)];
            A = point2-point1;
            B = point3-point1;
            Cx = A(2)*B(3)-B(2)*A(3);
            Cy = A(3)*B(1)-B(3)*A(1);
            Cz = A(1)*B(2)-A(2)*B(1);
            C = [Cx,Cy,Cz];
            Module_C = sqrt(Cx^2+Cy^2+Cz^2);
            Aw = Module_C/2;
            P_avg = (P(i,j)+P(i+1,j+1)+P(i,j+1))/3;
            F = F+P_avg*Aw;
            Fx = Fx+P_avg*Aw*Cx/Module_C;
            Fy = Fy+P_avg*Aw*Cy/Module_C;
            Fz = Fz+P_avg*Aw*Cz/Module_C;            
        end
    end
    for i = M-1
        for j = 1:n-1
            point1 = [xingxian(j,1,i),xingxian(j,2,i),xingxian(j,3,i)];
            point2 = [xingxian(j,1,i+1),xingxian(j,2,i+1),xingxian(j,3,i+1)];
            point3 = [xingxian(j+1,1,i),xingxian(j+1,2,i),xingxian(j+1,3,i)];
            A = point2-point1;
            B = point3-point1;
            Cx = A(2)*B(3)-B(2)*A(3);
            Cy = A(3)*B(1)-B(3)*A(1);
            Cz = A(1)*B(2)-A(2)*B(1);
            C = [Cx,Cy,Cz];
            Module_C = sqrt(Cx^2+Cy^2+Cz^2);
            Aw = Module_C/2;
            P_avg = (P(i,j)+P(i+1,j)+P(i,j+1))/3;
            F = F+P_avg*Aw;
            Fx = Fx+P_avg*Aw*Cx/Module_C;
            Fy = Fy+P_avg*Aw*Cy/Module_C;
            Fz = Fz+P_avg*Aw*Cz/Module_C;
        end
    end
end