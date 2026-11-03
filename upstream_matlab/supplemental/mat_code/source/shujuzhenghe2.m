function [data_xingxian]=shujuzhenghe2(x_y_z_xingxian,x_y_z_xingxian2,M)
    for i = 1:length(y_fct)
        for j = 1:length(x_y_z_xingxian(:,1,1))
            X(i,j) =  x_y_z_xingxian(j,1,i);
            Y(i,j) =  x_y_z_xingxian(j,2,i);
            Z(i,j) =  x_y_z_xingxian(j,3,i);
        end
    end   
    data_xingxian(1,2,:) = linspace(y_fct(1,1)+0.03,y_fct(1,end)-0.03,50);
    for i = 1:50
        data_xingxian(1,1,i) = interp1(y_fct,x_fct,data_xingxian(1,2,i));
        data_xingxian(1,3,i) = interp1(y_fct,z_fct,data_xingxian(1,2,i));
    end
    for i = 1:50
        data_xingxian(1:50,1,i) = linspace(data_xingxian(1,1,i),x_y_z_xingxian(1,1,1),50);
        data_xingxian(1:50,2,i) = data_xingxian(1,2,i);
        for j = 2:50
            data_xingxian(j,3,i) = griddata(X,Y,Z,data_xingxian(j,1,i),data_xingxian(j,2,i));
        end
    end

end